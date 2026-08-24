#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <string>

using namespace weft;

namespace {

class CheckRISCVResourcesPass final
    : public mlir::PassWrapper<CheckRISCVResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckRISCVResourcesPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-check-resources";
  }
  llvm::StringRef getDescription() const final {
    return "check the selected candidate without revising earlier decisions";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;
      if (problem.getStage() != "schedule") {
        problem.emitError("resource checking requires a scheduled candidate");
        signalPassFailure();
        return;
      }
      llvm::StringMap<int64_t> first;
      llvm::StringMap<int64_t> last;
      llvm::StringMap<std::string> producers;
      llvm::StringMap<bool> deferredLocalResults;
      llvm::StringMap<llvm::SmallVector<int64_t, 4>> useOrdinals;
      llvm::StringSet<> pipelineFrontiers;
      for (auto [ordinal, attribute] : llvm::enumerate(problem.getOperations())) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name =
            riscv_internal::string(operation, "name").value_or("");
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute idAttribute : results) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(idAttribute).getValue();
            first.try_emplace(id, ordinal);
            last[id] = ordinal;
            producers[id] = name.str();
            if (riscv_internal::string(operation, "realization").value_or("") ==
                "rvv.outer.deferred-reduction-product")
              deferredLocalResults[id] = true;
          }
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute idAttribute : operands) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(idAttribute).getValue();
            first.try_emplace(id, ordinal);
            last[id] = ordinal;
            auto &uses = useOrdinals[id];
            if (uses.empty() || uses.back() != static_cast<int64_t>(ordinal))
              uses.push_back(ordinal);
          }
        if (auto cluster =
                operation.getAs<mlir::DictionaryAttr>("local_cluster")) {
          int64_t depth =
              riscv_internal::integer(cluster, "pipeline_depth").value_or(1);
          if (depth > 1)
            if (auto frontier =
                    cluster.getAs<mlir::ArrayAttr>("frontier_values"))
              for (mlir::Attribute value : frontier)
                pipelineFrontiers.insert(
                    mlir::cast<mlir::StringAttr>(value).getValue());
        }
      }

      llvm::SmallVector<mlir::DictionaryAttr> values;
      llvm::StringMap<std::string> valueClasses;
      llvm::StringMap<std::string> resourceClassParents;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(value, "id");
        llvm::StringRef handoff =
            riscv_internal::string(value, "handoff_class").value_or(id);
        valueClasses[id] = handoff.str();
        resourceClassParents.try_emplace(handoff, handoff.str());
        int64_t begin = first.contains(id) ? first.lookup(id) : 0;
        int64_t end = last.contains(id) ? last.lookup(id) : begin;
        value = riscv_internal::set(
            value, "live_start", builder.getI64IntegerAttr(begin));
        value = riscv_internal::set(
            value, "live_end", builder.getI64IntegerAttr(end));
        if (deferredLocalResults.lookup(id)) {
          value = riscv_internal::set(
              value, "register_groups", builder.getI64IntegerAttr(0));
          value = riscv_internal::set(
              value, "storage",
              builder.getStringAttr("deferred-operation-cluster"));
          value = riscv_internal::set(
              value, "materialization",
              builder.getStringAttr("deferred-to-single-consumer"));
          value = riscv_internal::set(
              value, "materialization_derived_from",
              builder.getStringAttr(
                  "selected outer-contract plus accumulator-add local cluster"));
        }
        if (riscv_internal::string(value, "materialization").value_or("") ==
            "pending-liveness") {
          bool multipleUses = useOrdinals.lookup(id).size() > 1;
          value = riscv_internal::set(
              value, "materialization",
              builder.getStringAttr(multipleUses ? "shared-register"
                                                  : "reload-per-use"));
          value = riscv_internal::set(
              value, "materialization_derived_from",
              builder.getStringAttr(
                  multipleUses
                      ? "live interval initially shared; resource peak may spill it"
                      : "single memory consumer does not require a persistent register"));
        }
        values.push_back(value);
      }

      auto findResourceClass = [&](llvm::StringRef value) {
        std::string root = value.str();
        while (resourceClassParents.lookup(root) != root)
          root = resourceClassParents.lookup(root);
        return root;
      };
      auto uniteResourceClasses = [&](llvm::StringRef lhs, llvm::StringRef rhs) {
        std::string lhsRoot = findResourceClass(lhs);
        std::string rhsRoot = findResourceClass(rhs);
        if (lhsRoot != rhsRoot)
          resourceClassParents[rhsRoot] = lhsRoot;
      };
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        auto local = operation.getAs<mlir::DictionaryAttr>("local_operation");
        auto tiedOperand =
            local ? local.getAs<mlir::IntegerAttr>("tied_operand")
                  : mlir::IntegerAttr();
        auto tiedResult =
            local ? local.getAs<mlir::IntegerAttr>("tied_result")
                  : mlir::IntegerAttr();
        auto operands = operation.getAs<mlir::ArrayAttr>("operands");
        auto results = operation.getAs<mlir::ArrayAttr>("results");
        if (!tiedOperand || !tiedResult || !operands || !results ||
            tiedOperand.getInt() < 0 || tiedResult.getInt() < 0 ||
            tiedOperand.getInt() >= static_cast<int64_t>(operands.size()) ||
            tiedResult.getInt() >= static_cast<int64_t>(results.size()))
          continue;
        llvm::StringRef operandId =
            mlir::cast<mlir::StringAttr>(operands[tiedOperand.getInt()])
                .getValue();
        llvm::StringRef resultId =
            mlir::cast<mlir::StringAttr>(results[tiedResult.getInt()]).getValue();
        uniteResourceClasses(valueClasses.lookup(operandId),
                             valueClasses.lookup(resultId));
      }
      for (mlir::DictionaryAttr &value : values) {
        llvm::StringRef id = *riscv_internal::string(value, "id");
        value = riscv_internal::set(
            value, "resource_class",
            builder.getStringAttr(findResourceClass(valueClasses.lookup(id))));
      }

      auto targetFragments =
          problem.getTarget().getAs<mlir::ArrayAttr>("matrix_fragments");
      constexpr int64_t reserved = 2;
      int64_t budget =
          *riscv_internal::integer(problem.getTarget(), "vector_registers");

      struct Peak {
        int64_t groups = 2;
        int64_t ordinal = -1;
        int64_t fragmentGroups = 0;
        int64_t temporaryGroups = 0;
        llvm::SmallVector<std::string> classes;
      };
      auto measurePeak = [&]() {
        Peak peak;
        for (int64_t ordinal = 0;
             ordinal < static_cast<int64_t>(problem.getOperations().size());
             ++ordinal) {
          llvm::StringMap<int64_t> live;
          for (mlir::DictionaryAttr value : values) {
            int64_t begin = *riscv_internal::integer(value, "live_start");
            int64_t end = *riscv_internal::integer(value, "live_end");
            int64_t groups =
                riscv_internal::integer(value, "register_groups").value_or(0);
            if (!groups)
              continue;
            llvm::StringRef id = *riscv_internal::string(value, "id");
            llvm::StringRef materialization =
                riscv_internal::string(value, "materialization").value_or("");
            bool liveNow = ordinal >= begin && ordinal <= end;
            if (materialization == "reload-per-use" ||
                materialization == "rematerialize-per-register-part")
              liveNow = llvm::is_contained(useOrdinals.lookup(id), ordinal);
            if (!liveNow)
              continue;
            llvm::StringRef resourceClass =
                riscv_internal::string(value, "resource_class").value_or(id);
            live[resourceClass] = std::max(live.lookup(resourceClass), groups);
          }
          int64_t total = reserved;
          for (const auto &entry : live)
            total += entry.getValue();
          int64_t fragmentGroups = 0;
          int64_t temporaryGroups = 0;
          auto operation = mlir::cast<mlir::DictionaryAttr>(
              problem.getOperations()[ordinal]);
          llvm::StringRef realization =
              riscv_internal::string(operation, "realization").value_or("");
          if (realization.consume_front("matrix.") && targetFragments)
            for (mlir::Attribute fragmentAttribute : targetFragments) {
              auto fragment =
                  mlir::cast<mlir::DictionaryAttr>(fragmentAttribute);
              if (riscv_internal::string(fragment, "identity").value_or("") ==
                  realization)
                fragmentGroups +=
                    riscv_internal::integer(fragment, "fixed_resource_groups")
                        .value_or(0);
            }
          if (auto local =
                  operation.getAs<mlir::DictionaryAttr>("local_operation"))
            temporaryGroups =
                riscv_internal::integer(local, "temporary_vector_groups")
                    .value_or(0);
          total += fragmentGroups + temporaryGroups;
          if (total <= peak.groups)
            continue;
          peak.groups = total;
          peak.ordinal = ordinal;
          peak.fragmentGroups = fragmentGroups;
          peak.temporaryGroups = temporaryGroups;
          peak.classes.clear();
          for (const auto &entry : live)
            peak.classes.push_back(entry.getKey().str() + ":" +
                                   std::to_string(entry.getValue()));
          if (temporaryGroups)
            peak.classes.push_back("primitive-temporaries:" +
                                   std::to_string(temporaryGroups));
          llvm::sort(peak.classes);
        }
        return peak;
      };

      Peak peak = measurePeak();
      llvm::SmallVector<std::string> reloaded;
      llvm::SmallVector<std::string> rematerialized;
      auto isPureRematerializable = [](llvm::StringRef producer) {
        return producer == "weft_kernel.iota" ||
               producer == "weft_kernel.extract" ||
               producer == "weft_kernel.lookup" ||
               producer == "weft_kernel.unary" ||
               producer == "weft_kernel.binary" ||
               producer == "weft_kernel.cast" ||
               producer == "weft_kernel.widen" ||
               producer == "weft_kernel.narrow";
      };
      while (peak.groups > budget) {
        int64_t bestScore = -1;
        size_t bestIndex = values.size();
        bool rematerialize = false;
        for (auto [index, value] : llvm::enumerate(values)) {
          llvm::StringRef id = *riscv_internal::string(value, "id");
          llvm::StringRef materialization =
              riscv_internal::string(value, "materialization").value_or("");
          std::string producer = producers.lookup(id);
          const bool reload = producer == "weft_kernel.admit" &&
                              materialization == "shared-register" &&
                              !pipelineFrontiers.contains(id);
          const int64_t perPart =
              riscv_internal::integer(value, "register_groups_per_part")
                  .value_or(0);
          const int64_t groups =
              riscv_internal::integer(value, "register_groups").value_or(0);
          const bool recompute = isPureRematerializable(producer) &&
                                 !pipelineFrontiers.contains(id) &&
                                 materialization !=
                                     "rematerialize-per-register-part" &&
                                 groups > perPart && perPart > 0 &&
                                 !useOrdinals.lookup(id).empty();
          if (!reload && !recompute)
            continue;
          int64_t span = *riscv_internal::integer(value, "live_end") -
                         *riscv_internal::integer(value, "live_start") + 1;
          int64_t score = (groups - (recompute ? perPart : 0)) * span;
          if (score > bestScore) {
            bestScore = score;
            bestIndex = index;
            rematerialize = recompute;
          }
        }
        if (bestIndex == values.size())
          break;
        mlir::DictionaryAttr value = values[bestIndex];
        llvm::StringRef id = *riscv_internal::string(value, "id");
        if (rematerialize) {
          rematerialized.push_back(id.str());
          value = riscv_internal::set(
              value, "materialization",
              builder.getStringAttr("rematerialize-per-register-part"));
          value = riscv_internal::set(
              value, "register_groups",
              builder.getI64IntegerAttr(
                  riscv_internal::integer(value, "register_groups_per_part")
                      .value_or(0)));
          value = riscv_internal::set(
              value, "storage", builder.getStringAttr("deferred-expression"));
          value = riscv_internal::set(
              value, "materialization_derived_from",
              builder.getStringAttr(
                  "pure computed aggregate exceeds the live register budget; emit one register coordinate at each use"));
        } else {
          reloaded.push_back(id.str());
          value = riscv_internal::set(
              value, "materialization",
              builder.getStringAttr("reload-per-use"));
          value = riscv_internal::set(
              value, "materialization_derived_from",
              builder.getStringAttr(
                  "live-range peak exceeded target registers; reload source at uses"));
        }
        values[bestIndex] = value;
        peak = measurePeak();
      }

      auto resources = riscv_internal::dictionary(
          builder,
          {{"vector_register_budget", builder.getI64IntegerAttr(budget)},
           {"peak_vector_groups", builder.getI64IntegerAttr(peak.groups)},
           {"peak_live_classes", riscv_internal::strings(builder, peak.classes)},
           {"peak_operation_ordinal", builder.getI64IntegerAttr(peak.ordinal)},
           {"peak_fragment_groups",
            builder.getI64IntegerAttr(peak.fragmentGroups)},
           {"peak_temporary_groups",
            builder.getI64IntegerAttr(peak.temporaryGroups)},
           {"reserved_vector_groups", builder.getI64IntegerAttr(reserved)},
           {"spill", builder.getStringAttr(reloaded.empty()
                                               ? (rematerialized.empty()
                                                      ? "none"
                                                      : "rematerialize-computed-values")
                                               : (rematerialized.empty()
                                                      ? "reload-admitted-values"
                                                      : "reload-and-rematerialize"))},
           {"reloaded_values", riscv_internal::strings(builder, reloaded)},
           {"rematerialized_values",
            riscv_internal::strings(builder, rematerialized)},
           {"stack_bytes", builder.getI64IntegerAttr(0)},
           {"cost", builder.getI64IntegerAttr(peak.groups)},
           {"decision_owner", builder.getStringAttr("global-resource-check")}});
      llvm::SmallVector<mlir::Attribute> valueAttributes(values.begin(),
                                                          values.end());
      problem.setValuesAttr(builder.getArrayAttr(valueAttributes));
      problem.setResourcesAttr(resources);
      if (peak.groups > budget) {
        std::string reason =
            "selected entity decisions require " +
            std::to_string(peak.groups) + " vector groups at operation " +
            std::to_string(peak.ordinal) + " but the target budget is " +
            std::to_string(budget);
        if (!peak.classes.empty()) {
          reason += ": ";
          for (auto [index, value] : llvm::enumerate(peak.classes)) {
            if (index)
              reason += ", ";
            reason += value;
          }
        }
        if (!rematerialized.empty()) {
          reason += "; rematerialized computed values: ";
          for (auto [index, value] : llvm::enumerate(rematerialized)) {
            if (index)
              reason += ", ";
            reason += value;
          }
        }
        reason += "; live values: ";
        bool firstValue = true;
        for (mlir::DictionaryAttr value : values) {
          int64_t begin = *riscv_internal::integer(value, "live_start");
          int64_t end = *riscv_internal::integer(value, "live_end");
          if (peak.ordinal < begin || peak.ordinal > end)
            continue;
          llvm::StringRef id = *riscv_internal::string(value, "id");
          int64_t groups =
              riscv_internal::integer(value, "register_groups").value_or(0);
          if (!groups)
            continue;
          if (!firstValue)
            reason += ", ";
          firstValue = false;
          reason += id.str() + "=" + producers.lookup(id) + "/" +
                    riscv_internal::string(value, "materialization")
                        .value_or("")
                        .str() +
                    "/" + std::to_string(groups) + "g/" +
                    std::to_string(riscv_internal::integer(
                                       value, "register_groups_per_part")
                                       .value_or(0)) +
                    "gpp/" + std::to_string(useOrdinals.lookup(id).size()) +
                    "uses";
        }
        resources = riscv_internal::set(
            resources, "invalid_reason",
            builder.getStringAttr(reason));
        problem.setResourcesAttr(resources);
        problem.setStageAttr(builder.getStringAttr("invalid"));
      } else {
        problem.setStageAttr(builder.getStringAttr("resources"));
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createCheckRISCVResourcesPass() {
  return std::make_unique<CheckRISCVResourcesPass>();
}
