#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
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
      llvm::StringMap<llvm::SmallVector<int64_t, 4>> useOrdinals;
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
      }

      llvm::SmallVector<mlir::DictionaryAttr> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(value, "id");
        int64_t begin = first.contains(id) ? first.lookup(id) : 0;
        int64_t end = last.contains(id) ? last.lookup(id) : begin;
        value = riscv_internal::set(
            value, "live_start", builder.getI64IntegerAttr(begin));
        value = riscv_internal::set(
            value, "live_end", builder.getI64IntegerAttr(end));
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
            if (materialization == "reload-per-use")
              liveNow = llvm::is_contained(useOrdinals.lookup(id), ordinal);
            if (!liveNow)
              continue;
            llvm::StringRef handoff =
                riscv_internal::string(value, "handoff_class").value_or(id);
            live[handoff] = std::max(live.lookup(handoff), groups);
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
      while (peak.groups > budget) {
        int64_t bestScore = -1;
        size_t bestIndex = values.size();
        for (auto [index, value] : llvm::enumerate(values)) {
          llvm::StringRef id = *riscv_internal::string(value, "id");
          if (producers.lookup(id) != "weft_kernel.admit" ||
              riscv_internal::string(value, "materialization").value_or("") !=
                  "shared-register")
            continue;
          int64_t groups =
              riscv_internal::integer(value, "register_groups").value_or(0);
          int64_t span = *riscv_internal::integer(value, "live_end") -
                         *riscv_internal::integer(value, "live_start") + 1;
          int64_t score = groups * span;
          if (score > bestScore) {
            bestScore = score;
            bestIndex = index;
          }
        }
        if (bestIndex == values.size())
          break;
        mlir::DictionaryAttr value = values[bestIndex];
        reloaded.push_back(riscv_internal::string(value, "id")->str());
        value = riscv_internal::set(
            value, "materialization",
            builder.getStringAttr("reload-per-use"));
        value = riscv_internal::set(
            value, "materialization_derived_from",
            builder.getStringAttr(
                "live-range peak exceeded target registers; reload source at uses"));
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
                                               ? "none"
                                               : "reload-admitted-values")},
           {"reloaded_values", riscv_internal::strings(builder, reloaded)},
           {"stack_bytes", builder.getI64IntegerAttr(0)},
           {"cost", builder.getI64IntegerAttr(peak.groups)},
           {"decision_owner", builder.getStringAttr("global-resource-check")}});
      llvm::SmallVector<mlir::Attribute> valueAttributes(values.begin(),
                                                          values.end());
      problem.setValuesAttr(builder.getArrayAttr(valueAttributes));
      problem.setResourcesAttr(resources);
      if (peak.groups > budget) {
        resources = riscv_internal::set(
            resources, "invalid_reason",
            builder.getStringAttr(
                "selected entity decisions exceed the vector-register budget"));
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
