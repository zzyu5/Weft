#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

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
      for (auto [ordinal, attribute] : llvm::enumerate(problem.getOperations())) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute idAttribute : ids) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(idAttribute).getValue();
              first.try_emplace(id, ordinal);
              last[id] = ordinal;
            }
      }

      llvm::SmallVector<mlir::Attribute> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(value, "id");
        int64_t begin = first.contains(id) ? first.lookup(id) : 0;
        int64_t end = last.contains(id) ? last.lookup(id) : begin;
        value = riscv_internal::set(
            value, "live_start", builder.getI64IntegerAttr(begin));
        value = riscv_internal::set(
            value, "live_end", builder.getI64IntegerAttr(end));
        values.push_back(value);
      }

      auto targetFragments =
          problem.getTarget().getAs<mlir::ArrayAttr>("matrix_fragments");
      constexpr int64_t reserved = 2;
      int64_t peak = reserved;
      int64_t peakOrdinal = -1;
      int64_t peakFragmentGroups = 0;
      llvm::SmallVector<std::string> peakClasses;
      for (int64_t ordinal = 0; ordinal <
                                static_cast<int64_t>(problem.getOperations().size());
           ++ordinal) {
        llvm::StringMap<int64_t> live;
        for (mlir::Attribute attribute : values) {
          auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
          int64_t begin = *riscv_internal::integer(value, "live_start");
          int64_t end = *riscv_internal::integer(value, "live_end");
          if (ordinal < begin || ordinal > end)
            continue;
          int64_t groups =
              riscv_internal::integer(value, "register_groups").value_or(0);
          if (!groups)
            continue;
          llvm::StringRef id = *riscv_internal::string(value, "id");
          llvm::StringRef handoff =
              riscv_internal::string(value, "handoff_class").value_or(id);
          live[handoff] = std::max(live.lookup(handoff), groups);
        }
        int64_t total = 0;
        for (const auto &entry : live)
          total += entry.getValue();
        int64_t fragmentGroups = 0;
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
        total += fragmentGroups + reserved;
        if (total > peak) {
          peak = total;
          peakOrdinal = ordinal;
          peakFragmentGroups = fragmentGroups;
          peakClasses.clear();
          for (const auto &entry : live)
            peakClasses.push_back(entry.getKey().str() + ":" +
                                  std::to_string(entry.getValue()));
          llvm::sort(peakClasses);
        }
      }
      int64_t budget =
          *riscv_internal::integer(problem.getTarget(), "vector_registers");
      auto resources = riscv_internal::dictionary(
          builder,
          {{"vector_register_budget", builder.getI64IntegerAttr(budget)},
           {"peak_vector_groups", builder.getI64IntegerAttr(peak)},
           {"peak_live_classes", riscv_internal::strings(builder, peakClasses)},
           {"peak_operation_ordinal", builder.getI64IntegerAttr(peakOrdinal)},
           {"peak_fragment_groups",
            builder.getI64IntegerAttr(peakFragmentGroups)},
           {"reserved_vector_groups", builder.getI64IntegerAttr(reserved)},
           {"spill", builder.getStringAttr("none")},
           {"stack_bytes", builder.getI64IntegerAttr(0)},
           {"cost", builder.getI64IntegerAttr(peak)},
           {"decision_owner", builder.getStringAttr("global-resource-check")}});
      problem.setValuesAttr(builder.getArrayAttr(values));
      problem.setResourcesAttr(resources);
      if (peak > budget) {
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
