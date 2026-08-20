#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <string>

using namespace weft;

namespace {

bool targetFlag(mlir::DictionaryAttr target, llvm::StringRef name) {
  if (auto value = target.getAs<mlir::BoolAttr>(name))
    return value.getValue();
  return false;
}

struct EncodingFacts {
  struct BaseLayout {
    std::string bitOrder;
    std::string byteOrder;
    int64_t alignment = 0;
    int64_t storageBits = 0;
  };
  llvm::StringMap<std::string> sourceByDerivedFamily;
  llvm::StringMap<std::string> packingByField;
  llvm::StringMap<BaseLayout> baseLayouts;
};

std::string fieldKey(llvm::StringRef family, llvm::StringRef field) {
  return (family + "::" + field).str();
}

EncodingFacts collectEncodingFacts(mlir::ModuleOp module) {
  EncodingFacts facts;
  module.walk([&](kernel::DeriveOp derive) {
    facts.sourceByDerivedFamily[derive.getResultFamily()] =
        derive.getSourceFamily().str();
  });
  module.walk([&](kernel::EncodingDeclOp declaration) {
    facts.baseLayouts[declaration.getSymName()] =
        {declaration.getBitOrder().str(), declaration.getByteOrder().str(),
         static_cast<int64_t>(declaration.getAlignment()),
         static_cast<int64_t>(declaration.getStorageBits())};
    for (auto [nameAttribute, packingAttribute] :
         llvm::zip_equal(declaration.getFieldNames(),
                         declaration.getFieldPacking())) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>(nameAttribute).getValue();
      llvm::StringRef packing =
          mlir::cast<mlir::StringAttr>(packingAttribute).getValue();
      facts.packingByField[fieldKey(declaration.getSymName(), name)] =
          packing.str();
    }
  });
  return facts;
}

llvm::StringRef resolveBaseFamily(const EncodingFacts &facts,
                                  llvm::StringRef family) {
  llvm::SmallSet<std::string, 4> visited;
  while (true) {
    auto found = facts.sourceByDerivedFamily.find(family);
    if (found == facts.sourceByDerivedFamily.end())
      break;
    if (!visited.insert(family.str()).second)
      break;
    family = found->second;
  }
  return family;
}

struct DerivedInterleave {
  std::string family;
  int64_t rows = 0;
};

llvm::SmallVector<DerivedInterleave>
collectDerivedInterleaves(mlir::ModuleOp module) {
  llvm::SmallVector<DerivedInterleave> result;
  module.walk([&](kernel::DeriveOp derive) {
    int64_t rows = 0;
    for (mlir::Attribute attribute : derive.getParameters()) {
      llvm::StringRef parameter =
          mlir::cast<mlir::StringAttr>(attribute).getValue();
      if (parameter.consume_front("rows="))
        parameter.getAsInteger(10, rows);
    }
    if (rows > 0)
      result.push_back({derive.getResultFamily().str(), rows});
  });
  return result;
}

llvm::SmallVector<std::string> matrixRealizations(
    mlir::DictionaryAttr target, mlir::DictionaryAttr operation,
    const llvm::StringMap<mlir::DictionaryAttr> &values) {
  auto operands = operation.getAs<mlir::ArrayAttr>("operands");
  auto results = operation.getAs<mlir::ArrayAttr>("results");
  if (!operands || operands.size() < 2 || !results || results.empty())
    return {};
  auto lookupSEW = [&](mlir::Attribute id) -> std::optional<int64_t> {
    auto found = values.find(mlir::cast<mlir::StringAttr>(id).getValue());
    if (found == values.end())
      return std::nullopt;
    return riscv_internal::integer(found->second, "logical_sew");
  };
  auto lhsSEW = lookupSEW(operands[0]);
  auto rhsSEW = lookupSEW(operands[1]);
  auto accumulatorSEW = lookupSEW(results[0]);
  if (!lhsSEW || !rhsSEW || !accumulatorSEW)
    return {};
  llvm::SmallVector<std::string> realizations;
  auto fragments = target.getAs<mlir::ArrayAttr>("matrix_fragments");
  for (mlir::Attribute attribute : fragments) {
    auto fragment = mlir::cast<mlir::DictionaryAttr>(attribute);
    if (*riscv_internal::integer(fragment, "lhs_sew") != *lhsSEW ||
        *riscv_internal::integer(fragment, "rhs_sew") != *rhsSEW ||
        *riscv_internal::integer(fragment, "accumulator_sew") !=
            *accumulatorSEW)
      continue;
    realizations.push_back(
        "matrix." + riscv_internal::string(fragment, "identity")->str());
  }
  return realizations;
}

class ConstrainRISCVInstructionsPass final
    : public mlir::PassWrapper<ConstrainRISCVInstructionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstrainRISCVInstructionsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-constrain-instructions";
  }
  llvm::StringRef getDescription() const final {
    return "couple decode, instruction, memory, broadcast and derived layout domains";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    EncodingFacts encodingFacts = collectEncodingFacts(module);
    llvm::SmallVector<DerivedInterleave> interleaves =
        collectDerivedInterleaves(module);
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() != "representations") {
        problem.emitError(
            "instruction constraints require a representations-stage problem");
        signalPassFailure();
        return;
      }
      bool hasProduct = false;
      bool hasNibbleField = false;
      llvm::SmallVector<
          std::pair<std::string, llvm::SmallVector<std::string>>>
          instructionDomains;
      llvm::StringSet<> usedEncodingFamilies;
      llvm::StringMap<mlir::DictionaryAttr> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        values[*riscv_internal::string(value, "id")] = value;
        if (auto family = value.getAs<mlir::StringAttr>("encoding_family"))
          usedEncodingFamilies.insert(family.getValue());
      }
      llvm::SmallVector<mlir::Attribute> plannedOperations;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name =
            mlir::cast<mlir::StringAttr>(operation.get("name")).getValue();
        llvm::StringRef engine =
            mlir::cast<mlir::StringAttr>(operation.get("engine")).getValue();
        llvm::StringRef operationId =
            mlir::cast<mlir::StringAttr>(operation.get("id")).getValue();
        bool isProduct = false;
        llvm::SmallVector<std::string> realizations;
        llvm::SmallVector<std::string> memoryForms;
        if (name == "weft_kernel.mac_pairs" ||
            name == "weft_kernel.mac_groups") {
          isProduct = true;
          hasProduct = true;
          if (engine == "wide" || engine.empty()) {
            if (targetFlag(problem.getTarget(), "has_widening_integer"))
              realizations.push_back("rvv.vwmaccsu");
          }
        } else if (name == "weft_kernel.dot" ||
                   name == "weft_kernel.contract" ||
                   name == "weft_kernel.outer_contract") {
          isProduct = true;
          hasProduct = true;
          if (engine == "wide" || engine.empty())
            realizations.push_back("rvv.reduction-product");
          if (engine == "matrix" || engine.empty()) {
            llvm::SmallVector<std::string> matrix = matrixRealizations(
                problem.getTarget(), operation, values);
            realizations.append(matrix.begin(), matrix.end());
          }
        } else if (name == "weft_kernel.admit" ||
                   name == "weft_kernel.commit") {
          realizations.push_back("transfer.rvv");
          memoryForms.push_back("unit-stride");
          memoryForms.push_back("strided");
        } else if (name == "weft_kernel.field") {
          realizations.push_back("encoded-field-view");
          auto sourceAttributes =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto fieldName = sourceAttributes
                               ? sourceAttributes.getAs<mlir::StringAttr>("name")
                               : mlir::StringAttr();
          auto ownerId = operands && !operands.empty()
                             ? mlir::dyn_cast<mlir::StringAttr>(operands[0])
                             : mlir::StringAttr();
          auto owner = ownerId ? values.find(ownerId.getValue()) : values.end();
          auto family = owner != values.end()
                            ? owner->second.getAs<mlir::StringAttr>(
                                  "encoding_family")
                            : mlir::StringAttr();
          if (!fieldName || !family) {
            problem.emitError(
                "encoded field lacks an owner encoding family or field name");
            signalPassFailure();
            return;
          }
          llvm::StringRef baseFamily =
              resolveBaseFamily(encodingFacts, family.getValue());
          auto packing = encodingFacts.packingByField.find(
              fieldKey(baseFamily, fieldName.getValue()));
          if (packing == encodingFacts.packingByField.end()) {
            problem.emitError()
                << "field '" << fieldName.getValue()
                << "' has no packing fact in encoding '" << baseFamily << "'";
            signalPassFailure();
            return;
          }
          operation = riscv_internal::set(
              operation, "field_packing",
              builder.getStringAttr(packing->second));
          llvm::StringRef packingSpelling = packing->second;
          if (packingSpelling.starts_with("nibble:"))
            hasNibbleField = true;
        } else if (name == "weft_kernel.widen") {
          realizations.push_back("rvv.widen");
        } else if (name == "weft_kernel.reduce") {
          realizations.push_back("rvv.reduce");
        } else if (name == "weft_kernel.fold2") {
          realizations.push_back("ordered-pair-fold");
        } else if (name == "weft_kernel.binary" ||
                   name == "weft_kernel.unary" ||
                   name == "weft_kernel.cast") {
          realizations.push_back("mapped-pointwise");
        } else if (name == "weft_kernel.for" || name == "weft_kernel.if" ||
                   name == "weft_kernel.while") {
          realizations.push_back("ordered-scalar-control");
        } else if (name == "weft_kernel.level") {
          realizations.push_back("logical-level");
        } else {
          realizations.push_back("structural");
        }
        if (isProduct && realizations.empty()) {
          problem.emitError()
              << "operation " << operationId << " with engine role '" << engine
              << "' has no legal target instruction";
          signalPassFailure();
          return;
        }
        if (isProduct) {
          bool requiresSubbyteUnpack = false;
          if (name == "weft_kernel.mac_pairs" ||
              name == "weft_kernel.mac_groups") {
            auto operands = operation.getAs<mlir::ArrayAttr>("operands");
            for (mlir::Attribute operandAttribute : operands) {
              auto operandId = mlir::cast<mlir::StringAttr>(operandAttribute);
              auto operand = values.find(operandId.getValue());
              if (operand != values.end() &&
                  riscv_internal::integer(operand->second, "logical_sew")
                          .value_or(8) < 8)
                requiresSubbyteUnpack = true;
            }
          }
          std::string instructionKey =
              (requiresSubbyteUnpack ? "instruction_subbyte."
                                     : "instruction.") +
              operationId.str();
          operation = riscv_internal::set(
              operation, "instruction_domain_key",
              builder.getStringAttr(instructionKey));
          instructionDomains.push_back({instructionKey, realizations});
        }
        operation = riscv_internal::set(
            operation, "realization_domain",
            riscv_internal::strings(builder, realizations));
        operation = riscv_internal::set(
            operation, "memory_form_domain",
            riscv_internal::strings(builder, memoryForms));
        plannedOperations.push_back(operation);
      }
      mlir::DictionaryAttr domains = problem.getDecisionDomains();
      auto laneAxes = domains.getAs<mlir::DenseI64ArrayAttr>("lane_axis");
      bool hasVectorLane = laneAxes && llvm::any_of(
                                            laneAxes.asArrayRef(),
                                            [](int64_t axis) { return axis > 0; });
      llvm::SmallVector<std::string> nibbleMethods{"none"};
      if (hasNibbleField) {
        nibbleMethods = {"and-shift"};
        if (targetFlag(problem.getTarget(), "has_indexed_memory"))
          nibbleMethods.push_back("indexed-gather");
      }
      llvm::SmallVector<std::string> interleaveOrders;
      llvm::StringSet<> usedInterleaves;
      for (const DerivedInterleave &interleave : interleaves) {
        if (!usedEncodingFamilies.contains(interleave.family))
          continue;
        usedInterleaves.insert(interleave.family);
        interleaveOrders.push_back(interleave.family + ".rows" +
                                   std::to_string(interleave.rows) +
                                   ".pair-major");
        interleaveOrders.push_back(interleave.family + ".rows" +
                                   std::to_string(interleave.rows) +
                                   ".row-major");
      }
      if (usedInterleaves.size() > 1) {
        problem.emitError(
            "one physical assignment currently supports one derived interleave family; multiple families are explicitly unsupported");
        signalPassFailure();
        return;
      }
      if (interleaveOrders.empty())
        interleaveOrders.push_back("not-applicable");
      domains = riscv_internal::set(
          domains, "nibble_unpack",
          riscv_internal::strings(builder, nibbleMethods));
      for (const auto &entry : instructionDomains)
        domains = riscv_internal::set(
            domains, entry.first,
            riscv_internal::strings(builder, entry.second));
      domains = riscv_internal::set(
          domains, "scale_broadcast",
          riscv_internal::strings(
              builder, hasProduct && hasVectorLane
                           ? llvm::SmallVector<std::string>{"lane-aligned",
                                                            "scalar-splat"}
                           : llvm::SmallVector<std::string>{"not-applicable"}));
      domains = riscv_internal::set(
          domains, "byte_interleave",
          riscv_internal::strings(builder, interleaveOrders));
      domains = riscv_internal::set(
          domains, "load_stride_alignment",
          riscv_internal::strings(builder,
                                  {"unit-stride", "strided"}));
      domains = riscv_internal::set(
          domains, "partial_layout",
          riscv_internal::strings(
              builder, hasProduct && hasVectorLane
                           ? llvm::SmallVector<std::string>{"group-major",
                                                            "lane-major"}
                           : llvm::SmallVector<std::string>{"not-applicable"}));
      domains = riscv_internal::set(
          domains, "horizontal_reduce",
          riscv_internal::strings(
              builder, hasProduct && hasVectorLane
                           ? llvm::SmallVector<std::string>{"streamed",
                                                            "end-of-sub"}
                           : llvm::SmallVector<std::string>{"not-applicable"}));

      llvm::SmallVector<std::string> constraints;
      for (mlir::Attribute attribute : problem.getConstraints())
        constraints.push_back(
            mlir::cast<mlir::StringAttr>(attribute).getValue().str());
      constraints.push_back(
          "mac-instruction <-> nibble-unpack <-> derived-byte-interleave");
      constraints.push_back(
          "partial-layout <-> horizontal-reduce <-> value-lmul");
      constraints.push_back(
          "load-form is derived from selected layout and logical axis relation");
      constraints.push_back(
          "explicit engine role is a hard domain restriction, never a hint");
      llvm::SmallVector<mlir::Attribute> layoutSources;
      for (const DerivedInterleave &interleave : interleaves) {
        if (!usedEncodingFamilies.contains(interleave.family))
          continue;
        llvm::StringRef baseFamily =
            resolveBaseFamily(encodingFacts, interleave.family);
        auto base = encodingFacts.baseLayouts.find(baseFamily);
        if (base == encodingFacts.baseLayouts.end()) {
          problem.emitError() << "derived encoding '" << interleave.family
                              << "' has no base layout facts";
          signalPassFailure();
          return;
        }
        layoutSources.push_back(riscv_internal::dictionary(
            builder,
            {{"family", builder.getStringAttr(interleave.family)},
             {"source_family", builder.getStringAttr(baseFamily)},
             {"rows", builder.getI64IntegerAttr(interleave.rows)},
             {"bit_order", builder.getStringAttr(base->second.bitOrder)},
             {"byte_order", builder.getStringAttr(base->second.byteOrder)},
             {"alignment", builder.getI64IntegerAttr(base->second.alignment)},
             {"base_record_storage_bits",
              builder.getI64IntegerAttr(base->second.storageBits)}}));
      }
      mlir::DictionaryAttr resourceModel = riscv_internal::set(
          problem.getResourceModel(), "derived_layout_sources",
          builder.getArrayAttr(layoutSources));
      problem.setOperationsAttr(builder.getArrayAttr(plannedOperations));
      problem.setDecisionDomainsAttr(domains);
      problem.setConstraintsAttr(riscv_internal::strings(builder, constraints));
      problem.setResourceModelAttr(resourceModel);
      problem.setStageAttr(builder.getStringAttr("instructions"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createConstrainRISCVInstructionsPass() {
  return std::make_unique<ConstrainRISCVInstructionsPass>();
}
