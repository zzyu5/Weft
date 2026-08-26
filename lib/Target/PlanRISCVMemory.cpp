#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <numeric>
#include <optional>

using namespace weft;

namespace {

riscv::AccessAttr makeAccess(mlir::Builder &builder, llvm::StringRef form,
                             llvm::StringRef mapping, int64_t alignment,
                             int64_t group = 0, int64_t layer = 0,
                             int64_t joinFields = 0, int64_t joinLowBits = 0,
                             int64_t joinRole = 0,
                             int64_t bitOffset = 0, int64_t storageBits = 0,
                             llvm::StringRef order = "none") {
  return riscv::AccessAttr::get(
      builder.getContext(), form, mapping, std::max<int64_t>(1, alignment),
      form == "indexed" ? 32 : 0, form == "segment" ? 2 : 0, group,
      layer, joinFields, joinLowBits, joinRole, bitOffset, storageBits, order);
}

riscv::LeafAttr transferLeaf(mlir::Builder &builder, llvm::StringRef family,
                             llvm::StringRef instruction) {
  return riscv_internal::leaf(builder, "transfer", family, instruction,
                              instruction, 0, 0);
}

mlir::LogicalResult verifyAccessCapability(mlir::Operation *operation,
                                           riscv::AccessAttr access) {
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  if (!kernel)
    return operation->emitError("physical memory edge is outside a RISC-V kernel");
  auto target = kernel.getTarget();
  if (access.getForm() == "indexed" && !target.getHasIndexedMemory())
    return operation->emitError(
        "selected indexed memory form is unsupported by the target profile");
  if (access.getForm() == "segment" && !target.getHasSegmentMemory())
    return operation->emitError(
        "selected segment memory form is unsupported by the target profile");
  return mlir::success();
}

bool laneTraversesRecords(riscv::FieldOp operation, mlir::Type physicalType) {
  auto field = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto physical = mlir::dyn_cast<riscv::ValueType>(physicalType);
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
  if (!field || !physical || physical.getLayout().getCarrier() != "rvv" ||
      facts.logicalRank < 0 ||
      facts.logicalRank > static_cast<int64_t>(field.getShape().size()))
    return false;
  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(physical.getAxisIds().asArrayRef(),
                 physical.getLayout().getLaneFactors().asArrayRef()))
    if (factor > 1) {
      if (laneAxis)
        return false;
      laneAxis = axis;
    }
  if (!laneAxis)
    return false;
  const int64_t outerRank =
      static_cast<int64_t>(field.getShape().size()) - facts.logicalRank;
  return llvm::is_contained(
      field.getAxisIds().asArrayRef().take_front(outerRank), laneAxis);
}

riscv::AccessAttr fieldAccess(mlir::Builder &builder,
                              riscv::FieldOp operation,
                              mlir::Type physicalType) {
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
  llvm::StringRef form = facts.mapping == "natural" ? "unit" : "indexed";
  if ((facts.mapping == "natural" && facts.scalarPerRecord) ||
      laneTraversesRecords(operation, physicalType))
    form = "strided";
  return makeAccess(builder, form, facts.mapping, facts.alignment, facts.group,
                    facts.layer, facts.joinFields, facts.joinLowBits,
                    facts.joinRole, facts.bitOffset, facts.storageBits,
                    facts.order);
}

class PlanRISCVMemoryPass
    : public mlir::PassWrapper<PlanRISCVMemoryPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-plan-memory";
  }
  llvm::StringRef getDescription() const override {
    return "Select and materialize memory forms from descriptors and value layouts";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    bool failed = false;
    llvm::SmallVector<riscv::LoadOp> deadTableLoads;
    llvm::SmallVector<riscv::MaterializeOp> deadTableMaterializations;
    getOperation().walk([&](riscv::LoadOp operation) {
      auto memory = operation.getRegion().getType();
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      llvm::StringRef mapping = encoding.getKind() == "dense" ? "dense" : "opaque";
      auto access = encoding.getKind() == "dense"
                        ? riscv_internal::denseAccess(
                              builder, memory, operation.getResult().getType())
                        : makeAccess(builder, "unit", mapping,
                                     memory.getAlignment());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      llvm::StringRef form = access.getForm();
      operation.setAccessAttr(access);
      operation.setLeafAttr(
          transferLeaf(builder, "load", ("rvv.load." + form).str()));
    });
    getOperation().walk([&](riscv::StoreOp operation) {
      auto memory = operation.getRegion().getType();
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      llvm::StringRef mapping = encoding.getKind() == "dense" ? "dense" : "opaque";
      auto access = encoding.getKind() == "dense"
                        ? riscv_internal::denseAccess(
                              builder, memory, operation.getValue().getType())
                        : makeAccess(builder, "unit", mapping,
                                     memory.getAlignment());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      llvm::StringRef form = access.getForm();
      operation.setAccessAttr(access);
      auto value =
          mlir::dyn_cast<riscv::ValueType>(operation.getValue().getType());
      const bool scalar = !value || value.getLayout().getCarrier() == "scalar";
      operation.setLeafAttr(transferLeaf(
          builder, "store",
          scalar ? "scalar.store" : ("rvv.store." + form).str()));
    });
    getOperation().walk([&](riscv::FieldOp operation) {
      riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
      if (facts.mapping == "opaque") {
        operation.emitError("encoded field has no complete storage mapping");
        failed = true;
        return;
      }
      auto access =
          fieldAccess(builder, operation, operation.getResult().getType());
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      if (access.getForm() == "indexed" && !target.getHasIndexedMemory()) {
        operation.emitError(
            "encoded field requires indexed memory unsupported by the target profile");
        failed = true;
        return;
      }
      operation.setAccessAttr(access);
      operation.setLeafAttr(transferLeaf(
          builder, "encoded-field", ("rvv.encoded." + facts.mapping).str()));
    });
    getOperation().walk([&](riscv::ExtractOp operation) {
      riscv::AccessAttr access;
      auto field = riscv_internal::sourceField(operation.getInput());
      auto inputLayout = riscv_internal::layoutOf(operation.getInput().getType());
      const bool gather = llvm::any_of(
          operation.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                   "gather";
          });
      if (gather && inputLayout && inputLayout.getCarrier() == "rvv") {
        operation.setAccessAttr(makeAccess(builder, "register", "natural", 1));
        operation.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "extract", "rvv.extract.vrgather",
            "rvv.extract.vrgather", 0, 0));
        return;
      }
      // Encoded fields use the local carrier because their bytes are not yet a
      // numerical register value.  They nevertheless retain an encoded memory
      // edge selected on FieldOp; do not misclassify that representation as a
      // compiler-created local array.
      if (!field && inputLayout && inputLayout.getCarrier() == "local") {
        if (gather) {
          auto target =
              operation->getParentOfType<riscv::KernelOp>().getTarget();
          if (!target.getHasIndexedMemory()) {
            operation.emitError(
                "local vector gather is unsupported by the target profile");
            failed = true;
            return;
          }
          operation.setAccessAttr(makeAccess(builder, "indexed", "dense", 1));
          operation.setLeafAttr(riscv_internal::leaf(
              builder, "rvv", "local-gather", "rvv.local-gather",
              "rvv.local-gather", 0, 0));
        } else {
          operation.setAccessAttr(makeAccess(builder, "local", "dense", 1));
          operation.setLeafAttr(transferLeaf(builder, "local-extract",
                                             "local.extract"));
        }
        return;
      }
      if (field)
        access = field.getAccess();
      else
        access = makeAccess(builder, "unit", "dense", 1);
      llvm::StringRef form = access.getForm();
      // A shaped gather selector is a real indexed access irrespective of the
      // producer's storage mapping.
      if (llvm::any_of(operation.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() == "gather";
          }))
        form = "indexed";
      if (field && laneTraversesRecords(field, operation.getResult().getType()))
        form = "strided";
      access = makeAccess(builder, form, access.getMapping(),
                          access.getAlignment(), access.getGroupSize(),
                          access.getLayerSize(), access.getJoinFields(),
                          access.getJoinLowBits(), access.getJoinRole(),
                          access.getBitOffset(), access.getStorageBits(),
                          access.getOrder());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation.setAccessAttr(access);
      operation.setLeafAttr(transferLeaf(
          builder, "extract", ("rvv.extract." + form).str()));
    });
    getOperation().walk([&](riscv::ConvertLayoutOp operation) {
      auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      if (!result || result.getLayout().getCarrier() != "rvv")
        return;

      riscv::AccessAttr access;
      mlir::Value source = riscv_internal::stripRepresentationConversions(
          operation.getInput());
      if (auto extract = source.getDefiningOp<riscv::ExtractOp>()) {
        if (!riscv_internal::sourceField(extract.getInput()))
          return;
        access = extract.getAccess();
      } else if (auto field = source.getDefiningOp<riscv::FieldOp>()) {
        access = fieldAccess(builder, field, operation.getResult().getType());
      } else {
        return;
      }
      if (access.getForm() != "unit" && access.getForm() != "strided" &&
          access.getForm() != "indexed" && access.getForm() != "segment")
        return;
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation->setAttr("source_access", access);
    });
    getOperation().walk([&](riscv::Fold2Op operation) {
      auto field = riscv_internal::sourceField(operation.getInput());
      if (!field || field.getAccess().getMapping() != "natural" ||
          field.getAccess().getBitOffset() % 8 ||
          riscv_internal::logicalBitWidth(operation.getInput().getType()) != 16) {
        operation.emitError(
            "pair fold requires one byte-aligned natural i16 encoded field edge");
        failed = true;
        return;
      }
      operation.setAccessAttr(field.getAccess());
    });
    getOperation().walk([&](riscv::UpdateOp operation) {
      auto inputLayout = riscv_internal::layoutOf(operation.getInput().getType());
      if (!inputLayout || inputLayout.getCarrier() != "local")
        return;
      operation.setLeafAttr(
          transferLeaf(builder, "local-update", "local.update"));
    });
    getOperation().walk([&](riscv::LookupOp operation) {
      // An unmaterialized admitted table remains an addressable memory edge.
      // Explicit materialize carries cross-use lifetime, so its selected RVV
      // value is consumed by a register gather instead of being reconstructed
      // as an indexed memory access at every use.
      auto tableLoad = operation.getTable().getDefiningOp<riscv::LoadOp>();
      auto tableMaterialize =
          operation.getTable().getDefiningOp<riscv::MaterializeOp>();
      if (!tableLoad && tableMaterialize)
        tableLoad =
            tableMaterialize.getInput().getDefiningOp<riscv::LoadOp>();
      auto indexLayout = riscv_internal::layoutOf(operation.getIndices().getType());
      bool indexed = indexLayout && indexLayout.getCarrier() == "rvv";
      auto tableLayout = riscv_internal::layoutOf(operation.getTable().getType());
      auto resultLayout = riscv_internal::layoutOf(operation.getResult().getType());
      auto tableParts = tableLayout
                            ? riscv_internal::staticProduct(
                                  tableLayout.getTimeFactors().asArrayRef())
                            : std::optional<int64_t>();
      auto tableReplicas = tableLayout
                               ? riscv_internal::staticProduct(
                                     tableLayout.getReplicaFactors().asArrayRef())
                               : std::optional<int64_t>();
      bool registerTable =
          indexed && tableLayout && resultLayout &&
          tableLayout.getCarrier() == "rvv" &&
          tableLayout.getSew() == resultLayout.getSew() &&
          tableLayout.getLmulEighths() == resultLayout.getLmulEighths() &&
          indexLayout.getSew() == resultLayout.getSew() &&
          indexLayout.getLmulEighths() == resultLayout.getLmulEighths() &&
          tableParts && *tableParts == 1 && tableReplicas &&
          *tableReplicas == 1;
      if (!registerTable && tableLoad) {
        operation->setOperand(0, tableLoad.getRegion());
        if (!llvm::is_contained(deadTableLoads, tableLoad))
          deadTableLoads.push_back(tableLoad);
        if (tableMaterialize &&
            !llvm::is_contained(deadTableMaterializations,
                                tableMaterialize))
          deadTableMaterializations.push_back(tableMaterialize);
      }
      if (registerTable) {
        operation.setAccessAttr(makeAccess(builder, "register", "natural", 1));
        operation.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "lookup", "rvv.vrgather", "rvv.vrgather", 0,
            0));
        return;
      }
      if (mlir::isa<riscv::ValueType>(operation.getTable().getType())) {
        operation.emitError(
            "materialized lookup table has no complete register or memory realization");
        failed = true;
        return;
      }
      llvm::StringRef form = indexed ? "indexed" : "unit";
      if (indexed &&
          !operation->getParentOfType<riscv::KernelOp>()
               .getTarget()
               .getHasIndexedMemory()) {
        operation.emitError(
            "vector lookup requires indexed memory unsupported by the target profile");
        failed = true;
        return;
      }
      operation.setAccessAttr(makeAccess(builder, form, "natural", 1));
      operation.setLeafAttr(riscv_internal::leaf(
          builder, indexed ? "rvv" : "scalar", "lookup",
          indexed ? "rvv.vluxei" : "scalar.lookup",
          indexed ? "rvv.vluxei" : "scalar.lookup", 0, 0));
    });
    for (riscv::MaterializeOp materialize : deadTableMaterializations)
      if (materialize && materialize.getResult().use_empty())
        materialize.erase();
    for (riscv::LoadOp load : deadTableLoads)
      if (load && load.getResult().use_empty())
        load.erase();
    getOperation().walk([&](mlir::Operation *operation) {
      if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                     riscv::OuterContractOp>(operation))
        return;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
      int64_t laneAxis = 0;
      if (result) {
        for (auto [axis, factor] :
             llvm::zip(result.getAxisIds().asArrayRef(),
                       result.getLayout().getLaneFactors().asArrayRef()))
          if (factor > 1) {
            laneAxis = axis;
            break;
          }
      }
      if (!laneAxis)
        if (auto over =
                operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
            over && !over.empty())
          laneAxis = over.asArrayRef().front();
      if (!laneAxis) {
        operation->emitError(
            "scalar contraction has no reduction axis for lane-memory planning");
        failed = true;
        return;
      }
      auto hasAxis = [&](mlir::Value value, int64_t axis) {
        return axis && llvm::is_contained(
                           riscv_internal::logicalAxes(value.getType()), axis);
      };
      unsigned operandIndex =
          hasAxis(operation->getOperand(1), laneAxis) ? 1 : 0;
      if (!laneAxis) {
        auto lhsLayout =
            riscv_internal::layoutOf(operation->getOperand(0).getType());
        auto rhsLayout =
            riscv_internal::layoutOf(operation->getOperand(1).getType());
        if (rhsLayout && rhsLayout.getCarrier() == "rvv" &&
            (!lhsLayout || lhsLayout.getCarrier() != "rvv"))
          operandIndex = 1;
      }
      operation->setAttr("lane_operand",
                         builder.getStringAttr(operandIndex ? "rhs" : "lhs"));
      auto accessFor = [&](mlir::Value value) {
        mlir::Operation *producer = value.getDefiningOp();
        while (auto conversion =
                   mlir::dyn_cast_or_null<riscv::ConvertLayoutOp>(producer))
          producer = conversion.getInput().getDefiningOp();
        return producer ? producer->getAttrOfType<riscv::AccessAttr>("access")
                        : riscv::AccessAttr();
      };
      riscv::AccessAttr access = accessFor(operation->getOperand(operandIndex));
      if (!access) {
        auto layout = riscv_internal::layoutOf(
            operation->getOperand(operandIndex).getType());
        if (layout && layout.getCarrier() == "rvv") {
          operation->setAttr("lane_memory_form",
                             builder.getStringAttr("register"));
          return;
        }
        operation->emitError(
            "selected contraction lane operand has neither memory access nor register representation");
        failed = true;
        return;
      }
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation->setAttr("lane_memory_form",
                         builder.getStringAttr(access.getForm()));
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createPlanRISCVMemoryPass() {
  return std::make_unique<PlanRISCVMemoryPass>();
}
