#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>

using namespace weft;

namespace {

llvm::StringRef layoutKind(mlir::ArrayAttr layouts) {
  if (!layouts || layouts.empty())
    return {};
  auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(layouts[0]);
  auto kind = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                         : mlir::StringAttr();
  return kind ? kind.getValue() : llvm::StringRef();
}

struct FieldFacts {
  llvm::StringRef mapping = "opaque";
  bool scalarPerRecord = false;
  int64_t group = 0;
  int64_t layer = 0;
  int64_t joinFields = 0;
  int64_t joinLowBits = 0;
  int64_t joinRole = 0;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
  llvm::StringRef order = "none";
};

FieldFacts fieldFacts(riscv::FieldOp operation) {
  auto ownerElement = riscv_internal::logicalElement(operation.getOwner().getType());
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(ownerElement);
  if (!encoding)
    return {};
  llvm::StringRef base = riscv_internal::baseEncodingFamily(operation, encoding);
  riscv::EncodingDeclOp declaration =
      riscv_internal::findEncoding(operation, base);
  if (!declaration)
    return {};
  size_t index = declaration.getFieldNames().size();
  for (auto [position, name] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(name).getValue() == operation.getName()) {
      index = position;
      break;
    }
  if (index == declaration.getFieldNames().size())
    return {};
  FieldFacts result;
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[index]);
  result.scalarPerRecord = fieldShape.empty();
  result.bitOffset = declaration.getFieldBitOffsets()[index];
  result.storageBits = declaration.getFieldStorageBits()[index];
  auto layouts = mlir::cast<mlir::ArrayAttr>(declaration.getFieldLayouts()[index]);
  llvm::StringRef kind = layoutKind(layouts);
  if (kind == "natural") {
    result.mapping = "natural";
  } else if (kind == "joined") {
    result.mapping = "joined";
    auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    result.group = joined.getAs<mlir::IntegerAttr>("size").getInt();
    result.joinFields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
    result.joinLowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
    result.joinRole = joined.getAs<mlir::IntegerAttr>("role").getInt();
    result.order = joined.getAs<mlir::StringAttr>("order").getValue();
  } else if (kind == "grouped" && layouts.size() == 2 &&
             layoutKind(mlir::ArrayAttr::get(operation.getContext(),
                                             {layouts[1]})) == "layered") {
    result.mapping = "grouped_layered";
    auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
    result.group = grouped.getAs<mlir::IntegerAttr>("size").getInt();
    result.layer = layered.getAs<mlir::IntegerAttr>("size").getInt();
    result.order = layered.getAs<mlir::StringAttr>("order").getValue();
  }
  return result;
}

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
      operation.setLeafAttr(
          transferLeaf(builder, "store", ("rvv.store." + form).str()));
    });
    getOperation().walk([&](riscv::FieldOp operation) {
      FieldFacts facts = fieldFacts(operation);
      if (facts.mapping == "opaque") {
        operation.emitError("encoded field has no complete storage mapping");
        failed = true;
        return;
      }
      llvm::StringRef form = facts.mapping == "natural" ? "unit" : "indexed";
      // A scalar field repeated once per encoded record is contiguous only
      // within a record.  Vectorizing an outer logical axis therefore requires
      // the record-byte stride selected from the owner descriptor.
      if (facts.mapping == "natural" && facts.scalarPerRecord)
        form = "strided";
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      if (form == "indexed" && !target.getHasIndexedMemory()) {
        operation.emitError(
            "encoded field requires indexed memory unsupported by the target profile");
        failed = true;
        return;
      }
      operation.setAccessAttr(makeAccess(
          builder, form, facts.mapping, 1, facts.group, facts.layer,
          facts.joinFields, facts.joinLowBits, facts.joinRole, facts.bitOffset,
          facts.storageBits, facts.order));
      operation.setLeafAttr(transferLeaf(
          builder, "encoded-field", ("rvv.encoded." + facts.mapping).str()));
    });
    getOperation().walk([&](riscv::ExtractOp operation) {
      riscv::AccessAttr access;
      auto field = operation.getInput().getDefiningOp<riscv::FieldOp>();
      auto inputLayout = riscv_internal::layoutOf(operation.getInput().getType());
      // Encoded fields use the local carrier because their bytes are not yet a
      // numerical register value.  They nevertheless retain an encoded memory
      // edge selected on FieldOp; do not misclassify that representation as a
      // compiler-created local array.
      if (!field && inputLayout && inputLayout.getCarrier() == "local") {
        const bool gather = llvm::any_of(
            operation.getSelectors(), [](mlir::Attribute selector) {
              return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                     "gather";
            });
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
    getOperation().walk([&](riscv::Fold2Op operation) {
      auto field = operation.getInput().getDefiningOp<riscv::FieldOp>();
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
      // A lookup table is an addressable memory value, not an RVV value that
      // must first be loaded in full.  Preserve the explicit descriptor edge
      // and let the lookup leaf load exactly the selected entries/window.
      if (auto tableLoad = operation.getTable().getDefiningOp<riscv::LoadOp>()) {
        operation->setOperand(0, tableLoad.getRegion());
        if (!llvm::is_contained(deadTableLoads, tableLoad))
          deadTableLoads.push_back(tableLoad);
      }
      auto indexLayout = riscv_internal::layoutOf(operation.getIndices().getType());
      bool indexed = indexLayout && indexLayout.getCarrier() == "rvv";
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
    for (riscv::LoadOp load : deadTableLoads)
      if (load && load.getResult().use_empty())
        load.erase();
    getOperation().walk([&](mlir::Operation *operation) {
      if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                     riscv::OuterContractOp>(operation))
        return;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
      if (!result) {
        operation->emitError(
            "contraction result has no physical value for lane-memory planning");
        failed = true;
        return;
      }
      int64_t laneAxis = 0;
      for (auto [axis, factor] :
           llvm::zip(result.getAxisIds().asArrayRef(),
                     result.getLayout().getLaneFactors().asArrayRef()))
        if (factor > 1) {
          laneAxis = axis;
          break;
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
