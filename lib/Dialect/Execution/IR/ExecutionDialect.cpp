#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"

#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace weft::execution;

#include "Weft/Dialect/Execution/IR/ExecutionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Execution/IR/ExecutionOps.cpp.inc"

namespace {

enum class RecordKind {
  Meta,
  Axis,
  Memory,
  Reduce,
  Scan,
  Summary,
  Contract,
  Math,
  Primitive,
};

PlanOp parentPlan(mlir::Operation *operation) {
  return operation->getParentOfType<PlanOp>();
}

weft::kernel::KernelOp canonicalKernel(PlanOp plan) {
  auto module = plan->getParentOfType<mlir::ModuleOp>();
  auto reference =
      plan->getAttrOfType<mlir::FlatSymbolRefAttr>("kernel_ref");
  if (!module || !reference)
    return {};
  return module.lookupSymbol<weft::kernel::KernelOp>(reference.getValue());
}

std::optional<int64_t> anchorOf(mlir::Operation *operation) {
  if (auto anchor = operation->getAttrOfType<mlir::IntegerAttr>("anchor"))
    return anchor.getInt();
  return std::nullopt;
}

std::optional<RecordKind> requiredRecordKind(mlir::Operation *operation) {
  using namespace weft::kernel;
  if (mlir::isa<MetaValueOp>(operation))
    return RecordKind::Meta;
  if (mlir::isa<VLAOp, BlockAxisOp>(operation))
    return RecordKind::Axis;
  if (mlir::isa<LoadOp, StoreOp, PrefetchOp, AtomicAddOp, FenceOp>(operation))
    return RecordKind::Memory;
  if (mlir::isa<ReduceOp>(operation))
    return RecordKind::Reduce;
  if (mlir::isa<ScanOp>(operation))
    return RecordKind::Scan;
  if (mlir::isa<SummaryFoldOp>(operation))
    return RecordKind::Summary;
  if (mlir::isa<ContractOp, weft::extension::BlockScaledContractOp>(operation))
    return RecordKind::Contract;
  if (auto unary = mlir::dyn_cast<UnaryOp>(operation);
      unary && unary.getKind() != "neg")
    return RecordKind::Math;
  if (mlir::isa<PermuteOp, LookupOp, DecodeOp, WidenOp, NarrowOp>(operation))
    return RecordKind::Primitive;
  return std::nullopt;
}

std::optional<RecordKind> recordKind(mlir::Operation *operation) {
  return llvm::TypeSwitch<mlir::Operation *, std::optional<RecordKind>>(operation)
      .Case<MetaBindingOp>([](auto) { return RecordKind::Meta; })
      .Case<AxisPlanOp>([](auto) { return RecordKind::Axis; })
      .Case<MemoryPlanOp>([](auto) { return RecordKind::Memory; })
      .Case<ReducePlanOp>([](auto) { return RecordKind::Reduce; })
      .Case<ScanPlanOp>([](auto) { return RecordKind::Scan; })
      .Case<SummaryPlanOp>([](auto) { return RecordKind::Summary; })
      .Case<ContractPlanOp>([](auto) { return RecordKind::Contract; })
      .Case<MathPlanOp>([](auto) { return RecordKind::Math; })
      .Case<PrimitivePlanOp>([](auto) { return RecordKind::Primitive; })
      .Default([](auto) { return std::nullopt; });
}

mlir::LogicalResult verifyRecordAnchor(mlir::Operation *record,
                                       RecordKind expected) {
  PlanOp plan = parentPlan(record);
  if (!plan)
    return record->emitOpError("must be nested directly in a selected plan");
  auto anchor = anchorOf(record);
  if (!anchor || *anchor < 0)
    return record->emitOpError("requires a non-negative canonical anchor");
  auto kernel = canonicalKernel(plan);
  if (!kernel)
    return record->emitOpError("plan does not reference a canonical kernel");
  mlir::Operation *matched = nullptr;
  kernel.walk([&](mlir::Operation *candidate) {
    auto value = candidate->getAttrOfType<mlir::IntegerAttr>(
        kCanonicalAnchorAttr);
    if (value && value.getInt() == *anchor)
      matched = candidate;
  });
  if (!matched)
    return record->emitOpError("references a missing canonical anchor");
  if (requiredRecordKind(matched) != expected)
    return record->emitOpError("record kind does not match its canonical anchor");
  return mlir::success();
}

bool validProvider(llvm::StringRef provider) {
  return llvm::StringSwitch<bool>(provider)
      .Cases("scalar", "rvv", true)
      .Default(false);
}

mlir::LogicalResult verifyProviderRecord(mlir::Operation *operation,
                                         RecordKind kind) {
  if (mlir::failed(verifyRecordAnchor(operation, kind)))
    return mlir::failure();
  auto provider = operation->getAttrOfType<mlir::StringAttr>("provider");
  auto strategy = operation->getAttrOfType<mlir::StringAttr>("strategy");
  if (!provider || !validProvider(provider.getValue()) || !strategy ||
      strategy.getValue().empty())
    return operation->emitOpError("requires known provider and strategy");
  PlanOp plan = parentPlan(operation);
  bool hasRVV = plan->getAttrOfType<mlir::BoolAttr>("has_rvv").getValue();
  if (provider.getValue() == "rvv" && !hasRVV)
    return operation->emitOpError("selects RVV for a target without V");
  return mlir::success();
}

} // namespace

mlir::LogicalResult PlanOp::verify() {
  auto kernel = canonicalKernel(*this);
  if (!kernel)
    return emitOpError("kernel_ref must resolve to one canonical kernel");
  if (getXlen() != 32 && getXlen() != 64)
    return emitOpError("xlen must be 32 or 64");
  if (getVlenBits() < 0 || getVectorRegisters() <= 0)
    return emitOpError("target vector facts are invalid");
  if (!mlir::isa<EndOp>(getBody().front().getTerminator()))
    return emitOpError("plan body must terminate with weft_execution.end");

  llvm::DenseMap<int64_t, RecordKind> required;
  bool malformedAnchor = false;
  kernel.walk([&](mlir::Operation *operation) {
    auto kind = requiredRecordKind(operation);
    if (!kind)
      return;
    auto anchor = operation->getAttrOfType<mlir::IntegerAttr>(
        kCanonicalAnchorAttr);
    if (!anchor || anchor.getInt() < 0 ||
        !required.try_emplace(anchor.getInt(), *kind).second)
      malformedAnchor = true;
  });
  if (malformedAnchor)
    return emitOpError("canonical planned anchors must be unique non-negative integers");

  llvm::DenseSet<int64_t> covered;
  for (mlir::Operation &operation : getBody().front().without_terminator()) {
    auto kind = recordKind(&operation);
    auto anchor = anchorOf(&operation);
    if (!kind || !anchor)
      return emitOpError("plan body contains a non-plan record");
    auto expected = required.find(*anchor);
    if (expected == required.end() || expected->second != *kind)
      return emitOpError("record does not match a required canonical anchor");
    if (!covered.insert(*anchor).second)
      return emitOpError("canonical anchor has more than one selected record");
  }
  if (covered.size() != required.size())
    return emitOpError("every canonical realization anchor needs one selected record");
  return mlir::success();
}

mlir::LogicalResult EndOp::verify() {
  if (!parentPlan(getOperation()))
    return emitOpError("must terminate weft_execution.plan");
  return mlir::success();
}

mlir::LogicalResult MetaBindingOp::verify() {
  return verifyRecordAnchor(getOperation(), RecordKind::Meta);
}

mlir::LogicalResult AxisPlanOp::verify() {
  if (mlir::failed(verifyRecordAnchor(getOperation(), RecordKind::Axis)))
    return mlir::failure();
  if (!validProvider(getProvider()) ||
      (getRealization() != "scalar" && getRealization() != "rvv"))
    return emitOpError("axis realization must select scalar or RVV");
  if (getUnroll() <= 0)
    return emitOpError("axis unroll must be positive");
  if (getRealization() == "scalar") {
    if (getProvider() != "scalar" || getSew() != 0 || getLmul() != "none")
      return emitOpError("scalar axis must use scalar/SEW=0/LMUL=none");
  } else {
    PlanOp plan = parentPlan(getOperation());
    if (getProvider() != "rvv" ||
        !plan->getAttrOfType<mlir::BoolAttr>("has_rvv").getValue())
      return emitOpError("RVV axis requires the RVV provider and target V");
    if (!llvm::is_contained({8, 16, 32, 64}, getSew()) ||
        !llvm::is_contained({"mf8", "mf4", "mf2", "m1", "m2", "m4", "m8"},
                            getLmul()))
      return emitOpError("RVV axis has invalid SEW or LMUL");
  }
  return mlir::success();
}

mlir::LogicalResult MemoryPlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Memory);
}

mlir::LogicalResult ReducePlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Reduce);
}

mlir::LogicalResult ScanPlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Scan);
}

mlir::LogicalResult SummaryPlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Summary);
}

mlir::LogicalResult ContractPlanOp::verify() {
  if (mlir::failed(verifyProviderRecord(getOperation(), RecordKind::Contract)))
    return mlir::failure();
  if (getMicroM() <= 0 || getMicroN() <= 0 || getMicroK() <= 0)
    return emitOpError("contract microtile dimensions must be positive");
  return mlir::success();
}

mlir::LogicalResult MathPlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Math);
}

mlir::LogicalResult PrimitivePlanOp::verify() {
  return verifyProviderRecord(getOperation(), RecordKind::Primitive);
}

void WEFTExecutionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Execution/IR/ExecutionOps.cpp.inc"
      >();
}
