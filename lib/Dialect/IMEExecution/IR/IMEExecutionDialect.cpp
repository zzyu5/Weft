#include "Weft/Dialect/IMEExecution/IR/IMEExecutionDialect.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace weft::ime_execution;

#include "Weft/Dialect/IMEExecution/IR/IMEExecutionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");

weft::execution::GroupOp findGroup(weft::execution::PlanOp plan, int64_t id) {
  for (mlir::Operation &operation : plan.getBody().front())
    if (auto group = llvm::dyn_cast<weft::execution::GroupOp>(operation))
      if (group.getIdAttr().getInt() == id)
        return group;
  return {};
}

weft::kernel::KernelOp resolveKernel(weft::execution::PlanOp plan) {
  auto module = plan->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return llvm::dyn_cast_or_null<weft::kernel::KernelOp>(
      mlir::SymbolTable::lookupSymbolIn(module, plan.getKernelAttr()));
}

mlir::Operation *findNode(weft::kernel::KernelOp kernel, int64_t id) {
  mlir::Operation *found = nullptr;
  kernel->walk([&](mlir::Operation *operation) {
    auto node = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    if (node && node.getInt() == id)
      found = operation;
  });
  return found;
}

std::optional<int64_t> findValueGroup(weft::execution::PlanOp plan,
                                      mlir::Value value) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return std::nullopt;
  auto node = result.getOwner()->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
  if (!node)
    return std::nullopt;
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto layout = llvm::dyn_cast<weft::execution::ValueLayoutOp>(operation);
    if (layout && layout.getSourceNodeAttr().getInt() == node.getInt() &&
        layout.getSourceResultAttr().getInt() ==
            static_cast<int64_t>(result.getResultNumber()))
      return layout.getGroupAttr().getInt();
  }
  return std::nullopt;
}

bool isIntegerZero(mlir::Value value) {
  if (auto cast = value.getDefiningOp<weft::kernel::CastOp>())
    return isIntegerZero(cast.getInput());
  auto constant = value.getDefiningOp<weft::kernel::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer && integer.getInt() == 0;
}

std::optional<int64_t> resolveSelectedExtent(
    weft::execution::PlanOp plan, weft::kernel::KernelOp kernel,
    mlir::Value value, int64_t axis) {
  auto extent = weft::kernel::deriveLogicalExtent(value, axis);
  if (!extent)
    return std::nullopt;
  if (extent->constant)
    return extent->constant;
  auto meta = extent->dynamic.getDefiningOp<weft::kernel::MetaValueOp>();
  if (!meta)
    return std::nullopt;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(meta.getInput());
  if (!argument || argument.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  int64_t argumentIndex = argument.getArgNumber();
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto binding = llvm::dyn_cast<weft::execution::MetaBindingOp>(operation);
    if (binding && binding.getArgumentAttr().getInt() == argumentIndex)
      return binding.getValueAttr().getInt();
  }
  return std::nullopt;
}

bool isSignedInteger(mlir::Type type, unsigned width) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.isSigned() && integer.getWidth() == width;
}

bool roleLayoutClosureMatches(weft::execution::PlanOp plan, mlir::Value value,
                              int64_t expectedGroup,
                              llvm::DenseSet<mlir::Value> &visited,
                              bool recurseDefinition = true) {
  if (!value || !visited.insert(value).second)
    return true;
  if (auto block = mlir::dyn_cast<weft::kernel::BlockType>(value.getType())) {
    if (block.getShape().size() == 2) {
      auto selectedGroup = findValueGroup(plan, value);
      if (!selectedGroup || *selectedGroup != expectedGroup)
        return false;
    }
  }
  if (!recurseDefinition)
    return true;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return true;
  return llvm::all_of(definition->getOperands(), [&](mlir::Value operand) {
    return roleLayoutClosureMatches(plan, operand, expectedGroup, visited);
  });
}

bool roleLayoutClosureMatches(weft::execution::PlanOp plan, mlir::Value value,
                              int64_t expectedGroup,
                              bool recurseDefinition = true) {
  llvm::DenseSet<mlir::Value> visited;
  return roleLayoutClosureMatches(plan, value, expectedGroup, visited,
                                  recurseDefinition);
}

} // namespace

bool weft::ime_execution::hasXsmtvdotiiExtension(llvm::StringRef target) {
  constexpr llvm::StringLiteral extension("xsmtvdotii");
  llvm::SmallVector<llvm::StringRef, 8> components;
  target.split(components, '_', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef component : components) {
    if (!component.consume_front(extension))
      continue;
    if (component.empty())
      return true;

    size_t index = 0;
    while (index < component.size() && llvm::isDigit(component[index]))
      ++index;
    if (index == 0 || index == component.size() || component[index] != 'p')
      continue;
    size_t minorBegin = ++index;
    while (index < component.size() && llvm::isDigit(component[index]))
      ++index;
    if (minorBegin != index && index == component.size())
      return true;
  }
  return false;
}

mlir::LogicalResult ContractConfigOp::verify() {
  auto plan = getOperation()->getParentOfType<weft::execution::PlanOp>();
  if (!plan || getOperation()->getParentOp() != plan.getOperation())
    return emitOpError("must be nested directly in weft_execution.plan");
  if (!hasXsmtvdotiiExtension(plan.getTarget()))
    return emitOpError(
        "requires a selected target carrying the xsmtvdotii IME extension");
  if (getMacMAttr().getInt() != 4 || getMacNAttr().getInt() != 4 ||
      getMacKAttr().getInt() != 8)
    return emitOpError("first verified IME fragment must be exactly 4x4x8");
  if (getVlenBitsAttr().getInt() != 256)
    return emitOpError("first verified IME fragment requires VLEN=256");
  if (getStrategy() != "fragment_tiled")
    return emitOpError("strategy must be fragment_tiled");

  llvm::ArrayRef<int64_t> groupIds = getGroups();
  llvm::SmallDenseSet<int64_t, 4> distinctGroups(groupIds.begin(),
                                                 groupIds.end());
  if (groupIds.size() != 3 || distinctGroups.size() != 3)
    return emitOpError(
        "groups must contain three distinct lhs, rhs, and result role groups");
  llvm::SmallVector<weft::execution::GroupOp, 3> groups;
  for (int64_t id : groupIds) {
    auto group = findGroup(plan, id);
    if (!group)
      return emitOpError("references an unknown shared layout group");
    if (group.getOwnerAttr().getValue() != getSymName())
      return emitOpError(
          "must be the common owner symbol referenced by every role group");
    auto layout = group.getLayout();
    if (layout.getRank() != 2 || !layout.getVectorAxes().empty() ||
        layout.getOrder().size() != 2 || layout.getOrder()[0] != 1 ||
        layout.getOrder()[1] != 0)
      return emitOpError(
          "IME fragments require serial rank-two K/column-fastest shared "
          "layouts; fragment lane mapping remains owner-local");
    groups.push_back(group);
  }

  for (mlir::Operation &operation : plan.getBody().front()) {
    auto other = llvm::dyn_cast<ContractConfigOp>(operation);
    if (other && other.getOperation() != getOperation() &&
        other.getSourceNodeAttr() == getSourceNodeAttr())
      return emitOpError(
          "duplicates the IME realization for one canonical contraction");
  }

  auto kernel = resolveKernel(plan);
  auto contract =
      kernel ? llvm::dyn_cast_or_null<weft::kernel::ContractOp>(
                   findNode(kernel, getSourceNodeAttr().getInt()))
             : weft::kernel::ContractOp();
  if (!contract)
    return emitOpError("source_node must reference weft_kernel.contract");
  auto lhs = mlir::dyn_cast<weft::kernel::BlockType>(contract.getLhs().getType());
  auto rhs = mlir::dyn_cast<weft::kernel::BlockType>(contract.getRhs().getType());
  auto init = mlir::dyn_cast<weft::kernel::BlockType>(contract.getInit().getType());
  auto result =
      mlir::dyn_cast<weft::kernel::BlockType>(contract.getResult().getType());
  if (!lhs || !rhs || !init || !result || lhs.getShape().size() != 2 ||
      rhs.getShape().size() != 2 || init.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      !isSignedInteger(lhs.getElementType(), 8) ||
      !isSignedInteger(rhs.getElementType(), 8) ||
      !isSignedInteger(init.getElementType(), 32) ||
      !isSignedInteger(result.getElementType(), 32))
    return emitOpError(
        "first IME owner supports signed rank-two si8 x si8 -> si32 only");
  if (contract.getLhsAxes() != llvm::ArrayRef<int64_t>({1}) ||
      contract.getRhsAxes() != llvm::ArrayRef<int64_t>({1}))
    return emitOpError(
        "first IME owner requires stored [M,K] x [N,K] contraction axes 1,1");
  if (contract.getOrdered())
    return emitOpError(
        "fragment-tiled vmadot does not implement ordered contraction semantics");

  auto lhsM = resolveSelectedExtent(plan, kernel, contract.getLhs(), 0);
  auto lhsK = resolveSelectedExtent(plan, kernel, contract.getLhs(), 1);
  auto rhsN = resolveSelectedExtent(plan, kernel, contract.getRhs(), 0);
  auto rhsK = resolveSelectedExtent(plan, kernel, contract.getRhs(), 1);
  auto resultM = resolveSelectedExtent(plan, kernel, contract.getResult(), 0);
  auto resultN = resolveSelectedExtent(plan, kernel, contract.getResult(), 1);
  if (!lhsM || !lhsK || !rhsN || !rhsK || !resultM || !resultN ||
      *lhsM <= 0 || *lhsK <= 0 || *rhsN <= 0 || *rhsK <= 0 ||
      *resultM <= 0 || *resultN <= 0 || *lhsM != *resultM ||
      *rhsN != *resultN || *lhsK != *rhsK)
    return emitOpError(
        "selected logical M, N, and K extents must resolve to positive, "
        "role-consistent integers");

  auto lhsLoad = contract.getLhs().getDefiningOp<weft::kernel::LoadOp>();
  auto rhsLoad = contract.getRhs().getDefiningOp<weft::kernel::LoadOp>();
  auto initSplat = contract.getInit().getDefiningOp<weft::kernel::SplatOp>();
  if (!lhsLoad || !rhsLoad || lhsLoad == rhsLoad || !initSplat ||
      !isIntegerZero(lhsLoad.getOther()) ||
      !isIntegerZero(rhsLoad.getOther()) ||
      !isIntegerZero(initSplat.getValue()))
    return emitOpError(
        "fragment-tiled IME requires two direct masked loads with zero fill and "
        "a zero-splat si32 accumulator");

  unsigned resultStoreCount = 0;
  bool hasOtherResultUse = false;
  weft::kernel::StoreOp resultStore;
  for (mlir::Operation *user : contract.getResult().getUsers()) {
    auto store = llvm::dyn_cast<weft::kernel::StoreOp>(user);
    if (store && store.getValue() == contract.getResult()) {
      ++resultStoreCount;
      resultStore = store;
    } else
      hasOtherResultUse = true;
  }
  if (resultStoreCount != 1 || hasOtherResultUse)
    return emitOpError(
        "each IME contraction site requires one direct result store with no "
        "other result users");

  auto lhsGroup = findValueGroup(plan, contract.getLhs());
  auto rhsGroup = findValueGroup(plan, contract.getRhs());
  auto initGroup = findValueGroup(plan, contract.getInit());
  auto resultGroup = findValueGroup(plan, contract.getResult());
  if (!lhsGroup || !rhsGroup || !initGroup || !resultGroup ||
      *lhsGroup != groupIds[0] || *rhsGroup != groupIds[1] ||
      *initGroup != groupIds[2] || *resultGroup != groupIds[2])
    return emitOpError(
        "groups must match canonical lhs-load, rhs-load, accumulator, and result "
        "value layouts in role order");
  bool lhsRole = roleLayoutClosureMatches(plan, contract.getLhs(), groupIds[0]);
  bool rhsRole = roleLayoutClosureMatches(plan, contract.getRhs(), groupIds[1]);
  bool resultRole =
      roleLayoutClosureMatches(plan, contract.getResult(), groupIds[2],
                               /*recurseDefinition=*/false) &&
      roleLayoutClosureMatches(plan, contract.getInit(), groupIds[2]) &&
      roleLayoutClosureMatches(plan, resultStore.getPointer(), groupIds[2]) &&
      roleLayoutClosureMatches(plan, resultStore.getMask(), groupIds[2]);
  if (!lhsRole || !rhsRole || !resultRole)
    return emitOpError(
        "every rank-two pointer, mask, accumulator, and data value in a role "
        "closure must use that role's selected group");
  return mlir::success();
}

void WEFTIMEExecutionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionOps.cpp.inc"
      >();
}
