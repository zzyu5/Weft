#include "Weft/Dialect/RVVExecution/IR/RVVExecutionDialect.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft::rvv_execution;

#include "Weft/Dialect/RVVExecution/IR/RVVExecutionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionOps.cpp.inc"

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
  auto node = result.getOwner()->getAttrOfType<mlir::IntegerAttr>(
      kNodeAttrName);
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

std::optional<int64_t> findUseGroup(weft::execution::PlanOp plan,
                                    mlir::Operation *consumer,
                                    int64_t operandIndex) {
  auto node = consumer->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
  if (!node || operandIndex < 0 ||
      operandIndex >= static_cast<int64_t>(consumer->getNumOperands()))
    return std::nullopt;
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto conversion =
        llvm::dyn_cast<weft::execution::LayoutConversionOp>(operation);
    if (conversion && conversion.getConsumerNodeAttr().getInt() == node.getInt() &&
        conversion.getConsumerOperandAttr().getInt() == operandIndex)
      return conversion.getGroupAttr().getInt();
  }
  return findValueGroup(plan, consumer->getOperand(operandIndex));
}

std::optional<int64_t> getPhysicalSEW(mlir::Type type) {
  auto block = mlir::dyn_cast<weft::kernel::BlockType>(type);
  if (!block)
    return std::nullopt;
  mlir::Type element = block.getElementType();
  if (mlir::isa<weft::kernel::PtrType>(element) || element.isIndex())
    return 64;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
    if (integer.getWidth() == 1)
      return 0;
    return integer.getWidth();
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(element))
    return floating.getWidth();
  return std::nullopt;
}

bool hasLegalLMUL(int64_t sew, int64_t laneRatio) {
  if (sew == 0)
    return true;
  if ((sew * 8) % laneRatio != 0)
    return false;
  int64_t eighths = sew * 8 / laneRatio;
  return eighths == 4 || eighths == 8 || eighths == 16 ||
         eighths == 32 || eighths == 64;
}

} // namespace

mlir::LogicalResult GroupConfigOp::verify() {
  auto plan = getOperation()->getParentOfType<weft::execution::PlanOp>();
  if (!plan || getOperation()->getParentOp() != plan.getOperation())
    return emitOpError("must be nested directly in weft_execution.plan");
  int64_t groupId = getGroupAttr().getInt();
  weft::execution::GroupOp group = findGroup(plan, groupId);
  if (!group)
    return emitOpError("references an unknown shared layout group");
  if (group.getOwnerAttr().getValue() != getSymName())
    return emitOpError("must be the owner symbol referenced by its layout group");
  size_t vectorAxisCount = group.getLayout().getVectorAxes().size();
  if (vectorAxisCount > 1)
    return emitOpError("supports at most one shared SIMD vector axis");
  mlir::IntegerAttr laneRatioAttr = getLaneRatioAttr();
  if (vectorAxisCount == 0 && laneRatioAttr)
    return emitOpError("serial RVV-owned groups must not carry lane_ratio");
  if (vectorAxisCount == 1 && !laneRatioAttr)
    return emitOpError("vectorized RVV groups require lane_ratio");
  int64_t laneRatio = laneRatioAttr ? laneRatioAttr.getInt() : 0;
  if (laneRatioAttr && laneRatio != 8 && laneRatio != 16 &&
      laneRatio != 32 && laneRatio != 64)
    return emitOpError("lane_ratio must be one of 8, 16, 32, or 64");
  int64_t registerBudget = getRegisterBudgetAttr().getInt();
  if (registerBudget <= 0 || registerBudget > 32)
    return emitOpError("register_budget must be in [1, 32]");
  weft::kernel::KernelOp kernel = resolveKernel(plan);
  if (!kernel)
    return emitOpError("cannot resolve the canonical source kernel");
  auto verifyMemberType = [&](mlir::Type type,
                              llvm::Twine identity) -> mlir::LogicalResult {
    if (!laneRatioAttr)
      return mlir::success();
    auto sew = getPhysicalSEW(type);
    if (!sew || !hasLegalLMUL(*sew, laneRatio))
      return emitOpError()
             << "lane_ratio does not produce a legal RVV LMUL for " << identity;
    return mlir::success();
  };
  for (mlir::Operation &operation : plan.getBody().front()) {
    if (auto valueLayout =
            llvm::dyn_cast<weft::execution::ValueLayoutOp>(operation)) {
      if (valueLayout.getGroupAttr().getInt() != groupId)
        continue;
      int64_t sourceNode = valueLayout.getSourceNodeAttr().getInt();
      int64_t sourceResult = valueLayout.getSourceResultAttr().getInt();
      mlir::Operation *node = findNode(kernel, sourceNode);
      if (!node || sourceResult < 0 ||
          sourceResult >= static_cast<int64_t>(node->getNumResults()))
        return emitOpError("cannot resolve a group member in the source kernel");
      if (mlir::failed(verifyMemberType(
              node->getResult(sourceResult).getType(),
              llvm::Twine("source node ") + llvm::Twine(sourceNode) +
                  " result " + llvm::Twine(sourceResult))))
        return mlir::failure();
      continue;
    }
    auto conversion =
        llvm::dyn_cast<weft::execution::LayoutConversionOp>(operation);
    if (!conversion || conversion.getGroupAttr().getInt() != groupId)
      continue;
    mlir::Operation *consumer =
        findNode(kernel, conversion.getConsumerNodeAttr().getInt());
    int64_t operand = conversion.getConsumerOperandAttr().getInt();
    if (!consumer || operand < 0 ||
        operand >= static_cast<int64_t>(consumer->getNumOperands()))
      return emitOpError(
          "cannot resolve an incoming conversion in the source kernel");
    if (mlir::failed(verifyMemberType(
            consumer->getOperand(operand).getType(),
            llvm::Twine("consumer node ") +
                llvm::Twine(conversion.getConsumerNodeAttr().getInt()) +
              " operand " + llvm::Twine(operand))))
      return mlir::failure();
  }

  mlir::LogicalResult completeness = mlir::success();
  kernel->walk([&](mlir::Operation *operation) {
    if (mlir::failed(completeness))
      return;
    auto node = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    if (!node)
      return;
    if (auto unary = llvm::dyn_cast<weft::kernel::UnaryOp>(operation)) {
      auto resultGroup = findValueGroup(plan, unary.getResult());
      if (unary.getKind() != "exp" || !resultGroup || *resultGroup != groupId)
        return;
      int64_t count = 0;
      for (mlir::Operation &record : plan.getBody().front())
        if (auto config = llvm::dyn_cast<UnaryConfigOp>(record))
          count += config.getSourceNodeAttr().getInt() == node.getInt() &&
                   config.getGroupAttr().getInt() == groupId;
      if (count != 1) {
        emitOpError()
            << "requires exactly one unary_config for canonical exp node "
            << node.getInt();
        completeness = mlir::failure();
      }
      return;
    }
    if (auto reduction = llvm::dyn_cast<weft::kernel::ReduceOp>(operation)) {
      auto useGroup = findUseGroup(plan, reduction.getOperation(), 0);
      if (!useGroup || *useGroup != groupId)
        return;
      int64_t count = 0;
      for (mlir::Operation &record : plan.getBody().front())
        if (auto config = llvm::dyn_cast<ReductionConfigOp>(record))
          count += config.getSourceNodeAttr().getInt() == node.getInt() &&
                   config.getGroupAttr().getInt() == groupId;
      if (count != 1) {
        emitOpError()
            << "requires exactly one reduction_config for canonical reduction "
               "node "
            << node.getInt();
        completeness = mlir::failure();
      }
      return;
    }
    auto contract = llvm::dyn_cast<weft::kernel::ContractOp>(operation);
    if (!contract)
      return;
    auto lhsGroup = findUseGroup(plan, contract.getOperation(), 0);
    auto rhsGroup = findUseGroup(plan, contract.getOperation(), 1);
    auto resultGroup = findValueGroup(plan, contract.getResult());
    bool participates = (lhsGroup && *lhsGroup == groupId) ||
                        (rhsGroup && *rhsGroup == groupId) ||
                        (resultGroup && *resultGroup == groupId);
    if (!participates)
      return;
    int64_t count = 0;
    for (mlir::Operation &record : plan.getBody().front()) {
      auto config = llvm::dyn_cast<ContractConfigOp>(record);
      if (!config || config.getSourceNodeAttr().getInt() != node.getInt())
        continue;
      count += llvm::is_contained(config.getGroups(), groupId);
    }
    if (count != 1) {
      emitOpError()
          << "requires exactly one contract_config covering canonical contract "
             "node "
          << node.getInt();
      completeness = mlir::failure();
    }
  });
  if (mlir::failed(completeness))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult UnaryConfigOp::verify() {
  auto plan = getOperation()->getParentOfType<weft::execution::PlanOp>();
  if (!plan || getOperation()->getParentOp() != plan.getOperation())
    return emitOpError("must be nested directly in weft_execution.plan");
  weft::execution::GroupOp selectedGroup =
      findGroup(plan, getGroupAttr().getInt());
  if (!selectedGroup)
    return emitOpError("references an unknown shared layout group");
  auto owner = llvm::dyn_cast_or_null<GroupConfigOp>(
      mlir::SymbolTable::lookupSymbolIn(plan, selectedGroup.getOwnerAttr()));
  if (!owner || owner.getGroupAttr().getInt() != getGroupAttr().getInt())
    return emitOpError("group must be owned by a matching RVV group_config");
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto other = llvm::dyn_cast<UnaryConfigOp>(operation);
    if (other && other.getOperation() != getOperation() &&
        other.getSourceNodeAttr() == getSourceNodeAttr())
      return emitOpError(
          "duplicates the RVV strategy for one canonical unary operation");
  }
  if (getStrategy() != "exp_poly_v1")
    return emitOpError("strategy must be exp_poly_v1");

  weft::kernel::KernelOp kernel = resolveKernel(plan);
  auto unary = kernel
                   ? llvm::dyn_cast_or_null<weft::kernel::UnaryOp>(
                         findNode(kernel, getSourceNodeAttr().getInt()))
                   : weft::kernel::UnaryOp();
  if (!unary)
    return emitOpError("source_node must reference weft_kernel.unary");
  auto input = mlir::dyn_cast<weft::kernel::BlockType>(unary.getInput().getType());
  auto result =
      mlir::dyn_cast<weft::kernel::BlockType>(unary.getResult().getType());
  if (unary.getKind() != "exp" || !input || !result ||
      !input.getElementType().isF32() || result != input)
    return emitOpError(
        "exp_poly_v1 requires a canonical shape-preserving f32 block exp");
  auto inputGroup = findUseGroup(plan, unary.getOperation(), 0);
  auto resultGroup = findValueGroup(plan, unary.getResult());
  if (!inputGroup || !resultGroup ||
      *inputGroup != getGroupAttr().getInt() ||
      *resultGroup != getGroupAttr().getInt())
    return emitOpError(
        "group must be the effective input-use and result layout of the "
        "canonical exponential");
  auto vectorAxes = selectedGroup.getLayout().getVectorAxes();
  if (vectorAxes.size() != 1 ||
      input.getShape()[vectorAxes.front()] == 1)
    return emitOpError("exp_poly_v1 requires one selected RVV vector axis");
  return mlir::success();
}

mlir::LogicalResult ReductionConfigOp::verify() {
  auto plan = getOperation()->getParentOfType<weft::execution::PlanOp>();
  if (!plan || getOperation()->getParentOp() != plan.getOperation())
    return emitOpError("must be nested directly in weft_execution.plan");
  weft::execution::GroupOp selectedGroup =
      findGroup(plan, getGroupAttr().getInt());
  if (!selectedGroup)
    return emitOpError("references an unknown shared layout group");
  auto owner = llvm::dyn_cast_or_null<GroupConfigOp>(
      mlir::SymbolTable::lookupSymbolIn(plan, selectedGroup.getOwnerAttr()));
  if (!owner || owner.getGroupAttr().getInt() != getGroupAttr().getInt())
    return emitOpError("group must be owned by a matching RVV group_config");
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto other = llvm::dyn_cast<ReductionConfigOp>(operation);
    if (other && other.getOperation() != getOperation() &&
        other.getSourceNodeAttr() == getSourceNodeAttr())
      return emitOpError(
          "duplicates the RVV strategy for one canonical reduction");
  }
  if (getStrategy() != "tree" && getStrategy() != "ordered")
    return emitOpError("strategy must be tree or ordered");
  weft::kernel::KernelOp kernel = resolveKernel(plan);
  auto reduce = kernel
                    ? llvm::dyn_cast_or_null<weft::kernel::ReduceOp>(
                          findNode(kernel, getSourceNodeAttr().getInt()))
                    : weft::kernel::ReduceOp();
  if (!reduce)
    return emitOpError("source_node must reference weft_kernel.reduce");
  auto input = mlir::dyn_cast<weft::kernel::BlockType>(
      reduce.getInput().getType());
  bool supportedKind = reduce &&
                       (reduce.getKind() == "sum" || reduce.getKind() == "max" ||
                        reduce.getKind() == "min");
  if (!input || !input.getElementType().isF32() || !supportedKind)
    return emitOpError(
        "first RVV reduction slice supports only f32 sum/max/min over a "
        "logical block");
  int64_t rank = input.getShape().size();
  int64_t axis = reduce.getAxisAttr().getInt();
  if (rank == 1) {
    if (!reduce.getResult().getType().isF32())
      return emitOpError(
          "rank-one f32 reduction must produce a scalar f32 result");
  } else if (rank == 2) {
    auto result = mlir::dyn_cast<weft::kernel::BlockType>(
        reduce.getResult().getType());
    int64_t projectedAxis = axis == 0 ? 1 : 0;
    if (!result || result.getShape().size() != 1 ||
        !result.getElementType().isF32() ||
        result.getShape().front() != input.getShape()[projectedAxis])
      return emitOpError(
          "rank-two f32 reduction must remove exactly its selected vector axis");
  } else {
    return emitOpError("first RVV reduction slice supports rank one or two");
  }
  auto inputGroup = findUseGroup(plan, reduce.getOperation(), 0);
  if (!inputGroup || *inputGroup != getGroupAttr().getInt())
    return emitOpError(
        "group must be the effective selected layout of the reduction input use");
  weft::execution::GroupOp group = findGroup(plan, *inputGroup);
  if (!group || group.getLayout().getVectorAxes().size() != 1 ||
      group.getLayout().getVectorAxes().front() !=
          reduce.getAxisAttr().getInt())
    return emitOpError(
        "first RVV reduction slice must reduce the selected vector axis");
  if ((reduce.getOrdered() && getStrategy() != "ordered") ||
      (!reduce.getOrdered() && getStrategy() != "tree"))
    return emitOpError("strategy conflicts with canonical ordered semantics");
  return mlir::success();
}

mlir::LogicalResult ContractConfigOp::verify() {
  auto plan = getOperation()->getParentOfType<weft::execution::PlanOp>();
  if (!plan || getOperation()->getParentOp() != plan.getOperation())
    return emitOpError("must be nested directly in weft_execution.plan");
  llvm::ArrayRef<int64_t> groupIds = getGroups();
  if (groupIds.empty())
    return emitOpError("groups must contain canonical contraction roles");
  llvm::SmallVector<weft::execution::GroupOp, 3> groups;
  for (int64_t id : groupIds) {
    weft::execution::GroupOp group = findGroup(plan, id);
    if (!group)
      return emitOpError("references an unknown shared layout group");
    auto owner = llvm::dyn_cast_or_null<GroupConfigOp>(
        mlir::SymbolTable::lookupSymbolIn(plan, group.getOwnerAttr()));
    if (!owner || owner.getGroupAttr().getInt() != id)
      return emitOpError(
          "every contraction role group must have a matching RVV group_config");
    groups.push_back(group);
  }
  for (mlir::Operation &operation : plan.getBody().front()) {
    auto other = llvm::dyn_cast<ContractConfigOp>(operation);
    if (other && other.getOperation() != getOperation() &&
        other.getSourceNodeAttr() == getSourceNodeAttr())
      return emitOpError(
          "duplicates the RVV strategy for one canonical contraction");
  }
  if (getStrategy() != "tree" && getStrategy() != "ordered" &&
      getStrategy() != "sequential")
    return emitOpError("strategy must be tree, ordered, or sequential");

  weft::kernel::KernelOp kernel = resolveKernel(plan);
  auto contract = kernel
                      ? llvm::dyn_cast_or_null<weft::kernel::ContractOp>(
                            findNode(kernel, getSourceNodeAttr().getInt()))
                      : weft::kernel::ContractOp();
  if (!contract)
    return emitOpError("source_node must reference weft_kernel.contract");
  bool blockedResult =
      mlir::isa<weft::kernel::BlockType>(contract.getResult().getType());
  size_t expectedGroups = blockedResult ? 3 : 2;
  if (groupIds.size() != expectedGroups)
    return emitOpError()
           << "groups must contain lhs, rhs"
           << (blockedResult ? ", and result" : "")
           << " role groups in canonical operand order";
  auto lhs = mlir::dyn_cast<weft::kernel::BlockType>(contract.getLhs().getType());
  auto rhs = mlir::dyn_cast<weft::kernel::BlockType>(contract.getRhs().getType());
  if (!lhs || !rhs || !lhs.getElementType().isF32() ||
      !rhs.getElementType().isF32() || contract.getLhsAxes().size() != 1 ||
      contract.getRhsAxes().size() != 1)
    return emitOpError("RVV contract roles require f32 blocks and one axis pair");
  auto lhsGroup = findUseGroup(plan, contract.getOperation(), 0);
  auto rhsGroup = findUseGroup(plan, contract.getOperation(), 1);
  if (!lhsGroup || !rhsGroup || *lhsGroup != groupIds[0] ||
      *rhsGroup != groupIds[1])
    return emitOpError(
        "groups must match the effective lhs and rhs operand-use layouts");

  bool rankOne = lhs.getShape().size() == 1 && rhs.getShape().size() == 1 &&
                 contract.getResult().getType().isF32() &&
                 contract.getLhsAxes().front() == 0 &&
                 contract.getRhsAxes().front() == 0;
  auto result = mlir::dyn_cast<weft::kernel::BlockType>(
      contract.getResult().getType());
  bool rankTwo = lhs.getShape().size() == 2 && rhs.getShape().size() == 2 &&
                 result && result.getShape().size() == 2 &&
                 result.getElementType().isF32() &&
                 contract.getLhsAxes().front() == 1 &&
                 contract.getRhsAxes().front() == 0;
  if (!rankOne && !rankTwo)
    return emitOpError(
        "RVV contract slice supports full rank-one dot or rank-two f32 "
        "lhs[M,K] x rhs[K,N]");

  auto init = mlir::dyn_cast<weft::kernel::BlockType>(
      contract.getInit().getType());
  bool scalarInit = contract.getInit().getType().isF32();
  bool blockedRankTwoInit =
      rankTwo && init && init.getElementType().isF32() && result &&
      init.getShape() == result.getShape();
  if ((rankOne && !scalarInit) ||
      (rankTwo && !scalarInit && !blockedRankTwoInit))
    return emitOpError(
        "rank-one contraction requires scalar f32 init; rank-two contraction "
        "requires scalar f32 or result-shaped f32 block init");

  if (rankOne) {
    if ((contract.getOrdered() && getStrategy() != "ordered") ||
        (!contract.getOrdered() && getStrategy() != "tree"))
      return emitOpError("strategy conflicts with canonical ordered semantics");
    for (weft::execution::GroupOp group : groups)
      if (group.getLayout().getRank() != 1 ||
          group.getLayout().getVectorAxes().size() != 1 ||
          group.getLayout().getVectorAxes().front() != 0)
        return emitOpError(
            "rank-one RVV contraction requires every operand role to use its "
            "only vector axis");
  } else {
    if (getStrategy() != "sequential")
      return emitOpError(
          "rank-two outer-product currently requires sequential K strategy");
    if (groupIds[0] == groupIds[1] || groupIds[0] == groupIds[2] ||
        groupIds[1] == groupIds[2])
      return emitOpError(
          "rank-two RVV contraction requires distinct lhs, rhs, and result "
          "role groups");
    auto resultGroup = findValueGroup(plan, contract.getResult());
    if (!resultGroup || *resultGroup != groupIds[2])
      return emitOpError(
          "third group must match the canonical contraction result layout");
    auto lhsLayout = groups[0].getLayout();
    auto rhsLayout = groups[1].getLayout();
    auto resultLayout = groups[2].getLayout();
    if (lhsLayout.getRank() != 2 || !lhsLayout.getVectorAxes().empty())
      return emitOpError(
          "rank-two outer-product lhs group must use a serial blocked layout");
    auto isNVectorLayout = [](auto layout) {
      return layout.getRank() == 2 && layout.getOrder().size() == 2 &&
             layout.getOrder().front() == 1 && layout.getOrder().back() == 0 &&
             layout.getVectorAxes().size() == 1 &&
             layout.getVectorAxes().front() == 1;
    };
    if (!isNVectorLayout(rhsLayout) || !isNVectorLayout(resultLayout))
      return emitOpError(
          "rank-two outer-product rhs and result groups must vectorize local "
          "axis 1 with vector-fastest traversal");
    auto rhsOwner = llvm::cast<GroupConfigOp>(
        mlir::SymbolTable::lookupSymbolIn(plan, groups[1].getOwnerAttr()));
    auto resultOwner = llvm::cast<GroupConfigOp>(
        mlir::SymbolTable::lookupSymbolIn(plan, groups[2].getOwnerAttr()));
    if (!rhsOwner.getLaneRatioAttr() || !resultOwner.getLaneRatioAttr() ||
        rhsOwner.getLaneRatioAttr().getInt() !=
            resultOwner.getLaneRatioAttr().getInt())
      return emitOpError(
          "rank-two outer-product rhs and result groups require one lane ratio");
  }
  return mlir::success();
}

void WEFTRVVExecutionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionOps.cpp.inc"
      >();
}
