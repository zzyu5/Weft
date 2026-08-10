#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace weft::execution;

#include "Weft/Dialect/Execution/IR/ExecutionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Execution/IR/ExecutionOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");

weft::kernel::KernelOp resolveKernel(PlanOp plan) {
  auto module = plan->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};
  return llvm::dyn_cast_or_null<weft::kernel::KernelOp>(
      mlir::SymbolTable::lookupSymbolIn(module, plan.getKernelAttr()));
}

bool hasMappedLayout(
    mlir::Value value,
    const llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> &valueGroups) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;
  auto node = result.getOwner()->getAttrOfType<mlir::IntegerAttr>(
      kNodeAttrName);
  return node && valueGroups.contains(
                     std::make_pair(node.getInt(),
                                    static_cast<int64_t>(result.getResultNumber())));
}

bool isUnmaterializedViewProvenance(
    mlir::Value value,
    const llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> &valueGroups,
    llvm::DenseSet<mlir::Value> &visited) {
  if (hasMappedLayout(value, valueGroups))
    return true;
  if (!weft::kernel::isIndexCoordinateProvenance(value) ||
      !visited.insert(value).second || value.use_empty())
    return false;
  mlir::Operation *definition = value.getDefiningOp();
  bool coordinateDefinition =
      llvm::isa_and_nonnull<weft::kernel::ArangeOp,
                            weft::kernel::ExpandDimsOp>(definition);
  if (auto binary =
          llvm::dyn_cast_or_null<weft::kernel::BinaryOp>(definition)) {
    auto result = mlir::dyn_cast<weft::kernel::BlockType>(
        binary.getResult().getType());
    coordinateDefinition |= result && result.getElementType().isIndex();
  }
  if (!coordinateDefinition)
    return false;
  for (mlir::Operation *user : value.getUsers()) {
    mlir::Value next;
    if (auto expand = llvm::dyn_cast<weft::kernel::ExpandDimsOp>(user)) {
      if (expand.getInput() != value)
        return false;
      next = expand.getResult();
    } else if (auto binary = llvm::dyn_cast<weft::kernel::BinaryOp>(user)) {
      auto input = mlir::dyn_cast<weft::kernel::BlockType>(value.getType());
      auto result = mlir::dyn_cast<weft::kernel::BlockType>(
          binary.getResult().getType());
      if (!input || !result || !input.getElementType().isIndex() ||
          !result.getElementType().isIndex() ||
          input.getShape().size() != result.getShape().size() ||
          (binary.getLhs() != value && binary.getRhs() != value))
        return false;
      next = binary.getResult();
    } else {
      return false;
    }
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    if (!isUnmaterializedViewProvenance(next, valueGroups, branchVisited))
      return false;
  }
  return true;
}

bool isUnmaterializedViewProvenance(
    mlir::Value value,
    const llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> &valueGroups) {
  llvm::DenseSet<mlir::Value> visited;
  return isUnmaterializedViewProvenance(value, valueGroups, visited);
}

void collectReachableProjectionStores(
    mlir::Value value, llvm::DenseSet<mlir::Operation *> &stores,
    llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return;
  for (mlir::Operation *user : value.getUsers()) {
    if (llvm::isa<weft::kernel::StoreOp>(user)) {
      stores.insert(user);
      continue;
    }
    if (!llvm::isa<weft::kernel::BinaryOp, weft::kernel::CompareOp,
                   weft::kernel::CastOp, weft::kernel::UnaryOp,
                   weft::kernel::PtrAddOp>(user))
      continue;
    for (mlir::Value result : user->getResults())
      collectReachableProjectionStores(result, stores, visited);
  }
}

llvm::DenseSet<mlir::Value> collectDerivedSerialProjectionValues(
    weft::kernel::KernelOp source, llvm::DenseMap<int64_t, GroupOp> &groups,
    const llvm::DenseMap<mlir::Value, int64_t> &mappedGroups,
    const llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t>
        &conversionGroups,
    bool &hasConflictingGroups) {
  llvm::DenseSet<mlir::Value> serialValues;
  llvm::DenseMap<mlir::Value, int64_t> serialProjectionGroups;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 2>>
      projectionResultsByStore;
  source->walk([&](weft::kernel::ReduceOp reduction) {
    auto node = reduction->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    if (!node)
      return;
    auto converted = conversionGroups.find(std::make_pair(node.getInt(), 0));
    std::optional<int64_t> selectedGroup;
    if (converted != conversionGroups.end())
      selectedGroup = converted->second;
    else if (auto mapped = mappedGroups.find(reduction.getInput());
             mapped != mappedGroups.end())
      selectedGroup = mapped->second;
    if (!selectedGroup)
      return;
    auto group = groups.find(*selectedGroup);
    if (group == groups.end() || group->second.getLayout().getRank() != 2 ||
        group->second.getLayout().getVectorAxes().size() != 1 ||
        reduction.getAxisAttr().getInt() !=
            group->second.getLayout().getVectorAxes().front())
      return;
    auto result = mlir::dyn_cast<weft::kernel::BlockType>(
        reduction.getResult().getType());
    if (result && result.getShape().size() == 1) {
      serialValues.insert(reduction.getResult());
      serialProjectionGroups.try_emplace(reduction.getResult(), *selectedGroup);
      llvm::DenseSet<mlir::Operation *> stores;
      llvm::DenseSet<mlir::Value> visited;
      collectReachableProjectionStores(reduction.getResult(), stores, visited);
      for (mlir::Operation *store : stores)
        projectionResultsByStore[store].push_back(reduction.getResult());
    }
  });
  source->walk([&](weft::kernel::ArangeOp arange) {
    auto result = mlir::cast<weft::kernel::BlockType>(arange.getResult().getType());
    if (result.getShape().size() != 1)
      return;
    llvm::DenseSet<mlir::Operation *> stores;
    llvm::DenseSet<mlir::Value> visited;
    collectReachableProjectionStores(arange.getResult(), stores, visited);
    for (mlir::Operation *store : stores) {
      auto projections = projectionResultsByStore.find(store);
      if (projections == projectionResultsByStore.end())
        continue;
      if (llvm::any_of(projections->second, [&](mlir::Value projection) {
            return weft::kernel::haveSameLogicalExtent(
                arange.getResult(), 0, projection, 0);
          })) {
        serialValues.insert(arange.getResult());
        return;
      }
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    source->walk([&](mlir::Operation *operation) {
      if (!llvm::isa<weft::kernel::BinaryOp, weft::kernel::CompareOp,
                     weft::kernel::CastOp, weft::kernel::UnaryOp,
                     weft::kernel::PtrAddOp>(operation))
        return;
      bool hasBlockOperand = false;
      bool allSerial = true;
      llvm::SmallDenseSet<int64_t, 2> operandGroups;
      for (mlir::Value operand : operation->getOperands()) {
        if (!mlir::isa<weft::kernel::BlockType>(operand.getType()))
          continue;
        hasBlockOperand = true;
        allSerial &= serialValues.contains(operand);
        auto selected = serialProjectionGroups.find(operand);
        if (selected != serialProjectionGroups.end())
          operandGroups.insert(selected->second);
      }
      if (operandGroups.size() > 1) {
        hasConflictingGroups = true;
        return;
      }
      if (!hasBlockOperand || !allSerial)
        return;
      for (mlir::Value result : operation->getResults()) {
        auto type = mlir::dyn_cast<weft::kernel::BlockType>(result.getType());
        if (!type || type.getShape().size() != 1)
          continue;
        mlir::Value representative;
        for (mlir::Value operand : operation->getOperands())
          if (mlir::isa<weft::kernel::BlockType>(operand.getType())) {
            representative = operand;
            break;
          }
        if (!representative || !weft::kernel::haveSameLogicalExtent(
                                   representative, 0, result, 0))
          continue;
        changed |= serialValues.insert(result).second;
        if (operandGroups.size() == 1)
          serialProjectionGroups.try_emplace(result, *operandGroups.begin());
      }
    });
  }
  return serialValues;
}

} // namespace

mlir::LogicalResult PlanOp::verify() {
  if (!llvm::isa_and_present<mlir::ModuleOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in a module");
  if (getTarget().trim().empty() || getTarget().trim() != getTarget())
    return emitOpError("target must be a non-empty, already-trimmed identity");

  weft::kernel::KernelOp source = resolveKernel(*this);
  if (!source)
    return emitOpError() << "references unknown weft_kernel.kernel "
                         << getKernelAttr();

  llvm::DenseMap<int64_t, mlir::Operation *> nodes;
  bool invalidNode = false;
  source->walk([&](mlir::Operation *operation) {
    auto id = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    if (!id || id.getInt() < 0 || !nodes.try_emplace(id.getInt(), operation).second)
      invalidNode = true;
  });
  if (invalidNode)
    return emitOpError(
        "requires every canonical source operation to carry one unique "
        "non-negative weft_execution.node index");

  llvm::DenseMap<int64_t, GroupOp> groups;
  llvm::SmallDenseSet<int64_t, 4> taskAxes;
  llvm::SmallDenseSet<int64_t, 4> metaArguments;
  llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> valueGroups;
  llvm::DenseMap<mlir::Value, int64_t> mappedValueGroups;

  for (mlir::Operation &operation : getBody().front()) {
    if (auto task = llvm::dyn_cast<TaskBindingOp>(operation)) {
      if (!taskAxes.insert(task.getAxisAttr().getInt()).second)
        return task.emitOpError("duplicates a task-axis binding");
      continue;
    }
    if (auto group = llvm::dyn_cast<GroupOp>(operation)) {
      if (!groups.try_emplace(group.getIdAttr().getInt(), group).second)
        return group.emitOpError("duplicates a layout group id");
      continue;
    }
    if (auto meta = llvm::dyn_cast<MetaBindingOp>(operation)) {
      int64_t argumentIndex = meta.getArgumentAttr().getInt();
      if (argumentIndex < 0 ||
          argumentIndex >=
              static_cast<int64_t>(source.getBody().front().getNumArguments()))
        return meta.emitOpError("argument is outside the canonical entry block");
      auto type = source.getBody().front().getArgument(argumentIndex).getType();
      auto constexprType = mlir::dyn_cast<weft::kernel::ConstexprType>(type);
      if (!constexprType || !constexprType.getValueType().isIndex())
        return meta.emitOpError("must bind an index constexpr argument");
      if (!metaArguments.insert(argumentIndex).second)
        return meta.emitOpError("duplicates a constexpr argument binding");
      continue;
    }
  }

  for (mlir::Operation &operation : getBody().front()) {
    if (llvm::isa<TaskBindingOp, MetaBindingOp, GroupOp, ValueLayoutOp,
                  LayoutConversionOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() == "weft_execution")
      return operation.emitOpError(
          "is not a recognized shared selected-execution record");
    if (!llvm::isa<SelectedOwnerOpInterface>(operation))
      return operation.emitOpError(
          "owner-local selected-execution records must implement "
          "SelectedOwnerOpInterface");
    auto group = operation.getAttrOfType<mlir::IntegerAttr>("group");
    auto groupList = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("groups");
    if (!group && !groupList)
      return operation.emitOpError(
          "owner-local selected-execution records must carry either one "
          "integer group or a non-empty groups back-reference");
    if (group && groupList)
      return operation.emitOpError(
          "owner-local selected-execution record cannot carry both group and "
          "groups back-references");
    if (group) {
      if (!groups.contains(group.getInt()))
        return operation.emitOpError(
            "owner-local selected-execution record references an unknown "
            "layout group");
      continue;
    }
    if (groupList.empty())
      return operation.emitOpError(
          "owner-local groups back-reference must not be empty");
    for (int64_t id : groupList.asArrayRef())
      if (!groups.contains(id))
        return operation.emitOpError(
            "owner-local selected-execution record references an unknown "
            "layout group in groups");
  }

  int64_t gridRank = source.getGridRankAttr().getInt();
  if (taskAxes.size() != static_cast<size_t>(gridRank))
    return emitOpError("requires exactly one task binding for every grid axis");
  for (int64_t axis = 0; axis < gridRank; ++axis)
    if (!taskAxes.contains(axis))
      return emitOpError() << "is missing task binding for grid axis " << axis;

  llvm::SmallDenseSet<int64_t, 4> expectedMetaArguments;
  for (auto [index, argument] :
       llvm::enumerate(source.getBody().front().getArguments()))
    if (mlir::isa<weft::kernel::ConstexprType>(argument.getType()))
      expectedMetaArguments.insert(index);
  bool exactMetaBindings = metaArguments.size() == expectedMetaArguments.size();
  for (int64_t argument : expectedMetaArguments)
    exactMetaBindings &= metaArguments.contains(argument);
  if (!exactMetaBindings)
    return emitOpError(
        "requires exactly one meta_binding for every constexpr argument");

  for (mlir::Operation &operation : getBody().front()) {
    auto valueLayout = llvm::dyn_cast<ValueLayoutOp>(operation);
    if (!valueLayout)
      continue;
    int64_t groupId = valueLayout.getGroupAttr().getInt();
    auto group = groups.find(groupId);
    if (group == groups.end())
      return valueLayout.emitOpError("references an unknown layout group");
    int64_t sourceNode = valueLayout.getSourceNodeAttr().getInt();
    mlir::Operation *node = nodes.lookup(sourceNode);
    if (!node)
      return valueLayout.emitOpError("references an unknown canonical node");
    int64_t resultIndex = valueLayout.getSourceResultAttr().getInt();
    if (resultIndex < 0 ||
        resultIndex >= static_cast<int64_t>(node->getNumResults()))
      return valueLayout.emitOpError("source_result is outside the node results");
    auto block = mlir::dyn_cast<weft::kernel::BlockType>(
        node->getResult(resultIndex).getType());
    if (!block)
      return valueLayout.emitOpError("must reference a canonical block result");
    if (group->second.getLayout().getRank() !=
        static_cast<int64_t>(block.getShape().size()))
      return valueLayout.emitOpError(
          "layout rank does not match the canonical block result");
    mlir::Operation *scope =
        nodes.lookup(group->second.getScopeNodeAttr().getInt());
    if (!scope)
      return valueLayout.emitOpError(
          "layout group references an unknown canonical scope");
    if (node->getBlock()->getParentOp() != scope)
      return valueLayout.emitOpError(
          "canonical block result is not defined directly in its layout "
          "group scope");
    auto key = std::make_pair(sourceNode, resultIndex);
    if (!valueGroups.try_emplace(key, groupId).second)
      return valueLayout.emitOpError("duplicates a canonical block result mapping");
    mappedValueGroups.try_emplace(node->getResult(resultIndex), groupId);
  }

  llvm::DenseMap<std::pair<int64_t, int64_t>, int64_t> conversionGroups;
  llvm::DenseSet<int64_t> groupsWithIncomingConversions;
  for (mlir::Operation &operation : getBody().front()) {
    auto conversion = llvm::dyn_cast<LayoutConversionOp>(operation);
    if (!conversion)
      continue;
    int64_t targetGroupId = conversion.getGroupAttr().getInt();
    auto targetGroup = groups.find(targetGroupId);
    if (targetGroup == groups.end())
      return conversion.emitOpError("references an unknown target group");
    int64_t consumerNode = conversion.getConsumerNodeAttr().getInt();
    mlir::Operation *consumer = nodes.lookup(consumerNode);
    if (!consumer)
      return conversion.emitOpError("consumer_node is not a canonical node");
    int64_t consumerOperand = conversion.getConsumerOperandAttr().getInt();
    if (consumerOperand < 0 ||
        consumerOperand >= static_cast<int64_t>(consumer->getNumOperands()))
      return conversion.emitOpError(
          "consumer_operand is outside the canonical operand list");
    mlir::Value sourceValue = consumer->getOperand(consumerOperand);
    auto sourceType =
        mlir::dyn_cast<weft::kernel::BlockType>(sourceValue.getType());
    if (!sourceType)
      return conversion.emitOpError(
          "consumer operand must be a canonical block value");
    auto sourceGroup = mappedValueGroups.find(sourceValue);
    if (sourceGroup == mappedValueGroups.end())
      return conversion.emitOpError(
          "consumer operand has no native value_layout group");
    if (sourceGroup->second == targetGroupId)
      return conversion.emitOpError(
          "source and target layout groups must be distinct");
    GroupOp sourceGroupOp = groups.lookup(sourceGroup->second);
    if (targetGroup->second.getLayout().getRank() !=
        static_cast<int64_t>(sourceType.getShape().size()))
      return conversion.emitOpError(
          "target layout rank must match the canonical operand rank");
    if (sourceGroupOp.getLayout() == targetGroup->second.getLayout() &&
        sourceGroupOp.getOwnerAttr() == targetGroup->second.getOwnerAttr())
      return conversion.emitOpError(
          "conversion cannot preserve both layout and execution owner");
    mlir::Operation *targetScope =
        nodes.lookup(targetGroup->second.getScopeNodeAttr().getInt());
    bool directUseInTargetScope =
        targetScope && consumer->getBlock()->getParentOp() == targetScope;
    bool entersCarriedLoop = false;
    if (auto loop = llvm::dyn_cast<weft::kernel::ForOp>(consumer)) {
      int64_t carried = consumerOperand - 3;
      if (targetScope == loop.getOperation() && carried >= 0 &&
          carried < static_cast<int64_t>(loop.getNumResults())) {
        auto yield = llvm::cast<weft::kernel::YieldOp>(
            loop.getBody().front().getTerminator());
        auto yieldedGroup = mappedValueGroups.find(yield.getValues()[carried]);
        auto resultGroup = mappedValueGroups.find(loop.getResult(carried));
        entersCarriedLoop =
            yieldedGroup != mappedValueGroups.end() &&
            yieldedGroup->second == targetGroupId &&
            resultGroup != mappedValueGroups.end() &&
            resultGroup->second == sourceGroup->second;
      }
    }
    bool exitsCarriedLoop = false;
    if (auto yield = llvm::dyn_cast<weft::kernel::YieldOp>(consumer)) {
      auto loop = yield->getParentOfType<weft::kernel::ForOp>();
      if (loop && loop->getBlock() && consumerOperand >= 0 &&
          consumerOperand < static_cast<int64_t>(loop.getNumResults()) &&
          targetScope == loop->getBlock()->getParentOp()) {
        auto initGroup =
            mappedValueGroups.find(loop.getInitArgs()[consumerOperand]);
        auto resultGroup =
            mappedValueGroups.find(loop.getResult(consumerOperand));
        exitsCarriedLoop =
            initGroup != mappedValueGroups.end() &&
            initGroup->second == targetGroupId &&
            resultGroup != mappedValueGroups.end() &&
            resultGroup->second == targetGroupId;
      }
    }
    if (!directUseInTargetScope && !entersCarriedLoop && !exitsCarriedLoop)
      return conversion.emitOpError(
          "consumer must be defined directly in the target group scope or be "
          "a canonical for/yield carried-state boundary");
    auto key = std::make_pair(consumerNode, consumerOperand);
    if (!conversionGroups.try_emplace(key, targetGroupId).second)
      return conversion.emitOpError(
          "duplicates the selected conversion for one canonical operand use");
    groupsWithIncomingConversions.insert(targetGroupId);
  }

  bool hasConflictingSerialGroups = false;
  llvm::DenseSet<mlir::Value> derivedSerialValues =
      collectDerivedSerialProjectionValues(source, groups, mappedValueGroups,
                                           conversionGroups,
                                           hasConflictingSerialGroups);
  if (hasConflictingSerialGroups)
    return emitOpError(
        "canonical serial projection combines values selected in different "
        "physical layout groups without an explicit use conversion");
  for (mlir::Value value : derivedSerialValues)
    if (mappedValueGroups.contains(value))
      return emitOpError(
          "derived rank-two reduction projections must not carry an "
          "independent native value_layout");
  bool missingValue = false;
  source->walk([&](mlir::Operation *operation) {
    auto id = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    if (!id)
      return;
    for (auto [index, result] : llvm::enumerate(operation->getResults())) {
      if (!mlir::isa<weft::kernel::BlockType>(result.getType()))
        continue;
      auto key = std::make_pair(id.getInt(), static_cast<int64_t>(index));
      if (!valueGroups.contains(key) &&
          !isUnmaterializedViewProvenance(result, valueGroups) &&
          !derivedSerialValues.contains(result))
        missingValue = true;
    }
  });
  if (missingValue)
    return emitOpError(
        "requires one value_layout for every physically realized canonical "
        "block result; only index-coordinate/expand_dims provenance and selected "
        "vector-axis reduction projections may remain derived");

  for (auto [id, group] : groups) {
    mlir::Operation *owner =
        mlir::SymbolTable::lookupSymbolIn(*this, group.getOwnerAttr());
    if (!owner)
      return group.emitOpError() << "references unknown owner config "
                                 << group.getOwnerAttr();
    if (owner->getParentOp() != getOperation())
      return group.emitOpError(
          "owner config must be nested directly in the selected plan");
    if (!llvm::isa<SelectedOwnerOpInterface>(owner))
      return group.emitOpError(
          "owner config must implement SelectedOwnerOpInterface");
    auto ownerGroup = owner->getAttrOfType<mlir::IntegerAttr>("group");
    auto ownerGroups = owner->getAttrOfType<mlir::DenseI64ArrayAttr>("groups");
    if (ownerGroup && ownerGroups)
      return group.emitOpError(
          "owner config cannot carry both group and groups back-references");
    bool ownsGroup = ownerGroup ? ownerGroup.getInt() == id
                                : ownerGroups && !ownerGroups.empty() &&
                                      llvm::is_contained(ownerGroups.asArrayRef(),
                                                         id);
    if (!ownsGroup)
      return group.emitOpError(
          "owner config must carry a matching group or groups back-reference");
    bool hasMember = llvm::any_of(valueGroups, [&](const auto &entry) {
      return entry.second == id;
    }) || groupsWithIncomingConversions.contains(id);
    if (!hasMember)
      return group.emitOpError(
          "has neither canonical block-result members nor incoming conversions");
    mlir::Operation *scope = nodes.lookup(group.getScopeNodeAttr().getInt());
    if (!scope)
      return group.emitOpError("scope_node references an unknown canonical node");
    if (!llvm::isa<weft::kernel::KernelOp, weft::kernel::ForOp>(scope))
      return group.emitOpError(
          "scope_node must reference a canonical kernel or structured loop");
  }
  return mlir::success();
}

mlir::LogicalResult TaskBindingOp::verify() {
  if (!llvm::isa_and_present<PlanOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in weft_execution.plan");
  if (getAxisAttr().getInt() < 0)
    return emitOpError("axis must be non-negative");
  if (getMapping() != "abi" && getMapping() != "serial" &&
      getMapping() != "hart")
    return emitOpError("mapping must be abi, serial, or hart");
  return mlir::success();
}

mlir::LogicalResult MetaBindingOp::verify() {
  if (!llvm::isa_and_present<PlanOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in weft_execution.plan");
  if (getArgumentAttr().getInt() < 0)
    return emitOpError("argument must be non-negative");
  if (getValueAttr().getInt() <= 0)
    return emitOpError("first execution slice requires a positive integer value");
  return mlir::success();
}

mlir::LogicalResult GroupOp::verify() {
  if (!llvm::isa_and_present<PlanOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in weft_execution.plan");
  if (getIdAttr().getInt() < 0 || getScopeNodeAttr().getInt() < 0)
    return emitOpError("id and scope_node must be non-negative");
  return mlir::success();
}

mlir::LogicalResult ValueLayoutOp::verify() {
  if (!llvm::isa_and_present<PlanOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in weft_execution.plan");
  if (getSourceNodeAttr().getInt() < 0 ||
      getSourceResultAttr().getInt() < 0 || getGroupAttr().getInt() < 0)
    return emitOpError(
        "source_node, source_result and group must be non-negative");
  return mlir::success();
}

mlir::LogicalResult LayoutConversionOp::verify() {
  if (!llvm::isa_and_present<PlanOp>(getOperation()->getParentOp()))
    return emitOpError("must be nested directly in weft_execution.plan");
  if (getConsumerNodeAttr().getInt() < 0 ||
      getConsumerOperandAttr().getInt() < 0 || getGroupAttr().getInt() < 0)
    return emitOpError(
        "consumer_node, consumer_operand and group must be non-negative");
  return mlir::success();
}

void WEFTExecutionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Execution/IR/ExecutionOps.cpp.inc"
      >();
}
