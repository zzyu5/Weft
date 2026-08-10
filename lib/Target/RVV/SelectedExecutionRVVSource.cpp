#include "Weft/Target/RVV/SelectedExecutionRVVSource.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionDialect.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace weft::target::rvv {
namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");

llvm::Error makeEmissionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("selected Weft RVV source emission failed: ") + message,
      llvm::errc::invalid_argument);
}

bool isCIdentifier(llvm::StringRef value) {
  if (value.empty())
    return false;
  unsigned char first = static_cast<unsigned char>(value.front());
  if (!std::isalpha(first) && value.front() != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_';
  });
}

bool isBlocked(mlir::Value value) {
  return mlir::isa<kernel::BlockType>(value.getType());
}

bool containsCanonicalStore(mlir::Operation &operation) {
  bool found = false;
  operation.walk([&](kernel::StoreOp) { found = true; });
  return found;
}

bool hasInterveningStore(mlir::Block &block, mlir::Operation *producer,
                         mlir::Operation *consumer) {
  bool afterProducer = false;
  for (mlir::Operation &operation : block) {
    if (&operation == producer) {
      afterProducer = true;
      continue;
    }
    if (&operation == consumer)
      return false;
    if (afterProducer && containsCanonicalStore(operation))
      return true;
  }
  return true;
}

std::string lmulSpelling(int64_t eighths) {
  return llvm::StringSwitch<std::string>(llvm::Twine(eighths).str())
      .Case("4", "mf2")
      .Case("8", "m1")
      .Case("16", "m2")
      .Case("32", "m4")
      .Case("64", "m8")
      .Default("");
}

struct VectorConfig {
  int64_t sew = 0;
  int64_t laneRatio = 0;
  std::string lmul;
  std::string suffix;
  std::string cType;
  bool mask = false;
  bool floating = false;
  bool unsignedInteger = false;
  bool address = false;
};

struct GroupInfo {
  execution::GroupOp group;
  rvv_execution::GroupConfigOp config;
  int64_t maxSEW = 0;
};

struct ContractSelection {
  llvm::SmallVector<int64_t, 3> groups;
  std::string strategy;
};

struct LaneExpression {
  std::string expression;
  std::string stride;
};

struct PointerLaneExpression {
  std::string pointer;
  std::string elementStride;
};

enum class Phase { Invariant, Vector, Post };

class KernelSourceEmitter {
public:
  KernelSourceEmitter(mlir::ModuleOp module, execution::PlanOp plan,
                      llvm::raw_ostream &os)
      : module(module), plan(plan), os(os) {}

  llvm::Error emit() {
    if (llvm::Error error = initialize())
      return error;
    if (!isCIdentifier(kernel.getSymName()))
      return makeEmissionError(llvm::Twine("kernel symbol '") +
                               kernel.getSymName() +
                               "' is not a valid C identifier");

    mlir::Block &entry = kernel.getBody().front();
    llvm::SmallVector<std::string, 8> parameters;
    auto argNames = kernel.getArgNames();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
      auto nameAttr = mlir::cast<mlir::StringAttr>(argNames[index]);
      llvm::StringRef name = nameAttr.getValue();
      if (!isCIdentifier(name))
        return makeEmissionError(llvm::Twine("kernel argument '") + name +
                                 "' is not a valid C identifier");
      if (auto meta = mlir::dyn_cast<kernel::ConstexprType>(argument.getType())) {
        (void)meta;
        auto binding = metaBindings.find(index);
        if (binding == metaBindings.end())
          return makeEmissionError("missing selected constexpr binding");
        expressions[argument] = std::to_string(binding->second);
        continue;
      }

      auto cType = getScalarCType(argument.getType());
      if (!cType)
        return cType.takeError();
      if (mlir::isa<kernel::PtrType>(argument.getType()) &&
          !pointerReachesStore(argument))
        *cType = "const " + *cType;
      parameters.push_back(*cType + " " + name.str());
      expressions[argument] = name.str();
    }

    for (int64_t axis = 0; axis < kernel.getGridRankAttr().getInt(); ++axis) {
      auto mapping = taskBindings.find(axis);
      if (mapping == taskBindings.end() || mapping->second != "abi")
        return makeEmissionError(
            "source emitter currently requires every task axis to use ABI binding");
      std::string name = "__weft_task_" + std::to_string(axis);
      parameters.push_back("size_t " + name);
      taskNames.try_emplace(axis, std::move(name));
    }

    os << "extern \"C\" void " << kernel.getSymName() << "(";
    llvm::interleaveComma(parameters, os);
    os << ") {\n";
    llvm::DenseMap<mlir::Value, std::string> parameterExpressions = expressions;
    auto entryGroup = scopeGroups.find(kernel.getOperation());
    if (entryGroup != scopeGroups.end()) {
      llvm::Error error = llvm::Error::success();
      if (entryGroup->second.size() == 1) {
        GroupInfo *group = findGroup(entryGroup->second.front());
        if (!group)
          return makeEmissionError("entry layout group cannot be resolved");
        int64_t rank = group->group.getLayout().getRank();
        error = rank == 1   ? emitRank1Scope(entry, *group, 1)
                : rank == 2 ? emitRank2Scope(entry, *group, 1)
                            : emitRankNElementwiseScope(entry, *group, 1);
      } else if (llvm::any_of(contractSelections, [&](const auto &entry) {
                   return entry.first->getBlock() == &kernel.getBody().front() &&
                          entry.second.groups.size() > 1;
                 })) {
        error = emitRank2ContractScope(entry, entryGroup->second, 1);
      } else {
        for (int64_t groupID : entryGroup->second) {
          GroupInfo *group = findGroup(groupID);
          if (!group || group->group.getLayout().getRank() != 2)
            return makeEmissionError(
                "ordinary multi-group source emission currently requires "
                "rank-two RVV groups");
          llvm::DenseSet<mlir::Operation *> operations =
              collectRank2GroupOperations(entry, groupID);
          if (operations.empty())
            return makeEmissionError(
                "selected rank-two group has no canonical operation slice");
          expressions = parameterExpressions;
          vectorValues.clear();
          pointerStrides.clear();
          line(1, "{");
          error = emitRank2Scope(entry, *group, 2, &operations);
          line(1, "}");
          if (error)
            break;
        }
      }
      if (error)
        return error;
    } else {
      for (mlir::Operation &operation : entry) {
        if (llvm::isa<kernel::ReturnOp>(operation))
          continue;
        if (llvm::Error error = emitScalarOperation(operation, 1))
          return error;
      }
    }
    os << "}\n\n";
    return llvm::Error::success();
  }

private:
  llvm::Error initialize() {
    kernel = llvm::dyn_cast_or_null<kernel::KernelOp>(
        mlir::SymbolTable::lookupSymbolIn(module, plan.getKernelAttr()));
    if (!kernel)
      return makeEmissionError("plan does not resolve to a canonical kernel");

    kernel->walk([&](mlir::Operation *operation) {
      auto id = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
      if (id)
        nodes.try_emplace(id.getInt(), operation);
    });

    for (mlir::Operation &operation : plan.getBody().front()) {
      if (auto binding = llvm::dyn_cast<execution::TaskBindingOp>(operation)) {
        taskBindings.try_emplace(binding.getAxisAttr().getInt(),
                                 binding.getMapping().str());
        continue;
      }
      if (auto binding = llvm::dyn_cast<execution::MetaBindingOp>(operation)) {
        metaBindings.try_emplace(binding.getArgumentAttr().getInt(),
                                 binding.getValueAttr().getInt());
        continue;
      }
      if (llvm::isa<execution::ValueLayoutOp, execution::LayoutConversionOp,
                    rvv_execution::GroupConfigOp>(operation))
        continue;
      if (auto unary =
              llvm::dyn_cast<rvv_execution::UnaryConfigOp>(operation)) {
        mlir::Operation *source =
            nodes.lookup(unary.getSourceNodeAttr().getInt());
        if (!llvm::isa_and_nonnull<kernel::UnaryOp>(source))
          return makeEmissionError(
              "unary_config does not resolve to a canonical unary operation");
        if (!unaryStrategies.try_emplace(source, unary.getStrategy().str()).second)
          return makeEmissionError(
              "canonical unary operation has duplicate selected strategies");
        continue;
      }
      if (auto reduction =
              llvm::dyn_cast<rvv_execution::ReductionConfigOp>(operation)) {
        mlir::Operation *source =
            nodes.lookup(reduction.getSourceNodeAttr().getInt());
        if (!llvm::isa_and_nonnull<kernel::ReduceOp>(source))
          return makeEmissionError(
              "reduction_config does not resolve to a canonical reduction");
        if (!reductionStrategies
                 .try_emplace(source, reduction.getStrategy().str())
                 .second)
          return makeEmissionError(
              "canonical reduction has duplicate selected strategies");
        reductionGroups.try_emplace(source,
                                    reduction.getGroupAttr().getInt());
        continue;
      }
      if (auto contract =
              llvm::dyn_cast<rvv_execution::ContractConfigOp>(operation)) {
        mlir::Operation *source =
            nodes.lookup(contract.getSourceNodeAttr().getInt());
        if (!llvm::isa_and_nonnull<kernel::ContractOp>(source))
          return makeEmissionError(
              "contract_config does not resolve to a canonical contraction");
        ContractSelection selection;
        selection.groups.append(contract.getGroups().begin(),
                                contract.getGroups().end());
        selection.strategy = contract.getStrategy().str();
        if (!contractSelections
                 .try_emplace(source, std::move(selection))
                 .second)
          return makeEmissionError(
              "canonical contraction has duplicate selected strategies");
        continue;
      }
      auto group = llvm::dyn_cast<execution::GroupOp>(operation);
      if (!group)
        return makeEmissionError(
            llvm::Twine("unsupported selected-plan record ") +
            operation.getName().getStringRef());
      auto owner = llvm::dyn_cast_or_null<rvv_execution::GroupConfigOp>(
          mlir::SymbolTable::lookupSymbolIn(plan, group.getOwnerAttr()));
      if (!owner)
        return makeEmissionError("layout group is not owned by an RVV config");
      int64_t id = group.getIdAttr().getInt();
      groups.try_emplace(id, GroupInfo{group, owner, 0});
      mlir::Operation *scope = nodes.lookup(group.getScopeNodeAttr().getInt());
      if (!scope)
        return makeEmissionError("layout group has an unknown source scope");
      scopeGroups[scope].push_back(id);
    }

    for (mlir::Operation &operation : plan.getBody().front()) {
      auto valueLayout = llvm::dyn_cast<execution::ValueLayoutOp>(operation);
      if (!valueLayout)
        continue;
      GroupInfo *group = findGroup(valueLayout.getGroupAttr().getInt());
      mlir::Operation *source =
          nodes.lookup(valueLayout.getSourceNodeAttr().getInt());
      int64_t result = valueLayout.getSourceResultAttr().getInt();
      if (!group || !source || result < 0 ||
          result >= static_cast<int64_t>(source->getNumResults()))
        return makeEmissionError("value_layout cannot be resolved");
      mlir::Value sourceValue = source->getResult(result);
      if (!valueGroups.try_emplace(sourceValue,
                                   valueLayout.getGroupAttr().getInt())
               .second)
        return makeEmissionError("canonical value has duplicate layout groups");
      if (!group->group.getLayout().getVectorAxes().empty()) {
        auto config = getVectorConfig(source->getResult(result).getType(), *group);
        if (!config)
          return config.takeError();
        if (!config->mask)
          group->maxSEW = std::max(group->maxSEW, config->sew);
      }
    }

    for (mlir::Operation &operation : plan.getBody().front()) {
      auto conversion = llvm::dyn_cast<execution::LayoutConversionOp>(operation);
      if (!conversion)
        continue;
      mlir::Operation *consumer =
          nodes.lookup(conversion.getConsumerNodeAttr().getInt());
      int64_t operandIndex = conversion.getConsumerOperandAttr().getInt();
      GroupInfo *target = findGroup(conversion.getGroupAttr().getInt());
      if (!consumer || operandIndex < 0 ||
          operandIndex >= static_cast<int64_t>(consumer->getNumOperands()) ||
          !target)
        return makeEmissionError("layout_conversion cannot be resolved");
      mlir::Value source = consumer->getOperand(operandIndex);
      auto sourceGroupID = valueGroups.find(source);
      if (sourceGroupID == valueGroups.end())
        return makeEmissionError(
            "layout_conversion source has no native selected group");
      GroupInfo *sourceGroup = findGroup(sourceGroupID->second);
      if (!sourceGroup)
        return makeEmissionError("layout_conversion source group is unknown");
      bool entersLoop = false;
      if (auto loop = llvm::dyn_cast<kernel::ForOp>(consumer)) {
        int64_t carried = operandIndex - 3;
        if (carried >= 0 &&
            carried < static_cast<int64_t>(loop.getNumResults()) &&
            nodes.lookup(target->group.getScopeNodeAttr().getInt()) ==
                loop.getOperation()) {
          auto yield = llvm::cast<kernel::YieldOp>(
              loop.getBody().front().getTerminator());
          auto yieldedGroup = valueGroups.find(yield.getValues()[carried]);
          auto resultGroup = valueGroups.find(loop.getResult(carried));
          entersLoop = yieldedGroup != valueGroups.end() &&
                       yieldedGroup->second ==
                           target->group.getIdAttr().getInt() &&
                       resultGroup != valueGroups.end() &&
                       resultGroup->second == sourceGroupID->second;
        }
      }
      bool exitsLoop = false;
      if (auto yield = llvm::dyn_cast<kernel::YieldOp>(consumer)) {
        auto loop = yield->getParentOfType<kernel::ForOp>();
        if (loop && loop->getBlock() && operandIndex >= 0 &&
            operandIndex < static_cast<int64_t>(loop.getNumResults()) &&
            nodes.lookup(target->group.getScopeNodeAttr().getInt()) ==
                loop->getBlock()->getParentOp()) {
          auto initGroup = valueGroups.find(loop.getInitArgs()[operandIndex]);
          auto resultGroup = valueGroups.find(loop.getResult(operandIndex));
          exitsLoop = initGroup != valueGroups.end() &&
                      initGroup->second == target->group.getIdAttr().getInt() &&
                      resultGroup != valueGroups.end() &&
                      resultGroup->second == target->group.getIdAttr().getInt();
        }
      }
      bool rematerializesSplat = false;
      bool rematerializesLoad = false;
      if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(consumer)) {
        mlir::Operation *sourceScope = nodes.lookup(
            sourceGroup->group.getScopeNodeAttr().getInt());
        mlir::Operation *targetScope =
            nodes.lookup(target->group.getScopeNodeAttr().getInt());
        auto vectorAxes = target->group.getLayout().getVectorAxes();
        rematerializesSplat =
            operandIndex == 0 && source.getDefiningOp<kernel::SplatOp>() &&
            sourceScope && sourceScope == targetScope &&
            consumer->getBlock()->getParentOp() == targetScope &&
            vectorAxes.size() == 1 &&
            vectorAxes.front() == reduction.getAxisAttr().getInt();
        auto load = source.getDefiningOp<kernel::LoadOp>();
        rematerializesLoad =
            operandIndex == 0 && load && load->getBlock() == consumer->getBlock() &&
            sourceScope && sourceScope == targetScope &&
            consumer->getBlock()->getParentOp() == targetScope &&
            vectorAxes.size() == 1 &&
            vectorAxes.front() == reduction.getAxisAttr().getInt() &&
            !hasInterveningStore(*consumer->getBlock(), load.getOperation(),
                                 consumer);
      }
      if (entersLoop || exitsLoop) {
        if (sourceGroup->group.getLayout() != target->group.getLayout())
          return makeEmissionError(
              "carried state conversion must preserve its blocked layout");
        auto sourceRatio = sourceGroup->config.getLaneRatioAttr();
        auto targetRatio = target->config.getLaneRatioAttr();
        if (static_cast<bool>(sourceRatio) != static_cast<bool>(targetRatio) ||
            (sourceRatio && sourceRatio.getInt() != targetRatio.getInt()))
          return makeEmissionError(
              "carried state conversion requires one physical lane ratio");
      } else if (!rematerializesSplat && !rematerializesLoad) {
        return makeEmissionError(
            "ordinary layout_conversion currently supports only a canonical "
            "splat or store-free masked load rematerialized for a reduction "
            "use");
      }
      if (!conversionGroups
               .try_emplace(std::make_pair(consumer, operandIndex),
                            conversion.getGroupAttr().getInt())
               .second)
        return makeEmissionError("duplicate emitted layout conversion use edge");
      auto &incoming =
          incomingConversionValues[conversion.getGroupAttr().getInt()];
      if (rematerializesSplat || rematerializesLoad) {
        llvm::DenseSet<mlir::Value> visited;
        collectRematerializedProducerValues(source, *consumer->getBlock(),
                                            incoming, visited);
      } else {
        incoming.insert(source);
      }
      if (!target->group.getLayout().getVectorAxes().empty()) {
        auto config = getVectorConfig(source.getType(), *target);
        if (!config)
          return config.takeError();
        if (!config->mask)
          target->maxSEW = std::max(target->maxSEW, config->sew);
      }
    }
    return llvm::Error::success();
  }

  GroupInfo *findGroup(int64_t id) {
    auto iterator = groups.find(id);
    return iterator == groups.end() ? nullptr : &iterator->second;
  }

  void collectRematerializedProducerValues(
      mlir::Value value, mlir::Block &block,
      llvm::DenseSet<mlir::Value> &values,
      llvm::DenseSet<mlir::Value> &visited) const {
    if (!value || !visited.insert(value).second)
      return;
    if (isBlocked(value))
      values.insert(value);
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != &block)
      return;
    for (mlir::Value operand : definition->getOperands())
      collectRematerializedProducerValues(operand, block, values, visited);
  }

  llvm::Expected<GroupInfo *> findScopeGroup(mlir::Operation *scope) {
    auto id = scopeGroups.find(scope);
    if (id == scopeGroups.end())
      return makeEmissionError("vectorized source scope has no layout group");
    if (id->second.size() != 1)
      return makeEmissionError(
          "generic vector scope requires exactly one selected layout group");
    GroupInfo *group = findGroup(id->second.front());
    if (!group)
      return makeEmissionError("source scope references an unknown layout group");
    return group;
  }

  llvm::Expected<VectorConfig> getVectorConfig(mlir::Type type,
                                                GroupInfo &group) const {
    auto block = mlir::dyn_cast<kernel::BlockType>(type);
    if (!block)
      return makeEmissionError("expected a canonical block type");
    mlir::IntegerAttr laneRatio = group.config.getLaneRatioAttr();
    if (!laneRatio)
      return makeEmissionError(
          "vector configuration requested from a serial selected group");
    int64_t ratio = laneRatio.getInt();
    mlir::Type element = block.getElementType();
    VectorConfig config;
    config.laneRatio = ratio;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
      if (integer.getWidth() == 1) {
        config.mask = true;
        config.cType = "vbool" + std::to_string(ratio) + "_t";
        return config;
      }
      config.sew = integer.getWidth();
      config.unsignedInteger = integer.isUnsigned();
    } else if (element.isIndex()) {
      config.sew = 64;
      config.unsignedInteger = true;
    } else if (auto floating = mlir::dyn_cast<mlir::FloatType>(element)) {
      config.sew = floating.getWidth();
      config.floating = true;
    } else if (mlir::isa<kernel::PtrType>(element)) {
      config.sew = 64;
      config.unsignedInteger = true;
      config.address = true;
    } else {
      return makeEmissionError("unsupported canonical block element type");
    }

    if ((config.sew * 8) % ratio != 0)
      return makeEmissionError("selected lane ratio does not form an RVV LMUL");
    config.lmul = lmulSpelling(config.sew * 8 / ratio);
    if (config.lmul.empty())
      return makeEmissionError("selected lane ratio forms an unsupported RVV LMUL");
    if (config.floating) {
      config.suffix = "f" + std::to_string(config.sew) + config.lmul;
      config.cType = "vfloat" + std::to_string(config.sew) + config.lmul + "_t";
    } else {
      std::string prefix = config.unsignedInteger ? "u" : "i";
      config.suffix = prefix + std::to_string(config.sew) + config.lmul;
      config.cType = "v" + (config.unsignedInteger ? std::string("uint")
                                                    : std::string("int")) +
                     std::to_string(config.sew) + config.lmul + "_t";
    }
    return config;
  }

  llvm::Expected<std::string> getScalarCType(mlir::Type type) const {
    if (type.isIndex())
      return std::string("size_t");
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
      if (integer.getWidth() == 1)
        return std::string("bool");
      if (integer.getWidth() == 8 || integer.getWidth() == 16 ||
          integer.getWidth() == 32 || integer.getWidth() == 64)
        return std::string(integer.isUnsigned() ? "uint" : "int") +
               std::to_string(integer.getWidth()) + "_t";
    }
    if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
      if (floating.getWidth() == 32)
        return std::string("float");
      if (floating.getWidth() == 64)
        return std::string("double");
    }
    if (auto pointer = mlir::dyn_cast<kernel::PtrType>(type)) {
      auto element = getScalarCType(pointer.getElementType());
      if (!element)
        return element.takeError();
      return *element + "*";
    }
    return makeEmissionError("unsupported scalar or pointer type in C ABI");
  }

  bool pointerReachesStore(mlir::Value value) {
    llvm::DenseSet<mlir::Value> visited;
    return pointerReachesStore(value, visited);
  }

  bool pointerReachesStore(mlir::Value value,
                           llvm::DenseSet<mlir::Value> &visited) {
    if (!visited.insert(value).second)
      return false;
    for (mlir::Operation *user : value.getUsers()) {
      if (auto store = llvm::dyn_cast<kernel::StoreOp>(user))
        if (store.getPointer() == value)
          return true;
      if (auto pointerAdd = llvm::dyn_cast<kernel::PtrAddOp>(user))
        if (pointerAdd.getBase() == value &&
            pointerReachesStore(pointerAdd.getResult(), visited))
          return true;
    }
    return false;
  }

  std::string valueName(mlir::Value value) const {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      auto node = result.getOwner()->getAttrOfType<mlir::IntegerAttr>(
          kNodeAttrName);
      if (node)
        return "__weft_v" + std::to_string(node.getInt()) + "_" +
               std::to_string(result.getResultNumber());
    }
    return "__weft_value";
  }

  llvm::Expected<std::string> lookup(mlir::Value value) const {
    auto iterator = expressions.find(value);
    if (iterator == expressions.end())
      return makeEmissionError("SSA value has no emitted source binding");
    return iterator->second;
  }

  llvm::Expected<std::string>
  lookupVectorMask(mlir::Value mask, GroupInfo &group,
                   llvm::StringRef vl) const {
    auto expression = lookup(mask);
    if (!expression)
      return expression.takeError();

    auto block = mlir::dyn_cast<kernel::BlockType>(mask.getType());
    if (block) {
      auto config = getVectorConfig(mask.getType(), group);
      if (!config)
        return config.takeError();
      if (!config->mask)
        return makeEmissionError(
            "blocked RVV memory mask must have i1 elements");
      if (vectorValues.contains(mask))
        return *expression;

      auto layout = group.group.getLayout();
      auto vectorAxes = layout.getVectorAxes();
      if (vectorAxes.size() != 1 ||
          static_cast<int64_t>(block.getShape().size()) != layout.getRank() ||
          block.getShape()[vectorAxes.front()] != 1)
        return makeEmissionError(
            "lane-varying RVV memory mask was not materialized as a predicate "
            "vector");
    } else if (!mask.getType().isInteger(1)) {
      return makeEmissionError("scalar RVV memory mask must be i1");
    }

    int64_t ratio = group.config.getLaneRatioAttr().getInt();
    std::string suffix =
        "_m_b" + std::to_string(ratio) + "(" + vl.str() + ")";
    return "(" + *expression + " ? __riscv_vmset" + suffix +
           " : __riscv_vmclr" + suffix + ")";
  }

  void line(unsigned indent, llvm::StringRef text) {
    os.indent(indent * 2) << text << "\n";
  }

  llvm::Expected<std::string> constantLiteral(mlir::TypedAttr attribute) const {
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
      if (integer.getType().isInteger(1))
        return std::string(integer.getInt() ? "true" : "false");
      return std::to_string(integer.getInt());
    }
    if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attribute)) {
      std::string literal;
      llvm::raw_string_ostream stream(literal);
      stream << floating.getValueAsDouble();
      stream.flush();
      if (!llvm::StringRef(literal).contains('.') &&
          !llvm::StringRef(literal).contains_insensitive('e'))
        literal += ".0";
      if (mlir::cast<mlir::FloatType>(floating.getType()).getWidth() == 32)
        literal += "f";
      return literal;
    }
    return makeEmissionError("constant has no supported C literal");
  }

  bool isNegativeIndexConstant(mlir::Value value) const {
    auto constant = value.getDefiningOp<kernel::ConstantOp>();
    if (!constant)
      return false;
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    return integer && integer.getInt() < 0;
  }

  bool operationDependsOnBlockedValue(
      mlir::Operation &operation,
      const llvm::DenseSet<mlir::Value> &dependentValues) const {
    return llvm::any_of(operation.getOperands(), [&](mlir::Value operand) {
             return isBlocked(operand) || dependentValues.contains(operand);
           }) ||
           llvm::any_of(operation.getResults(), isBlocked);
  }

  void collectArangeAxes(mlir::Value value, int64_t logicalAxis,
                         int64_t groupID, int64_t targetRank,
                         llvm::DenseSet<mlir::Value> &visited,
                         llvm::DenseSet<int64_t> &axes) const {
    if (!visited.insert(value).second)
      return;
    auto valueType = mlir::dyn_cast<kernel::BlockType>(value.getType());
    if (!valueType ||
        static_cast<int64_t>(valueType.getShape().size()) > targetRank)
      return;
    auto mapped = valueGroups.find(value);
    auto incoming = incomingConversionValues.find(groupID);
    bool reachesGroup =
        (mapped != valueGroups.end() && mapped->second == groupID) ||
        (incoming != incomingConversionValues.end() &&
         incoming->second.contains(value));
    if (reachesGroup) {
      if (static_cast<int64_t>(valueType.getShape().size()) == targetRank)
        axes.insert(logicalAxis);
    }
    for (mlir::Operation *user : value.getUsers()) {
      if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(user)) {
        if (expand.getInput() != value)
          continue;
        int64_t insertedAxis = expand.getAxisAttr().getInt();
      int64_t expandedAxis =
          logicalAxis >= insertedAxis ? logicalAxis + 1 : logicalAxis;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      collectArangeAxes(expand.getResult(), expandedAxis, groupID, targetRank,
                        branchVisited, axes);
        continue;
      }
      if (auto splat = llvm::dyn_cast<kernel::SplatOp>(user)) {
        if (splat.getShapeLike() == value) {
          llvm::DenseSet<mlir::Value> branchVisited = visited;
          collectArangeAxes(splat.getResult(), logicalAxis, groupID, targetRank,
                            branchVisited, axes);
        }
        continue;
      }
      auto binary = llvm::dyn_cast<kernel::BinaryOp>(user);
      if (!binary || !valueType.getElementType().isIndex() ||
          (binary.getLhs() != value && binary.getRhs() != value))
        continue;
      auto resultType = mlir::dyn_cast<kernel::BlockType>(
          binary.getResult().getType());
      if (!resultType || !resultType.getElementType().isIndex() ||
          resultType.getShape().size() != valueType.getShape().size() ||
          !kernel::isIndexCoordinateProvenance(binary.getResult()))
        continue;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      collectArangeAxes(binary.getResult(), logicalAxis, groupID, targetRank,
                        branchVisited, axes);
    }
  }

  std::optional<int64_t> tryArangeAxis(kernel::ArangeOp arange,
                                       GroupInfo &group) const {
    llvm::DenseSet<mlir::Value> visited;
    llvm::DenseSet<int64_t> axes;
    collectArangeAxes(arange.getResult(), 0,
                      group.group.getIdAttr().getInt(),
                      group.group.getLayout().getRank(), visited, axes);
    if (axes.size() != 1)
      return std::nullopt;
    return *axes.begin();
  }

  llvm::Expected<int64_t> arangeAxis(kernel::ArangeOp arange,
                                     GroupInfo &group) const {
    auto axis = tryArangeAxis(arange, group);
    if (!axis)
      return makeEmissionError(
          "arange must reach exactly one logical axis of its selected layout "
          "through canonical coordinate/expand_dims provenance");
    return *axis;
  }

  llvm::Expected<std::string> projectedScalarCType(mlir::Type type) const {
    if (auto block = mlir::dyn_cast<kernel::BlockType>(type))
      return getScalarCType(block.getElementType());
    return getScalarCType(type);
  }

  llvm::Error emitSerialProjectionOperation(mlir::Operation &operation,
                                            llvm::StringRef serialCoordinate,
                                            unsigned indent) {
    if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
      if (!expressions.contains(reduction.getResult()))
        return makeEmissionError(
            "serial reduction projection has no row-local accumulator");
      return llvm::Error::success();
    }
    if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation)) {
      auto start = lookup(arange.getStart());
      if (!start)
        return start.takeError();
      expressions[arange.getResult()] =
          "(" + *start + " + " + serialCoordinate.str() + ")";
      return llvm::Error::success();
    }
    if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(operation)) {
      auto lhs = lookup(binary.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(binary.getRhs());
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef op = llvm::StringSwitch<llvm::StringRef>(binary.getKind())
                               .Case("add", "+")
                               .Case("sub", "-")
                               .Case("mul", "*")
                               .Case("div", "/")
                               .Case("mod", "%")
                               .Case("and", "&")
                               .Case("or", "|")
                               .Case("xor", "^")
                               .Default("");
      std::string expression;
      if (!op.empty())
        expression = "(" + *lhs + " " + op.str() + " " + *rhs + ")";
      else if (binary.getKind() == "max")
        expression = "(" + *lhs + " > " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else if (binary.getKind() == "min")
        expression = "(" + *lhs + " < " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else
        return makeEmissionError(
            "unsupported serial projection binary operation");
      std::string name = valueName(binary.getResult());
      line(indent, "auto " + name + " = " + expression + ";");
      expressions[binary.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation)) {
      auto input = lookup(unary.getInput());
      if (!input)
        return input.takeError();
      std::string expression;
      if (unary.getKind() == "neg")
        expression = "(-" + *input + ")";
      else if (unary.getKind() == "rsqrt")
        expression = "(1.0f / sqrtf(" + *input + "))";
      else if (unary.getKind() == "exp")
        expression = "expf(" + *input + ")";
      else
        return makeEmissionError(
            "unsupported serial projection unary operation");
      std::string name = valueName(unary.getResult());
      line(indent, "auto " + name + " = " + expression + ";");
      expressions[unary.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto cast = llvm::dyn_cast<kernel::CastOp>(operation)) {
      auto input = lookup(cast.getInput());
      if (!input)
        return input.takeError();
      auto type = projectedScalarCType(cast.getResult().getType());
      if (!type)
        return type.takeError();
      std::string name = valueName(cast.getResult());
      line(indent, *type + " " + name + " = static_cast<" + *type + ">(" +
                       *input + ");");
      expressions[cast.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto compare = llvm::dyn_cast<kernel::CompareOp>(operation)) {
      auto lhs = lookup(compare.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(compare.getRhs());
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef predicate =
          llvm::StringSwitch<llvm::StringRef>(compare.getPredicate())
              .Case("eq", "==")
              .Case("ne", "!=")
              .Case("lt", "<")
              .Case("le", "<=")
              .Case("gt", ">")
              .Case("ge", ">=")
              .Default("");
      if (predicate.empty())
        return makeEmissionError(
            "unsupported serial projection comparison predicate");
      std::string name = valueName(compare.getResult());
      line(indent, "bool " + name + " = (" + *lhs + " " + predicate.str() +
                       " " + *rhs + ");");
      expressions[compare.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto pointerAdd = llvm::dyn_cast<kernel::PtrAddOp>(operation)) {
      auto base = lookup(pointerAdd.getBase());
      if (!base)
        return base.takeError();
      auto offset = lookup(pointerAdd.getOffset());
      if (!offset)
        return offset.takeError();
      std::string name = valueName(pointerAdd.getResult());
      line(indent, "auto *" + name + " = " + *base + " + " + *offset + ";");
      expressions[pointerAdd.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto store = llvm::dyn_cast<kernel::StoreOp>(operation)) {
      auto pointer = lookup(store.getPointer());
      if (!pointer)
        return pointer.takeError();
      auto value = lookup(store.getValue());
      if (!value)
        return value.takeError();
      auto mask = lookup(store.getMask());
      if (!mask)
        return mask.takeError();
      line(indent, "if (" + *mask + ") *" + *pointer + " = " + *value + ";");
      return llvm::Error::success();
    }
    return makeEmissionError(llvm::Twine("no serial projection rule for ") +
                             operation.getName().getStringRef());
  }

  llvm::Error emitRank1Scope(mlir::Block &block, GroupInfo &group,
                             unsigned indent) {
    auto selectedLayout = group.group.getLayout();
    if (selectedLayout.getRank() != 1 ||
        selectedLayout.getVectorAxes().size() != 1 ||
        selectedLayout.getVectorAxes().front() != 0)
      return makeEmissionError(
          "rank-1 source emission requires its only logical axis to be vector");

    llvm::DenseMap<mlir::Operation *, Phase> phases;
    llvm::DenseSet<mlir::Value> postValues;
    for (mlir::Operation &operation : block) {
      bool blockDependent =
          llvm::any_of(operation.getOperands(), isBlocked) ||
          llvm::any_of(operation.getResults(), isBlocked);
      bool postDependent = llvm::any_of(operation.getOperands(), [&](mlir::Value value) {
        return postValues.contains(value);
      });
      Phase phase = blockDependent ? Phase::Vector
                    : postDependent ? Phase::Post
                                    : Phase::Invariant;
      if (llvm::isa<kernel::ReduceOp, kernel::ContractOp>(operation)) {
        phase = Phase::Vector;
        for (mlir::Value result : operation.getResults())
          postValues.insert(result);
      } else if (phase == Phase::Post) {
        for (mlir::Value result : operation.getResults())
          postValues.insert(result);
      }
      phases.try_emplace(&operation, phase);
    }

    for (mlir::Operation &operation : block) {
      if (llvm::isa<kernel::ReturnOp>(operation) ||
          phases.lookup(&operation) != Phase::Invariant)
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent))
        return error;
    }

    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    for (mlir::Operation &operation : block) {
      auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation);
      if (!reduction)
        continue;
      if ((reduction.getKind() != "sum" && reduction.getKind() != "max" &&
           reduction.getKind() != "min") ||
          !reduction.getResult().getType().isF32())
        return makeEmissionError(
            "rank-1 source emission supports only f32 sum/max/min reduction");
      auto init = lookup(reduction.getInit());
      if (!init)
        return init.takeError();
      std::string name = valueName(reduction.getResult());
      line(indent, "float " + name + " = " + *init + ";");
      expressions[reduction.getResult()] = name;
      reductions.try_emplace(reduction, std::move(name));
    }
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : block) {
      auto contract = llvm::dyn_cast<kernel::ContractOp>(operation);
      if (!contract)
        continue;
      if (!contract.getResult().getType().isF32())
        return makeEmissionError(
            "rank-1 source emission supports only scalar f32 contraction");
      auto init = lookup(contract.getInit());
      if (!init)
        return init.takeError();
      std::string name = valueName(contract.getResult());
      line(indent, "float " + name + " = " + *init + ";");
      expressions[contract.getResult()] = name;
      contracts.try_emplace(contract, std::move(name));
    }

    std::optional<std::string> extent;
    for (mlir::Operation &operation : block) {
      auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation);
      if (!arange)
        continue;
      auto currentExtent = lookup(arange.getExtent());
      if (!currentExtent)
        return currentExtent.takeError();
      if (extent && *extent != *currentExtent)
        return makeEmissionError(
            "one rank-1 layout group requires a common logical extent");
      extent = *currentExtent;
    }
    if (!extent || group.maxSEW == 0)
      return makeEmissionError(
          "rank-1 layout group requires an arange extent and physical SEW");

    int64_t groupID = group.group.getIdAttr().getInt();
    int64_t ratio = group.config.getLaneRatioAttr().getInt();
    std::string setvlLMUL = lmulSpelling(group.maxSEW * 8 / ratio);
    if (setvlLMUL.empty())
      return makeEmissionError("cannot form rank-1 selected setvl configuration");
    std::string lane = "__weft_lane_g" + std::to_string(groupID);
    std::string vl = "__weft_vl_g" + std::to_string(groupID);
    line(indent, "for (size_t " + lane + " = 0; " + lane + " < " + *extent +
                     ";) {");
    line(indent + 1,
         "size_t " + vl + " = __riscv_vsetvl_e" +
             std::to_string(group.maxSEW) + setvlLMUL + "(" + *extent + " - " +
             lane + ");");
    for (mlir::Operation &operation : block) {
      if (phases.lookup(&operation) != Phase::Vector)
        continue;
      if (llvm::Error error = emitVectorOperation(
              operation, group, lane, vl, reductions, contracts, indent + 1))
        return error;
    }
    line(indent + 1, lane + " += " + vl + ";");
    line(indent, "}");

    for (mlir::Operation &operation : block) {
      if (llvm::isa<kernel::ReturnOp>(operation) ||
          phases.lookup(&operation) != Phase::Post)
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error emitRankNElementwiseScope(
      mlir::Block &block, GroupInfo &group, unsigned indent,
      const llvm::DenseSet<mlir::Operation *> *operationSlice = nullptr) {
    auto isActive = [&](mlir::Operation &operation) {
      return !operationSlice || operationSlice->contains(&operation);
    };
    auto selectedLayout = group.group.getLayout();
    int64_t rank = selectedLayout.getRank();
    if (rank <= 2 || selectedLayout.getVectorAxes().size() != 1)
      return makeEmissionError(
          "generic rank-N source emission requires rank greater than two and "
          "exactly one selected vector axis");
    int64_t vectorAxis = selectedLayout.getVectorAxes().front();
    if (selectedLayout.getOrder().size() != static_cast<size_t>(rank) ||
        selectedLayout.getOrder().front() != vectorAxis)
      return makeEmissionError(
          "generic rank-N source emission requires vector-fastest traversal");
    if (group.maxSEW == 0)
      return makeEmissionError("rank-N RVV layout group has no physical SEW");

    llvm::DenseMap<mlir::Operation *, bool> blockDependent;
    llvm::DenseSet<mlir::Value> dependentValues;
    for (mlir::Operation &operation : block) {
      if (!isActive(operation))
        continue;
      if (llvm::isa<kernel::ReduceOp, kernel::ContractOp>(operation))
        return makeEmissionError(
            "generic rank-N source emission does not yet realize higher-rank "
            "reduction or contraction");
      bool dependent = operationDependsOnBlockedValue(operation, dependentValues);
      blockDependent.try_emplace(&operation, dependent);
      if (dependent)
        for (mlir::Value result : operation.getResults())
          dependentValues.insert(result);
    }

    for (mlir::Operation &operation : block) {
      if (!isActive(operation) ||
          llvm::isa<kernel::ReturnOp, kernel::YieldOp>(operation) ||
          blockDependent.lookup(&operation))
        continue;
      if (llvm::isa<kernel::ForOp>(operation))
        return makeEmissionError(
            "generic rank-N physical scope cannot contain an unrelated loop");
      if (llvm::Error error = emitScalarOperation(operation, indent))
        return error;
    }

    llvm::DenseMap<int64_t, std::string> axisExtents;
    if (llvm::Error error = collectGroupAxisExtents(group, axisExtents))
      return error;
    for (int64_t axis = 0; axis < rank; ++axis)
      if (!axisExtents.contains(axis))
        return makeEmissionError(
            "rank-N layout requires one canonical logical extent per axis");

    int64_t groupID = group.group.getIdAttr().getInt();
    llvm::SmallVector<int64_t, 4> serialAxes;
    for (int64_t axis : selectedLayout.getOrder())
      if (axis != vectorAxis)
        serialAxes.push_back(axis);
    llvm::SmallVector<int64_t, 4> outerToInner(serialAxes.rbegin(),
                                               serialAxes.rend());
    activeAxisGroup = &group;
    activeVectorAxis = vectorAxis;
    activeSerialCoordinates.clear();

    unsigned currentIndent = indent;
    for (int64_t axis : outerToInner) {
      std::string coordinate = "__weft_axis" + std::to_string(axis) + "_g" +
                               std::to_string(groupID);
      line(currentIndent,
           "for (size_t " + coordinate + " = 0; " + coordinate + " < " +
               axisExtents.lookup(axis) + "; ++" + coordinate + ") {");
      activeSerialCoordinates.try_emplace(axis, coordinate);
      ++currentIndent;
    }

    std::string lane = "__weft_lane_g" + std::to_string(groupID);
    std::string vl = "__weft_vl_g" + std::to_string(groupID);
    line(currentIndent,
         "for (size_t " + lane + " = 0; " + lane + " < " +
             axisExtents.lookup(vectorAxis) + ";) {");
    int64_t ratio = group.config.getLaneRatioAttr().getInt();
    std::string setvlLMUL = lmulSpelling(group.maxSEW * 8 / ratio);
    if (setvlLMUL.empty()) {
      activeAxisGroup = nullptr;
      activeSerialCoordinates.clear();
      return makeEmissionError(
          "cannot form generic rank-N selected setvl configuration");
    }
    line(currentIndent + 1,
         "size_t " + vl + " = __riscv_vsetvl_e" +
             std::to_string(group.maxSEW) + setvlLMUL + "(" +
             axisExtents.lookup(vectorAxis) + " - " + lane + ");");
    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : block) {
      if (!isActive(operation) ||
          llvm::isa<kernel::ReturnOp, kernel::YieldOp>(operation) ||
          !blockDependent.lookup(&operation))
        continue;
      if (llvm::isa<kernel::ForOp>(operation)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return makeEmissionError(
            "generic rank-N blocked operation slice cannot contain a loop");
      }
      if (llvm::Error error = emitVectorOperation(
              operation, group, lane, vl, reductions, contracts,
              currentIndent + 1)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return error;
      }
    }
    line(currentIndent + 1, lane + " += " + vl + ";");
    line(currentIndent, "}");
    while (currentIndent > indent) {
      --currentIndent;
      line(currentIndent, "}");
    }
    activeAxisGroup = nullptr;
    activeSerialCoordinates.clear();
    return llvm::Error::success();
  }

  llvm::Error emitRank2Scope(
      mlir::Block &block, GroupInfo &group, unsigned indent,
      const llvm::DenseSet<mlir::Operation *> *operationSlice = nullptr) {
    auto isActive = [&](mlir::Operation &operation) {
      return !operationSlice || operationSlice->contains(&operation);
    };
    auto selectedLayout = group.group.getLayout();
    if (selectedLayout.getRank() != 2 ||
        selectedLayout.getVectorAxes().size() != 1)
      return makeEmissionError(
          "rank-2 source emission requires exactly one selected vector axis");
    int64_t vectorAxis = selectedLayout.getVectorAxes().front();
    int64_t serialAxis = vectorAxis == 0 ? 1 : 0;
    if (selectedLayout.getOrder().size() != 2 ||
        selectedLayout.getOrder().front() != vectorAxis ||
        selectedLayout.getOrder().back() != serialAxis)
      return makeEmissionError(
          "first rank-2 source slice requires vector-fastest traversal");
    if (group.maxSEW == 0)
      return makeEmissionError("rank-2 RVV layout group has no physical SEW");

    llvm::DenseMap<mlir::Operation *, bool> blockDependent;
    llvm::DenseSet<mlir::Value> dependentValues;
    for (mlir::Operation &operation : block) {
      if (!isActive(operation))
        continue;
      bool dependent = operationDependsOnBlockedValue(operation, dependentValues);
      blockDependent.try_emplace(&operation, dependent);
      if (dependent)
        for (mlir::Value result : operation.getResults())
          dependentValues.insert(result);
    }

    for (mlir::Operation &operation : block) {
      if (!isActive(operation) || llvm::isa<kernel::ReturnOp>(operation) ||
          blockDependent.lookup(&operation))
        continue;
      if (llvm::isa<kernel::ForOp>(operation))
        return makeEmissionError(
            "rank-2 physical groups cannot contain an unrelated scalar loop");
      if (llvm::Error error = emitScalarOperation(operation, indent))
        return error;
    }

    llvm::DenseMap<int64_t, std::string> axisExtents;
    if (llvm::Error error = collectGroupAxisExtents(group, axisExtents))
      return error;
    if (!axisExtents.contains(serialAxis) ||
        !axisExtents.contains(vectorAxis))
      return makeEmissionError(
          "rank-2 layout requires one structural arange extent per axis");

    llvm::DenseMap<mlir::Operation *, Phase> phases;
    llvm::DenseSet<mlir::Value> postValues;
    for (mlir::Operation &operation : block) {
      if (!isActive(operation))
        continue;
      if (llvm::isa<kernel::ReturnOp>(operation)) {
        phases.try_emplace(&operation, Phase::Invariant);
        continue;
      }
      bool postDependent = llvm::any_of(operation.getOperands(), [&](mlir::Value value) {
        return postValues.contains(value);
      });
      bool serialArange = false;
      if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation)) {
        if (!tryArangeAxis(arange, group)) {
          auto extent = lookup(arange.getExtent());
          if (!extent)
            return extent.takeError();
          serialArange = *extent == axisExtents.lookup(serialAxis);
        }
      }
      Phase phase = serialArange || postDependent
                        ? Phase::Post
                        : blockDependent.lookup(&operation) ? Phase::Vector
                                                            : Phase::Invariant;
      if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
        if (reduction.getAxisAttr().getInt() != vectorAxis)
          return makeEmissionError(
              "rank-2 reduction must project the selected vector axis");
        phase = Phase::Vector;
        postValues.insert(reduction.getResult());
      } else if (phase == Phase::Post) {
        for (mlir::Value result : operation.getResults())
          postValues.insert(result);
      }
      phases.try_emplace(&operation, phase);
    }

    int64_t groupID = group.group.getIdAttr().getInt();
    std::string serial = "__weft_axis" + std::to_string(serialAxis) + "_g" +
                         std::to_string(groupID);
    std::string lane = "__weft_lane_g" + std::to_string(groupID);
    std::string vl = "__weft_vl_g" + std::to_string(groupID);
    line(indent, "for (size_t " + serial + " = 0; " + serial + " < " +
                     axisExtents.lookup(serialAxis) + "; ++" + serial + ") {");

    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : block) {
      if (!isActive(operation))
        continue;
      auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation);
      if (!reduction)
        continue;
      auto result = mlir::dyn_cast<kernel::BlockType>(
          reduction.getResult().getType());
      bool supportedKind = reduction.getKind() == "sum" ||
                           reduction.getKind() == "max" ||
                           reduction.getKind() == "min";
      if (!result || result.getShape().size() != 1 ||
          !result.getElementType().isF32() || !supportedKind)
        return makeEmissionError(
            "rank-2 source emission supports f32 sum/max/min to one serial "
            "axis");
      auto init = lookup(reduction.getInit());
      if (!init)
        return init.takeError();
      std::string name = valueName(reduction.getResult());
      line(indent + 1, "float " + name + " = " + *init + ";");
      expressions[reduction.getResult()] = name;
      reductions.try_emplace(reduction, std::move(name));
    }

    line(indent + 1, "for (size_t " + lane + " = 0; " + lane + " < " +
                         axisExtents.lookup(vectorAxis) + ";) {");

    int64_t ratio = group.config.getLaneRatioAttr().getInt();
    std::string setvlLMUL = lmulSpelling(group.maxSEW * 8 / ratio);
    if (setvlLMUL.empty())
      return makeEmissionError("cannot form rank-2 selected setvl configuration");
    line(indent + 2,
         "size_t " + vl + " = __riscv_vsetvl_e" +
             std::to_string(group.maxSEW) + setvlLMUL + "(" +
             axisExtents.lookup(vectorAxis) + " - " + lane + ");");

    activeAxisGroup = &group;
    activeVectorAxis = vectorAxis;
    activeSerialCoordinates.clear();
    activeSerialCoordinates.try_emplace(serialAxis, serial);
    for (mlir::Operation &operation : block) {
      if (!isActive(operation) || phases.lookup(&operation) != Phase::Vector)
        continue;
      if (auto loop = llvm::dyn_cast<kernel::ForOp>(operation)) {
        if (llvm::Error error = emitRank2CarriedContractLoop(
                loop, group, lane, vl, indent + 2)) {
          activeAxisGroup = nullptr;
          activeSerialCoordinates.clear();
          return error;
        }
        continue;
      }
      if (llvm::Error error = emitVectorOperation(
              operation, group, lane, vl, reductions, contracts, indent + 2)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return error;
      }
    }
    line(indent + 2, lane + " += " + vl + ";");
    line(indent + 1, "}");

    for (mlir::Operation &operation : block) {
      if (!isActive(operation) || phases.lookup(&operation) != Phase::Post)
        continue;
      if (llvm::Error error =
              emitSerialProjectionOperation(operation, serial, indent + 1)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return error;
      }
    }
    activeAxisGroup = nullptr;
    activeSerialCoordinates.clear();
    line(indent, "}");
    return llvm::Error::success();
  }

  llvm::Error emitRank2CarriedContractLoop(
      kernel::ForOp loop, GroupInfo &outerGroup, llvm::StringRef lane,
      llvm::StringRef vl, unsigned indent) {
    (void)lane;
    auto scope = scopeGroups.find(loop.getOperation());
    if (scope == scopeGroups.end() || scope->second.size() != 3)
      return makeEmissionError(
          "block-carried rank-two loop requires exactly three body role groups");
    if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1 ||
        !isBlocked(loop.getInitArgs().front()) ||
        !isBlocked(loop.getResult(0)))
      return makeEmissionError(
          "first block-carried RVV loop requires exactly one block state");

    mlir::Block &body = loop.getBody().front();
    auto yield = llvm::dyn_cast<kernel::YieldOp>(body.getTerminator());
    if (!yield || body.getNumArguments() != 2 || yield.getValues().size() != 1)
      return makeEmissionError(
          "block-carried rank-two loop has an unsupported canonical body");

    kernel::ContractOp contract;
    const ContractSelection *selection = nullptr;
    for (auto &[source, current] : contractSelections) {
      auto candidate = llvm::dyn_cast<kernel::ContractOp>(source);
      if (!candidate || candidate->getBlock() != &body ||
          current.groups.size() != 3)
        continue;
      if (contract)
        return makeEmissionError(
            "one block-carried loop currently supports one rank-two contraction");
      contract = candidate;
      selection = &current;
    }
    if (!contract || !selection || selection->strategy != "sequential" ||
        contract.getInit() != body.getArgument(1) ||
        yield.getValues().front() != contract.getResult())
      return makeEmissionError(
          "carried loop must yield one sequential rank-two contraction of its "
          "body accumulator argument");

    llvm::SmallDenseSet<int64_t, 4> scopeSet(scope->second.begin(),
                                             scope->second.end());
    if (scopeSet.size() != 3 ||
        llvm::any_of(selection->groups,
                     [&](int64_t id) { return !scopeSet.contains(id); }))
      return makeEmissionError(
          "carried contraction roles must exactly cover its body groups");
    GroupInfo *lhsGroup = findGroup(selection->groups[0]);
    GroupInfo *rhsGroup = findGroup(selection->groups[1]);
    GroupInfo *resultGroup = findGroup(selection->groups[2]);
    if (!lhsGroup || !rhsGroup || !resultGroup ||
        !lhsGroup->group.getLayout().getVectorAxes().empty() ||
        rhsGroup->group.getLayout().getVectorAxes() !=
            llvm::ArrayRef<int64_t>({1}) ||
        resultGroup->group.getLayout() != outerGroup.group.getLayout())
      return makeEmissionError(
          "carried contraction requires serial lhs and identical N-vector "
          "rhs/result/outer layouts");

    int64_t outerGroupID = outerGroup.group.getIdAttr().getInt();
    auto initNative = valueGroups.find(loop.getInitArgs().front());
    auto resultNative = valueGroups.find(loop.getResult(0));
    auto entryConversion =
        conversionGroups.find(std::make_pair(loop.getOperation(), int64_t{3}));
    auto exitConversion = conversionGroups.find(
        std::make_pair(yield.getOperation(), int64_t{0}));
    if (initNative == valueGroups.end() || resultNative == valueGroups.end() ||
        initNative->second != outerGroupID ||
        resultNative->second != outerGroupID ||
        entryConversion == conversionGroups.end() ||
        entryConversion->second != selection->groups[2] ||
        exitConversion == conversionGroups.end() ||
        exitConversion->second != outerGroupID)
      return makeEmissionError(
          "carried contraction is missing its selected entry/exit use edges");

    auto outerRatio = outerGroup.config.getLaneRatioAttr();
    auto rhsRatio = rhsGroup->config.getLaneRatioAttr();
    auto resultRatio = resultGroup->config.getLaneRatioAttr();
    if (!outerRatio || !rhsRatio || !resultRatio ||
        outerRatio.getInt() != rhsRatio.getInt() ||
        outerRatio.getInt() != resultRatio.getInt())
      return makeEmissionError(
          "carried contraction requires one outer/rhs/result lane ratio");

    llvm::DenseMap<int64_t, std::string> lhsExtents;
    llvm::DenseMap<int64_t, std::string> rhsExtents;
    llvm::DenseMap<int64_t, std::string> resultExtents;
    if (llvm::Error error = collectGroupAxisExtents(*lhsGroup, lhsExtents))
      return error;
    if (llvm::Error error = collectGroupAxisExtents(*rhsGroup, rhsExtents))
      return error;
    if (llvm::Error error =
            collectGroupAxisExtents(*resultGroup, resultExtents))
      return error;
    if (lhsExtents.lookup(0) != resultExtents.lookup(0) ||
        lhsExtents.lookup(1) != rhsExtents.lookup(0) ||
        rhsExtents.lookup(1) != resultExtents.lookup(1))
      return makeEmissionError(
          "carried contraction role extents disagree after canonical "
          "provenance resolution");

    auto resultConfig =
        getVectorConfig(contract.getResult().getType(), *resultGroup);
    if (!resultConfig || !resultConfig->floating || resultConfig->sew != 32)
      return resultConfig
                 ? makeEmissionError(
                       "carried contraction result must be an f32 RVV vector")
                 : resultConfig.takeError();
    auto init = lookup(loop.getInitArgs().front());
    if (!init)
      return init.takeError();
    auto lower = lookup(loop.getLower());
    if (!lower)
      return lower.takeError();
    auto upper = lookup(loop.getUpper());
    if (!upper)
      return upper.takeError();
    auto step = lookup(loop.getStep());
    if (!step)
      return step.takeError();

    llvm::DenseSet<mlir::Operation *> lhsOperations;
    llvm::DenseSet<mlir::Value> lhsVisited;
    collectProducerOperations(contract.getLhs(), body, lhsOperations,
                              lhsVisited);
    llvm::DenseSet<mlir::Operation *> rhsOperations;
    llvm::DenseSet<mlir::Value> rhsVisited;
    collectProducerOperations(contract.getRhs(), body, rhsOperations,
                              rhsVisited);
    auto resultOperations =
        collectGroupProducerOperations(body, selection->groups[2]);
    llvm::DenseMap<mlir::Operation *, bool> blockDependent;
    llvm::DenseSet<mlir::Value> dependentValues;
    for (mlir::Operation &operation : body) {
      bool dependent = operationDependsOnBlockedValue(operation, dependentValues);
      blockDependent.try_emplace(&operation, dependent);
      if (dependent)
        for (mlir::Value result : operation.getResults())
          dependentValues.insert(result);
    }
    for (mlir::Operation &operation : body) {
      if (!blockDependent.lookup(&operation))
        continue;
      if (llvm::isa<kernel::ContractOp>(operation) &&
          &operation != contract.getOperation())
        return makeEmissionError(
            "carried contraction body contains an unselected contraction");
      bool accounted = &operation == contract.getOperation() ||
                       lhsOperations.contains(&operation) ||
                       rhsOperations.contains(&operation) ||
                       resultOperations.contains(&operation);
      if (!accounted)
        return makeEmissionError(
            llvm::Twine("carried contraction body has an unaccounted blocked ") +
            "operation " + operation.getName().getStringRef());
      if (resultOperations.contains(&operation) &&
          &operation != contract.getOperation() &&
          !llvm::isa<kernel::YieldOp>(operation))
        return makeEmissionError(
            "post-contract block operations in a carried body are not yet "
            "implemented");
    }

    std::string accumulator = valueName(loop.getResult(0));
    std::string induction =
        "__weft_i" + std::to_string(getNode(loop.getOperation()));
    std::string contractionAxis =
        "__weft_contract_k_" + std::to_string(getNode(contract.getOperation()));
    line(indent, resultConfig->cType + " " + accumulator + " = " + *init +
                     ";");
    expressions[loop.getResult(0)] = accumulator;
    vectorValues.insert(loop.getResult(0));
    line(indent, "for (size_t " + induction + " = " + *lower + "; " +
                     induction + " < " + *upper + "; " + induction + " += " +
                     *step + ") {");
    expressions[body.getArgument(0)] = induction;
    expressions[body.getArgument(1)] = accumulator;
    vectorValues.insert(body.getArgument(1));

    for (mlir::Operation &operation : body) {
      if (llvm::isa<kernel::YieldOp>(operation) ||
          blockDependent.lookup(&operation))
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent + 1))
        return error;
    }
    line(indent + 1, "for (size_t " + contractionAxis + " = 0; " +
                         contractionAxis + " < " + lhsExtents.lookup(1) +
                         "; ++" + contractionAxis + ") {");

    GroupInfo *savedAxisGroup = activeAxisGroup;
    int64_t savedVectorAxis = activeVectorAxis;
    auto savedSerialCoordinates = activeSerialCoordinates;
    auto outerM = savedSerialCoordinates.find(0);
    if (savedAxisGroup != &outerGroup || savedVectorAxis != 1 ||
        outerM == savedSerialCoordinates.end())
      return makeEmissionError(
          "carried contraction has no active outer M/N physical coordinates");

    activeAxisGroup = rhsGroup;
    activeVectorAxis = 1;
    activeSerialCoordinates.clear();
    activeSerialCoordinates.try_emplace(0, contractionAxis);
    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : body) {
      if (!rhsOperations.contains(&operation) ||
          llvm::isa<kernel::ContractOp, kernel::YieldOp>(operation))
        continue;
      bool blocked = llvm::any_of(operation.getOperands(), isBlocked) ||
                     llvm::any_of(operation.getResults(), isBlocked);
      if (!blocked)
        continue;
      if (llvm::Error error = emitVectorOperation(
              operation, *rhsGroup, lane, vl, reductions, contracts,
              indent + 2))
        return error;
    }
    auto rhs = lookup(contract.getRhs());
    if (!rhs)
      return rhs.takeError();

    llvm::DenseMap<mlir::Operation *, std::string> lhsCoordinates;
    kernel->walk([&](kernel::ArangeOp arange) {
      auto axis = tryArangeAxis(arange, *lhsGroup);
      if (!axis)
        return;
      lhsCoordinates.try_emplace(arange.getOperation(),
                                 *axis == 0 ? outerM->second
                                            : contractionAxis);
    });
    auto lhs =
        deriveScalarCoordinateExpression(contract.getLhs(), lhsCoordinates);
    if (!lhs)
      return lhs.takeError();
    std::string product = valueName(contract.getResult()) + "_product";
    line(indent + 2,
         resultConfig->cType + " " + product + " = __riscv_vfmul_vf_" +
             resultConfig->suffix + "(" + *rhs + ", " + *lhs + ", " +
             vl.str() + ");");
    line(indent + 2,
         accumulator + " = __riscv_vfadd_vv_" + resultConfig->suffix + "(" +
             accumulator + ", " + product + ", " + vl.str() + ");");
    line(indent + 1, "}");
    expressions[contract.getResult()] = accumulator;
    vectorValues.insert(contract.getResult());
    line(indent, "}");

    activeAxisGroup = savedAxisGroup;
    activeVectorAxis = savedVectorAxis;
    activeSerialCoordinates = std::move(savedSerialCoordinates);
    return llvm::Error::success();
  }

  llvm::Error collectGroupAxisExtents(
      GroupInfo &group,
      llvm::DenseMap<int64_t, std::string> &axisExtents) {
    int64_t rank = group.group.getLayout().getRank();
    int64_t groupID = group.group.getIdAttr().getInt();
    llvm::DenseMap<int64_t, mlir::Value> representatives;
    llvm::SmallDenseSet<int64_t, 4> singletonAxes;
    llvm::DenseSet<mlir::Value> visited;
    auto collectValue = [&](mlir::Value value) -> llvm::Error {
      if (!visited.insert(value).second)
        return llvm::Error::success();
      auto block = mlir::dyn_cast<kernel::BlockType>(value.getType());
      if (!block || static_cast<int64_t>(block.getShape().size()) != rank)
        return llvm::Error::success();
      for (int64_t axis = 0; axis < rank; ++axis) {
        if (block.getShape()[axis] == 1) {
          singletonAxes.insert(axis);
          continue;
        }
        auto fact = kernel::deriveLogicalExtent(value, axis);
        if (!fact)
          return makeEmissionError(
              "value-local layout axis has no canonical logical extent");
        auto existing = representatives.find(axis);
        if (existing != representatives.end() &&
            !kernel::haveSameLogicalExtent(existing->second, axis, value, axis))
          return makeEmissionError(
              "one value-local layout axis has inconsistent canonical extents");
        std::string expression;
        if (fact->constant)
          expression = std::to_string(*fact->constant);
        else {
          auto dynamic = lookup(fact->dynamic);
          if (!dynamic)
            return dynamic.takeError();
          expression = *dynamic;
        }
        representatives.try_emplace(axis, value);
        axisExtents.try_emplace(axis, std::move(expression));
      }
      return llvm::Error::success();
    };
    for (const auto &[value, selectedGroup] : valueGroups) {
      if (selectedGroup != groupID)
        continue;
      if (llvm::Error error = collectValue(value))
        return error;
    }
    auto incoming = incomingConversionValues.find(groupID);
    if (incoming != incomingConversionValues.end())
      for (mlir::Value value : incoming->second)
        if (llvm::Error error = collectValue(value))
          return error;
    for (int64_t axis = 0; axis < rank; ++axis)
      if (!axisExtents.contains(axis) && singletonAxes.contains(axis))
        axisExtents.try_emplace(axis, "1");
    if (static_cast<int64_t>(axisExtents.size()) != rank)
      return makeEmissionError(
          "layout group requires one canonical logical extent per axis");
    return llvm::Error::success();
  }

  void collectProducerOperations(mlir::Value value, mlir::Block &block,
                                 llvm::DenseSet<mlir::Operation *> &operations,
                                 llvm::DenseSet<mlir::Value> &visited) {
    if (!isBlocked(value) || !visited.insert(value).second)
      return;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != &block)
      return;
    operations.insert(definition);
    if (llvm::isa<kernel::ContractOp>(definition))
      return;
    for (mlir::Value operand : definition->getOperands())
      collectProducerOperations(operand, block, operations, visited);
  }

  std::optional<int64_t> effectiveUseGroup(mlir::Operation &consumer,
                                           int64_t operandIndex) const {
    auto converted =
        conversionGroups.find(std::make_pair(&consumer, operandIndex));
    if (converted != conversionGroups.end())
      return converted->second;
    if (operandIndex < 0 ||
        operandIndex >= static_cast<int64_t>(consumer.getNumOperands()))
      return std::nullopt;
    auto native = valueGroups.find(consumer.getOperand(operandIndex));
    return native == valueGroups.end() ? std::nullopt
                                       : std::optional<int64_t>(native->second);
  }

  void collectOperationAndProducers(
      mlir::Operation *operation, mlir::Block &block,
      llvm::DenseSet<mlir::Operation *> &operations) {
    if (!operation || operation->getBlock() != &block ||
        llvm::isa<kernel::ReturnOp>(operation) ||
        !operations.insert(operation).second)
      return;
    for (mlir::Value operand : operation->getOperands())
      collectOperationAndProducers(operand.getDefiningOp(), block, operations);
  }

  void collectResultConsumers(mlir::Value value, mlir::Block &block,
                              int64_t groupID,
                              llvm::DenseSet<mlir::Operation *> &operations,
                              llvm::DenseSet<mlir::Value> &visited) {
    if (!visited.insert(value).second)
      return;
    for (mlir::Operation *user : value.getUsers()) {
      if (user->getBlock() != &block || llvm::isa<kernel::ReturnOp>(user))
        continue;
      if (auto selected = reductionGroups.find(user);
          selected != reductionGroups.end() && selected->second != groupID)
        continue;
      collectOperationAndProducers(user, block, operations);
      for (mlir::Value result : user->getResults())
        collectResultConsumers(result, block, groupID, operations, visited);
    }
  }

  llvm::DenseSet<mlir::Operation *>
  collectRank2GroupOperations(mlir::Block &block, int64_t groupID) {
    llvm::DenseSet<mlir::Operation *> operations;
    for (const auto &[value, selectedGroup] : valueGroups)
      if (selectedGroup == groupID)
        collectOperationAndProducers(value.getDefiningOp(), block, operations);

    for (const auto &[source, selectedGroup] : reductionGroups) {
      if (selectedGroup != groupID || source->getBlock() != &block)
        continue;
      collectOperationAndProducers(source, block, operations);
      llvm::DenseSet<mlir::Value> visited;
      for (mlir::Value result : source->getResults())
        collectResultConsumers(result, block, groupID, operations, visited);
    }

    for (mlir::Operation &operation : block) {
      if (!llvm::isa<kernel::StoreOp>(operation))
        continue;
      bool hasSelectedOperand = false;
      bool allSelectedOperandsUseGroup = true;
      for (auto [index, operand] : llvm::enumerate(operation.getOperands())) {
        if (!isBlocked(operand))
          continue;
        auto selected =
            effectiveUseGroup(operation, static_cast<int64_t>(index));
        if (!selected)
          continue;
        hasSelectedOperand = true;
        allSelectedOperandsUseGroup &= *selected == groupID;
      }
      if (hasSelectedOperand && allSelectedOperandsUseGroup)
        collectOperationAndProducers(&operation, block, operations);
    }
    return operations;
  }

  llvm::DenseSet<mlir::Operation *>
  collectGroupProducerOperations(mlir::Block &block, int64_t groupID) {
    llvm::DenseSet<mlir::Operation *> operations;
    llvm::DenseSet<mlir::Value> visited;
    for (const auto &[value, selectedGroup] : valueGroups)
      if (selectedGroup == groupID)
        collectProducerOperations(value, block, operations, visited);
    for (mlir::Operation &operation : block) {
      bool consumesGroup = llvm::any_of(operation.getOperands(), [&](mlir::Value value) {
        auto selected = valueGroups.find(value);
        return selected != valueGroups.end() && selected->second == groupID;
      });
      if (!consumesGroup)
        continue;
      operations.insert(&operation);
      if (llvm::isa<kernel::ContractOp>(operation))
        continue;
      for (mlir::Value operand : operation.getOperands())
        collectProducerOperations(operand, block, operations, visited);
    }
    return operations;
  }

  llvm::Expected<std::string> deriveScalarCoordinateExpression(
      mlir::Value value,
      const llvm::DenseMap<mlir::Operation *, std::string> &arangeCoordinates) {
    if (!isBlocked(value))
      return lookup(value);
    if (auto arange = value.getDefiningOp<kernel::ArangeOp>()) {
      auto coordinate = arangeCoordinates.find(arange.getOperation());
      if (coordinate == arangeCoordinates.end())
        return makeEmissionError(
            "scalar contract projection has no coordinate for canonical arange");
      auto start = lookup(arange.getStart());
      if (!start)
        return start.takeError();
      return "(" + *start + " + " + coordinate->second + ")";
    }
    if (auto expand = value.getDefiningOp<kernel::ExpandDimsOp>())
      return deriveScalarCoordinateExpression(expand.getInput(),
                                              arangeCoordinates);
    if (auto splat = value.getDefiningOp<kernel::SplatOp>())
      return lookup(splat.getValue());
    if (auto cast = value.getDefiningOp<kernel::CastOp>()) {
      auto input = deriveScalarCoordinateExpression(cast.getInput(),
                                                    arangeCoordinates);
      if (!input)
        return input.takeError();
      auto element = mlir::cast<kernel::BlockType>(cast.getResult().getType())
                         .getElementType();
      auto type = getScalarCType(element);
      if (!type)
        return type.takeError();
      return "static_cast<" + *type + ">(" + *input + ")";
    }
    if (auto unary = value.getDefiningOp<kernel::UnaryOp>()) {
      auto input = deriveScalarCoordinateExpression(unary.getInput(),
                                                    arangeCoordinates);
      if (!input)
        return input.takeError();
      if (unary.getKind() == "neg")
        return "(-" + *input + ")";
      return makeEmissionError(
          "scalar contract projection supports only unary negation");
    }
    if (auto binary = value.getDefiningOp<kernel::BinaryOp>()) {
      auto lhs = deriveScalarCoordinateExpression(binary.getLhs(),
                                                  arangeCoordinates);
      if (!lhs)
        return lhs.takeError();
      auto rhs = deriveScalarCoordinateExpression(binary.getRhs(),
                                                  arangeCoordinates);
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef spelling =
          llvm::StringSwitch<llvm::StringRef>(binary.getKind())
              .Case("add", "+")
              .Case("sub", "-")
              .Case("mul", "*")
              .Case("div", "/")
              .Case("mod", "%")
              .Case("and", "&")
              .Case("or", "|")
              .Case("xor", "^")
              .Default("");
      if (!spelling.empty())
        return "(" + *lhs + " " + spelling.str() + " " + *rhs + ")";
      if (binary.getKind() == "max")
        return "(" + *lhs + " > " + *rhs + " ? " + *lhs + " : " + *rhs +
               ")";
      if (binary.getKind() == "min")
        return "(" + *lhs + " < " + *rhs + " ? " + *lhs + " : " + *rhs +
               ")";
      return makeEmissionError(
          "scalar contract projection has unsupported binary operation");
    }
    if (auto compare = value.getDefiningOp<kernel::CompareOp>()) {
      auto lhs = deriveScalarCoordinateExpression(compare.getLhs(),
                                                  arangeCoordinates);
      if (!lhs)
        return lhs.takeError();
      auto rhs = deriveScalarCoordinateExpression(compare.getRhs(),
                                                  arangeCoordinates);
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef spelling =
          llvm::StringSwitch<llvm::StringRef>(compare.getPredicate())
              .Case("eq", "==")
              .Case("ne", "!=")
              .Case("lt", "<")
              .Case("le", "<=")
              .Case("gt", ">")
              .Case("ge", ">=")
              .Default("");
      if (spelling.empty())
        return makeEmissionError(
            "scalar contract projection has unsupported comparison");
      return "(" + *lhs + " " + spelling.str() + " " + *rhs + ")";
    }
    if (auto pointerAdd = value.getDefiningOp<kernel::PtrAddOp>()) {
      auto base = deriveScalarCoordinateExpression(pointerAdd.getBase(),
                                                   arangeCoordinates);
      if (!base)
        return base.takeError();
      auto offset = deriveScalarCoordinateExpression(pointerAdd.getOffset(),
                                                     arangeCoordinates);
      if (!offset)
        return offset.takeError();
      return "(" + *base + " + " + *offset + ")";
    }
    if (auto load = value.getDefiningOp<kernel::LoadOp>()) {
      auto pointer = deriveScalarCoordinateExpression(load.getPointer(),
                                                      arangeCoordinates);
      if (!pointer)
        return pointer.takeError();
      auto mask = deriveScalarCoordinateExpression(load.getMask(),
                                                   arangeCoordinates);
      if (!mask)
        return mask.takeError();
      auto other = deriveScalarCoordinateExpression(load.getOther(),
                                                    arangeCoordinates);
      if (!other)
        return other.takeError();
      return "(" + *mask + " ? *" + *pointer + " : " + *other + ")";
    }
    return makeEmissionError(
        "blocked value has no scalar contract-coordinate expression");
  }

  llvm::Error emitRank2ContractScope(
      mlir::Block &block, llvm::ArrayRef<int64_t> scopeGroupIDs,
      unsigned indent) {
    kernel::ContractOp contract;
    const ContractSelection *selection = nullptr;
    for (auto &[source, current] : contractSelections) {
      auto candidate = llvm::dyn_cast<kernel::ContractOp>(source);
      if (!candidate || candidate->getBlock() != &block ||
          current.groups.size() != 3)
        continue;
      if (contract)
        return makeEmissionError(
            "one multi-group source scope currently supports one contraction");
      contract = candidate;
      selection = &current;
    }
    if (!contract || !selection)
      return makeEmissionError(
          "multi-group source scope has no selected rank-two contraction");
    if (selection->strategy != "sequential")
      return makeEmissionError(
          "rank-two contract source emission requires sequential K strategy");

    llvm::SmallDenseSet<int64_t, 4> scopeSet(scopeGroupIDs.begin(),
                                             scopeGroupIDs.end());
    if (scopeGroupIDs.size() != 3 || scopeSet.size() != 3)
      return makeEmissionError(
          "first rank-two contract source slice requires exactly three "
          "distinct groups in its canonical scope");
    for (int64_t id : selection->groups)
      if (!scopeSet.contains(id))
        return makeEmissionError(
            "contraction role group is outside its canonical source scope");
    GroupInfo *lhsGroup = findGroup(selection->groups[0]);
    GroupInfo *rhsGroup = findGroup(selection->groups[1]);
    GroupInfo *resultGroup = findGroup(selection->groups[2]);
    if (!lhsGroup || !rhsGroup || !resultGroup)
      return makeEmissionError("rank-two contraction group cannot be resolved");
    auto rhsVectorAxes = rhsGroup->group.getLayout().getVectorAxes();
    auto resultVectorAxes = resultGroup->group.getLayout().getVectorAxes();
    if (!lhsGroup->group.getLayout().getVectorAxes().empty() ||
        rhsVectorAxes.size() != 1 || rhsVectorAxes.front() != 1 ||
        resultVectorAxes.size() != 1 || resultVectorAxes.front() != 1)
      return makeEmissionError(
          "rank-two contraction requires serial lhs and N-vector rhs/result");
    auto rhsLaneRatio = rhsGroup->config.getLaneRatioAttr();
    auto resultLaneRatio = resultGroup->config.getLaneRatioAttr();
    if (!rhsLaneRatio || !resultLaneRatio ||
        rhsLaneRatio.getInt() != resultLaneRatio.getInt())
      return makeEmissionError(
          "rank-two contraction requires one rhs/result lane ratio");

    llvm::DenseMap<mlir::Operation *, bool> blockDependent;
    llvm::DenseSet<mlir::Value> dependentValues;
    for (mlir::Operation &operation : block) {
      bool dependent = operationDependsOnBlockedValue(operation, dependentValues);
      blockDependent.try_emplace(&operation, dependent);
      if (dependent)
        for (mlir::Value result : operation.getResults())
          dependentValues.insert(result);
    }
    for (mlir::Operation &operation : block) {
      if (llvm::isa<kernel::ReturnOp>(operation) ||
          blockDependent.lookup(&operation))
        continue;
      if (llvm::isa<kernel::ForOp>(operation))
        return makeEmissionError(
            "rank-two contract source slice does not support nested loops");
      if (llvm::Error error = emitScalarOperation(operation, indent))
        return error;
    }

    llvm::DenseMap<int64_t, std::string> lhsExtents;
    llvm::DenseMap<int64_t, std::string> rhsExtents;
    llvm::DenseMap<int64_t, std::string> resultExtents;
    if (llvm::Error error =
            collectGroupAxisExtents(*lhsGroup, lhsExtents))
      return error;
    if (llvm::Error error =
            collectGroupAxisExtents(*rhsGroup, rhsExtents))
      return error;
    if (llvm::Error error =
            collectGroupAxisExtents(*resultGroup, resultExtents))
      return error;
    if (lhsExtents.lookup(0) != resultExtents.lookup(0) ||
        lhsExtents.lookup(1) != rhsExtents.lookup(0) ||
        rhsExtents.lookup(1) != resultExtents.lookup(1))
      return makeEmissionError(
          "rank-two contraction role extents disagree after canonical "
          "provenance resolution");

    auto resultConfig = getVectorConfig(contract.getResult().getType(),
                                        *resultGroup);
    if (!resultConfig || !resultConfig->floating || resultConfig->sew != 32)
      return resultConfig
                 ? makeEmissionError(
                       "rank-two contraction result must be an f32 RVV vector")
                 : resultConfig.takeError();
    int64_t maxSEW = std::max(rhsGroup->maxSEW, resultGroup->maxSEW);
    if (maxSEW == 0)
      return makeEmissionError(
          "rank-two contraction has no physical vector SEW");
    std::string setvlLMUL =
        lmulSpelling(maxSEW * 8 / resultLaneRatio.getInt());
    if (setvlLMUL.empty())
      return makeEmissionError(
          "rank-two contraction cannot form selected setvl configuration");
    auto init = lookup(contract.getInit());
    if (!init)
      return init.takeError();

    llvm::DenseSet<mlir::Operation *> lhsOperations;
    llvm::DenseSet<mlir::Value> lhsVisited;
    collectProducerOperations(contract.getLhs(), block, lhsOperations,
                              lhsVisited);
    llvm::DenseSet<mlir::Operation *> rhsOperations;
    llvm::DenseSet<mlir::Value> rhsVisited;
    collectProducerOperations(contract.getRhs(), block, rhsOperations,
                              rhsVisited);
    auto resultOperations =
        collectGroupProducerOperations(block, selection->groups[2]);
    for (mlir::Operation &operation : block) {
      if (!blockDependent.lookup(&operation) ||
          llvm::isa<kernel::ReturnOp>(operation))
        continue;
      if (llvm::isa<kernel::ContractOp>(operation) &&
          &operation != contract.getOperation())
        return makeEmissionError(
            "rank-two contract scope contains an unselected contraction");
      bool accounted = &operation == contract.getOperation() ||
                       lhsOperations.contains(&operation) ||
                       rhsOperations.contains(&operation) ||
                       resultOperations.contains(&operation);
      if (!accounted)
        return makeEmissionError(
            llvm::Twine("rank-two contract scope has an unaccounted blocked ") +
            "operation " + operation.getName().getStringRef());
    }
    std::string m = "__weft_contract_m_" +
                    std::to_string(getNode(contract.getOperation()));
    std::string n = "__weft_contract_n_" +
                    std::to_string(getNode(contract.getOperation()));
    std::string k = "__weft_contract_k_" +
                    std::to_string(getNode(contract.getOperation()));
    std::string vl = "__weft_contract_vl_" +
                     std::to_string(getNode(contract.getOperation()));
    std::string accumulator = valueName(contract.getResult()) + "_acc";

    line(indent, "for (size_t " + m + " = 0; " + m + " < " +
                     resultExtents.lookup(0) + "; ++" + m + ") {");
    line(indent + 1, "for (size_t " + n + " = 0; " + n + " < " +
                         resultExtents.lookup(1) + ";) {");
    line(indent + 2,
         "size_t " + vl + " = __riscv_vsetvl_e" +
             std::to_string(maxSEW) + setvlLMUL + "(" +
             resultExtents.lookup(1) + " - " + n + ");");
    line(indent + 2,
         resultConfig->cType + " " + accumulator + " = __riscv_vfmv_v_f_" +
             resultConfig->suffix + "(" + *init + ", " + vl + ");");
    line(indent + 2, "for (size_t " + k + " = 0; " + k + " < " +
                         lhsExtents.lookup(1) + "; ++" + k + ") {");

    activeAxisGroup = rhsGroup;
    activeVectorAxis = 1;
    activeSerialCoordinates.clear();
    activeSerialCoordinates.try_emplace(0, k);
    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : block) {
      if (!rhsOperations.contains(&operation) ||
          llvm::isa<kernel::ContractOp>(operation))
        continue;
      bool blocked = llvm::any_of(operation.getOperands(), isBlocked) ||
                     llvm::any_of(operation.getResults(), isBlocked);
      if (!blocked)
        continue;
      if (llvm::Error error = emitVectorOperation(
              operation, *rhsGroup, n, vl, reductions, contracts, indent + 3)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return error;
      }
    }
    auto rhs = lookup(contract.getRhs());
    if (!rhs) {
      activeAxisGroup = nullptr;
      activeSerialCoordinates.clear();
      return rhs.takeError();
    }

    llvm::DenseMap<mlir::Operation *, std::string> lhsCoordinates;
    for (mlir::Operation &operation : block) {
      auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation);
      if (!arange)
        continue;
      auto axis = tryArangeAxis(arange, *lhsGroup);
      if (!axis)
        continue;
      lhsCoordinates.try_emplace(arange.getOperation(), *axis == 0 ? m : k);
    }
    auto lhs = deriveScalarCoordinateExpression(contract.getLhs(),
                                                lhsCoordinates);
    if (!lhs) {
      activeAxisGroup = nullptr;
      activeSerialCoordinates.clear();
      return lhs.takeError();
    }
    std::string product = accumulator + "_product";
    line(indent + 3,
         resultConfig->cType + " " + product + " = __riscv_vfmul_vf_" +
             resultConfig->suffix + "(" + *rhs + ", " + *lhs + ", " + vl +
             ");");
    line(indent + 3,
         accumulator + " = __riscv_vfadd_vv_" + resultConfig->suffix + "(" +
             accumulator + ", " + product + ", " + vl + ");");
    line(indent + 2, "}");

    activeAxisGroup = resultGroup;
    activeVectorAxis = 1;
    activeSerialCoordinates.clear();
    activeSerialCoordinates.try_emplace(0, m);
    expressions[contract.getResult()] = accumulator;
    vectorValues.insert(contract.getResult());
    for (mlir::Operation &operation : block) {
      if (!resultOperations.contains(&operation) ||
          llvm::isa<kernel::ContractOp>(operation))
        continue;
      bool blocked = llvm::any_of(operation.getOperands(), isBlocked) ||
                     llvm::any_of(operation.getResults(), isBlocked);
      if (!blocked)
        continue;
      if (llvm::Error error = emitVectorOperation(
              operation, *resultGroup, n, vl, reductions, contracts,
              indent + 2)) {
        activeAxisGroup = nullptr;
        activeSerialCoordinates.clear();
        return error;
      }
    }
    activeAxisGroup = nullptr;
    activeSerialCoordinates.clear();
    line(indent + 2, n + " += " + vl + ";");
    line(indent + 1, "}");
    line(indent, "}");
    return llvm::Error::success();
  }

  llvm::Error emitScalarOperation(mlir::Operation &operation,
                                  unsigned indent) {
    if (auto constant = llvm::dyn_cast<kernel::ConstantOp>(operation)) {
      auto literal = constantLiteral(
          mlir::cast<mlir::TypedAttr>(constant.getValue()));
      if (!literal)
        return literal.takeError();
      std::string name = valueName(constant.getResult());
      auto type = getScalarCType(constant.getResult().getType());
      if (!type)
        return type.takeError();
      line(indent, *type + " " + name + " = " + *literal + ";");
      expressions[constant.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto meta = llvm::dyn_cast<kernel::MetaValueOp>(operation)) {
      auto input = lookup(meta.getInput());
      if (!input)
        return input.takeError();
      expressions[meta.getResult()] = *input;
      return llvm::Error::success();
    }
    if (auto task = llvm::dyn_cast<kernel::TaskIdOp>(operation)) {
      auto name = taskNames.find(task.getAxisAttr().getInt());
      if (name == taskNames.end())
        return makeEmissionError("task_id has no selected ABI task binding");
      expressions[task.getResult()] = name->second;
      return llvm::Error::success();
    }
    if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(operation)) {
      if (isBlocked(binary.getResult()))
        return makeEmissionError("blocked binary reached scalar emission");
      auto lhs = lookup(binary.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(binary.getRhs());
      if (!rhs)
        return rhs.takeError();
      std::string expression;
      llvm::StringRef op = llvm::StringSwitch<llvm::StringRef>(binary.getKind())
                               .Case("add", "+")
                               .Case("sub", "-")
                               .Case("mul", "*")
                               .Case("div", "/")
                               .Case("mod", "%")
                               .Case("and", "&")
                               .Case("or", "|")
                               .Case("xor", "^")
                               .Default("");
      if (!op.empty())
        expression = "(" + *lhs + " " + op.str() + " " + *rhs + ")";
      else if (binary.getKind() == "max")
        expression = "(" + *lhs + " > " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else if (binary.getKind() == "min")
        expression = "(" + *lhs + " < " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else
        return makeEmissionError("unsupported scalar binary operation");
      std::string name = valueName(binary.getResult());
      line(indent, "auto " + name + " = " + expression + ";");
      expressions[binary.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation)) {
      if (isBlocked(unary.getResult()))
        return makeEmissionError("blocked unary reached scalar emission");
      auto input = lookup(unary.getInput());
      if (!input)
        return input.takeError();
      std::string expression;
      if (unary.getKind() == "neg")
        expression = "(-" + *input + ")";
      else if (unary.getKind() == "rsqrt")
        expression = "(1.0f / sqrtf(" + *input + "))";
      else if (unary.getKind() == "exp")
        expression = "expf(" + *input + ")";
      else
        return makeEmissionError("unsupported scalar unary operation");
      std::string name = valueName(unary.getResult());
      line(indent, "auto " + name + " = " + expression + ";");
      expressions[unary.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto cast = llvm::dyn_cast<kernel::CastOp>(operation)) {
      if (isBlocked(cast.getResult()))
        return makeEmissionError("blocked cast emission is not implemented");
      auto input = lookup(cast.getInput());
      if (!input)
        return input.takeError();
      auto type = getScalarCType(cast.getResult().getType());
      if (!type)
        return type.takeError();
      std::string name = valueName(cast.getResult());
      line(indent, *type + " " + name + " = static_cast<" + *type + ">(" +
                       *input + ");");
      expressions[cast.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto compare = llvm::dyn_cast<kernel::CompareOp>(operation)) {
      if (isBlocked(compare.getResult()))
        return makeEmissionError("blocked compare reached scalar emission");
      auto lhs = lookup(compare.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(compare.getRhs());
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef predicate =
          llvm::StringSwitch<llvm::StringRef>(compare.getPredicate())
              .Case("eq", "==")
              .Case("ne", "!=")
              .Case("lt", "<")
              .Case("le", "<=")
              .Case("gt", ">")
              .Case("ge", ">=")
              .Default("");
      if (predicate.empty())
        return makeEmissionError("unsupported scalar comparison predicate");
      std::string name = valueName(compare.getResult());
      line(indent, "bool " + name + " = (" + *lhs + " " + predicate.str() +
                       " " + *rhs + ");");
      expressions[compare.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto pointerAdd = llvm::dyn_cast<kernel::PtrAddOp>(operation)) {
      if (isBlocked(pointerAdd.getResult()))
        return makeEmissionError("blocked pointer add reached scalar emission");
      auto base = lookup(pointerAdd.getBase());
      if (!base)
        return base.takeError();
      auto offset = lookup(pointerAdd.getOffset());
      if (!offset)
        return offset.takeError();
      std::string name = valueName(pointerAdd.getResult());
      line(indent, "auto *" + name + " = " + *base + " + " + *offset + ";");
      expressions[pointerAdd.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto load = llvm::dyn_cast<kernel::LoadOp>(operation)) {
      if (isBlocked(load.getResult()) || isBlocked(load.getPointer()) ||
          isBlocked(load.getMask()))
        return makeEmissionError("blocked load reached scalar emission");
      auto pointer = lookup(load.getPointer());
      if (!pointer)
        return pointer.takeError();
      auto mask = lookup(load.getMask());
      if (!mask)
        return mask.takeError();
      auto other = lookup(load.getOther());
      if (!other)
        return other.takeError();
      auto type = getScalarCType(load.getResult().getType());
      if (!type)
        return type.takeError();
      std::string name = valueName(load.getResult());
      line(indent, *type + " " + name + " = " + *mask + " ? *" + *pointer +
                       " : " + *other + ";");
      expressions[load.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (auto store = llvm::dyn_cast<kernel::StoreOp>(operation)) {
      if (isBlocked(store.getPointer()) || isBlocked(store.getValue()) ||
          isBlocked(store.getMask()))
        return makeEmissionError("blocked store reached scalar emission");
      auto pointer = lookup(store.getPointer());
      if (!pointer)
        return pointer.takeError();
      auto value = lookup(store.getValue());
      if (!value)
        return value.takeError();
      auto mask = lookup(store.getMask());
      if (!mask)
        return mask.takeError();
      line(indent, "if (" + *mask + ") *" + *pointer + " = " + *value + ";");
      return llvm::Error::success();
    }
    if (auto loop = llvm::dyn_cast<kernel::ForOp>(operation))
      return emitFor(loop, indent);
    if (llvm::isa<kernel::YieldOp, kernel::ReturnOp>(operation))
      return llvm::Error::success();
    if (llvm::isa<kernel::ArangeOp, kernel::ReduceOp, kernel::ContractOp>(
            operation))
      return makeEmissionError("blocked operation reached scalar emission");
    return makeEmissionError(llvm::Twine("no source emission rule for ") +
                             operation.getName().getStringRef());
  }

  llvm::Error emitFor(kernel::ForOp loop, unsigned indent) {
    auto groupID = scopeGroups.find(loop.getOperation());
    if (groupID == scopeGroups.end())
      return emitScalarFor(loop, indent);
    if (groupID->second.size() != 1)
      return makeEmissionError(
          "nested multi-group contraction emission is not yet supported");
    int64_t selectedGroupID = groupID->second.front();
    GroupInfo *group = findGroup(selectedGroupID);
    if (!group)
      return makeEmissionError("loop layout group cannot be resolved");

    if (group->group.getLayout().getRank() > 2) {
      if (!loop.getInitArgs().empty() || loop.getNumResults() != 0)
        return makeEmissionError(
            "generic rank-N loop scope does not yet support carried state");
      auto lower = lookup(loop.getLower());
      if (!lower)
        return lower.takeError();
      auto upper = lookup(loop.getUpper());
      if (!upper)
        return upper.takeError();
      auto step = lookup(loop.getStep());
      if (!step)
        return step.takeError();
      std::string induction =
          "__weft_i" + std::to_string(getNode(loop.getOperation()));
      line(indent, "for (size_t " + induction + " = " + *lower + "; " +
                       induction + " < " + *upper + "; " + induction +
                       " += " + *step + ") {");
      mlir::Block &body = loop.getBody().front();
      expressions[body.getArgument(0)] = induction;
      if (llvm::Error error =
              emitRankNElementwiseScope(body, *group, indent + 1))
        return error;
      line(indent, "}");
      return llvm::Error::success();
    }

    auto lower = lookup(loop.getLower());
    if (!lower)
      return lower.takeError();
    auto upper = lookup(loop.getUpper());
    if (!upper)
      return upper.takeError();
    auto step = lookup(loop.getStep());
    if (!step)
      return step.takeError();

    llvm::SmallVector<std::string, 4> carried;
    for (auto [index, values] :
         llvm::enumerate(llvm::zip(loop.getInitArgs(), loop.getResults()))) {
      auto [init, result] = values;
      auto initExpression = lookup(init);
      if (!initExpression)
        return initExpression.takeError();
      auto type = getScalarCType(result.getType());
      if (!type)
        return type.takeError();
      std::string name = valueName(result);
      line(indent, *type + " " + name + " = " + *initExpression + ";");
      expressions[result] = name;
      carried.push_back(std::move(name));
      (void)index;
    }

    std::string induction = "__weft_i" +
                            std::to_string(getNode(loop.getOperation()));
    line(indent, "for (size_t " + induction + " = " + *lower + "; " +
                     induction + " < " + *upper + "; " + induction + " += " +
                     *step + ") {");
    mlir::Block &body = loop.getBody().front();
    expressions[body.getArgument(0)] = induction;
    for (auto [argument, name] :
         llvm::zip(body.getArguments().drop_front(), carried))
      expressions[argument] = name;

    llvm::DenseMap<mlir::Operation *, Phase> phases;
    llvm::DenseSet<mlir::Value> postValues;
    for (mlir::Operation &operation : body) {
      bool blockDependent =
          llvm::any_of(operation.getOperands(), isBlocked) ||
          llvm::any_of(operation.getResults(), isBlocked);
      bool postDependent = llvm::any_of(operation.getOperands(), [&](mlir::Value value) {
        return postValues.contains(value);
      });
      Phase phase = blockDependent ? Phase::Vector
                    : postDependent ? Phase::Post
                                    : Phase::Invariant;
      phases.try_emplace(&operation, phase);
      if (llvm::isa<kernel::ReduceOp, kernel::ContractOp>(operation)) {
        phase = Phase::Vector;
        phases[&operation] = phase;
        for (mlir::Value result : operation.getResults())
          postValues.insert(result);
      } else if (phase == Phase::Post) {
        for (mlir::Value result : operation.getResults())
          postValues.insert(result);
      }
    }

    for (mlir::Operation &operation : body) {
      if (llvm::isa<kernel::YieldOp>(operation) ||
          phases.lookup(&operation) != Phase::Invariant)
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent + 1))
        return error;
    }

    llvm::DenseMap<kernel::ReduceOp, std::string> reductions;
    for (mlir::Operation &operation : body) {
      auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation);
      if (!reduction)
        continue;
      if ((reduction.getKind() != "sum" && reduction.getKind() != "max" &&
           reduction.getKind() != "min") ||
          !reduction.getResult().getType().isF32())
        return makeEmissionError(
            "first selected RVV source slice supports only f32 sum/max/min "
            "reduction");
      auto init = lookup(reduction.getInit());
      if (!init)
        return init.takeError();
      std::string name = valueName(reduction.getResult());
      line(indent + 1, "float " + name + " = " + *init + ";");
      expressions[reduction.getResult()] = name;
      reductions.try_emplace(reduction, std::move(name));
    }
    llvm::DenseMap<kernel::ContractOp, std::string> contracts;
    for (mlir::Operation &operation : body) {
      auto contract = llvm::dyn_cast<kernel::ContractOp>(operation);
      if (!contract)
        continue;
      if (!contract.getResult().getType().isF32())
        return makeEmissionError(
            "first selected RVV source slice supports scalar f32 contraction");
      auto init = lookup(contract.getInit());
      if (!init)
        return init.takeError();
      std::string name = valueName(contract.getResult());
      line(indent + 1, "float " + name + " = " + *init + ";");
      expressions[contract.getResult()] = name;
      contracts.try_emplace(contract, std::move(name));
    }

    std::optional<std::string> extent;
    for (mlir::Operation &operation : body) {
      auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation);
      if (!arange)
        continue;
      auto currentExtent = lookup(arange.getExtent());
      if (!currentExtent)
        return currentExtent.takeError();
      if (extent && *extent != *currentExtent)
        return makeEmissionError(
            "one RVV layout group requires a common logical block extent");
      extent = *currentExtent;
    }
    if (!extent)
      return makeEmissionError("RVV layout group has no structural arange extent");
    if (group->maxSEW == 0)
      return makeEmissionError("RVV layout group has no physical vector SEW");

    int64_t ratio = group->config.getLaneRatioAttr().getInt();
    std::string setvlLMUL = lmulSpelling(group->maxSEW * 8 / ratio);
    if (setvlLMUL.empty())
      return makeEmissionError("cannot form the selected setvl configuration");
    std::string lane = "__weft_lane_g" + std::to_string(selectedGroupID);
    std::string vl = "__weft_vl_g" + std::to_string(selectedGroupID);
    line(indent + 1, "for (size_t " + lane + " = 0; " + lane + " < " +
                         *extent + ";) {");
    line(indent + 2,
         "size_t " + vl + " = __riscv_vsetvl_e" +
             std::to_string(group->maxSEW) + setvlLMUL + "(" + *extent + " - " +
             lane + ");");
    for (mlir::Operation &operation : body) {
      if (phases.lookup(&operation) != Phase::Vector)
        continue;
      if (llvm::Error error =
              emitVectorOperation(operation, *group, lane, vl, reductions,
                                  contracts, indent + 2))
        return error;
    }
    line(indent + 2, lane + " += " + vl + ";");
    line(indent + 1, "}");

    for (mlir::Operation &operation : body) {
      if (llvm::isa<kernel::YieldOp>(operation) ||
          phases.lookup(&operation) != Phase::Post)
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent + 1))
        return error;
    }

    auto yield = llvm::cast<kernel::YieldOp>(body.getTerminator());
    for (auto [name, yielded] : llvm::zip(carried, yield.getValues())) {
      auto expression = lookup(yielded);
      if (!expression)
        return expression.takeError();
      line(indent + 1, name + " = " + *expression + ";");
    }
    line(indent, "}");
    return llvm::Error::success();
  }

  llvm::Error emitScalarFor(kernel::ForOp loop, unsigned indent) {
    if (llvm::any_of(loop.getInitArgs(), isBlocked) ||
        llvm::any_of(loop.getResults(), isBlocked))
      return makeEmissionError("block-carried scalar loop is not supported");
    auto lower = lookup(loop.getLower());
    if (!lower)
      return lower.takeError();
    auto upper = lookup(loop.getUpper());
    if (!upper)
      return upper.takeError();
    auto step = lookup(loop.getStep());
    if (!step)
      return step.takeError();
    llvm::SmallVector<std::string, 4> carried;
    for (auto [init, result] : llvm::zip(loop.getInitArgs(), loop.getResults())) {
      auto value = lookup(init);
      if (!value)
        return value.takeError();
      auto type = getScalarCType(result.getType());
      if (!type)
        return type.takeError();
      std::string name = valueName(result);
      line(indent, *type + " " + name + " = " + *value + ";");
      expressions[result] = name;
      carried.push_back(std::move(name));
    }
    std::string induction = "__weft_i" +
                            std::to_string(getNode(loop.getOperation()));
    line(indent, "for (size_t " + induction + " = " + *lower + "; " +
                     induction + " < " + *upper + "; " + induction + " += " +
                     *step + ") {");
    mlir::Block &body = loop.getBody().front();
    expressions[body.getArgument(0)] = induction;
    for (auto [argument, name] :
         llvm::zip(body.getArguments().drop_front(), carried))
      expressions[argument] = name;
    for (mlir::Operation &operation : body) {
      if (llvm::isa<kernel::YieldOp>(operation))
        continue;
      if (llvm::Error error = emitScalarOperation(operation, indent + 1))
        return error;
    }
    auto yield = llvm::cast<kernel::YieldOp>(body.getTerminator());
    for (auto [name, yielded] : llvm::zip(carried, yield.getValues())) {
      auto expression = lookup(yielded);
      if (!expression)
        return expression.takeError();
      line(indent + 1, name + " = " + *expression + ";");
    }
    line(indent, "}");
    return llvm::Error::success();
  }

  llvm::Error emitVectorOperation(
      mlir::Operation &operation, GroupInfo &group,
      llvm::StringRef lane, llvm::StringRef vl,
      const llvm::DenseMap<kernel::ReduceOp, std::string> &reductions,
      const llvm::DenseMap<kernel::ContractOp, std::string> &contracts,
      unsigned indent) {
    if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation))
      return emitArange(arange, group, lane, vl, indent);
    if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(operation)) {
      auto input = lookup(expand.getInput());
      if (!input)
        return input.takeError();
      expressions[expand.getResult()] = *input;
      if (vectorValues.contains(expand.getInput()))
        vectorValues.insert(expand.getResult());
      return llvm::Error::success();
    }
    if (auto splat = llvm::dyn_cast<kernel::SplatOp>(operation)) {
      auto value = lookup(splat.getValue());
      if (!value)
        return value.takeError();
      auto config = getVectorConfig(splat.getResult().getType(), group);
      if (!config)
        return config.takeError();
      std::string name = valueName(splat.getResult());
      if (config->mask) {
        std::string suffix = "_m_b" + std::to_string(config->laneRatio) +
                             "(" + vl.str() + ")";
        line(indent, config->cType + " " + name + " = (" + *value +
                         " ? __riscv_vmset" + suffix + " : __riscv_vmclr" +
                         suffix + ");");
      } else if (config->floating && config->sew == 32) {
        line(indent, config->cType + " " + name + " = __riscv_vfmv_v_f_" +
                         config->suffix + "(" + *value + ", " + vl.str() +
                         ");");
      } else {
        return makeEmissionError(
            "first selected RVV splat slice supports f32 data and i1 masks");
      }
      expressions[splat.getResult()] = std::move(name);
      vectorValues.insert(splat.getResult());
      return llvm::Error::success();
    }
    if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(operation))
      return emitVectorBinary(binary, group, vl, indent);
    if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation))
      return emitVectorUnary(unary, group, vl, indent);
    if (auto compare = llvm::dyn_cast<kernel::CompareOp>(operation))
      return emitVectorCompare(compare, group, vl, indent);
    if (auto pointerAdd = llvm::dyn_cast<kernel::PtrAddOp>(operation)) {
      if (!isBlocked(pointerAdd.getResult()))
        return emitScalarOperation(operation, indent);
      auto pointer = pointerLaneExpression(pointerAdd.getResult(), lane);
      if (!pointer)
        return pointer.takeError();
      expressions[pointerAdd.getResult()] = pointer->pointer;
      pointerStrides[pointerAdd.getResult()] = pointer->elementStride;
      return llvm::Error::success();
    }
    if (auto load = llvm::dyn_cast<kernel::LoadOp>(operation))
      return emitVectorLoad(load, group, vl, indent);
    if (auto store = llvm::dyn_cast<kernel::StoreOp>(operation))
      return emitVectorStore(store, group, vl, indent);
    if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
      auto accumulator = reductions.find(reduction);
      if (accumulator == reductions.end())
        return makeEmissionError("reduction accumulator was not materialized");
      return emitVectorReduction(reduction, group, vl, accumulator->second,
                                 indent);
    }
    if (auto contract = llvm::dyn_cast<kernel::ContractOp>(operation)) {
      auto accumulator = contracts.find(contract);
      if (accumulator == contracts.end())
        return makeEmissionError(
            "contraction accumulator was not materialized");
      return emitVectorContract(contract, group, vl, accumulator->second,
                                indent);
    }
    if (llvm::isa<kernel::CastOp>(operation))
      return makeEmissionError("blocked cast RVV emission is not implemented");
    return makeEmissionError(llvm::Twine("no RVV vector rule for ") +
                             operation.getName().getStringRef());
  }

  llvm::Error emitArange(kernel::ArangeOp arange, GroupInfo &group,
                         llvm::StringRef lane, llvm::StringRef vl,
                         unsigned indent) {
    auto config = getVectorConfig(arange.getResult().getType(), group);
    if (!config)
      return config.takeError();
    auto start = lookup(arange.getStart());
    if (!start)
      return start.takeError();
    if (activeAxisGroup == &group) {
      auto axis = arangeAxis(arange, group);
      if (!axis)
        return axis.takeError();
      if (*axis != activeVectorAxis) {
        auto coordinate = activeSerialCoordinates.find(*axis);
        if (coordinate == activeSerialCoordinates.end())
          return makeEmissionError(
              "serial arange axis has no active physical coordinate");
        expressions[arange.getResult()] =
            "(" + *start + " + " + coordinate->second + ")";
        return llvm::Error::success();
      }
    }
    std::string name = valueName(arange.getResult());
    line(indent, config->cType + " " + name + " = __riscv_vid_v_" +
                     config->suffix + "(" + vl.str() + ");");
    line(indent, name + " = __riscv_vadd_vx_" + config->suffix + "(" + name +
                     ", static_cast<uint64_t>(" + *start + " + " + lane.str() +
                     "), " + vl.str() + ");");
    expressions[arange.getResult()] = std::move(name);
    vectorValues.insert(arange.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorBinary(kernel::BinaryOp binary,
                               GroupInfo &group, llvm::StringRef vl,
                               unsigned indent) {
    auto config = getVectorConfig(binary.getResult().getType(), group);
    if (!config)
      return config.takeError();
    bool lhsVector = vectorValues.contains(binary.getLhs());
    bool rhsVector = vectorValues.contains(binary.getRhs());
    if (!lhsVector && !rhsVector) {
      auto lhs = lookup(binary.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(binary.getRhs());
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef op = llvm::StringSwitch<llvm::StringRef>(binary.getKind())
                               .Case("add", "+")
                               .Case("sub", "-")
                               .Case("mul", "*")
                               .Case("div", "/")
                               .Case("mod", "%")
                               .Case("and", "&")
                               .Case("or", "|")
                               .Case("xor", "^")
                               .Default("");
      std::string name = valueName(binary.getResult());
      std::string expression;
      if (!op.empty())
        expression = "(" + *lhs + " " + op.str() + " " + *rhs + ")";
      else if (binary.getKind() == "max")
        expression = "(" + *lhs + " > " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else if (binary.getKind() == "min")
        expression = "(" + *lhs + " < " + *rhs + " ? " + *lhs + " : " +
                     *rhs + ")";
      else
        return makeEmissionError(
            "unsupported scalarized blocked binary operation");
      line(indent, "auto " + name + " = " + expression + ";");
      expressions[binary.getResult()] = std::move(name);
      return llvm::Error::success();
    }

    if (config->mask) {
      llvm::StringRef kind = binary.getKind();
      if (kind != "and" && kind != "or" && kind != "xor")
        return makeEmissionError("RVV mask values support only and, or, and xor");
      std::string name = valueName(binary.getResult());
      if (lhsVector && rhsVector) {
        auto lhs = lookup(binary.getLhs());
        if (!lhs)
          return lhs.takeError();
        auto rhs = lookup(binary.getRhs());
        if (!rhs)
          return rhs.takeError();
        std::string mnemonic =
            kind == "and" ? "vmand" : kind == "or" ? "vmor" : "vmxor";
        line(indent, config->cType + " " + name + " = __riscv_" + mnemonic +
                         "_mm_b" + std::to_string(config->laneRatio) + "(" +
                         *lhs + ", " + *rhs + ", " + vl.str() + ");");
      } else {
        mlir::Value vectorValue = lhsVector ? binary.getLhs() : binary.getRhs();
        mlir::Value scalarValue = lhsVector ? binary.getRhs() : binary.getLhs();
        auto vectorExpression = lookup(vectorValue);
        if (!vectorExpression)
          return vectorExpression.takeError();
        auto scalarExpression = lookup(scalarValue);
        if (!scalarExpression)
          return scalarExpression.takeError();
        std::string maskSuffix =
            "_m_b" + std::to_string(config->laneRatio) + "(" + vl.str() + ")";
        std::string expression;
        if (kind == "and")
          expression = "(" + *scalarExpression + " ? " + *vectorExpression +
                       " : __riscv_vmclr" + maskSuffix + ")";
        else if (kind == "or")
          expression = "(" + *scalarExpression + " ? __riscv_vmset" +
                       maskSuffix + " : " + *vectorExpression + ")";
        else
          expression = "(" + *scalarExpression + " ? __riscv_vmnot_m_b" +
                       std::to_string(config->laneRatio) + "(" +
                       *vectorExpression + ", " + vl.str() + ") : " +
                       *vectorExpression + ")";
        line(indent, config->cType + " " + name + " = " + expression + ";");
      }
      expressions[binary.getResult()] = std::move(name);
      vectorValues.insert(binary.getResult());
      return llvm::Error::success();
    }

    std::string mnemonic;
    if (config->floating)
      mnemonic = llvm::StringSwitch<std::string>(binary.getKind())
                     .Case("add", "vfadd")
                     .Case("sub", "vfsub")
                     .Case("mul", "vfmul")
                     .Case("div", "vfdiv")
                     .Case("max", "vfmax")
                     .Case("min", "vfmin")
                     .Default("");
    else
      mnemonic = llvm::StringSwitch<std::string>(binary.getKind())
                     .Case("add", "vadd")
                     .Case("sub", "vsub")
                     .Case("mul", "vmul")
                     .Case("and", "vand")
                     .Case("or", "vor")
                     .Case("xor", "vxor")
                     .Case("max", config->unsignedInteger ? "vmaxu" : "vmax")
                     .Case("min", config->unsignedInteger ? "vminu" : "vmin")
                     .Default("");
    if (mnemonic.empty())
      return makeEmissionError("unsupported RVV vector binary operation");

    std::string lhs;
    std::string rhs;
    std::string form;
    if (lhsVector && rhsVector) {
      auto lhsValue = lookup(binary.getLhs());
      if (!lhsValue)
        return lhsValue.takeError();
      auto rhsValue = lookup(binary.getRhs());
      if (!rhsValue)
        return rhsValue.takeError();
      lhs = *lhsValue;
      rhs = *rhsValue;
      form = "vv";
    } else {
      bool commutative = binary.getKind() == "add" || binary.getKind() == "mul" ||
                         binary.getKind() == "and" || binary.getKind() == "or" ||
                         binary.getKind() == "xor" || binary.getKind() == "max" ||
                         binary.getKind() == "min";
      mlir::Value vectorValue = lhsVector ? binary.getLhs() : binary.getRhs();
      mlir::Value scalarValue = lhsVector ? binary.getRhs() : binary.getLhs();
      if (!lhsVector && !commutative)
        return makeEmissionError(
            "scalar-left non-commutative RVV binary is not implemented");
      auto vectorExpression = lookup(vectorValue);
      if (!vectorExpression)
        return vectorExpression.takeError();
      auto scalarExpression = lookup(scalarValue);
      if (!scalarExpression)
        return scalarExpression.takeError();
      lhs = *vectorExpression;
      rhs = *scalarExpression;
      form = config->floating ? "vf" : "vx";
    }

    std::string name = valueName(binary.getResult());
    line(indent, config->cType + " " + name + " = __riscv_" + mnemonic + "_" +
                     form + "_" + config->suffix + "(" + lhs + ", " + rhs +
                     ", " + vl.str() + ");");
    expressions[binary.getResult()] = std::move(name);
    vectorValues.insert(binary.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorExp(kernel::UnaryOp unary, GroupInfo &group,
                            llvm::StringRef vl, unsigned indent) {
    auto strategy = unaryStrategies.find(unary.getOperation());
    if (strategy == unaryStrategies.end() ||
        strategy->second != "exp_poly_v1")
      return makeEmissionError(
          "canonical vector exponential has no supported selected strategy");
    if (!vectorValues.contains(unary.getInput()))
      return makeEmissionError(
          "exp_poly_v1 requires an input varying along the selected vector axis");
    auto input = lookup(unary.getInput());
    if (!input)
      return input.takeError();
    auto config = getVectorConfig(unary.getResult().getType(), group);
    if (!config)
      return config.takeError();
    if (!config->floating || config->sew != 32)
      return makeEmissionError("exp_poly_v1 requires an f32 RVV vector");

    std::string floatType = config->cType;
    std::string floatSuffix = config->suffix;
    std::string uintType = "vuint32" + config->lmul + "_t";
    std::string uintSuffix = "u32" + config->lmul;
    std::string maskType =
        "vbool" + std::to_string(config->laneRatio) + "_t";
    std::string maskSuffix = "b" + std::to_string(config->laneRatio);
    std::string base = valueName(unary.getResult()) + "_exp";
    auto name = [&](llvm::StringRef suffix) {
      return base + "_" + suffix.str();
    };
    auto emitFloat = [&](llvm::StringRef local, const std::string &expression) {
      line(indent, floatType + " " + name(local) + " = " + expression + ";");
    };
    auto emitUInt = [&](llvm::StringRef local, const std::string &expression) {
      line(indent, uintType + " " + name(local) + " = " + expression + ";");
    };
    auto emitMask = [&](llvm::StringRef local, const std::string &expression) {
      line(indent, maskType + " " + name(local) + " = " + expression + ";");
    };
    auto call = [&](llvm::StringRef mnemonic, llvm::StringRef suffix,
                    const std::string &arguments) {
      return "__riscv_" + mnemonic.str() + "_" + suffix.str() + "(" +
             arguments + ")";
    };
    auto withVL = [&](std::initializer_list<std::string> arguments) {
      std::string joined;
      llvm::raw_string_ostream stream(joined);
      llvm::interleaveComma(arguments, stream);
      if (!arguments.size())
        stream << vl;
      else
        stream << ", " << vl;
      stream.flush();
      return joined;
    };

    // The polynomial itself is evaluated only on finite lanes.  This avoids
    // feeding NaN/Inf bit patterns through the integer exponent construction;
    // canonical special values are restored after the finite approximation.
    emitFloat("input_abs", call("vfabs_v", floatSuffix,
                                 withVL({*input})));
    emitMask("finite", call("vmfle_vf", floatSuffix + "_" + maskSuffix,
                             withVL({name("input_abs"),
                                     "0x1.fffffep127f"})));
    emitFloat("zero", call("vfmv_v_f", floatSuffix, withVL({"0.0f"})));
    emitFloat("finite_input",
              call("vmerge_vvm", floatSuffix,
                   withVL({name("zero"), *input, name("finite")})));
    std::string finiteInput = name("finite_input");

    emitFloat("r", call("vfmv_v_f", floatSuffix,
                         withVL({"0x1.8p23f"})));
    emitFloat("z", call("vfmacc_vf", floatSuffix,
                         withVL({name("r"), "0x1.715476p+0f", finiteInput})));
    emitFloat("n", call("vfsub_vv", floatSuffix,
                         withVL({name("z"), name("r")})));
    emitFloat("b0", call("vfnmsac_vf", floatSuffix,
                          withVL({finiteInput, "0x1.62e4p-1f", name("n")})));
    emitFloat("b", call("vfnmsac_vf", floatSuffix,
                         withVL({name("b0"), "0x1.7f7d1cp-20f", name("n")})));
    emitUInt("zbits", call("vreinterpret_v", floatSuffix + "_" + uintSuffix,
                            name("z")));
    emitUInt("e", call("vsll_vx", uintSuffix,
                        withVL({name("zbits"), "23"})));
    emitUInt("kbits", call("vadd_vx", uintSuffix,
                            withVL({name("e"), "0x3f800000u"})));
    emitFloat("k", call("vreinterpret_v", uintSuffix + "_" + floatSuffix,
                         name("kbits")));
    emitFloat("absn", call("vfabs_v", floatSuffix,
                            withVL({name("n")})));
    emitMask("extreme", call("vmfgt_vf", floatSuffix + "_" + maskSuffix,
                              withVL({name("absn"), "126.0f"})));
    emitFloat("u", call("vfmul_vv", floatSuffix,
                         withVL({name("b"), name("b")})));
    emitFloat("ja", call("vfmul_vf", floatSuffix,
                          withVL({name("b"), "0x1.ffffecp-1f"})));
    emitFloat("jc0", call("vfmv_v_f", floatSuffix,
                           withVL({"0x1.fffdb6p-2f"})));
    emitFloat("jia", call("vfmacc_vf", floatSuffix,
                           withVL({name("jc0"), "0x1.555e66p-3f",
                                   name("b")})));
    emitFloat("jc1", call("vfmv_v_f", floatSuffix,
                           withVL({"0x1.573e2ep-5f"})));
    emitFloat("jib", call("vfmacc_vf", floatSuffix,
                           withVL({name("jc1"), "0x1.0e4020p-7f",
                                   name("b")})));
    emitFloat("jmid", call("vfmacc_vv", floatSuffix,
                            withVL({name("jia"), name("jib"), name("u")})));
    emitFloat("j", call("vfmacc_vv", floatSuffix,
                         withVL({name("ja"), name("jmid"), name("u")})));
    emitFloat("fast", call("vfmacc_vv", floatSuffix,
                            withVL({name("k"), name("k"), name("j")})));

    std::string resultName = valueName(unary.getResult());
    line(indent, floatType + " " + resultName + " = " + name("fast") + ";");
    std::string population = name("population");
    line(indent,
         "size_t " + population + " = __riscv_vcpop_m_" + maskSuffix + "(" +
             name("extreme") + ", " + vl.str() + ");");
    line(indent, "if (" + population + " != 0) {");
    unsigned nested = indent + 1;
    auto emitNestedMask =
        [&](llvm::StringRef local, const std::string &expression) {
      line(nested, maskType + " " + name(local) + " = " + expression + ";");
    };
    auto emitNestedUInt =
        [&](llvm::StringRef local, const std::string &expression) {
      line(nested, uintType + " " + name(local) + " = " + expression + ";");
    };
    auto emitNestedFloat =
        [&](llvm::StringRef local, const std::string &expression) {
      line(nested, floatType + " " + name(local) + " = " + expression + ";");
    };
    emitNestedMask("negative",
                   call("vmfle_vf", floatSuffix + "_" + maskSuffix,
                        withVL({name("n"), "0.0f"})));
    emitNestedUInt("dzero", call("vmv_v_x", uintSuffix, withVL({"0u"})));
    emitNestedUInt("d", call("vmerge_vxm", uintSuffix,
                              withVL({name("dzero"), "0x82000000u",
                                      name("negative")})));
    emitNestedUInt("s1bits", call("vadd_vx", uintSuffix,
                                   withVL({name("d"), "0x7f000000u"})));
    emitNestedFloat("s1",
                    call("vreinterpret_v", uintSuffix + "_" + floatSuffix,
                         name("s1bits")));
    emitNestedUInt("s2bits", call("vsub_vv", uintSuffix,
                                   withVL({name("e"), name("d")})));
    emitNestedFloat("s2",
                    call("vreinterpret_v", uintSuffix + "_" + floatSuffix,
                         name("s2bits")));
    emitNestedFloat("wide0", call("vfmacc_vv", floatSuffix,
                                   withVL({name("s2"), name("s2"),
                                           name("j")})));
    emitNestedFloat("wide", call("vfmul_vv", floatSuffix,
                                  withVL({name("wide0"), name("s1")})));
    emitNestedFloat("range", call("vmerge_vvm", floatSuffix,
                                   withVL({name("fast"), name("wide"),
                                           name("extreme")})));
    emitNestedMask("overflow",
                   call("vmfgt_vf", floatSuffix + "_" + maskSuffix,
                        withVL({name("absn"), "192.0f"})));
    emitNestedFloat("s1sq", call("vfmul_vv", floatSuffix,
                                  withVL({name("s1"), name("s1")})));
    emitNestedFloat("slow", call("vmerge_vvm", floatSuffix,
                                  withVL({name("range"), name("s1sq"),
                                          name("overflow")})));
    line(nested, resultName + " = " + name("slow") + ";");
    line(indent, "}");

    emitMask("positive_inf",
             call("vmfeq_vf", floatSuffix + "_" + maskSuffix,
                  withVL({*input, "__builtin_inff()"})));
    emitFloat("positive_inf_value",
              call("vfmv_v_f", floatSuffix,
                   withVL({"__builtin_inff()"})));
    line(indent,
         resultName + " = " +
             call("vmerge_vvm", floatSuffix,
                  withVL({resultName, name("positive_inf_value"),
                          name("positive_inf")})) +
             ";");
    emitMask("negative_inf",
             call("vmfeq_vf", floatSuffix + "_" + maskSuffix,
                  withVL({*input, "-__builtin_inff()"})));
    emitFloat("negative_inf_value",
              call("vfmv_v_f", floatSuffix, withVL({"0.0f"})));
    line(indent,
         resultName + " = " +
             call("vmerge_vvm", floatSuffix,
                  withVL({resultName, name("negative_inf_value"),
                          name("negative_inf")})) +
             ";");
    emitMask("nan", call("vmfne_vv", floatSuffix + "_" + maskSuffix,
                          withVL({*input, *input})));
    line(indent, resultName + " = " +
                     call("vmerge_vvm", floatSuffix,
                          withVL({resultName, *input, name("nan")})) +
                     ";");
    expressions[unary.getResult()] = resultName;
    vectorValues.insert(unary.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorUnary(kernel::UnaryOp unary, GroupInfo &group,
                              llvm::StringRef vl, unsigned indent) {
    auto input = lookup(unary.getInput());
    if (!input)
      return input.takeError();
    if (!vectorValues.contains(unary.getInput())) {
      if (unary.getKind() != "neg")
        return makeEmissionError(
            "unsupported scalarized blocked unary operation");
      std::string name = valueName(unary.getResult());
      line(indent, "auto " + name + " = (-" + *input + ");");
      expressions[unary.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    if (unary.getKind() == "exp")
      return emitVectorExp(unary, group, vl, indent);
    auto config = getVectorConfig(unary.getResult().getType(), group);
    if (!config)
      return config.takeError();
    if (unary.getKind() != "neg" || !config->floating || config->sew != 32)
      return makeEmissionError(
          "first selected RVV source slice supports f32 vector negation");
    std::string name = valueName(unary.getResult());
    line(indent, config->cType + " " + name + " = __riscv_vfneg_v_" +
                     config->suffix + "(" + *input + ", " + vl.str() + ");");
    expressions[unary.getResult()] = std::move(name);
    vectorValues.insert(unary.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorCompare(kernel::CompareOp compare,
                                GroupInfo &group, llvm::StringRef vl,
                                unsigned indent) {
    bool lhsVector = vectorValues.contains(compare.getLhs());
    bool rhsVector = vectorValues.contains(compare.getRhs());
    if (!lhsVector && !rhsVector) {
      auto lhs = lookup(compare.getLhs());
      if (!lhs)
        return lhs.takeError();
      auto rhs = lookup(compare.getRhs());
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef predicate =
          llvm::StringSwitch<llvm::StringRef>(compare.getPredicate())
              .Case("eq", "==")
              .Case("ne", "!=")
              .Case("lt", "<")
              .Case("le", "<=")
              .Case("gt", ">")
              .Case("ge", ">=")
              .Default("");
      if (predicate.empty())
        return makeEmissionError("unsupported scalarized comparison predicate");
      std::string name = valueName(compare.getResult());
      line(indent, "bool " + name + " = (" + *lhs + " " + predicate.str() +
                       " " + *rhs + ");");
      expressions[compare.getResult()] = std::move(name);
      return llvm::Error::success();
    }
    bool swapOperands = !lhsVector && rhsVector;
    mlir::Value vectorOperand =
        swapOperands ? compare.getRhs() : compare.getLhs();
    mlir::Value otherOperand =
        swapOperands ? compare.getLhs() : compare.getRhs();
    auto operandConfig = getVectorConfig(vectorOperand.getType(), group);
    if (!operandConfig)
      return operandConfig.takeError();
    auto resultConfig = getVectorConfig(compare.getResult().getType(), group);
    if (!resultConfig)
      return resultConfig.takeError();
    auto lhs = lookup(vectorOperand);
    if (!lhs)
      return lhs.takeError();
    auto rhs = lookup(otherOperand);
    if (!rhs)
      return rhs.takeError();

    llvm::StringRef predicate = compare.getPredicate();
    if (swapOperands)
      predicate = llvm::StringSwitch<llvm::StringRef>(predicate)
                      .Case("eq", "eq")
                      .Case("ne", "ne")
                      .Case("lt", "gt")
                      .Case("le", "ge")
                      .Case("gt", "lt")
                      .Case("ge", "le")
                      .Default("");
    std::string mnemonic;
    if (operandConfig->floating)
      mnemonic = llvm::StringSwitch<std::string>(predicate)
                     .Case("eq", "vmfeq")
                     .Case("ne", "vmfne")
                     .Case("lt", "vmflt")
                     .Case("le", "vmfle")
                     .Case("gt", "vmfgt")
                     .Case("ge", "vmfge")
                     .Default("");
    else
      mnemonic = llvm::StringSwitch<std::string>(predicate)
                     .Case("eq", "vmseq")
                     .Case("ne", "vmsne")
                     .Case("lt", operandConfig->unsignedInteger ? "vmsltu"
                                                                 : "vmslt")
                     .Case("le", operandConfig->unsignedInteger ? "vmsleu"
                                                                 : "vmsle")
                     .Case("gt", operandConfig->unsignedInteger ? "vmsgtu"
                                                                 : "vmsgt")
                     .Case("ge", operandConfig->unsignedInteger ? "vmsgeu"
                                                                 : "vmsge")
                     .Default("");
    if (mnemonic.empty())
      return makeEmissionError("unsupported RVV comparison predicate");
    std::string form = lhsVector && rhsVector
                           ? "vv"
                           : (operandConfig->floating ? "vf" : "vx");
    std::string name = valueName(compare.getResult());
    line(indent, resultConfig->cType + " " + name + " = __riscv_" + mnemonic +
                     "_" + form + "_" + operandConfig->suffix + "_b" +
                     std::to_string(operandConfig->laneRatio) + "(" + *lhs +
                     ", " + *rhs + ", " + vl.str() + ");");
    expressions[compare.getResult()] = std::move(name);
    vectorValues.insert(compare.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorLoad(kernel::LoadOp load, GroupInfo &group,
                             llvm::StringRef vl, unsigned indent) {
    auto config = getVectorConfig(load.getResult().getType(), group);
    if (!config)
      return config.takeError();
    if (!config->floating || config->sew != 32)
      return makeEmissionError(
          "first selected RVV source slice supports f32 blocked loads");
    auto pointer = lookup(load.getPointer());
    if (!pointer)
      return pointer.takeError();
    auto mask = lookupVectorMask(load.getMask(), group, vl);
    if (!mask)
      return mask.takeError();
    auto other = lookup(load.getOther());
    if (!other)
      return other.takeError();
    auto stride = pointerStrides.find(load.getPointer());
    if (stride == pointerStrides.end())
      return makeEmissionError(
          "blocked load pointer has no selected affine lane stride");
    std::string passthrough = valueName(load.getResult()) + "_passthrough";
    line(indent, config->cType + " " + passthrough + " = __riscv_vfmv_v_f_" +
                     config->suffix + "(" + *other + ", " + vl.str() + ");");
    std::string name = valueName(load.getResult());
    if (stride->second == "1") {
      line(indent, config->cType + " " + name + " = __riscv_vle" +
                       std::to_string(config->sew) + "_v_" + config->suffix +
                       "_tumu(" + *mask + ", " + passthrough + ", " + *pointer +
                       ", " + vl.str() + ");");
    } else {
      std::string byteStride =
          "static_cast<ptrdiff_t>(" + stride->second +
          ") * static_cast<ptrdiff_t>(sizeof(float))";
      line(indent, config->cType + " " + name + " = __riscv_vlse" +
                       std::to_string(config->sew) + "_v_" + config->suffix +
                       "_tumu(" + *mask + ", " + passthrough + ", " + *pointer +
                       ", " + byteStride + ", " + vl.str() + ");");
    }
    expressions[load.getResult()] = std::move(name);
    vectorValues.insert(load.getResult());
    return llvm::Error::success();
  }

  llvm::Error emitVectorStore(kernel::StoreOp store, GroupInfo &group,
                              llvm::StringRef vl, unsigned indent) {
    auto config = getVectorConfig(store.getValue().getType(), group);
    if (!config)
      return config.takeError();
    if (!config->floating || config->sew != 32)
      return makeEmissionError(
          "first selected RVV source slice supports f32 blocked stores");
    auto pointer = lookup(store.getPointer());
    if (!pointer)
      return pointer.takeError();
    auto value = lookup(store.getValue());
    if (!value)
      return value.takeError();
    auto mask = lookupVectorMask(store.getMask(), group, vl);
    if (!mask)
      return mask.takeError();
    auto stride = pointerStrides.find(store.getPointer());
    if (stride == pointerStrides.end())
      return makeEmissionError(
          "blocked store pointer has no selected affine lane stride");
    if (stride->second != "1")
      return makeEmissionError(
          "selected RVV vector store currently requires unit lane stride");
    line(indent, "__riscv_vse" + std::to_string(config->sew) + "_v_" +
                     config->suffix + "_m(" + *mask + ", " + *pointer + ", " +
                     *value + ", " + vl.str() + ");");
    return llvm::Error::success();
  }

  llvm::Error emitVectorReduction(kernel::ReduceOp reduction,
                                  GroupInfo &group, llvm::StringRef vl,
                                  llvm::StringRef accumulator,
                                  unsigned indent) {
    auto config = getVectorConfig(reduction.getInput().getType(), group);
    if (!config)
      return config.takeError();
    auto input = lookup(reduction.getInput());
    if (!input)
      return input.takeError();
    if (!config->floating || config->sew != 32 ||
        (reduction.getKind() != "sum" && reduction.getKind() != "max" &&
         reduction.getKind() != "min"))
      return makeEmissionError("unsupported RVV reduction configuration");
    std::string base = valueName(reduction.getResult());
    std::string seed = base + "_seed";
    std::string partial = base + "_partial";
    line(indent, "vfloat32m1_t " + seed +
                     " = __riscv_vfmv_v_f_f32m1(" + accumulator.str() +
                     ", 1);");
    auto strategy = reductionStrategies.find(reduction.getOperation());
    if (strategy == reductionStrategies.end())
      return makeEmissionError(
          "canonical reduction has no selected RVV reduction strategy");
    std::string mnemonic;
    if (reduction.getKind() == "sum")
      mnemonic = llvm::StringSwitch<std::string>(strategy->second)
                     .Case("ordered", "vfredosum")
                     .Case("tree", "vfredusum")
                     .Default("");
    else if (strategy->second == "ordered" || strategy->second == "tree")
      mnemonic = reduction.getKind() == "max" ? "vfredmax" : "vfredmin";
    if (mnemonic.empty())
      return makeEmissionError("selected RVV reduction strategy is unsupported");
    line(indent, "vfloat32m1_t " + partial + " = __riscv_" + mnemonic +
                     "_vs_" + config->suffix + "_f32m1(" + *input + ", " + seed +
                     ", " + vl.str() + ");");
    line(indent, accumulator.str() + " = __riscv_vfmv_f_s_f32m1_f32(" +
                     partial + ");");
    return llvm::Error::success();
  }

  llvm::Error emitVectorContract(kernel::ContractOp contract,
                                 GroupInfo &group, llvm::StringRef vl,
                                 llvm::StringRef accumulator,
                                 unsigned indent) {
    auto config = getVectorConfig(contract.getLhs().getType(), group);
    if (!config)
      return config.takeError();
    if (!config->floating || config->sew != 32 ||
        !vectorValues.contains(contract.getLhs()) ||
        !vectorValues.contains(contract.getRhs()))
      return makeEmissionError(
          "first selected RVV contract slice requires two f32 vectors");
    auto lhs = lookup(contract.getLhs());
    if (!lhs)
      return lhs.takeError();
    auto rhs = lookup(contract.getRhs());
    if (!rhs)
      return rhs.takeError();
    auto strategy = contractSelections.find(contract.getOperation());
    if (strategy == contractSelections.end())
      return makeEmissionError(
          "canonical contraction has no selected RVV strategy");
    std::string mnemonic =
        llvm::StringSwitch<std::string>(strategy->second.strategy)
                               .Case("ordered", "vfredosum")
                               .Case("tree", "vfredusum")
                               .Default("");
    if (mnemonic.empty())
      return makeEmissionError("selected RVV contract strategy is unsupported");

    std::string base = valueName(contract.getResult());
    std::string product = base + "_product";
    std::string seed = base + "_seed";
    std::string partial = base + "_partial";
    line(indent, config->cType + " " + product + " = __riscv_vfmul_vv_" +
                     config->suffix + "(" + *lhs + ", " + *rhs + ", " +
                     vl.str() + ");");
    line(indent, "vfloat32m1_t " + seed +
                     " = __riscv_vfmv_v_f_f32m1(" + accumulator.str() +
                     ", 1);");
    line(indent, "vfloat32m1_t " + partial + " = __riscv_" + mnemonic +
                     "_vs_" + config->suffix + "_f32m1(" + product + ", " +
                     seed + ", " + vl.str() + ");");
    line(indent, accumulator.str() + " = __riscv_vfmv_f_s_f32m1_f32(" +
                     partial + ");");
    return llvm::Error::success();
  }

  llvm::Expected<PointerLaneExpression>
  pointerLaneExpression(mlir::Value value, llvm::StringRef lane) {
    auto pointerAdd = value.getDefiningOp<kernel::PtrAddOp>();
    if (!pointerAdd)
      return makeEmissionError("blocked pointer is not defined by ptr_add");
    std::string base;
    std::string baseStride = "0";
    if (isBlocked(pointerAdd.getBase())) {
      auto nested = pointerLaneExpression(pointerAdd.getBase(), lane);
      if (!nested)
        return nested.takeError();
      base = nested->pointer;
      baseStride = nested->elementStride;
    } else {
      auto scalarBase = lookup(pointerAdd.getBase());
      if (!scalarBase)
        return scalarBase.takeError();
      base = *scalarBase;
    }
    std::string offset;
    std::string offsetStride = "0";
    if (isBlocked(pointerAdd.getOffset())) {
      auto laneOffset = deriveLaneExpression(pointerAdd.getOffset(), lane);
      if (!laneOffset)
        return laneOffset.takeError();
      offset = laneOffset->expression;
      offsetStride = laneOffset->stride;
    } else {
      auto scalarOffset = lookup(pointerAdd.getOffset());
      if (!scalarOffset)
        return scalarOffset.takeError();
      offset = *scalarOffset;
    }
    std::string combinedStride;
    if (baseStride == "0")
      combinedStride = offsetStride;
    else if (offsetStride == "0")
      combinedStride = baseStride;
    else
      combinedStride = "(" + baseStride + " + " + offsetStride + ")";
    return PointerLaneExpression{"(" + base + " + " + offset + ")",
                                 std::move(combinedStride)};
  }

  llvm::Expected<LaneExpression>
  deriveLaneExpression(mlir::Value value, llvm::StringRef lane) {
    if (!isBlocked(value)) {
      auto scalar = lookup(value);
      if (!scalar)
        return scalar.takeError();
      return LaneExpression{*scalar, "0"};
    }
    if (auto arange = value.getDefiningOp<kernel::ArangeOp>()) {
      auto start = lookup(arange.getStart());
      if (!start)
        return start.takeError();
      if (activeAxisGroup) {
        auto axis = arangeAxis(arange, *activeAxisGroup);
        if (!axis)
          return axis.takeError();
        if (*axis != activeVectorAxis) {
          auto coordinate = activeSerialCoordinates.find(*axis);
          if (coordinate == activeSerialCoordinates.end())
            return makeEmissionError(
                "serial pointer axis has no active physical coordinate");
          return LaneExpression{
              "(" + *start + " + " + coordinate->second + ")", "0"};
        }
      }
      return LaneExpression{"(" + *start + " + " + lane.str() + ")", "1"};
    }
    if (auto expand = value.getDefiningOp<kernel::ExpandDimsOp>())
      return deriveLaneExpression(expand.getInput(), lane);
    if (auto cast = value.getDefiningOp<kernel::CastOp>())
      return deriveLaneExpression(cast.getInput(), lane);
    auto binary = value.getDefiningOp<kernel::BinaryOp>();
    if (!binary || (binary.getKind() != "add" && binary.getKind() != "sub" &&
                    binary.getKind() != "mul"))
      return makeEmissionError(
          "contiguous pointer offset has no supported affine axis expression");
    auto lhs = deriveLaneExpression(binary.getLhs(), lane);
    if (!lhs)
      return lhs.takeError();
    auto rhs = deriveLaneExpression(binary.getRhs(), lane);
    if (!rhs)
      return rhs.takeError();
    if (binary.getKind() == "add") {
      std::string stride = lhs->stride == "0"
                               ? rhs->stride
                               : rhs->stride == "0"
                                     ? lhs->stride
                                     : "(" + lhs->stride + " + " + rhs->stride +
                                           ")";
      return LaneExpression{"(" + lhs->expression + " + " + rhs->expression +
                                ")",
                            std::move(stride)};
    }
    if (binary.getKind() == "sub") {
      if (rhs->stride != "0")
        return makeEmissionError(
            "selected pointer arithmetic does not support negative or "
            "cancelling lane strides");
      return LaneExpression{"(" + lhs->expression + " - " + rhs->expression +
                                ")",
                            lhs->stride};
    }
    if (lhs->stride == "0" && rhs->stride == "0")
      return LaneExpression{"(" + lhs->expression + " * " + rhs->expression +
                                ")",
                            "0"};
    if (lhs->stride != "0" && rhs->stride == "0") {
      if (isNegativeIndexConstant(binary.getRhs()))
        return makeEmissionError(
            "selected pointer arithmetic does not support negative lane "
            "multipliers");
      return LaneExpression{
          "(" + lhs->expression + " * " + rhs->expression + ")",
          "(" + lhs->stride + " * " + rhs->expression + ")"};
    }
    if (lhs->stride == "0" && rhs->stride != "0") {
      if (isNegativeIndexConstant(binary.getLhs()))
        return makeEmissionError(
            "selected pointer arithmetic does not support negative lane "
            "multipliers");
      return LaneExpression{
          "(" + lhs->expression + " * " + rhs->expression + ")",
          "(" + lhs->expression + " * " + rhs->stride + ")"};
    }
    return makeEmissionError(
        "vector-axis multiplication is not affine in one lane coordinate");
  }

  int64_t getNode(mlir::Operation *operation) const {
    auto node = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
    return node ? node.getInt() : -1;
  }

  mlir::ModuleOp module;
  execution::PlanOp plan;
  llvm::raw_ostream &os;
  kernel::KernelOp kernel;
  llvm::DenseMap<int64_t, mlir::Operation *> nodes;
  llvm::DenseMap<int64_t, GroupInfo> groups;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t, 4>> scopeGroups;
  llvm::DenseMap<mlir::Value, int64_t> valueGroups;
  llvm::DenseMap<std::pair<mlir::Operation *, int64_t>, int64_t>
      conversionGroups;
  llvm::DenseMap<int64_t, std::string> taskBindings;
  llvm::DenseMap<int64_t, std::string> taskNames;
  llvm::DenseMap<int64_t, int64_t> metaBindings;
  llvm::DenseMap<mlir::Operation *, std::string> unaryStrategies;
  llvm::DenseMap<mlir::Operation *, std::string> reductionStrategies;
  llvm::DenseMap<mlir::Operation *, int64_t> reductionGroups;
  llvm::DenseMap<mlir::Operation *, ContractSelection> contractSelections;
  llvm::DenseMap<mlir::Value, std::string> expressions;
  llvm::DenseSet<mlir::Value> vectorValues;
  llvm::DenseMap<mlir::Value, std::string> pointerStrides;
  llvm::DenseMap<int64_t, llvm::DenseSet<mlir::Value>>
      incomingConversionValues;
  GroupInfo *activeAxisGroup = nullptr;
  int64_t activeVectorAxis = -1;
  llvm::DenseMap<int64_t, std::string> activeSerialCoordinates;
};

} // namespace

llvm::Error emitSelectedExecutionRVVSource(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  if (mlir::failed(mlir::verify(module)))
    return makeEmissionError(
        "module verification failed before selected RVV source emission");
  llvm::SmallVector<execution::PlanOp, 4> plans(
      module.getOps<execution::PlanOp>());
  if (plans.empty())
    return makeEmissionError("module has no weft_execution.plan");
  llvm::StringSet<> plannedKernels;
  for (execution::PlanOp plan : plans)
    if (!plannedKernels.insert(plan.getKernelAttr().getValue()).second)
      return makeEmissionError(
          "multiple selected plans would emit the same canonical kernel symbol");

  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <math.h>\n";
  os << "#include <riscv_vector.h>\n\n";
  for (execution::PlanOp plan : plans) {
    KernelSourceEmitter emitter(module, plan, os);
    if (llvm::Error error = emitter.emit())
      return error;
  }
  return llvm::Error::success();
}

} // namespace weft::target::rvv
