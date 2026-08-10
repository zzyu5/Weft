#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Layout/IR/LayoutAlgebra.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace weft::transforms {

#define GEN_PASS_DEF_SELECTIMEEXECUTION
#include "Weft/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");

struct MetaBindingPlan {
  int64_t argument = -1;
  int64_t value = 0;
};

struct ValueLayoutPlan {
  int64_t sourceNode = -1;
  int64_t sourceResult = -1;
  int64_t group = -1;
};

struct ContractSitePlan {
  kernel::ContractOp contract;
  kernel::LoadOp lhsLoad;
  kernel::LoadOp rhsLoad;
  kernel::StoreOp store;
  int64_t logicalM = 0;
  int64_t logicalN = 0;
  int64_t logicalK = 0;
  int64_t lhsGroup = -1;
  int64_t rhsGroup = -1;
  int64_t resultGroup = -1;
  std::string ownerSymbol;
  llvm::DenseSet<mlir::Operation *> operations;
  llvm::DenseSet<mlir::Value> lhsValues;
  llvm::DenseSet<mlir::Value> rhsValues;
  llvm::DenseSet<mlir::Value> resultValues;
};

struct KernelPlan {
  kernel::KernelOp kernel;
  std::string symbol;
  std::string target;
  llvm::DenseMap<mlir::Operation *, int64_t> nodes;
  llvm::SmallVector<MetaBindingPlan, 4> metaBindings;
  llvm::SmallVector<ValueLayoutPlan, 32> valueLayouts;
  llvm::SmallVector<ContractSitePlan, 4> sites;
};

bool isSignedInteger(mlir::Type type, unsigned width) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.isSigned() && integer.getWidth() == width;
}

bool isIntegerZero(mlir::Value value) {
  if (auto cast = value.getDefiningOp<kernel::CastOp>())
    return isIntegerZero(cast.getInput());
  auto constant = value.getDefiningOp<kernel::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer && integer.getInt() == 0;
}

std::string makePlanSymbol(llvm::StringRef kernelName, llvm::StringRef target) {
  std::string symbol = (kernelName + "__" + target).str();
  for (char &character : symbol) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (!std::isalnum(byte) && character != '_')
      character = '_';
  }
  return symbol;
}

void collectOperationAndProducers(mlir::Operation *operation,
                                  mlir::Block &entry,
                                  llvm::DenseSet<mlir::Operation *> &operations) {
  if (!operation || operation->getBlock() != &entry ||
      llvm::isa<kernel::ReturnOp>(operation) ||
      !operations.insert(operation).second)
    return;
  for (mlir::Value operand : operation->getOperands())
    collectOperationAndProducers(operand.getDefiningOp(), entry, operations);
}

void collectValueAndProducers(mlir::Value value, mlir::Block &entry,
                              llvm::DenseSet<mlir::Value> &values) {
  if (!value || !values.insert(value).second)
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != &entry)
    return;
  for (mlir::Value operand : definition->getOperands())
    collectValueAndProducers(operand, entry, values);
}

mlir::Type getScalarElementType(mlir::Type type) {
  if (auto block = mlir::dyn_cast<kernel::BlockType>(type))
    return block.getElementType();
  return type;
}

bool isCoordinateIntegerType(mlir::Type type) {
  if (auto meta = mlir::dyn_cast<kernel::ConstexprType>(type))
    type = meta.getValueType();
  type = getScalarElementType(type);
  return type.isIndex() || type.isInteger(1);
}

bool hasScalarizableExpression(mlir::Value value,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  if (mlir::isa<mlir::BlockArgument>(value))
    return mlir::isa<kernel::PtrType>(value.getType()) ||
           isCoordinateIntegerType(value.getType());
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (llvm::isa<kernel::ConstantOp, kernel::TaskIdOp>(definition))
    return isCoordinateIntegerType(value.getType());
  if (auto meta = llvm::dyn_cast<kernel::MetaValueOp>(definition))
    return hasScalarizableExpression(meta.getInput(), visited);
  if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(definition))
    return isCoordinateIntegerType(value.getType()) &&
           hasScalarizableExpression(arange.getStart(), visited) &&
           hasScalarizableExpression(arange.getExtent(), visited);
  if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(definition))
    return isCoordinateIntegerType(value.getType()) &&
           hasScalarizableExpression(expand.getInput(), visited);
  if (auto splat = llvm::dyn_cast<kernel::SplatOp>(definition))
    return isCoordinateIntegerType(value.getType()) &&
           hasScalarizableExpression(splat.getValue(), visited) &&
           hasScalarizableExpression(splat.getShapeLike(), visited);
  if (auto cast = llvm::dyn_cast<kernel::CastOp>(definition))
    return isCoordinateIntegerType(value.getType()) &&
           isCoordinateIntegerType(cast.getInput().getType()) &&
           hasScalarizableExpression(cast.getInput(), visited);
  if (auto pointer = llvm::dyn_cast<kernel::PtrAddOp>(definition))
    return hasScalarizableExpression(pointer.getBase(), visited) &&
           hasScalarizableExpression(pointer.getOffset(), visited);
  if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(definition)) {
    llvm::StringRef kind = binary.getKind();
    if (!isCoordinateIntegerType(value.getType()) ||
        (kind != "add" && kind != "sub" && kind != "mul" && kind != "and" &&
         kind != "or" && kind != "xor" && kind != "max" && kind != "min"))
      return false;
    return hasScalarizableExpression(binary.getLhs(), visited) &&
           hasScalarizableExpression(binary.getRhs(), visited);
  }
  if (auto compare = llvm::dyn_cast<kernel::CompareOp>(definition)) {
    llvm::StringRef predicate = compare.getPredicate();
    if (!getScalarElementType(value.getType()).isInteger(1) ||
        (predicate != "eq" && predicate != "ne" && predicate != "lt" &&
         predicate != "le" && predicate != "gt" && predicate != "ge"))
      return false;
    return hasScalarizableExpression(compare.getLhs(), visited) &&
           hasScalarizableExpression(compare.getRhs(), visited);
  }
  return false;
}

bool hasScalarizableExpression(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return hasScalarizableExpression(value, visited);
}

std::optional<int64_t> resolveSelectedExtent(
    kernel::KernelOp source, mlir::Value value, int64_t axis,
    const llvm::DenseMap<int64_t, int64_t> &selectedMeta) {
  auto extent = kernel::deriveLogicalExtent(value, axis);
  if (!extent)
    return std::nullopt;
  if (extent->constant)
    return extent->constant;
  auto meta = extent->dynamic.getDefiningOp<kernel::MetaValueOp>();
  if (!meta)
    return std::nullopt;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(meta.getInput());
  if (!argument || argument.getOwner() != &source.getBody().front())
    return std::nullopt;
  auto binding = selectedMeta.find(argument.getArgNumber());
  return binding == selectedMeta.end()
             ? std::nullopt
             : std::optional<int64_t>(binding->second);
}

class SelectIMEExecutionPass final
    : public impl::SelectIMEExecutionBase<SelectIMEExecutionPass> {
public:
  using impl::SelectIMEExecutionBase<
      SelectIMEExecutionPass>::SelectIMEExecutionBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::StringRef targetRef(target);
    if (targetRef.empty() || !targetRef.starts_with("rv64") ||
        !weft::ime_execution::hasXsmtvdotiiExtension(targetRef)) {
      module.emitError(
          "weft-select-ime-execution requires an RV64 target carrying the "
          "xsmtvdotii extension");
      signalPassFailure();
      return;
    }
    if (vlenBits != 256) {
      module.emitError(
          "weft-select-ime-execution requires the explicit target capability "
          "vlen-bits=256");
      signalPassFailure();
      return;
    }
    if (blockElements < 0) {
      module.emitError("block-elements must be zero (unset) or positive");
      signalPassFailure();
      return;
    }
    if (blockElements > 0 && !metaBindings.empty()) {
      module.emitError(
          "block-elements and meta-bindings are mutually exclusive; choose "
          "either an explicit uniform value or per-parameter bindings");
      signalPassFailure();
      return;
    }

    llvm::StringMap<int64_t> namedMetaBindings;
    if (mlir::failed(parseNamedMetaBindings(module, namedMetaBindings))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<KernelPlan, 4> plans;
    llvm::StringSet<> usedNamedMetaBindings;
    for (kernel::KernelOp source : module.getOps<kernel::KernelOp>()) {
      auto plan = analyzeKernel(module, source, namedMetaBindings,
                                usedNamedMetaBindings);
      if (mlir::failed(plan)) {
        signalPassFailure();
        return;
      }
      plans.push_back(std::move(*plan));
    }
    if (plans.empty()) {
      module.emitError(
          "weft-select-ime-execution requires at least one canonical "
          "weft_kernel.kernel");
      signalPassFailure();
      return;
    }
    for (const auto &binding : namedMetaBindings) {
      if (usedNamedMetaBindings.contains(binding.getKey()))
        continue;
      module.emitError() << "meta binding '" << binding.getKey()
                         << "' matches no canonical constexpr parameter";
      signalPassFailure();
      return;
    }
    for (KernelPlan &plan : plans)
      materializePlan(module, plan);
  }

private:
  mlir::LogicalResult
  parseNamedMetaBindings(mlir::ModuleOp module,
                         llvm::StringMap<int64_t> &bindings) {
    for (const std::string &spelling : metaBindings) {
      llvm::StringRef specification(spelling);
      size_t separator = specification.find('=');
      if (separator == llvm::StringRef::npos || separator == 0 ||
          separator + 1 == specification.size() ||
          specification.find('=', separator + 1) != llvm::StringRef::npos) {
        module.emitError()
            << "meta binding '" << specification
            << "' must have exactly the form NAME=POSITIVE_INTEGER";
        return mlir::failure();
      }
      llvm::StringRef name = specification.take_front(separator);
      llvm::StringRef valueSpelling = specification.drop_front(separator + 1);
      int64_t value = 0;
      if (valueSpelling.getAsInteger(10, value) || value <= 0) {
        module.emitError()
            << "meta binding '" << specification
            << "' requires a positive base-10 integer value";
        return mlir::failure();
      }
      if (!bindings.try_emplace(name, value).second) {
        module.emitError() << "duplicates meta binding for canonical parameter '"
                           << name << "'";
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  mlir::FailureOr<KernelPlan> analyzeKernel(
      mlir::ModuleOp module, kernel::KernelOp source,
      const llvm::StringMap<int64_t> &namedMetaBindings,
      llvm::StringSet<> &usedNamedMetaBindings) {
    for (execution::PlanOp existing : module.getOps<execution::PlanOp>())
      if (existing.getKernelAttr().getValue() == source.getSymName()) {
        source.emitOpError("already has a selected execution plan");
        return mlir::failure();
      }
    bool hasStaleNode = false;
    source->walk([&](mlir::Operation *operation) {
      hasStaleNode |= operation->hasAttr(kNodeAttrName);
    });
    if (hasStaleNode) {
      source.emitOpError(
          "carries weft_execution.node indices without a selected plan");
      return mlir::failure();
    }
    KernelPlan plan;
    plan.kernel = source;
    plan.target = target;
    plan.symbol = makePlanSymbol(source.getSymName(), target);
    if (mlir::SymbolTable::lookupSymbolIn(module, plan.symbol)) {
      source.emitOpError() << "selected plan symbol @" << plan.symbol
                           << " already exists";
      return mlir::failure();
    }
    int64_t nextNode = 0;
    source->walk([&](mlir::Operation *operation) {
      plan.nodes.try_emplace(operation, nextNode++);
    });

    llvm::DenseMap<int64_t, int64_t> selectedMeta;
    for (auto [index, argument] :
         llvm::enumerate(source.getBody().front().getArguments())) {
      auto meta = mlir::dyn_cast<kernel::ConstexprType>(argument.getType());
      if (!meta)
        continue;
      if (!meta.getValueType().isIndex()) {
        source.emitOpError(
            "first IME execution slice supports only index constexpr bindings");
        return mlir::failure();
      }
      bool hasMaterializedUse = false;
      for (mlir::Operation *user : argument.getUsers()) {
        auto materialized = llvm::dyn_cast<kernel::MetaValueOp>(user);
        if (!materialized) {
          user->emitOpError("constexpr arguments must first pass through meta_value");
          return mlir::failure();
        }
        hasMaterializedUse |= !materialized.getResult().use_empty();
      }
      if (!hasMaterializedUse) {
        source.emitOpError("constexpr argument has no canonical scalar use");
        return mlir::failure();
      }
      auto name = mlir::cast<mlir::StringAttr>(source.getArgNames()[index])
                      .getValue();
      int64_t selectedValue = blockElements;
      if (selectedValue == 0) {
        auto selected = namedMetaBindings.find(name);
        if (selected == namedMetaBindings.end()) {
          source.emitOpError()
              << "has no selected value for canonical constexpr parameter '"
              << name
              << "'; pass a per-parameter meta binding or an explicit uniform "
                 "block-elements value";
          return mlir::failure();
        }
        selectedValue = selected->second;
        usedNamedMetaBindings.insert(name);
      }
      selectedMeta.try_emplace(index, selectedValue);
      plan.metaBindings.push_back(
          MetaBindingPlan{static_cast<int64_t>(index), selectedValue});
    }

    mlir::Block &entry = source.getBody().front();
    llvm::SmallVector<kernel::ContractOp, 4> contracts;
    bool hasNestedOrRegionOperation = false;
    source->walk([&](mlir::Operation *operation) {
      if (operation == source.getOperation())
        return;
      hasNestedOrRegionOperation |=
          operation->getBlock() != &entry || operation->getNumRegions() != 0;
      if (auto contract = llvm::dyn_cast<kernel::ContractOp>(operation))
        contracts.push_back(contract);
    });
    if (hasNestedOrRegionOperation || contracts.empty()) {
      source.emitOpError(
          "IME fragment tiling currently requires one flat canonical body with "
          "at least one contraction site");
      return mlir::failure();
    }

    for (auto [siteIndex, contract] : llvm::enumerate(contracts)) {
      ContractSitePlan site;
      site.contract = contract;
      site.lhsLoad = contract.getLhs().getDefiningOp<kernel::LoadOp>();
      site.rhsLoad = contract.getRhs().getDefiningOp<kernel::LoadOp>();
      auto initSplat = contract.getInit().getDefiningOp<kernel::SplatOp>();
      if (!site.lhsLoad || !site.rhsLoad || site.lhsLoad == site.rhsLoad ||
          !initSplat || !isIntegerZero(site.lhsLoad.getOther()) ||
          !isIntegerZero(site.rhsLoad.getOther()) ||
          !isIntegerZero(initSplat.getValue())) {
        contract.emitOpError(
            "must consume two distinct direct zero-filled loads and start from "
            "a zero-splat accumulator");
        return mlir::failure();
      }

      for (mlir::Operation *user : contract.getResult().getUsers()) {
        auto candidate = llvm::dyn_cast<kernel::StoreOp>(user);
        if (!candidate || candidate.getValue() != contract.getResult() ||
            site.store) {
          contract.emitOpError(
              "must feed exactly one direct store and have no other result use");
          return mlir::failure();
        }
        site.store = candidate;
      }
      if (!site.store) {
        contract.emitOpError("has no direct canonical result store");
        return mlir::failure();
      }

      auto lhs = mlir::dyn_cast<kernel::BlockType>(contract.getLhs().getType());
      auto rhs = mlir::dyn_cast<kernel::BlockType>(contract.getRhs().getType());
      auto init = mlir::dyn_cast<kernel::BlockType>(contract.getInit().getType());
      auto result =
          mlir::dyn_cast<kernel::BlockType>(contract.getResult().getType());
      if (!lhs || !rhs || !init || !result || lhs.getShape().size() != 2 ||
          rhs.getShape().size() != 2 || init.getShape().size() != 2 ||
          result.getShape().size() != 2 ||
          !isSignedInteger(lhs.getElementType(), 8) ||
          !isSignedInteger(rhs.getElementType(), 8) ||
          !isSignedInteger(init.getElementType(), 32) ||
          !isSignedInteger(result.getElementType(), 32) ||
          contract.getLhsAxes() != llvm::ArrayRef<int64_t>({1}) ||
          contract.getRhsAxes() != llvm::ArrayRef<int64_t>({1}) ||
          contract.getOrdered()) {
        contract.emitOpError(
            "IME owner requires unordered signed rank-two si8[M,K] x "
            "si8[N,K] -> si32[M,N]");
        return mlir::failure();
      }

      auto lhsM =
          resolveSelectedExtent(source, contract.getLhs(), 0, selectedMeta);
      auto lhsK =
          resolveSelectedExtent(source, contract.getLhs(), 1, selectedMeta);
      auto rhsN =
          resolveSelectedExtent(source, contract.getRhs(), 0, selectedMeta);
      auto rhsK =
          resolveSelectedExtent(source, contract.getRhs(), 1, selectedMeta);
      auto resultM =
          resolveSelectedExtent(source, contract.getResult(), 0, selectedMeta);
      auto resultN =
          resolveSelectedExtent(source, contract.getResult(), 1, selectedMeta);
      if (!lhsM || !lhsK || !rhsN || !rhsK || !resultM || !resultN ||
          *lhsM <= 0 || *lhsK <= 0 || *rhsN <= 0 || *rhsK <= 0 ||
          *resultM <= 0 || *resultN <= 0 || *lhsM != *resultM ||
          *rhsN != *resultN || *lhsK != *rhsK) {
        contract.emitOpError(
            "selected canonical M, N, and K extents must resolve to positive, "
            "role-consistent integers");
        return mlir::failure();
      }
      site.logicalM = *lhsM;
      site.logicalN = *rhsN;
      site.logicalK = *lhsK;

      for (mlir::Value expression :
           {site.lhsLoad.getPointer(), site.lhsLoad.getMask(),
            site.rhsLoad.getPointer(), site.rhsLoad.getMask(),
            site.store.getPointer(), site.store.getMask()}) {
        if (!hasScalarizableExpression(expression)) {
          mlir::Operation *definition = expression.getDefiningOp();
          (definition ? definition : contract.getOperation())
              ->emitOpError(
                  "has no safe structural index/mask projection for IME "
                  "fragment packing or scatter");
          return mlir::failure();
        }
      }

      collectOperationAndProducers(site.store.getOperation(), entry,
                                   site.operations);
      collectValueAndProducers(site.lhsLoad.getResult(), entry, site.lhsValues);
      collectValueAndProducers(site.rhsLoad.getResult(), entry, site.rhsValues);
      site.resultValues.insert(contract.getResult());
      collectValueAndProducers(contract.getInit(), entry, site.resultValues);
      collectValueAndProducers(site.store.getPointer(), entry, site.resultValues);
      collectValueAndProducers(site.store.getMask(), entry, site.resultValues);

      site.lhsGroup = static_cast<int64_t>(siteIndex) * 3;
      site.rhsGroup = site.lhsGroup + 1;
      site.resultGroup = site.lhsGroup + 2;
      site.ownerSymbol = "ime_c" + std::to_string(siteIndex);
      plan.sites.push_back(std::move(site));
    }

    llvm::DenseSet<mlir::Operation *> coveredOperations;
    llvm::DenseMap<mlir::Operation *, unsigned> siteLocalSemanticUses;
    for (const ContractSitePlan &site : plan.sites) {
      for (mlir::Operation *operation : site.operations) {
        coveredOperations.insert(operation);
        if (llvm::isa<kernel::LoadOp, kernel::StoreOp, kernel::ContractOp>(
                operation))
          ++siteLocalSemanticUses[operation];
      }
    }
    for (mlir::Operation &operation : entry) {
      if (llvm::isa<kernel::ReturnOp>(operation))
        continue;
      if (!coveredOperations.contains(&operation)) {
        operation.emitOpError(
            "is outside every selected contraction site; IME emission will not "
            "ignore canonical work");
        return mlir::failure();
      }
      if (llvm::isa<kernel::LoadOp, kernel::StoreOp, kernel::ContractOp>(
              operation) &&
          siteLocalSemanticUses.lookup(&operation) != 1) {
        operation.emitOpError(
            "must belong to exactly one IME contraction site; shared or "
            "overlapping physical work requires an explicit future layout edge");
        return mlir::failure();
      }
    }

    bool invalidLayoutRole = false;
    for (mlir::Operation &operation : entry) {
      for (auto [resultIndex, value] : llvm::enumerate(operation.getResults())) {
        auto block = mlir::dyn_cast<kernel::BlockType>(value.getType());
        if (!block)
          continue;
        if (block.getShape().size() != 2) {
          if (!kernel::isIndexCoordinateProvenance(value))
            invalidLayoutRole = true;
          continue;
        }
        llvm::SmallVector<int64_t, 2> roleGroups;
        for (const ContractSitePlan &site : plan.sites) {
          if (site.lhsValues.contains(value))
            roleGroups.push_back(site.lhsGroup);
          if (site.rhsValues.contains(value))
            roleGroups.push_back(site.rhsGroup);
          if (site.resultValues.contains(value))
            roleGroups.push_back(site.resultGroup);
        }
        if (roleGroups.size() != 1) {
          invalidLayoutRole = true;
          continue;
        }
        plan.valueLayouts.push_back(ValueLayoutPlan{
            plan.nodes.lookup(&operation), static_cast<int64_t>(resultIndex),
            roleGroups.front()});
      }
    }
    if (invalidLayoutRole) {
      source.emitOpError(
          "cannot assign every rank-two canonical block value to exactly one "
          "site-local lhs, rhs, or result IME role group");
      return mlir::failure();
    }
    return plan;
  }

  static mlir::Operation *createRecord(
      mlir::OpBuilder &builder, mlir::Location location, llvm::StringRef name,
      llvm::ArrayRef<mlir::NamedAttribute> attributes) {
    mlir::OperationState state(location, name);
    state.addAttributes(attributes);
    return builder.create(state);
  }

  void materializePlan(mlir::ModuleOp module, KernelPlan &plan) {
    mlir::OpBuilder builder(module.getContext());
    for (auto [operation, node] : plan.nodes)
      operation->setAttr(kNodeAttrName, builder.getI64IntegerAttr(node));

    builder.setInsertionPointToEnd(module.getBody());
    mlir::OperationState state(plan.kernel.getLoc(),
                               execution::PlanOp::getOperationName());
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(plan.symbol));
    state.addAttribute("kernel", mlir::FlatSymbolRefAttr::get(
                                     module.getContext(),
                                     plan.kernel.getSymName()));
    state.addAttribute("target", builder.getStringAttr(plan.target));
    state.addRegion();
    auto selected = llvm::cast<execution::PlanOp>(builder.create(state));
    selected.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&selected.getBody().front());

    for (int64_t axis = 0; axis < plan.kernel.getGridRankAttr().getInt(); ++axis)
      createRecord(builder, plan.kernel.getLoc(),
                   execution::TaskBindingOp::getOperationName(),
                   {builder.getNamedAttr("axis", builder.getI64IntegerAttr(axis)),
                    builder.getNamedAttr("mapping",
                                         builder.getStringAttr("abi"))});
    for (const MetaBindingPlan &binding : plan.metaBindings)
      createRecord(
          builder, plan.kernel.getLoc(),
          execution::MetaBindingOp::getOperationName(),
          {builder.getNamedAttr(
               "argument", builder.getI64IntegerAttr(binding.argument)),
           builder.getNamedAttr(
               "value", builder.getI64IntegerAttr(binding.value))});

    auto serialLayout = layout::getIdentityBlockedLayout(
        module.getContext(), 2, llvm::ArrayRef<int64_t>());
    int64_t scopeNode = plan.nodes.lookup(plan.kernel.getOperation());
    for (ContractSitePlan &site : plan.sites) {
      for (int64_t group :
           {site.lhsGroup, site.rhsGroup, site.resultGroup})
        createRecord(
            builder, site.contract.getLoc(),
            execution::GroupOp::getOperationName(),
            {builder.getNamedAttr("id", builder.getI64IntegerAttr(group)),
             builder.getNamedAttr("scope_node",
                                  builder.getI64IntegerAttr(scopeNode)),
             builder.getNamedAttr("layout", serialLayout),
             builder.getNamedAttr(
                 "owner", mlir::FlatSymbolRefAttr::get(module.getContext(),
                                                        site.ownerSymbol))});
    }
    for (const ValueLayoutPlan &value : plan.valueLayouts)
      createRecord(
          builder, plan.kernel.getLoc(),
          execution::ValueLayoutOp::getOperationName(),
          {builder.getNamedAttr(
               "source_node", builder.getI64IntegerAttr(value.sourceNode)),
           builder.getNamedAttr(
               "source_result", builder.getI64IntegerAttr(value.sourceResult)),
           builder.getNamedAttr("group",
                                builder.getI64IntegerAttr(value.group))});
    for (ContractSitePlan &site : plan.sites)
      createRecord(
          builder, site.contract.getLoc(),
          ime_execution::ContractConfigOp::getOperationName(),
          {builder.getNamedAttr(mlir::SymbolTable::getSymbolAttrName(),
                                builder.getStringAttr(site.ownerSymbol)),
           builder.getNamedAttr(
               "source_node",
               builder.getI64IntegerAttr(
                   plan.nodes.lookup(site.contract.getOperation()))),
           builder.getNamedAttr(
               "groups",
               builder.getDenseI64ArrayAttr(
                   {site.lhsGroup, site.rhsGroup, site.resultGroup})),
           builder.getNamedAttr("mac_m", builder.getI64IntegerAttr(4)),
           builder.getNamedAttr("mac_n", builder.getI64IntegerAttr(4)),
           builder.getNamedAttr("mac_k", builder.getI64IntegerAttr(8)),
           builder.getNamedAttr("vlen_bits",
                                builder.getI64IntegerAttr(vlenBits)),
           builder.getNamedAttr("strategy",
                                builder.getStringAttr("fragment_tiled"))});
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createSelectIMEExecutionPass() {
  return std::make_unique<SelectIMEExecutionPass>();
}

std::unique_ptr<::mlir::Pass>
createSelectIMEExecutionPass(SelectIMEExecutionOptions options) {
  return std::make_unique<SelectIMEExecutionPass>(std::move(options));
}

} // namespace weft::transforms
