#include "Weft/Target/IME/SelectedExecutionIMESource.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

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

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace weft::target::ime {
namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");
constexpr llvm::StringLiteral kHelperName(
    "__weft_ime_vmadot_s8s8_i32_4x4x8_kloop");

llvm::Error makeEmissionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("selected Weft IME source emission failed: ") + message,
      llvm::errc::invalid_argument);
}

bool isASCIIAlpha(char value) {
  return (value >= 'a' && value <= 'z') ||
         (value >= 'A' && value <= 'Z');
}

bool isASCIIDigit(char value) { return value >= '0' && value <= '9'; }

bool isCIdentifier(llvm::StringRef value) {
  if (value.empty() || (!isASCIIAlpha(value.front()) && value.front() != '_'))
    return false;
  return llvm::all_of(value.drop_front(), [](char character) {
    return isASCIIAlpha(character) || isASCIIDigit(character) ||
           character == '_';
  });
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

struct ContractSite {
  ime_execution::ContractConfigOp config;
  kernel::ContractOp contract;
  kernel::LoadOp lhsLoad;
  kernel::LoadOp rhsLoad;
  kernel::StoreOp store;
  int64_t logicalM = 0;
  int64_t logicalN = 0;
  int64_t logicalK = 0;
  int64_t macM = 0;
  int64_t macN = 0;
  int64_t macK = 0;
};

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
                               "' is not a valid portable C identifier");

    mlir::Block &entry = kernel.getBody().front();
    llvm::SmallVector<std::string, 12> parameters;
    auto argNames = kernel.getArgNames();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
      auto nameAttr = mlir::cast<mlir::StringAttr>(argNames[index]);
      llvm::StringRef name = nameAttr.getValue();
      if (!isCIdentifier(name))
        return makeEmissionError(llvm::Twine("kernel argument '") + name +
                                 "' is not a valid portable C identifier");
      if (mlir::isa<kernel::ConstexprType>(argument.getType())) {
        auto binding = metaBindings.find(index);
        if (binding == metaBindings.end())
          return makeEmissionError("missing selected constexpr binding");
        baseExpressions.try_emplace(argument, std::to_string(binding->second));
        continue;
      }
      auto cType = getScalarCType(argument.getType());
      if (!cType)
        return cType.takeError();
      if (mlir::isa<kernel::PtrType>(argument.getType()) &&
          !pointerReachesStore(argument))
        *cType = "const " + *cType;
      parameters.push_back(*cType + " " + name.str());
      baseExpressions.try_emplace(argument, name.str());
    }
    for (int64_t axis = 0; axis < kernel.getGridRankAttr().getInt(); ++axis) {
      auto mapping = taskBindings.find(axis);
      if (mapping == taskBindings.end() || mapping->second != "abi")
        return makeEmissionError(
            "IME source emission requires every task axis to use ABI binding");
      std::string name = "__weft_task_" + std::to_string(axis);
      parameters.push_back("size_t " + name);
      taskNames.try_emplace(axis, std::move(name));
    }

    os << "extern \"C\" void " << kernel.getSymName() << "(";
    llvm::interleaveComma(parameters, os);
    os << ") {\n";
    for (auto [siteIndex, site] : llvm::enumerate(sites))
      if (llvm::Error error = emitSite(site, siteIndex))
        return error;
    os << "}\n\n";
    return llvm::Error::success();
  }

private:
  llvm::Error initialize() {
    kernel = llvm::dyn_cast_or_null<kernel::KernelOp>(
        mlir::SymbolTable::lookupSymbolIn(module, plan.getKernelAttr()));
    if (!kernel)
      return makeEmissionError("plan does not resolve to a canonical kernel");

    bool duplicateNode = false;
    kernel->walk([&](mlir::Operation *operation) {
      auto id = operation->getAttrOfType<mlir::IntegerAttr>(kNodeAttrName);
      if (!id || !nodes.try_emplace(id.getInt(), operation).second)
        duplicateNode = true;
    });
    if (duplicateNode)
      return makeEmissionError(
          "canonical nodes are missing unique selected-execution identities");

    llvm::SmallVector<ime_execution::ContractConfigOp, 4> configs;
    for (mlir::Operation &operation : plan.getBody().front()) {
      if (auto binding = llvm::dyn_cast<execution::TaskBindingOp>(operation)) {
        if (!taskBindings
                 .try_emplace(binding.getAxisAttr().getInt(),
                              binding.getMapping().str())
                 .second)
          return makeEmissionError("duplicate task binding in selected plan");
        continue;
      }
      if (auto binding = llvm::dyn_cast<execution::MetaBindingOp>(operation)) {
        if (!metaBindings
                 .try_emplace(binding.getArgumentAttr().getInt(),
                              binding.getValueAttr().getInt())
                 .second)
          return makeEmissionError("duplicate meta binding in selected plan");
        continue;
      }
      if (auto config =
              llvm::dyn_cast<ime_execution::ContractConfigOp>(operation)) {
        configs.push_back(config);
        continue;
      }
      if (llvm::isa<execution::GroupOp, execution::ValueLayoutOp>(operation))
        continue;
      return makeEmissionError(llvm::Twine("unsupported selected-plan record ") +
                               operation.getName().getStringRef());
    }
    if (configs.empty())
      return makeEmissionError("selected plan has no IME contract_config");
    llvm::sort(configs, [](ime_execution::ContractConfigOp lhs,
                           ime_execution::ContractConfigOp rhs) {
      return lhs.getSourceNodeAttr().getInt() <
             rhs.getSourceNodeAttr().getInt();
    });

    for (ime_execution::ContractConfigOp config : configs) {
      if (config.getStrategy() != "fragment_tiled" ||
          config.getMacMAttr().getInt() != 4 ||
          config.getMacNAttr().getInt() != 4 ||
          config.getMacKAttr().getInt() != 8 ||
          config.getVlenBitsAttr().getInt() != 256)
        return makeEmissionError(
            "selected IME config is not the supported 4x4x8 VLEN=256 "
            "fragment_tiled realization");

      ContractSite site;
      site.config = config;
      site.macM = config.getMacMAttr().getInt();
      site.macN = config.getMacNAttr().getInt();
      site.macK = config.getMacKAttr().getInt();
      site.contract = llvm::dyn_cast_or_null<kernel::ContractOp>(
          nodes.lookup(config.getSourceNodeAttr().getInt()));
      if (!site.contract)
        return makeEmissionError(
            "IME contract_config does not resolve to a canonical contraction");
      site.lhsLoad =
          site.contract.getLhs().getDefiningOp<kernel::LoadOp>();
      site.rhsLoad =
          site.contract.getRhs().getDefiningOp<kernel::LoadOp>();
      if (!site.lhsLoad || !site.rhsLoad || site.lhsLoad == site.rhsLoad)
        return makeEmissionError(
            "selected IME contraction operands are not distinct canonical loads");
      auto lhs =
          mlir::dyn_cast<kernel::BlockType>(site.contract.getLhs().getType());
      auto rhs =
          mlir::dyn_cast<kernel::BlockType>(site.contract.getRhs().getType());
      auto init =
          mlir::dyn_cast<kernel::BlockType>(site.contract.getInit().getType());
      auto result =
          mlir::dyn_cast<kernel::BlockType>(site.contract.getResult().getType());
      auto initSplat =
          site.contract.getInit().getDefiningOp<kernel::SplatOp>();
      if (!lhs || !rhs || !init || !result || lhs.getShape().size() != 2 ||
          rhs.getShape().size() != 2 || init.getShape().size() != 2 ||
          result.getShape().size() != 2 ||
          !isSignedInteger(lhs.getElementType(), 8) ||
          !isSignedInteger(rhs.getElementType(), 8) ||
          !isSignedInteger(init.getElementType(), 32) ||
          !isSignedInteger(result.getElementType(), 32) ||
          site.contract.getLhsAxes() != llvm::ArrayRef<int64_t>({1}) ||
          site.contract.getRhsAxes() != llvm::ArrayRef<int64_t>({1}) ||
          site.contract.getOrdered() || !initSplat ||
          !isIntegerZero(initSplat.getValue()) ||
          !isIntegerZero(site.lhsLoad.getOther()) ||
          !isIntegerZero(site.rhsLoad.getOther()))
        return makeEmissionError(
            "selected vmadot helper requires unordered signed rank-two "
            "si8[M,K] x si8[N,K] -> si32[M,N], zero load fill, and a zero "
            "accumulator");
      for (mlir::Operation *user : site.contract.getResult().getUsers()) {
        auto candidate = llvm::dyn_cast<kernel::StoreOp>(user);
        if (!candidate || candidate.getValue() != site.contract.getResult() ||
            site.store)
          return makeEmissionError(
              "selected IME contraction result must have exactly one direct "
              "canonical store");
        site.store = candidate;
      }
      if (!site.store)
        return makeEmissionError(
            "selected IME contraction result has no canonical store");

      auto lhsM = resolveSelectedExtent(site.contract.getLhs(), 0);
      auto lhsK = resolveSelectedExtent(site.contract.getLhs(), 1);
      auto rhsN = resolveSelectedExtent(site.contract.getRhs(), 0);
      auto rhsK = resolveSelectedExtent(site.contract.getRhs(), 1);
      auto resultM = resolveSelectedExtent(site.contract.getResult(), 0);
      auto resultN = resolveSelectedExtent(site.contract.getResult(), 1);
      if (!lhsM || !lhsK || !rhsN || !rhsK || !resultM || !resultN ||
          *lhsM <= 0 || *lhsK <= 0 || *rhsN <= 0 || *rhsK <= 0 ||
          *resultM <= 0 || *resultN <= 0 || *lhsM != *resultM ||
          *rhsN != *resultN || *lhsK != *rhsK)
        return makeEmissionError(
            "canonical IME M, N, and K extents do not resolve to positive, "
            "role-consistent selected values");
      site.logicalM = *lhsM;
      site.logicalN = *rhsN;
      site.logicalK = *lhsK;
      sites.push_back(site);
    }

    mlir::Block &entry = kernel.getBody().front();
    bool hasNestedOrRegionOperation = false;
    kernel->walk([&](mlir::Operation *operation) {
      if (operation == kernel.getOperation())
        return;
      hasNestedOrRegionOperation |=
          operation->getBlock() != &entry || operation->getNumRegions() != 0;
    });
    if (hasNestedOrRegionOperation)
      return makeEmissionError(
          "fragment-tiled IME source requires a flat canonical kernel body");

    llvm::DenseSet<mlir::Operation *> coveredOperations;
    llvm::DenseMap<mlir::Operation *, unsigned> semanticMembership;
    for (ContractSite &site : sites) {
      llvm::DenseSet<mlir::Operation *> closure;
      collectOperationAndProducers(site.store.getOperation(), entry, closure);
      for (mlir::Operation *operation : closure) {
        coveredOperations.insert(operation);
        if (llvm::isa<kernel::LoadOp, kernel::StoreOp, kernel::ContractOp>(
                operation))
          ++semanticMembership[operation];
      }
    }
    for (mlir::Operation &operation : entry) {
      if (llvm::isa<kernel::ReturnOp>(operation))
        continue;
      if (!coveredOperations.contains(&operation))
        return makeEmissionError(
            "canonical work exists outside all selected IME contract sites");
      if (llvm::isa<kernel::LoadOp, kernel::StoreOp, kernel::ContractOp>(
              operation) &&
          semanticMembership.lookup(&operation) != 1)
        return makeEmissionError(
            "canonical load/store/contract work is shared by multiple IME "
            "sites without an explicit selected layout edge");
    }
    return llvm::Error::success();
  }

  std::optional<int64_t> resolveSelectedExtent(mlir::Value value,
                                               int64_t axis) {
    auto extent = kernel::deriveLogicalExtent(value, axis);
    if (!extent)
      return std::nullopt;
    if (extent->constant)
      return extent->constant;
    auto meta = extent->dynamic.getDefiningOp<kernel::MetaValueOp>();
    if (!meta)
      return std::nullopt;
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(meta.getInput());
    if (!argument || argument.getOwner() != &kernel.getBody().front())
      return std::nullopt;
    auto binding = metaBindings.find(argument.getArgNumber());
    return binding == metaBindings.end()
               ? std::nullopt
               : std::optional<int64_t>(binding->second);
  }

  llvm::Error emitSite(ContractSite &site, size_t siteIndex) {
    std::string prefix = "__weft_s" + std::to_string(siteIndex);
    std::string m0 = prefix + "_m0";
    std::string n0 = prefix + "_n0";
    std::string kf = prefix + "_kf";
    std::string mi = prefix + "_mi";
    std::string ni = prefix + "_ni";
    std::string ki = prefix + "_ki";
    std::string mCoordinate = "(" + m0 + " + " + mi + ")";
    std::string nCoordinate = "(" + n0 + " + " + ni + ")";
    std::string kCoordinate =
        "(" + kf + " * " + std::to_string(site.macK) + " + " + ki + ")";

    llvm::SmallVector<std::string, 2> lhsCoordinates{mCoordinate, kCoordinate};
    llvm::SmallVector<std::string, 2> rhsCoordinates{nCoordinate, kCoordinate};
    llvm::SmallVector<std::string, 2> resultCoordinates{mCoordinate,
                                                        nCoordinate};
    auto lhsPointer = scalarExpression(site.lhsLoad.getPointer(), lhsCoordinates);
    if (!lhsPointer)
      return lhsPointer.takeError();
    auto lhsMask = scalarExpression(site.lhsLoad.getMask(), lhsCoordinates);
    if (!lhsMask)
      return lhsMask.takeError();
    auto rhsPointer = scalarExpression(site.rhsLoad.getPointer(), rhsCoordinates);
    if (!rhsPointer)
      return rhsPointer.takeError();
    auto rhsMask = scalarExpression(site.rhsLoad.getMask(), rhsCoordinates);
    if (!rhsMask)
      return rhsMask.takeError();
    auto resultPointer =
        scalarExpression(site.store.getPointer(), resultCoordinates);
    if (!resultPointer)
      return resultPointer.takeError();
    auto resultMask = scalarExpression(site.store.getMask(), resultCoordinates);
    if (!resultMask)
      return resultMask.takeError();

    int64_t kFragments = 1 + (site.logicalK - 1) / site.macK;
    int64_t lhsFragmentElements = site.macM * site.macK;
    int64_t rhsFragmentElements = site.macN * site.macK;
    int64_t resultFragmentElements = site.macM * site.macN;
    if (kFragments >
            std::numeric_limits<int64_t>::max() / lhsFragmentElements ||
        kFragments >
            std::numeric_limits<int64_t>::max() / rhsFragmentElements)
      return makeEmissionError(
          "selected logical K extent overflows fragment scratch sizing");
    std::string aFragment = prefix + "_a";
    std::string bFragment = prefix + "_b";
    std::string cFragment = prefix + "_c";

    line(1, "for (size_t " + m0 + " = 0; " + m0 + " < " +
                std::to_string(site.logicalM) + "; " + m0 + " += " +
                std::to_string(site.macM) + ") {");
    line(2, "for (size_t " + n0 + " = 0; " + n0 + " < " +
                std::to_string(site.logicalN) + "; " + n0 + " += " +
                std::to_string(site.macN) + ") {");
    line(3, "alignas(32) int8_t " + aFragment + "[" +
                std::to_string(kFragments * lhsFragmentElements) + "] = {};");
    line(3, "alignas(32) int8_t " + bFragment + "[" +
                std::to_string(kFragments * rhsFragmentElements) + "] = {};");
    line(3, "alignas(32) int32_t " + cFragment + "[" +
                std::to_string(resultFragmentElements) + "] = {};");
    line(3, "for (size_t " + kf + " = 0; " + kf + " < " +
                std::to_string(kFragments) + "; ++" + kf + ") {");
    line(4, "for (size_t " + mi + " = 0; " + mi + " < " +
                std::to_string(site.macM) + "; ++" + mi + ") {");
    line(5, "for (size_t " + ki + " = 0; " + ki + " < " +
                std::to_string(site.macK) + "; ++" + ki + ") {");
    line(6, "if (" + mCoordinate + " < " +
                std::to_string(site.logicalM) + " && " + kCoordinate + " < " +
                std::to_string(site.logicalK) + " && (" + *lhsMask + "))");
    line(7, aFragment + "[" + kf + " * " +
                std::to_string(lhsFragmentElements) + " + " + mi + " * " +
                std::to_string(site.macK) + " + " + ki + "] = *" +
                *lhsPointer + ";");
    line(5, "}");
    line(4, "}");
    line(4, "for (size_t " + ni + " = 0; " + ni + " < " +
                std::to_string(site.macN) + "; ++" + ni + ") {");
    line(5, "for (size_t " + ki + " = 0; " + ki + " < " +
                std::to_string(site.macK) + "; ++" + ki + ") {");
    line(6, "if (" + nCoordinate + " < " +
                std::to_string(site.logicalN) + " && " + kCoordinate + " < " +
                std::to_string(site.logicalK) + " && (" + *rhsMask + "))");
    line(7, bFragment + "[" + kf + " * " +
                std::to_string(rhsFragmentElements) + " + " + ni + " * " +
                std::to_string(site.macK) + " + " + ki + "] = *" +
                *rhsPointer + ";");
    line(5, "}");
    line(4, "}");
    line(3, "}");
    line(3, (kHelperName + "(" + aFragment + ", " + bFragment + ", " +
             std::to_string(kFragments) + ", " + cFragment + ");")
                .str());
    line(3, "for (size_t " + mi + " = 0; " + mi + " < " +
                std::to_string(site.macM) + "; ++" + mi + ") {");
    line(4, "for (size_t " + ni + " = 0; " + ni + " < " +
                std::to_string(site.macN) + "; ++" + ni + ") {");
    line(5, "if (" + mCoordinate + " < " +
                std::to_string(site.logicalM) + " && " + nCoordinate + " < " +
                std::to_string(site.logicalN) + " && (" + *resultMask + "))");
    line(6, "*" + *resultPointer + " = " + cFragment + "[" + mi + " * " +
                std::to_string(site.macN) + " + " + ni + "];" );
    line(4, "}");
    line(3, "}");
    line(2, "}");
    line(1, "}");
    return llvm::Error::success();
  }

  llvm::Expected<std::string> getScalarCType(mlir::Type type) const {
    if (auto block = mlir::dyn_cast<kernel::BlockType>(type))
      return getScalarCType(block.getElementType());
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
    if (auto pointer = mlir::dyn_cast<kernel::PtrType>(type)) {
      auto element = getScalarCType(pointer.getElementType());
      if (!element)
        return element.takeError();
      return *element + "*";
    }
    return makeEmissionError("unsupported scalar or pointer type in IME C ABI");
  }

  bool pointerReachesStore(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    return pointerReachesStore(value, visited);
  }

  bool pointerReachesStore(mlir::Value value,
                           llvm::DenseSet<mlir::Value> &visited) const {
    if (!visited.insert(value).second)
      return false;
    for (mlir::Operation *user : value.getUsers()) {
      if (auto currentStore = llvm::dyn_cast<kernel::StoreOp>(user))
        if (currentStore.getPointer() == value)
          return true;
      if (auto pointer = llvm::dyn_cast<kernel::PtrAddOp>(user))
        if (pointer.getBase() == value &&
            pointerReachesStore(pointer.getResult(), visited))
          return true;
    }
    return false;
  }

  llvm::Expected<std::string> constantLiteral(mlir::Attribute attribute) const {
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
      if (integer.getType().isInteger(1))
        return std::string(integer.getInt() ? "true" : "false");
      return std::to_string(integer.getInt());
    }
    return makeEmissionError("IME coordinate expression requires integer constants");
  }

  llvm::Expected<llvm::SmallVector<std::string, 2>> projectCoordinates(
      mlir::Type resultType, mlir::Type operandType,
      llvm::ArrayRef<std::string> coordinates) const {
    auto operandBlock = mlir::dyn_cast<kernel::BlockType>(operandType);
    if (!operandBlock)
      return llvm::SmallVector<std::string, 2>();
    auto resultBlock = mlir::dyn_cast<kernel::BlockType>(resultType);
    if (!resultBlock || coordinates.size() != resultBlock.getShape().size() ||
        operandBlock.getShape().size() > resultBlock.getShape().size())
      return makeEmissionError(
          "cannot project scalar coordinates through canonical broadcasting");
    size_t offset = resultBlock.getShape().size() - operandBlock.getShape().size();
    llvm::SmallVector<std::string, 2> projected;
    projected.reserve(operandBlock.getShape().size());
    for (auto [axis, dimension] : llvm::enumerate(operandBlock.getShape())) {
      int64_t resultDimension = resultBlock.getShape()[offset + axis];
      if (dimension != 1 && !mlir::ShapedType::isDynamic(dimension) &&
          !mlir::ShapedType::isDynamic(resultDimension) &&
          dimension != resultDimension)
        return makeEmissionError(
            "canonical broadcast projection has incompatible static dimensions");
      projected.push_back(dimension == 1 ? "0" : coordinates[offset + axis]);
    }
    return projected;
  }

  llvm::Expected<std::string>
  scalarExpression(mlir::Value value,
                   llvm::ArrayRef<std::string> coordinates) const {
    auto base = baseExpressions.find(value);
    if (base != baseExpressions.end()) {
      if (!coordinates.empty())
        return makeEmissionError("scalar ABI value received block coordinates");
      return base->second;
    }
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return makeEmissionError("coordinate expression has no canonical definition");
    if (auto constant = llvm::dyn_cast<kernel::ConstantOp>(definition)) {
      if (!coordinates.empty() || !isCoordinateIntegerType(value.getType()))
        return makeEmissionError(
            "IME coordinate constants must be scalar index or i1 values");
      return constantLiteral(constant.getValue());
    }
    if (auto meta = llvm::dyn_cast<kernel::MetaValueOp>(definition))
      return scalarExpression(meta.getInput(), {});
    if (auto task = llvm::dyn_cast<kernel::TaskIdOp>(definition)) {
      if (!coordinates.empty())
        return makeEmissionError("task_id received block coordinates");
      auto name = taskNames.find(task.getAxisAttr().getInt());
      if (name == taskNames.end())
        return makeEmissionError("task_id has no selected ABI task binding");
      return name->second;
    }
    if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(definition)) {
      if (coordinates.size() != 1 ||
          !isCoordinateIntegerType(arange.getResult().getType()))
        return makeEmissionError("arange requires one index fragment coordinate");
      auto start = scalarExpression(arange.getStart(), {});
      if (!start)
        return start.takeError();
      return "(" + *start + " + " + coordinates.front() + ")";
    }
    if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(definition)) {
      int64_t axis = expand.getAxisAttr().getInt();
      if (axis < 0 || axis >= static_cast<int64_t>(coordinates.size()))
        return makeEmissionError("expand_dims coordinate axis is invalid");
      llvm::SmallVector<std::string, 2> inputCoordinates(coordinates.begin(),
                                                         coordinates.end());
      inputCoordinates.erase(inputCoordinates.begin() + axis);
      return scalarExpression(expand.getInput(), inputCoordinates);
    }
    if (auto splat = llvm::dyn_cast<kernel::SplatOp>(definition))
      return scalarExpression(splat.getValue(), {});
    if (auto cast = llvm::dyn_cast<kernel::CastOp>(definition)) {
      if (!isCoordinateIntegerType(cast.getInput().getType()) ||
          !isCoordinateIntegerType(cast.getResult().getType()))
        return makeEmissionError(
            "IME coordinate casts are limited to index and i1 values");
      auto inputCoordinates =
          projectCoordinates(cast.getResult().getType(), cast.getInput().getType(),
                             coordinates);
      if (!inputCoordinates)
        return inputCoordinates.takeError();
      auto input = scalarExpression(cast.getInput(), *inputCoordinates);
      if (!input)
        return input.takeError();
      auto type = getScalarCType(cast.getResult().getType());
      if (!type)
        return type.takeError();
      return "static_cast<" + *type + ">(" + *input + ")";
    }
    if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(definition)) {
      if (!isCoordinateIntegerType(binary.getResult().getType()))
        return makeEmissionError(
            "IME coordinate binary operations must produce index or i1");
      auto lhsCoordinates = projectCoordinates(
          binary.getResult().getType(), binary.getLhs().getType(), coordinates);
      if (!lhsCoordinates)
        return lhsCoordinates.takeError();
      auto rhsCoordinates = projectCoordinates(
          binary.getResult().getType(), binary.getRhs().getType(), coordinates);
      if (!rhsCoordinates)
        return rhsCoordinates.takeError();
      auto lhs = scalarExpression(binary.getLhs(), *lhsCoordinates);
      if (!lhs)
        return lhs.takeError();
      auto rhs = scalarExpression(binary.getRhs(), *rhsCoordinates);
      if (!rhs)
        return rhs.takeError();
      llvm::StringRef spelling =
          llvm::StringSwitch<llvm::StringRef>(binary.getKind())
              .Case("add", "+")
              .Case("sub", "-")
              .Case("mul", "*")
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
          "unsupported or potentially undefined scalarized binary operation");
    }
    if (auto compare = llvm::dyn_cast<kernel::CompareOp>(definition)) {
      auto lhsCoordinates = projectCoordinates(
          compare.getResult().getType(), compare.getLhs().getType(), coordinates);
      if (!lhsCoordinates)
        return lhsCoordinates.takeError();
      auto rhsCoordinates = projectCoordinates(
          compare.getResult().getType(), compare.getRhs().getType(), coordinates);
      if (!rhsCoordinates)
        return rhsCoordinates.takeError();
      auto lhs = scalarExpression(compare.getLhs(), *lhsCoordinates);
      if (!lhs)
        return lhs.takeError();
      auto rhs = scalarExpression(compare.getRhs(), *rhsCoordinates);
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
        return makeEmissionError("unsupported scalarized comparison predicate");
      return "(" + *lhs + " " + spelling.str() + " " + *rhs + ")";
    }
    if (auto pointer = llvm::dyn_cast<kernel::PtrAddOp>(definition)) {
      auto baseCoordinates = projectCoordinates(
          pointer.getResult().getType(), pointer.getBase().getType(), coordinates);
      if (!baseCoordinates)
        return baseCoordinates.takeError();
      auto offsetCoordinates = projectCoordinates(
          pointer.getResult().getType(), pointer.getOffset().getType(), coordinates);
      if (!offsetCoordinates)
        return offsetCoordinates.takeError();
      auto baseExpression = scalarExpression(pointer.getBase(), *baseCoordinates);
      if (!baseExpression)
        return baseExpression.takeError();
      auto offsetExpression =
          scalarExpression(pointer.getOffset(), *offsetCoordinates);
      if (!offsetExpression)
        return offsetExpression.takeError();
      return "(" + *baseExpression + " + " + *offsetExpression + ")";
    }
    return makeEmissionError(llvm::Twine("no scalar-coordinate rule for ") +
                             definition->getName().getStringRef());
  }

  void line(unsigned indent, llvm::StringRef text) {
    os.indent(indent * 2) << text << "\n";
  }

  mlir::ModuleOp module;
  execution::PlanOp plan;
  llvm::raw_ostream &os;
  kernel::KernelOp kernel;
  llvm::SmallVector<ContractSite, 4> sites;
  llvm::DenseMap<int64_t, mlir::Operation *> nodes;
  llvm::DenseMap<int64_t, std::string> taskBindings;
  llvm::DenseMap<int64_t, std::string> taskNames;
  llvm::DenseMap<int64_t, int64_t> metaBindings;
  llvm::DenseMap<mlir::Value, std::string> baseExpressions;
};

void emitHelper(llvm::raw_ostream &os) {
  os << "// Static spelling table: canonical signed si8 x si8 -> si32 plus "
        "selected 4x4x8/VLEN=256 maps to the real-K1-sealed vmadot leaf "
        "(encoding 0xe210312b).\n";
  os << "static inline void " << kHelperName
     << "(const int8_t *A, const int8_t *B, size_t kt, int32_t *C) {\n";
  os << "  if (kt == 0) {\n";
  os << "    for (size_t i = 0; i < 16; ++i) C[i] = 0;\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  __asm__ volatile(\n";
  os << "      \"vsetvli   t0, zero, e8, m1, ta, ma   \\n\\t\"\n";
  os << "      \"vmv.v.i   v2, 0                      \\n\\t\"\n";
  os << "      \"vmv.v.i   v3, 0                      \\n\\t\"\n";
  os << "      \"mv        t2, %[kt]                  \\n\\t\"\n";
  os << "      \"mv        t3, %[pa]                  \\n\\t\"\n";
  os << "      \"mv        t4, %[pb]                  \\n\\t\"\n";
  os << "      \"1:                                      \\n\\t\"\n";
  os << "      \"vle8.v    v0, (t3)                   \\n\\t\"\n";
  os << "      \"vle8.v    v1, (t4)                   \\n\\t\"\n";
  os << "      \"vmadot    v2, v0, v1                 \\n\\t\"\n";
  os << "      \"addi      t3, t3, 32                 \\n\\t\"\n";
  os << "      \"addi      t4, t4, 32                 \\n\\t\"\n";
  os << "      \"addi      t2, t2, -1                 \\n\\t\"\n";
  os << "      \"bnez      t2, 1b                     \\n\\t\"\n";
  os << "      \"vsetvli   t0, zero, e32, m1, ta, ma  \\n\\t\"\n";
  os << "      \"vse32.v   v2, (%[pc])                \\n\\t\"\n";
  os << "      \"addi      t5, %[pc], 32              \\n\\t\"\n";
  os << "      \"vse32.v   v3, (t5)                   \\n\\t\"\n";
  os << "      :\n";
  os << "      : [pa] \"r\"(A), [pb] \"r\"(B), [kt] \"r\"(kt), "
        "[pc] \"r\"(C)\n";
  os << "      : \"t0\", \"t2\", \"t3\", \"t4\", \"t5\", \"v0\", "
        "\"v1\", \"v2\", \"v3\", \"memory\");\n";
  os << "}\n\n";
}

} // namespace

llvm::Error emitSelectedExecutionIMESource(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  if (mlir::failed(mlir::verify(module)))
    return makeEmissionError(
        "module verification failed before selected IME source emission");
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
  os << "#include <stdint.h>\n\n";
  emitHelper(os);
  for (execution::PlanOp plan : plans) {
    KernelSourceEmitter emitter(module, plan, os);
    if (llvm::Error error = emitter.emit())
      return error;
  }
  return llvm::Error::success();
}

} // namespace weft::target::ime
