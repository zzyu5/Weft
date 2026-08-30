#include "RISCVIntrinsicC.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace weft;

namespace {

std::string identifier(llvm::StringRef source) {
  std::string result;
  result.reserve(source.size() + 1);
  for (char character : source)
    result.push_back(std::isalnum(static_cast<unsigned char>(character))
                         ? character
                         : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

std::string arrayString(mlir::ArrayAttr values, llvm::StringRef separator) {
  std::string result;
  for (auto [index, attribute] : llvm::enumerate(values)) {
    if (index)
      result += separator;
    result += mlir::cast<mlir::StringAttr>(attribute).getValue().str();
  }
  return result;
}

struct EncodingField {
  std::string name;
  mlir::Type type;
  llvm::SmallVector<int64_t> shape;
  int64_t logicalRank = 0;
  riscv::AccessAttr access;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
};

struct EncodingInfo {
  std::string family;
  std::string kind;
  int64_t storageBits = 0;
  int64_t alignment = 1;
  int64_t logicalElements = 0;
  llvm::SmallVector<EncodingField> fields;
};

struct StorageFragment {
  std::string byte;
  std::string shift;
  unsigned width = 0;
};

std::optional<StorageFragment>
singleStorageFragment(const EncodingField &field, llvm::StringRef logicalIndex,
                      unsigned logicalWidth) {
  if (!field.access)
    return std::nullopt;
  llvm::StringRef kind = field.access.getMapping();
  if (kind == "natural") {
    std::string bit = "(" + std::to_string(field.bitOffset) + " + (" +
                      logicalIndex.str() + ") * " +
                      std::to_string(logicalWidth) + ")";
    return StorageFragment{"(" + bit + " / 8)", "(" + bit + " % 8)",
                           logicalWidth};
  }
  if (kind != "grouped_layered")
    return std::nullopt;
  int64_t group = field.access.getGroupSize();
  int64_t layer = field.access.getLayerSize();
  llvm::StringRef order = field.access.getOrder();
  int64_t layers = group / layer;
  std::string within = "((" + logicalIndex.str() + ") % " +
                       std::to_string(group) + ")";
  std::string layerIndex = "(" + within + " / " + std::to_string(layer) + ")";
  if (order == "hi_first")
    layerIndex = "(" + std::to_string(layers - 1) + " - " + layerIndex + ")";
  std::string byte = "(" + std::to_string(field.bitOffset / 8) + " + ((" +
                     logicalIndex.str() + ") / " + std::to_string(group) + ") * " +
                     std::to_string(layer) + " + " + within + " % " +
                     std::to_string(layer) + ")";
  std::string shift = "(" + layerIndex + " * " +
                      std::to_string(logicalWidth) + ")";
  return StorageFragment{byte, shift, logicalWidth};
}

struct PointInfo {
  int64_t axis = 0;
  std::string base;
  std::string active;
  int64_t physicalExtent = 1;
};

struct MemoryInfo {
  std::string name;
  mlir::Type encoding;
  mlir::Type elementType;
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<std::string> extents;
  llvm::SmallVector<std::string> strides;
  llvm::SmallVector<std::string> origins;
  bool isConst = true;
  int64_t interleaveRows = 0;
  int64_t recordBytes = 0;
  int64_t logicalElements = 0;
};

struct SliceInfo {
  mlir::Value base;
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<std::string> selectors;
  llvm::SmallVector<std::pair<int64_t, mlir::Value>> localOffsets;
};

struct FieldInfo {
  mlir::Value owner;
  std::string name;
  std::optional<mlir::Value> index;
  std::string selector;
  llvm::SmallVector<mlir::Value, 2> relativeIndices;
  riscv::AccessAttr storageAccess;
  riscv::AccessAttr useAccess;
  mlir::Type elementType;
  llvm::SmallVector<int64_t> shape;
  int64_t logicalRank = 0;
  int64_t regularBase = 0;
  int64_t regularStride = 0;
  int64_t regularRepeat = 0;
};

struct Binding {
  enum class Kind {
    None,
    Scalar,
    ScalarTuple,
    Vector,
    Memory,
    Slice,
    Record,
    Field,
    LocalArray,
    Fragment,
    Window,
    PartialSet,
    Point,
    Domain,
  } kind = Kind::None;

  std::string scalar;
  std::string localElements;
  llvm::SmallVector<std::string> parts;
  MemoryInfo memory;
  SliceInfo slice;
  FieldInfo field;
  PointInfo point;
  mlir::Value parentDomain;
  riscv::DomainType domainType;
  std::string domainExtent;
  std::string domainPartition;
  std::string domainMultiplicity;
  std::string recordPointer;
  std::string fragmentFamily;
  std::string windowFamily;
  llvm::SmallVector<std::string> windowLhs;
  llvm::SmallVector<std::string> windowRhs;
  llvm::SmallVector<std::string> windowValidity;
  int64_t interleaveRows = 0;
  int64_t recordAxis = 0;
  int64_t recordElements = 0;
  int64_t recordStrideBytes = 0;
  llvm::SmallVector<std::pair<int64_t, std::string>> recordByteStrides;
  llvm::SmallVector<std::pair<int64_t, std::string>> recordOrigins;
};

class Emitter {
public:
  Emitter(mlir::ModuleOp module, riscv::KernelOp kernel,
          llvm::raw_ostream &output)
      : module(module), kernel(kernel), output(output) {
    collectEncodings();
    for (auto [index, symbol] : llvm::enumerate(kernel.getShapeSymbols()))
      axisSymbolIds[mlir::cast<mlir::StringAttr>(symbol).getValue()] = index + 1;
    for (auto [name, value] :
         llvm::zip(kernel.getParameterNames(), kernel.getParameterValues()))
      autoBindings[mlir::cast<mlir::StringAttr>(name).getValue()] = value;
  }

  Emitter(mlir::ModuleOp module, riscv::ArtifactPackOp artifact,
          llvm::raw_ostream &output)
      : module(module), artifact(artifact), output(output) {}

  mlir::LogicalResult emit() {
    if (!kernel.getBody().hasOneBlock())
      return fail(kernel, "intrinsic-C emission requires one kernel entry block");
    if (mlir::failed(validatePhysicalProgram()))
      return mlir::failure();
    mlir::FailureOr<std::string> arguments = functionArguments();
    if (mlir::failed(arguments))
      return mlir::failure();
    line("void " + identifier(kernel.getSymName()) + "(" + *arguments + ") {");
    ++indent;
    for (mlir::Attribute symbol : kernel.getShapeSymbols()) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>(symbol).getValue();
      if (!autoBindings.contains(name))
        line("(void)" + identifier(name) + ";");
    }
    if (mlir::failed(initializeKernelArguments()))
      return mlir::failure();
    if (mlir::failed(compileBlock(kernel.getBody().front(), {})))
      return mlir::failure();
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

  mlir::LogicalResult emitArtifact() {
    if (!artifact || !artifact.getSizeBody().hasOneBlock() ||
        !artifact.getPackBody().hasOneBlock())
      return fail(artifact, "artifact emission requires two closed physical regions");
    const std::string prefix = identifier(artifact.getSymName());
    bindings.clear();
    line("size_t " + prefix + "_packed_size(size_t M, size_t K) {");
    ++indent;
    if (mlir::failed(compileBlock(artifact.getSizeBody().front(),
                                  {scalar("M"), scalar("K")})))
      return mlir::failure();
    auto sizeYield = mlir::cast<riscv::ArtifactSizeYieldOp>(
        artifact.getSizeBody().front().getTerminator());
    Binding size = bindings.lookup(sizeYield.getValue());
    if (size.kind != Binding::Kind::Scalar)
      return fail(sizeYield, "artifact size region did not produce one scalar index");
    line("return " + size.scalar + ";");
    --indent;
    line("}");
    line("");

    bindings.clear();
    Binding source;
    source.kind = Binding::Kind::Memory;
    source.memory.name = "source";
    source.memory.isConst = true;
    Binding target;
    target.kind = Binding::Kind::Memory;
    target.memory.name = "target";
    target.memory.isConst = false;
    line("void " + prefix +
         "_pack(const uint8_t *restrict source, uint8_t *restrict target, size_t M, size_t K) {");
    ++indent;
    if (mlir::failed(compileBlock(
            artifact.getPackBody().front(),
            {source, target, scalar("M"), scalar("K")})))
      return mlir::failure();
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

private:
  mlir::LogicalResult validatePhysicalProgram();

  mlir::LogicalResult fail(mlir::Operation *operation, llvm::Twine message) {
    operation->emitError(message);
    return mlir::failure();
  }

  template <typename Op>
  mlir::LogicalResult fail(Op operation, llvm::Twine message) {
    operation.emitError(message);
    return mlir::failure();
  }

  void line(llvm::StringRef text = "") {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix) {
    return identifier(prefix) + "_" + std::to_string(nextName++);
  }

  riscv::LayoutAttr layoutOf(mlir::Value value) const {
    return riscv_internal::layoutOf(value.getType());
  }
  llvm::StringRef instructionOf(mlir::Operation *operation) const {
    if (auto leaf = operation
                        ? operation->getAttrOfType<riscv::LeafAttr>("leaf")
                        : riscv::LeafAttr())
      return leaf.getInstruction();
    return {};
  }
  bool isSharedMaterialization(mlir::Value value) const {
    auto materialize = value.getDefiningOp<riscv::RegisterMaterializeOp>();
    return materialize && materialize.getRealization() == "share";
  }

  void collectEncodings() {
    for (riscv::EncodingDeclOp declaration :
         module.getOps<riscv::EncodingDeclOp>()) {
      EncodingInfo info;
      info.family = declaration.getSymName().str();
      info.kind = declaration.getKind().str();
      info.storageBits = declaration.getStorageBits();
      info.alignment = declaration.getAlignment();
      info.logicalElements = declaration.getElements();
      auto names = declaration.getFieldNames();
      auto types = declaration.getFieldTypes();
      auto shapes = declaration.getFieldShapes();
      auto offsets = declaration.getFieldBitOffsets();
      auto storage = declaration.getFieldStorageBits();
      for (size_t index = 0; index < names.size(); ++index) {
        EncodingField field;
        field.name = mlir::cast<mlir::StringAttr>(names[index]).getValue().str();
        field.type = mlir::cast<mlir::TypeAttr>(types[index]).getValue();
        auto shape = mlir::cast<mlir::DenseI64ArrayAttr>(shapes[index]);
        field.shape.assign(shape.asArrayRef().begin(), shape.asArrayRef().end());
        field.bitOffset = offsets[index];
        field.storageBits = storage[index];
        info.fields.push_back(std::move(field));
      }
      encodings[info.family] = std::move(info);
    }
  }

  std::optional<std::string> scalarCType(mlir::Type type) const {
    if (type.isIndex())
      return "size_t";
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
      std::string prefix = integer.isUnsigned() ? "uint" : "int";
      return prefix + std::to_string(std::max(8u, integer.getWidth())) + "_t";
    }
    if (type.isF16())
      return "_Float16";
    if (type.isF32())
      return "float";
    if (type.isF64())
      return "double";
    return std::nullopt;
  }

  std::optional<mlir::Type> denseElementType(mlir::Type type) const {
    auto encoding = mlir::dyn_cast<kernel::EncodingType>(type);
    if (!encoding || encoding.getKind() != "dense")
      return std::nullopt;
    llvm::StringRef family = encoding.getFamily();
    if (family == "f16")
      return mlir::Float16Type::get(type.getContext());
    if (family == "bf16")
      return mlir::BFloat16Type::get(type.getContext());
    if (family == "f32")
      return mlir::Float32Type::get(type.getContext());
    if (family == "f64")
      return mlir::Float64Type::get(type.getContext());
    llvm::StringRef widthSpelling = family;
    mlir::IntegerType::SignednessSemantics signedness;
    if (widthSpelling.consume_front("i"))
      signedness = mlir::IntegerType::Signed;
    else if (widthSpelling.consume_front("u"))
      signedness = mlir::IntegerType::Unsigned;
    else
      widthSpelling = {};
    unsigned width = 0;
    if (!widthSpelling.empty() &&
        !widthSpelling.getAsInteger(10, width) && width > 0)
      return mlir::IntegerType::get(type.getContext(), width, signedness);
    auto info = encodings.find(encoding.getFamily());
    if (info != encodings.end() && info->second.fields.size() == 1)
      return info->second.fields.front().type;
    return std::nullopt;
  }

  std::optional<std::string> argumentCType(mlir::BlockArgument argument) {
    auto view = mlir::cast<riscv::MemDescType>(argument.getType());
    auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(
            kernel.getArgAccess()[argument.getArgNumber()])
            .getValue();
    bool writable = access == "write" || access == "readwrite";
    std::string element = "uint8_t";
    if (encoding.getKind() == "dense") {
      auto denseType = denseElementType(encoding);
      if (!denseType)
        return std::nullopt;
      auto fieldType = scalarCType(*denseType);
      if (!fieldType)
        return std::nullopt;
      element = *fieldType;
    }
    int64_t aliasSet = kernel.getArgAliasSets()[argument.getArgNumber()];
    size_t aliases = 0;
    for (int64_t candidate : kernel.getArgAliasSets())
      aliases += candidate == aliasSet;
    std::string qualifier = aliasSet >= 0 && aliases == 1 ? " *restrict" : " *";
    return (writable ? "" : "const ") + element + qualifier;
  }

  mlir::FailureOr<std::string> functionArguments() {
    llvm::SmallVector<std::string> arguments;
    auto names = kernel.getArgNames();
    for (auto [index, argument] :
         llvm::enumerate(kernel.getBody().front().getArguments())) {
      auto type = argumentCType(argument);
      if (!type) {
        kernel.emitError("intrinsic-C emission does not support a kernel argument type");
        return mlir::failure();
      }
      arguments.push_back(*type + " " +
                          identifier(mlir::cast<mlir::StringAttr>(names[index])
                                         .getValue()));
    }
    for (mlir::Attribute symbol : kernel.getShapeSymbols())
      if (!autoBindings.contains(
              mlir::cast<mlir::StringAttr>(symbol).getValue()))
        arguments.push_back("size_t " +
                            identifier(mlir::cast<mlir::StringAttr>(symbol)
                                           .getValue()));
    return llvm::join(arguments, ", ");
  }

  std::string resolve(llvm::StringRef spelling) const {
    if (spelling.consume_front("auto:")) {
      auto direct = autoBindings.find(spelling);
      if (direct != autoBindings.end())
        return std::to_string(direct->second);
      llvm::SmallVector<llvm::StringRef> choices;
      spelling.split(choices, '|', -1, false);
      for (llvm::StringRef choice : choices) {
        auto found = autoBindings.find(choice);
        if (found != autoBindings.end())
          return std::to_string(found->second);
      }
    }
    int64_t integer = 0;
    if (!spelling.getAsInteger(10, integer))
      return std::to_string(integer);
    return identifier(spelling);
  }

  std::optional<std::string> extentForShape(int64_t extent, int64_t axis) {
    if (extent > 0)
      return std::to_string(extent);
    for (mlir::Attribute symbol : kernel.getShapeSymbols()) {
      llvm::StringRef name = mlir::cast<mlir::StringAttr>(symbol).getValue();
      if (axisSymbolIds.lookup(name) == axis)
        return identifier(name);
    }
    return std::nullopt;
  }

  mlir::LogicalResult initializeKernelArguments() {
    mlir::Block &entry = kernel.getBody().front();
    auto names = kernel.getArgNames();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
      auto view = mlir::cast<riscv::MemDescType>(argument.getType());
      MemoryInfo memory;
      memory.name = identifier(
          mlir::cast<mlir::StringAttr>(names[index]).getValue());
      memory.encoding = view.getEncoding();
      if (auto elementType = denseElementType(view.getEncoding()))
        memory.elementType = *elementType;
      memory.axes.assign(view.getAxisIds().asArrayRef().begin(),
                         view.getAxisIds().asArrayRef().end());
      llvm::StringRef access = mlir::cast<mlir::StringAttr>(
                                   kernel.getArgAccess()[index])
                                   .getValue();
      memory.isConst = access != "write" && access != "readwrite";
      Binding binding;
      binding.kind = Binding::Kind::Memory;
      binding.memory = std::move(memory);
      bindings[argument] = std::move(binding);
    }
    return mlir::success();
  }

  // The remaining operation emitters consume typed physical operation
  // semantics and closed local leaves; they never inspect kernel names or a
  // side assignment.

  mlir::LogicalResult compileBlock(mlir::Block &block,
                                   llvm::ArrayRef<Binding> arguments) {
    conversionCaches.emplace_back();
    llvm::SmallVector<int64_t> physicalPointAxes;
    unsigned physicalPointLaneScopes = 0;
    auto leavePhysicalPoints = [&]() {
      while (physicalPointLaneScopes > 0) {
        laneVLStack.pop_back();
        --physicalPointLaneScopes;
      }
      for (int64_t axis : llvm::reverse(physicalPointAxes))
        axisScopes[axis].pop_back();
    };
    if (!arguments.empty()) {
      if (arguments.size() != block.getNumArguments())
        return fail(block.getParentOp(), "internal intrinsic-C block binding mismatch");
      for (auto [argument, binding] : llvm::zip(block.getArguments(), arguments))
        bindings[argument] = binding;
    }
    for (mlir::Operation &operation : block) {
      if (mlir::isa<riscv::ReturnOp, riscv::YieldOp,
                    riscv::ArtifactSizeYieldOp, riscv::ArtifactReturnOp,
                    mlir::scf::YieldOp,
                    mlir::scf::ConditionOp>(operation))
        continue;
      if (mlir::failed(compileOperation(operation))) {
        leavePhysicalPoints();
        conversionCaches.pop_back();
        return mlir::failure();
      }
      if (auto physicalPoint =
              mlir::dyn_cast<riscv::PhysicalPointOp>(operation)) {
        const Binding &binding = bindings.lookup(physicalPoint.getResult());
        axisScopes[binding.point.axis].push_back(binding.point);
        physicalPointAxes.push_back(binding.point.axis);
        int64_t lanes = 1;
        block.walk([&](mlir::Operation *nested) {
          for (mlir::Value result : nested->getResults())
            if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType());
                value && value.getLayout().getCarrier() == "rvv" &&
                llvm::is_contained(value.getAxisIds().asArrayRef(),
                                   binding.point.axis))
              lanes = std::max<int64_t>(lanes, value.getLayout().getVl());
        });
        if (lanes > 1) {
          laneVLStack.push_back("(" + binding.point.active + " < " +
                                std::to_string(lanes) + " ? " +
                                binding.point.active + " : " +
                                std::to_string(lanes) + ")");
          ++physicalPointLaneScopes;
        }
      }
    }
    leavePhysicalPoints();
    conversionCaches.pop_back();
    return mlir::success();
  }

  mlir::LogicalResult compileOperation(mlir::Operation &operation);
  mlir::LogicalResult compilePhysicalPoint(riscv::PhysicalPointOp point);
  mlir::LogicalResult compileFor(mlir::scf::ForOp operation);
  mlir::LogicalResult compileIf(mlir::scf::IfOp operation);
  mlir::LogicalResult compileWhile(mlir::scf::WhileOp operation);
  mlir::LogicalResult compileMemoryView(riscv::MemoryViewOp operation);
  mlir::LogicalResult compileStorageLoad(riscv::StorageLoadOp operation);
  mlir::LogicalResult compileStorageStore(riscv::StorageStoreOp operation);
  mlir::LogicalResult compileSlice(riscv::SliceOp slice);
  mlir::LogicalResult compileAdmit(riscv::LoadOp admit);
  mlir::LogicalResult compileStagedView(riscv::StagedViewOp staged);
  mlir::LogicalResult
  compileRegisterMaterialize(riscv::RegisterMaterializeOp materialize);
  mlir::LogicalResult
  compileLocalCapacityGuard(riscv::LocalCapacityGuardOp operation);
  mlir::LogicalResult compileLocalAlloc(riscv::LocalAllocOp operation);
  mlir::LogicalResult compileLocalBind(riscv::LocalBindOp operation);
  mlir::LogicalResult compileLocalLoad(riscv::LocalLoadOp operation);
  mlir::LogicalResult compileLocalStore(riscv::LocalStoreOp operation);
  mlir::LogicalResult
  compileRVVLocalMaterialize(riscv::RVVLocalMaterializeOp operation);
  mlir::LogicalResult compileSpill(riscv::SpillOp operation);
  mlir::LogicalResult compileReload(riscv::ReloadOp operation);
  mlir::LogicalResult compileIMEPack(riscv::IMEPackOp operation);
  mlir::LogicalResult compileIMEFragmentMMA(
      riscv::IMEFragmentMMAOp operation);
  mlir::LogicalResult compileIMEUnpack(riscv::IMEUnpackOp operation);
  mlir::LogicalResult compileConvertLayout(riscv::ConvertLayoutOp conversion);
  mlir::LogicalResult compileIota(riscv::IotaOp operation);
  mlir::LogicalResult
  compileRVVAxisBroadcast(riscv::RVVAxisBroadcastOp operation);
  mlir::LogicalResult compileNew(riscv::NewOp operation);
  mlir::LogicalResult compileField(riscv::FieldOp operation);
  mlir::LogicalResult compileExtract(riscv::ExtractOp operation);
  mlir::LogicalResult compileUpdate(riscv::UpdateOp operation);
  mlir::LogicalResult compileUnary(riscv::UnaryOp operation);
  mlir::LogicalResult compileCompare(riscv::CompareOp operation);
  template <typename ConversionOp>
  mlir::LogicalResult compileCastImpl(ConversionOp operation);
  mlir::LogicalResult compileCast(riscv::CastOp operation);
  mlir::LogicalResult compileNarrow(riscv::NarrowOp operation);
  mlir::LogicalResult compileWiden(riscv::WidenOp operation);
  mlir::LogicalResult compileReduce(riscv::ReduceOp operation);
  mlir::LogicalResult compileFold2(riscv::Fold2Op operation);
  mlir::LogicalResult compileLookup(riscv::LookupOp operation);
  mlir::LogicalResult compileBinary(riscv::BinaryOp operation);
  mlir::LogicalResult compileCommit(riscv::StoreOp operation);
  mlir::LogicalResult compileRVVSplat(riscv::RVVSplatOp operation);
  mlir::LogicalResult compileProjectReductionOperand(
      riscv::ProjectReductionOperandOp operation);
  mlir::LogicalResult
  compileRVVContractStep(riscv::RVVContractStepOp operation);
  mlir::LogicalResult compileRVVEncodedContractStep(
      riscv::RVVEncodedContractStepOp operation);
  mlir::LogicalResult
  compileRVVBitplaneMerge(riscv::RVVBitplaneMergeOp operation);
  mlir::LogicalResult
  compilePackedPlaneMerge(riscv::PackedPlaneMergeOp operation);
  mlir::LogicalResult
  compileRVVBitmaskDecode(riscv::RVVBitmaskDecodeOp operation);
  mlir::LogicalResult
  compileGroupedMacReduce(riscv::RVVGroupedMacReduceOp operation);
  mlir::LogicalResult
  compileGroupedMacLoad(riscv::RVVGroupedMacLoadOp operation);
  mlir::LogicalResult
  compileGroupedMacStep(riscv::RVVGroupedMacStepOp operation);
  mlir::LogicalResult
  compileEncodedDotLoad(riscv::RVVEncodedDotLoadOp operation);
  mlir::LogicalResult
  compileEncodedDotStep(riscv::RVVEncodedDotStepOp operation);
  mlir::LogicalResult compileRVVWidenDot(riscv::RVVWidenDotOp operation);
  mlir::LogicalResult
  compileRVVWidenMultiply(riscv::RVVWidenMultiplyOp operation);
  mlir::LogicalResult compileRVVWidenScalarMultiply(
      riscv::RVVWidenScalarMultiplyOp operation);
  mlir::LogicalResult compileRVVRegularRepeatIndex(
      riscv::RVVRegularRepeatIndexOp operation);
  mlir::LogicalResult compileRVVRegularRepeatGather(
      riscv::RVVRegularRepeatGatherOp operation);
  mlir::LogicalResult
  compileRVVStorageWindow(riscv::RVVStorageWindowOp operation);
  mlir::LogicalResult
  compileRVVLayeredStorageLoad(riscv::RVVLayeredStorageLoadOp operation);
  mlir::LogicalResult
  compileRVVLayeredStorageDecode(riscv::RVVLayeredStorageDecodeOp operation);
  mlir::LogicalResult
  compileRVVReplicaStorageLoad(riscv::RVVReplicaStorageLoadOp operation);
  mlir::LogicalResult
  compileRVVWidenAccumulate(riscv::RVVWidenAccumulateOp operation);
  mlir::LogicalResult
  compileRVVFinalizeWidenDot(riscv::RVVFinalizeWidenDotOp operation);
  mlir::LogicalResult compileRVVPartialSet(riscv::RVVPartialSetOp operation);
  mlir::LogicalResult
  compileRVVPartialCapture(riscv::RVVPartialCaptureOp operation);
  mlir::LogicalResult
  compileRVVPartialRepack(riscv::RVVPartialRepackOp operation);
  mlir::LogicalResult
  compileRVVPartialMerge(riscv::RVVPartialMergeOp operation);
  mlir::LogicalResult
  compileRVVPartialReduce(riscv::RVVPartialReduceOp operation);
  mlir::LogicalResult compileRVVPartialScaleCombine(
      riscv::RVVPartialScaleCombineOp operation);
  mlir::LogicalResult compileRVVPartialWidenScale(
      riscv::RVVPartialWidenScaleOp operation);
  mlir::LogicalResult
  compileRVVPartialCombine(riscv::RVVPartialCombineOp operation);
  mlir::LogicalResult
  compileRVVPartialFinalize(riscv::RVVPartialFinalizeOp operation);
  mlir::LogicalResult
  compileRVVAssembleReplicas(riscv::RVVAssembleReplicasOp operation);
  mlir::LogicalResult
  compileRVVWidenReduce(riscv::RVVWidenReduceOp operation);
  mlir::LogicalResult compileRVVPartitionedWidenReduceStore(
      riscv::RVVPartitionedWidenReduceStoreOp operation);
  mlir::LogicalResult
  compileRVVLayeredWindow(riscv::RVVLayeredWindowOp operation);
  mlir::LogicalResult
  compileRVVLayeredStream(riscv::RVVLayeredStreamOp operation);
  mlir::LogicalResult compileRVVProjectedLayeredStream(
      riscv::RVVProjectedLayeredStreamOp operation);
  mlir::LogicalResult
  compileRVVStreamReduce(riscv::RVVStreamReduceOp operation);
  mlir::LogicalResult compileRVVStreamDot(riscv::RVVStreamDotOp operation);
  mlir::LogicalResult
  compileRVVStreamContract(riscv::RVVStreamContractOp operation);

  Binding scalar(llvm::StringRef expression) const {
    Binding binding;
    binding.kind = Binding::Kind::Scalar;
    binding.scalar = expression.str();
    return binding;
  }

  std::string vectorType(mlir::Value value) const;
  std::string vectorSuffix(mlir::Value value) const;
  int64_t vectorPartCount(mlir::Value value) const;
  int64_t streamPartCount(mlir::Value value) const;
  int64_t registerPartCount(mlir::Value value) const;
  int64_t scalarPartCount(mlir::Value value) const;
  int64_t physicalLanes(mlir::Value value) const;
  int64_t laneAxisFor(mlir::Value value) const;
  std::optional<Binding::Kind> selectedBindingKind(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerAxesFor(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerExtentsFor(mlir::Value value) const;
  std::optional<llvm::SmallVector<int64_t, 4>>
  registerCoordinates(mlir::Value value, size_t registerPart) const;
  std::optional<size_t> projectPart(mlir::Value source, mlir::Value result,
                                    size_t resultPart) const;
  std::optional<size_t> mappedPart(mlir::Operation *operation, size_t operand,
                                   size_t resultPart) const;
  std::optional<size_t> projectRegisterPart(mlir::Value source,
                                            mlir::Value result,
                                            size_t resultRegisterPart) const;
  std::optional<std::string>
  extractVectorLane(mlir::Value source, const Binding &binding,
                    size_t sourcePart, int64_t laneOffset,
                    std::string *failureReason = nullptr);
  std::optional<std::string>
  extractLaneForRegisterBroadcast(mlir::Value source, mlir::Value result,
                                  size_t resultPart, const Binding &binding,
                                  std::string *failureReason = nullptr);
  mlir::FailureOr<Binding> projectBinding(mlir::Value source,
                                          mlir::Value result,
                                          const Binding &binding);
  mlir::LogicalResult assignBinding(mlir::Operation *operation,
                                    mlir::Value targetValue, Binding &target,
                                    mlir::Value sourceValue,
                                    const Binding &source,
                                    llvm::StringRef mismatch);
  std::string partOffset(mlir::Value value, size_t part) const;
  std::optional<int64_t> timeCoordinate(mlir::Value value, size_t part,
                                        int64_t axis) const;
  std::string partVL(mlir::Value value, size_t part) const;
  mlir::FailureOr<Binding>
  makeVector(mlir::Value value, llvm::StringRef prefix,
             std::optional<Binding> initial = std::nullopt,
             mlir::Value initialValue = {});
  mlir::FailureOr<Binding> declareMutableBinding(mlir::Value value,
                                                 llvm::StringRef prefix);
  mlir::FailureOr<Binding> loadDenseBlock(mlir::Value result,
                                          const Binding &memoryBlock);
  mlir::FailureOr<Binding> materializeNumeric(mlir::Value value,
                                              Binding binding);
  mlir::FailureOr<Binding> recordForSlice(mlir::Value value);
  std::optional<std::string>
  denseAddress(const SliceInfo &slice,
               llvm::ArrayRef<std::pair<int64_t, std::string>> offsets) const;
  std::optional<EncodingField> fieldFor(const Binding &fieldBinding) const;
  Binding emitInterleavedField(mlir::Value result, const Binding &fieldBinding,
                               llvm::StringRef logicalIndex);
  std::string laneVL() const {
    return laneVLStack.empty() ? "1" : laneVLStack.back();
  }

  mlir::ModuleOp module;
  riscv::KernelOp kernel;
  riscv::ArtifactPackOp artifact;
  llvm::raw_ostream &output;
  unsigned indent = 0;
  unsigned nextName = 0;
  llvm::DenseMap<mlir::Value, Binding> bindings;
  llvm::StringMap<EncodingInfo> encodings;
  llvm::StringMap<int64_t> autoBindings;
  llvm::StringMap<int64_t> axisSymbolIds;
  llvm::DenseMap<int64_t, llvm::SmallVector<PointInfo>> axisScopes;
  llvm::SmallVector<std::string> laneVLStack;
  llvm::SmallVector<llvm::StringMap<std::string>> conversionCaches;
};

namespace {

std::string lmulSpelling(int64_t eighths) {
  switch (eighths) {
  case 1:
    return "mf8";
  case 2:
    return "mf4";
  case 4:
    return "mf2";
  case 8:
    return "m1";
  case 16:
    return "m2";
  case 32:
    return "m4";
  case 64:
    return "m8";
  default:
    return {};
  }
}

std::string vectorSuffixFor(riscv::ValueType value) {
  char category = 'i';
  if (mlir::isa<mlir::FloatType>(value.getElementType()))
    category = 'f';
  else if (auto integer =
               mlir::dyn_cast<mlir::IntegerType>(value.getElementType()))
    category = integer.isUnsigned() ? 'u' : 'i';
  return std::string(1, category) +
         std::to_string(value.getLayout().getSew()) +
         lmulSpelling(value.getLayout().getLmulEighths());
}

std::string vectorTypeFor(riscv::ValueType value) {
  std::string suffix = vectorSuffixFor(value);
  const std::string prefix = suffix.front() == 'f'
                                 ? "vfloat"
                                 : suffix.front() == 'u' ? "vuint" : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t physicalLanesFor(riscv::ValueType value) {
  int64_t lanes = 1;
  for (int64_t factor : value.getLayout().getLaneFactors().asArrayRef())
    lanes *= factor;
  return std::max<int64_t>(1, lanes);
}

int64_t product(mlir::DenseI64ArrayAttr values) {
  return riscv_internal::staticProduct(values.asArrayRef()).value_or(0);
}

riscv::AccessAttr accessOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::AccessAttr>("access")
                   : riscv::AccessAttr();
}

riscv::LeafAttr leafOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::LeafAttr>("leaf")
                   : riscv::LeafAttr();
}

} // namespace

mlir::LogicalResult Emitter::validatePhysicalProgram() {
  bool invalid = false;
  kernel.walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults()) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
        auto layout = value.getLayout();
        if (!layout || layout.getCarrier() == "unassigned" ||
            (layout.getCarrier() == "rvv" &&
             (layout.getSew() <= 0 || layout.getLmulEighths() <= 0 ||
              layout.getVl() <= 0))) {
          operation->emitError("terminal emission received an incomplete physical value layout");
          invalid = true;
        }
      }
    }
    if (auto leaf = leafOf(operation); leaf &&
        (leaf.getEngine() == "unselected" || leaf.getInstruction().empty() ||
         leaf.getSpelling().empty())) {
      operation->emitError("terminal emission received an incomplete target leaf");
      invalid = true;
    }
    if (auto access = operation->getAttrOfType<riscv::AccessAttr>("access");
        access && access.getForm() == "unassigned") {
      operation->emitError("terminal emission received an unplanned memory edge");
      invalid = true;
    }
  });
  return mlir::failure(invalid);
}

std::string Emitter::vectorSuffix(mlir::Value value) const {
  riscv::LayoutAttr layout = layoutOf(value);
  int64_t sew = layout.getSew();
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  char category = 'i';
  if (mlir::isa<mlir::FloatType>(element))
    category = 'f';
  else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    category = integer.isUnsigned() ? 'u' : 'i';
  else if (element.isIndex())
    category = 'u';
  return std::string(1, category) + std::to_string(sew) +
         lmulSpelling(layout.getLmulEighths());
}

std::string Emitter::vectorType(mlir::Value value) const {
  std::string suffix = vectorSuffix(value);
  char category = suffix.front();
  std::string prefix = category == 'f' ? "vfloat" : category == 'u' ? "vuint"
                                                                       : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t Emitter::streamPartCount(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getTimeFactors()));
  return 1;
}

std::optional<Binding::Kind>
Emitter::selectedBindingKind(mlir::Value value) const {
  auto layout = layoutOf(value);
  if (!layout)
    return Binding::Kind::Scalar;
  llvm::StringRef carrier = layout.getCarrier();
  if (carrier == "rvv")
    return Binding::Kind::Vector;
  if (carrier == "scalar" && scalarPartCount(value) > 1)
    return Binding::Kind::ScalarTuple;
  if (carrier == "scalar")
    return Binding::Kind::Scalar;
  if (carrier == "local")
    return Binding::Kind::LocalArray;
  return std::nullopt;
}

int64_t Emitter::registerPartCount(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getReplicaFactors()));
  return 1;
}

int64_t Emitter::scalarPartCount(mlir::Value value) const {
  return registerPartCount(value) * streamPartCount(value);
}

llvm::SmallVector<int64_t, 4>
Emitter::registerAxesFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto layout = layoutOf(value))
    for (auto [axis, factor] : llvm::zip(layout.getAxisIds().asArrayRef(),
                                         layout.getReplicaFactors().asArrayRef()))
      if (factor > 1)
        result.push_back(axis);
  return result;
}

llvm::SmallVector<int64_t, 4>
Emitter::registerExtentsFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto layout = layoutOf(value))
    for (int64_t factor : layout.getReplicaFactors().asArrayRef())
      if (factor > 1)
        result.push_back(factor);
  if (!result.empty())
    return result;
  result.assign(registerAxesFor(value).size(), 1);
  if (result.size() == 1)
    result.front() = registerPartCount(value);
  return result;
}

std::optional<llvm::SmallVector<int64_t, 4>>
Emitter::registerCoordinates(mlir::Value value, size_t registerPart) const {
  llvm::SmallVector<int64_t, 4> axes = registerAxesFor(value);
  llvm::SmallVector<int64_t, 4> extents = registerExtentsFor(value);
  if (axes.size() != extents.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> coordinates(axes.size(), 0);
  int64_t remaining = static_cast<int64_t>(registerPart);
  for (int64_t index = static_cast<int64_t>(axes.size()) - 1; index >= 0;
       --index) {
    if (extents[index] <= 0)
      return std::nullopt;
    coordinates[index] = remaining % extents[index];
    remaining /= extents[index];
  }
  if (remaining != 0)
    return std::nullopt;
  return coordinates;
}

std::optional<size_t> Emitter::projectPart(mlir::Value source,
                                           mlir::Value result,
                                           size_t resultPart) const {
  const int64_t resultStreams = streamPartCount(result);
  const int64_t sourceStreams = streamPartCount(source);
  if (resultStreams <= 0 || sourceStreams <= 0)
    return std::nullopt;
  int64_t resultStream = resultPart % resultStreams;
  int64_t resultRegister = resultPart / resultStreams;
  auto resultLayout = layoutOf(result);
  auto sourceLayout = layoutOf(source);
  if (!resultLayout || !sourceLayout)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> resultTimeAxes;
  llvm::SmallVector<int64_t, 4> resultTimeExtents;
  llvm::SmallVector<int64_t, 4> resultTimeCoordinates;
  for (auto [axis, extent] :
       llvm::zip(resultLayout.getAxisIds().asArrayRef(),
                 resultLayout.getTimeFactors().asArrayRef())) {
    if (extent <= 0)
      return std::nullopt;
    if (extent > 1) {
      resultTimeAxes.push_back(axis);
      resultTimeExtents.push_back(extent);
      resultTimeCoordinates.push_back(0);
    }
  }
  for (int64_t index = static_cast<int64_t>(resultTimeAxes.size()) - 1;
       index >= 0; --index) {
    resultTimeCoordinates[index] = resultStream % resultTimeExtents[index];
    resultStream /= resultTimeExtents[index];
  }
  if (resultStream != 0)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> resultAxes = registerAxesFor(result);
  llvm::SmallVector<int64_t, 4> resultExtents = registerExtentsFor(result);
  if (resultAxes.size() != resultExtents.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> resultCoordinates(resultAxes.size(), 0);
  for (int64_t index = static_cast<int64_t>(resultAxes.size()) - 1; index >= 0;
       --index) {
    if (resultExtents[index] <= 0)
      return std::nullopt;
    resultCoordinates[index] = resultRegister % resultExtents[index];
    resultRegister /= resultExtents[index];
  }
  if (resultRegister != 0)
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> sourceAxes = registerAxesFor(source);
  llvm::SmallVector<int64_t, 4> sourceExtents = registerExtentsFor(source);
  if (sourceAxes.size() != sourceExtents.size())
    return std::nullopt;
  auto resultCoordinateForAxis = [&](int64_t axis) -> std::optional<int64_t> {
    if (auto found = llvm::find(resultAxes, axis); found != resultAxes.end())
      return resultCoordinates[static_cast<size_t>(found - resultAxes.begin())];
    if (auto found = llvm::find(resultTimeAxes, axis);
        found != resultTimeAxes.end())
      return resultTimeCoordinates[
          static_cast<size_t>(found - resultTimeAxes.begin())];
    return int64_t{0};
  };
  int64_t sourceRegister = 0;
  for (auto [axis, extent] : llvm::zip(sourceAxes, sourceExtents)) {
    auto coordinate = resultCoordinateForAxis(axis);
    if (!coordinate || extent <= 0)
      return std::nullopt;
    if (*coordinate >= extent)
      return std::nullopt;
    sourceRegister = sourceRegister * extent + *coordinate;
  }

  int64_t sourceStream = 0;
  for (auto [axis, extent] :
       llvm::zip(sourceLayout.getAxisIds().asArrayRef(),
                 sourceLayout.getTimeFactors().asArrayRef())) {
    if (extent <= 0)
      return std::nullopt;
    if (extent == 1)
      continue;
    auto coordinate = resultCoordinateForAxis(axis);
    if (!coordinate || *coordinate >= extent)
      return std::nullopt;
    sourceStream = sourceStream * extent + *coordinate;
  }
  const int64_t sourceLane = laneAxisFor(source);
  const int64_t resultLane = laneAxisFor(result);
  if (sourceLane > 0 && sourceLane != resultLane)
    return std::nullopt;
  const int64_t projected = sourceRegister * sourceStreams + sourceStream;
  if (projected < 0 || projected >= vectorPartCount(source))
    return std::nullopt;
  return static_cast<size_t>(projected);
}

std::optional<size_t> Emitter::mappedPart(mlir::Operation *operation,
                                          size_t operand,
                                          size_t resultPart) const {
  if (operand >= operation->getNumOperands() || operation->getNumResults() != 1)
    return std::nullopt;
  return projectPart(operation->getOperand(operand), operation->getResult(0),
                     resultPart);
}

std::optional<size_t>
Emitter::projectRegisterPart(mlir::Value source, mlir::Value result,
                             size_t resultRegisterPart) const {
  auto resultCoordinates = registerCoordinates(result, resultRegisterPart);
  llvm::SmallVector<int64_t, 4> resultAxes = registerAxesFor(result);
  llvm::SmallVector<int64_t, 4> sourceAxes = registerAxesFor(source);
  llvm::SmallVector<int64_t, 4> sourceExtents = registerExtentsFor(source);
  if (!resultCoordinates || resultCoordinates->size() != resultAxes.size() ||
      sourceAxes.size() != sourceExtents.size())
    return std::nullopt;
  int64_t sourceRegister = 0;
  for (auto [axis, extent] : llvm::zip(sourceAxes, sourceExtents)) {
    auto found = llvm::find(resultAxes, axis);
    if (found == resultAxes.end() || extent <= 0)
      return std::nullopt;
    size_t position = static_cast<size_t>(found - resultAxes.begin());
    if ((*resultCoordinates)[position] >= extent)
      return std::nullopt;
    sourceRegister = sourceRegister * extent + (*resultCoordinates)[position];
  }
  if (sourceRegister < 0 || sourceRegister >= registerPartCount(source))
    return std::nullopt;
  return static_cast<size_t>(sourceRegister);
}

std::optional<std::string>
Emitter::extractVectorLane(mlir::Value source, const Binding &binding,
                           size_t sourcePart, int64_t laneOffset,
                           std::string *failureReason) {
  auto reject = [&](llvm::StringRef reason) -> std::optional<std::string> {
    if (failureReason)
      *failureReason = reason.str();
    return std::nullopt;
  };
  if (binding.kind != Binding::Kind::Vector ||
      sourcePart >= binding.parts.size())
    return reject("source binding has no selected RVV vector part");
  if (laneOffset < 0 || laneOffset >= physicalLanes(source))
    return reject("selected RVV lane is outside the typed lane extent");
  const std::string cacheKey =
      std::to_string(reinterpret_cast<uintptr_t>(source.getAsOpaquePointer())) +
      ":part" + std::to_string(sourcePart) +
      ":lane" + std::to_string(laneOffset);
  for (const auto &scope : llvm::reverse(conversionCaches))
    if (auto found = scope.find(cacheKey); found != scope.end())
      return found->second;

  const std::string suffix = vectorSuffix(source);
  const std::string shifted =
      "__riscv_vslidedown_vx_" + suffix + "(" + binding.parts[sourcePart] +
      ", " + std::to_string(laneOffset) + ", " + partVL(source, sourcePart) +
      ")";
  mlir::Type element = riscv_internal::logicalElement(source.getType());
  auto sourceLayout = layoutOf(source);
  const unsigned physicalSew =
      sourceLayout ? static_cast<unsigned>(sourceLayout.getSew())
                   : riscv_internal::logicalBitWidth(source.getType());
  std::string expression;
  if (mlir::isa<mlir::FloatType>(element))
    expression = "__riscv_vfmv_f_s_" + suffix + "_f" +
                 std::to_string(physicalSew) + "(" + shifted + ")";
  else {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!integer)
      return reject("source lane element is neither integer nor floating point");
    expression = "__riscv_vmv_x_s_" + suffix + "_" +
                 std::string(integer.isUnsigned() ? "u" : "i") +
                 std::to_string(physicalSew) + "(" + shifted + ")";
  }
  auto type = scalarCType(element);
  if (!type)
    return reject("source lane element has no scalar C spelling");
  std::string materialized = fresh("layout_convert");
  line(*type + " " + materialized + " = " + expression + ";");
  if (!conversionCaches.empty())
    conversionCaches.back()[cacheKey] = materialized;
  return materialized;
}

std::optional<std::string> Emitter::extractLaneForRegisterBroadcast(
    mlir::Value source, mlir::Value result, size_t resultPart,
    const Binding &binding, std::string *failureReason) {
  auto reject = [&](llvm::StringRef reason) -> std::optional<std::string> {
    if (failureReason)
      *failureReason = reason.str();
    return std::nullopt;
  };
  if (binding.kind != Binding::Kind::Vector)
    return reject("source binding is not an RVV vector");
  if (laneAxisFor(source) == 0 || laneAxisFor(result) != 0)
    return reject("source/result lane roles do not describe lane-to-register transfer");
  const int64_t resultStreams = streamPartCount(result);
  if (resultStreams <= 0)
    return reject("result has no positive scalar stream count");
  const size_t resultRegister = resultPart / resultStreams;
  auto coordinates = registerCoordinates(result, resultRegister);
  auto axes = registerAxesFor(result);
  if (!coordinates || coordinates->size() != axes.size())
    return reject("result register coordinates are incomplete");
  int64_t laneOffset = 0;
  int64_t laneAxis = laneAxisFor(source);
  auto laneCoordinate = llvm::find(axes, laneAxis);
  auto resultLayout = layoutOf(result);
  auto resultAxis = resultLayout
                        ? llvm::find(resultLayout.getAxisIds().asArrayRef(),
                                     laneAxis)
                        : llvm::ArrayRef<int64_t>::iterator();
  if (!resultLayout || resultAxis == resultLayout.getAxisIds().asArrayRef().end())
    return reject("result layout does not preserve the source lane axis");
  const size_t axisPosition = static_cast<size_t>(
      resultAxis - resultLayout.getAxisIds().asArrayRef().begin());
  const int64_t timeFactor = resultLayout.getTimeFactors()[axisPosition];
  const int64_t replicaFactor =
      resultLayout.getReplicaFactors()[axisPosition];
  if (timeFactor > 1 && replicaFactor > 1)
    return reject("source lane axis is split across both result time and register coordinates");
  if (timeFactor > 1) {
    auto coordinate = timeCoordinate(result, resultPart, laneAxis);
    if (!coordinate)
      return reject("result time coordinate cannot be projected to the source lane axis");
    laneOffset = *coordinate;
  } else {
    if (laneCoordinate == axes.end())
      return reject("result register axes do not contain the source lane axis");
    laneOffset = (*coordinates)[laneCoordinate - axes.begin()];
  }
  auto sourceRegister =
      projectRegisterPart(source, result, resultRegister);
  if (!sourceRegister)
    return reject("source register part cannot be projected from the result coordinates");
  const int64_t laneStream = laneOffset / physicalLanes(source);
  laneOffset %= physicalLanes(source);
  const int64_t sourcePart =
      *sourceRegister * streamPartCount(source) + laneStream;
  if (sourcePart < 0 ||
      sourcePart >= static_cast<int64_t>(binding.parts.size()))
    return reject("projected source vector part is absent");
  return extractVectorLane(source, binding, sourcePart, laneOffset,
                           failureReason);
}

mlir::FailureOr<Binding>
Emitter::projectBinding(mlir::Value source, mlir::Value result,
                        const Binding &binding) {
  if (binding.kind == Binding::Kind::LocalArray)
    return binding;
  if (binding.kind == Binding::Kind::Scalar)
    return binding;
  if (binding.kind != Binding::Kind::ScalarTuple &&
      binding.kind != Binding::Kind::Vector)
    return binding;
  const int64_t parts = binding.kind == Binding::Kind::Vector
                            ? vectorPartCount(result)
                            : scalarPartCount(result);
  if (parts <= 0)
    return mlir::failure();
  Binding projected;
  projected.kind = binding.kind;
  for (int64_t part = 0; part < parts; ++part) {
    auto sourcePart = projectPart(source, result, part);
    if (!sourcePart || *sourcePart >= binding.parts.size())
      return mlir::failure();
    projected.parts.push_back(binding.parts[*sourcePart]);
  }
  if (projected.kind == Binding::Kind::ScalarTuple && parts == 1) {
    projected.kind = Binding::Kind::Scalar;
    projected.scalar = projected.parts.front();
    projected.parts.clear();
  }
  return projected;
}

mlir::LogicalResult
Emitter::assignBinding(mlir::Operation *operation, mlir::Value targetValue,
                       Binding &target, mlir::Value sourceValue,
                       const Binding &source, llvm::StringRef mismatch) {
  if (targetValue.getType() != sourceValue.getType())
    return fail(operation,
                mismatch.str() +
                    " (physical IR omitted a typed convert_layout operation)");
  if (source.kind == Binding::Kind::LocalArray &&
      (target.kind == Binding::Kind::None ||
       target.kind == Binding::Kind::LocalArray)) {
    if (target.kind == Binding::Kind::None || target.scalar.empty()) {
      target = source;
      return mlir::success();
    }
    if (target.scalar == source.scalar)
      return mlir::success();
    return fail(operation,
                mismatch.str() + " (distinct local arrays require an explicit copy op)");
  }
  if (target.kind == Binding::Kind::Scalar &&
      source.kind == Binding::Kind::Scalar) {
    line(target.scalar + " = " + source.scalar + ";");
    return mlir::success();
  }
  if (target.kind == Binding::Kind::Window &&
      source.kind == Binding::Kind::Window) {
    if (target.windowFamily != source.windowFamily ||
        target.windowLhs.size() != source.windowLhs.size() ||
        target.windowRhs.size() != source.windowRhs.size() ||
        target.windowValidity.size() != source.windowValidity.size())
      return fail(operation,
                  mismatch.str() + " (physical window schemas disagree)");
    for (auto [destination, expression] :
         llvm::zip(target.windowLhs, source.windowLhs))
      line(destination + " = " + expression + ";");
    for (auto [destination, expression] :
         llvm::zip(target.windowRhs, source.windowRhs))
      line(destination + " = " + expression + ";");
    for (auto [destination, expression] :
         llvm::zip(target.windowValidity, source.windowValidity))
      line(destination + " = " + expression + ";");
    return mlir::success();
  }
  if ((target.kind != Binding::Kind::Vector &&
       target.kind != Binding::Kind::ScalarTuple) ||
      target.kind != source.kind)
    return fail(operation,
                mismatch.str() + " (target/source binding kinds disagree: " +
                    std::to_string(static_cast<int>(target.kind)) + "/" +
                    std::to_string(static_cast<int>(source.kind)) + ")");
  const int64_t parts = target.kind == Binding::Kind::Vector
                            ? vectorPartCount(targetValue)
                            : scalarPartCount(targetValue);
  if (parts != static_cast<int64_t>(target.parts.size()))
    return fail(operation,
                mismatch.str() + " (target part count disagrees)");
  for (int64_t part = 0; part < parts; ++part) {
    auto sourcePart = projectPart(sourceValue, targetValue, part);
    if (!sourcePart || *sourcePart >= source.parts.size())
      return fail(operation,
                  mismatch.str() + " (source part projection disagrees)");
    line(target.parts[part] + " = " + source.parts[*sourcePart] + ";");
  }
  return mlir::success();
}

int64_t Emitter::physicalLanes(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getLaneFactors()));
  return 1;
}

int64_t Emitter::laneAxisFor(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    for (auto [axis, factor] : llvm::zip(layout.getAxisIds().asArrayRef(),
                                         layout.getLaneFactors().asArrayRef()))
      if (factor > 1)
        return axis;
  return 0;
}

std::string Emitter::partOffset(mlir::Value value, size_t part) const {
  const int64_t streams = streamPartCount(value);
  int64_t stream = static_cast<int64_t>(part % streams);
  auto layout = layoutOf(value);
  int64_t laneCoordinate = 0;
  if (layout) {
    auto axes = layout.getAxisIds().asArrayRef();
    auto extents = layout.getTimeFactors().asArrayRef();
    for (int64_t index = static_cast<int64_t>(axes.size()) - 1; index >= 0;
         --index) {
      int64_t coordinate = stream % extents[index];
      stream /= extents[index];
      if (axes[index] == laneAxisFor(value))
        laneCoordinate = coordinate;
    }
  }
  return std::to_string(laneCoordinate * physicalLanes(value));
}

std::optional<int64_t> Emitter::timeCoordinate(mlir::Value value, size_t part,
                                               int64_t axis) const {
  auto layout = layoutOf(value);
  const int64_t streams = streamPartCount(value);
  if (!layout || streams <= 0)
    return std::nullopt;
  int64_t stream = static_cast<int64_t>(part % streams);
  int64_t result = 0;
  bool found = false;
  for (int64_t index = static_cast<int64_t>(layout.getAxisIds().size()) - 1;
       index >= 0; --index) {
    const int64_t factor = layout.getTimeFactors()[index];
    if (factor <= 0)
      return std::nullopt;
    const int64_t coordinate = stream % factor;
    stream /= factor;
    if (layout.getAxisIds()[index] == axis) {
      result = coordinate;
      found = true;
    }
  }
  if (stream != 0 || !found)
    return std::nullopt;
  return result;
}

std::string Emitter::partVL(mlir::Value value, size_t part) const {
  const int64_t streams = streamPartCount(value);
  const int64_t laneAxis = laneAxisFor(value);
  if (auto layout = layoutOf(value); layout && layout.getValidity() == "full")
    return std::to_string(physicalLanes(value));
  auto scope = axisScopes.find(laneAxis);
  if (scope == axisScopes.end() || scope->second.empty())
    return std::to_string(physicalLanes(value));
  const std::string active = scope->second.back().active;
  const std::string offset = partOffset(value, part);
  const std::string lanes = std::to_string(physicalLanes(value));
  return "((" + active + " > " + offset + ") ? ((" + active + " - " +
         offset + " < " + lanes + ") ? " + active + " - " + offset + " : " +
         lanes + ") : 0)";
}

int64_t Emitter::vectorPartCount(mlir::Value value) const {
  return registerPartCount(value) * streamPartCount(value);
}

mlir::FailureOr<Binding>
Emitter::makeVector(mlir::Value value, llvm::StringRef prefix,
                    std::optional<Binding> initial,
                    mlir::Value initialValue) {
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t count = vectorPartCount(value);
  const std::string type = vectorType(value);
  const std::string suffix = vectorSuffix(value);
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  auto elementType = scalarCType(element);
  if (!elementType) {
    kernel.emitError("intrinsic-C emission does not support the selected vector element type");
    return mlir::failure();
  }
  for (int64_t index = 0; index < count; ++index) {
    std::string name = fresh(prefix);
    std::string expression;
    if (initial) {
      if (initial->kind == Binding::Kind::Scalar) {
        if (mlir::isa<mlir::FloatType>(element))
          expression = "__riscv_vfmv_v_f_" + suffix + "((" +
                       *elementType + ")(" + initial->scalar + "), " +
                       partVL(value, index) + ")";
        else
          expression = "__riscv_vmv_v_x_" + suffix + "((" +
                       *elementType + ")(" + initial->scalar + "), " +
                       partVL(value, index) + ")";
      } else if (initial->kind == Binding::Kind::Vector ||
                 initial->kind == Binding::Kind::ScalarTuple) {
        if (!initialValue) {
          kernel.emitError(
              "vector initialization from a structured value requires its physical source");
          return mlir::failure();
        }
        auto sourcePart = projectPart(initialValue, value, index);
        if (!sourcePart || *sourcePart >= initial->parts.size()) {
          kernel.emitError(
              "vector initializer cannot be projected to the selected result mapping");
          return mlir::failure();
        }
        if (initial->kind == Binding::Kind::Vector)
          expression = initial->parts[*sourcePart];
        else if (mlir::isa<mlir::FloatType>(element))
          expression = "__riscv_vfmv_v_f_" + suffix + "((" +
                       *elementType + ")(" + initial->parts[*sourcePart] +
                       "), " + partVL(value, index) + ")";
        else
          expression = "__riscv_vmv_v_x_" + suffix + "((" +
                       *elementType + ")(" + initial->parts[*sourcePart] +
                       "), " + partVL(value, index) + ")";
      }
    }
    if (expression.empty())
      expression = "(" + type + "){0}";
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  return result;
}

mlir::FailureOr<Binding>
Emitter::declareMutableBinding(mlir::Value value, llvm::StringRef prefix) {
  if (Binding existing = bindings.lookup(value);
      existing.kind == Binding::Kind::LocalArray)
    return existing;
  std::optional<Binding::Kind> selected = selectedBindingKind(value);
  if (!selected)
    return mlir::failure();

  Binding result;
  result.kind = *selected;
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  if (result.kind == Binding::Kind::Scalar) {
    auto type = scalarCType(element);
    if (!type)
      return mlir::failure();
    result.scalar = fresh(prefix);
    line(*type + " " + result.scalar + ";");
    return result;
  }

  int64_t count = 0;
  std::string type;
  if (result.kind == Binding::Kind::Vector) {
    count = vectorPartCount(value);
    type = vectorType(value);
  } else if (result.kind == Binding::Kind::ScalarTuple) {
    count = registerPartCount(value);
    auto scalarType = scalarCType(element);
    if (!scalarType)
      return mlir::failure();
    type = *scalarType;
  } else {
    return mlir::failure();
  }
  if (count <= 0)
    return mlir::failure();
  for (int64_t part = 0; part < count; ++part) {
    std::string name = fresh(prefix);
    line(type + " " + name + ";");
    result.parts.push_back(std::move(name));
  }
  return result;
}

std::optional<std::string> Emitter::denseAddress(
    const SliceInfo &slice,
    llvm::ArrayRef<std::pair<int64_t, std::string>> offsets) const {
  auto memoryIt = bindings.find(slice.base);
  if (memoryIt == bindings.end() || memoryIt->second.kind != Binding::Kind::Memory)
    return std::nullopt;
  const MemoryInfo &memory = memoryIt->second.memory;
  llvm::StringMap<std::string> offsetByAxis;
  for (auto [axis, expression] : offsets)
    offsetByAxis[std::to_string(axis)] = expression;
  std::string linear = "0";
  size_t domainCursor = 0;
  for (size_t dimension = 0; dimension < memory.axes.size(); ++dimension) {
    std::string coordinate = "0";
    if (dimension < slice.selectors.size() &&
        slice.selectors[dimension] != "all") {
      const std::string &selector = slice.selectors[dimension];
      if (domainCursor >= slice.indices.size())
        return std::nullopt;
      const Binding &index = bindings.lookup(slice.indices[domainCursor++]);
      if (selector == "domain")
        coordinate = index.point.base;
      else if (selector == "group_index")
        coordinate = "(" + index.point.base + " / " +
                     std::to_string(index.point.physicalExtent) + ")";
      else if (selector == "index")
        coordinate = index.scalar;
      else
        return std::nullopt;
    }
    for (auto [axis, value] : slice.localOffsets) {
      if (axis != memory.axes[dimension])
        continue;
      const Binding &offset = bindings.lookup(value);
      std::string expression;
      if (offset.kind == Binding::Kind::Scalar)
        expression = offset.scalar;
      else if (offset.kind == Binding::Kind::Point)
        expression = offset.point.base;
      else
        return std::nullopt;
      coordinate = "(" + coordinate + " + " + expression + ")";
    }
    auto extra = offsetByAxis.find(std::to_string(memory.axes[dimension]));
    if (extra != offsetByAxis.end())
      coordinate = "(" + coordinate + " + " + extra->second + ")";
    if (dimension < memory.origins.size() && memory.origins[dimension] != "0")
      coordinate = "(" + coordinate + " - " + memory.origins[dimension] + ")";
    linear = "(" + linear + " + (" + coordinate + ") * (" +
             memory.strides[dimension] + "))";
  }
  return memory.name + " + " + linear;
}

mlir::FailureOr<Binding>
Emitter::loadDenseBlock(mlir::Value result, const Binding &memoryBlock) {
  Binding loaded;
  loaded.kind = Binding::Kind::Vector;
  auto slice = memoryBlock.slice;
  const Binding &base = bindings.lookup(slice.base);
  if (base.kind != Binding::Kind::Memory) {
    result.getDefiningOp()->emitError("selected dense load has no memory-backed slice");
    return mlir::failure();
  }
  const MemoryInfo &memory = base.memory;
  mlir::Type resultElement = riscv_internal::logicalElement(result.getType());
  if (!memory.elementType || memory.elementType != resultElement) {
    result.getDefiningOp()->emitError(
        "dense load element type does not match its selected local value");
    return mlir::failure();
  }
  auto elementType = scalarCType(resultElement);
  unsigned sew = riscv_internal::logicalBitWidth(result.getType());
  if (!elementType || !sew)
    return mlir::failure();
  const int64_t laneAxis = laneAxisFor(result);
  llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(result);
  size_t laneDimension = llvm::find(memory.axes, laneAxis) - memory.axes.begin();
  if (laneDimension >= memory.strides.size()) {
    result.getDefiningOp()->emitError("selected dense load has no physical lane axis");
    return mlir::failure();
  }
  const std::string suffix = vectorSuffix(result);
  const std::string type = vectorType(result);
  mlir::Operation *transfer = result.getDefiningOp();
  while (transfer) {
    if (auto fresh = mlir::dyn_cast<riscv::NewOp>(transfer);
        fresh && fresh.getInitialized()) {
      transfer = fresh.getInitial().getDefiningOp();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(transfer)) {
      transfer = conversion.getInput().getDefiningOp();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(transfer)) {
      transfer = materialize.getInput().getDefiningOp();
      continue;
    }
    break;
  }
  riscv::AccessAttr edge = accessOf(transfer);
  llvm::StringRef memoryForm = edge ? edge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided") {
    result.getDefiningOp()->emitError(
        "dense load has no pass-selected unit or strided memory form");
    return mlir::failure();
  }
  const int64_t streams = streamPartCount(result);
  const int64_t totalParts = vectorPartCount(result);
  for (int64_t part = 0; part < totalParts; ++part) {
    auto coordinates = registerCoordinates(result, part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size()) {
      result.getDefiningOp()->emitError(
          "selected dense load has no register-coordinate mapping");
      return mlir::failure();
    }
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string active = "1";
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      offsets.push_back({axis, std::to_string(coordinate)});
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
    }
    if (streams > 1)
      offsets.push_back({laneAxis, partOffset(result, part)});
    auto address = denseAddress(slice, offsets);
    if (!address) {
      result.getDefiningOp()->emitError("selected dense load has no address relation");
      return mlir::failure();
    }
    std::string expression;
    if (memoryForm == "unit")
      expression = "__riscv_vle" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + partVL(result, part) + ")";
    else
      expression = "__riscv_vlse" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + memory.strides[laneDimension] +
                   " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                   partVL(result, part) + ")";
    std::string name = fresh("load");
    if (!registerAxes.empty()) {
      std::string zero = mlir::isa<mlir::FloatType>(resultElement)
                             ? "__riscv_vfmv_v_f_" + suffix + "(0.0, " +
                                   partVL(result, part) + ")"
                             : "__riscv_vmv_v_x_" + suffix + "(0, " +
                                   partVL(result, part) + ")";
      line(type + " " + name + " = " + zero + ";");
      line("if (" + active + ") " + name + " = " + expression + ";");
    } else {
      line(type + " " + name + " = " + expression + ";");
    }
    loaded.parts.push_back(std::move(name));
  }
  return loaded;
}

mlir::FailureOr<Binding> Emitter::materializeNumeric(mlir::Value value,
                                                     Binding binding) {
  if (binding.kind == Binding::Kind::Slice) {
    auto layout = layoutOf(value);
    llvm::StringRef carrier = layout ? layout.getCarrier() : llvm::StringRef("scalar");
    if (carrier == "scalar") {
      const int64_t parts = registerPartCount(value);
      if (parts > 1) {
        const Binding &base = bindings.lookup(binding.slice.base);
        mlir::Type element = riscv_internal::logicalElement(value.getType());
        auto type = scalarCType(element);
        auto axes = registerAxesFor(value);
        if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
            base.memory.elementType != element || !type) {
          value.getDefiningOp()->emitError(
              "register-replicated dense load has no matching scalar memory type");
          return mlir::failure();
        }
        Binding tuple;
        tuple.kind = Binding::Kind::ScalarTuple;
        for (int64_t part = 0; part < parts; ++part) {
          auto coordinates = registerCoordinates(value, part);
          if (!coordinates || coordinates->size() != axes.size()) {
            value.getDefiningOp()->emitError(
                "register-replicated dense load has no typed coordinate mapping");
            return mlir::failure();
          }
          llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
          for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
            offsets.push_back({axis, std::to_string(coordinate)});
          auto address = denseAddress(binding.slice, offsets);
          if (!address)
            return mlir::failure();
          tuple.parts.push_back("(*(const " + *type + " *)(" + *address + "))");
        }
        return tuple;
      }
      auto valueType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      auto layout = valueType ? valueType.getLayout() : riscv::LayoutAttr();
      if (!layout || product(layout.getTimeFactors()) != 1 ||
          product(layout.getLaneFactors()) != 1 ||
          product(layout.getReplicaFactors()) != 1 ||
          product(layout.getFragmentFactors()) != 1 ||
          product(layout.getLocalFactors()) != 1) {
        value.getDefiningOp()->emitError(
            "sequential dense load requires one selected physical element");
        return mlir::failure();
      }
      const Binding &base = bindings.lookup(binding.slice.base);
      mlir::Type element = riscv_internal::logicalElement(value.getType());
      if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
          base.memory.elementType != element) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no matching memory element");
        return mlir::failure();
      }
      auto address = denseAddress(binding.slice, {});
      auto type = scalarCType(element);
      if (!address || !type) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no address or C type");
        return mlir::failure();
      }
      return scalar("*(const " + *type + " *)(" + *address + ")");
    }
    if (carrier != "rvv") {
      value.getDefiningOp()->emitError(
          "dense slice has no selected scalar or vector load realization");
      return mlir::failure();
    }
    mlir::FailureOr<Binding> loaded = loadDenseBlock(value, binding);
    if (mlir::failed(loaded))
      return mlir::failure();
    // A selected physical SSA load is evaluated once.  Any reload or
    // rematerialization must already be an explicit physical operation; the
    // terminal emitter must not silently duplicate the producer per consumer.
    bindings[value] = *loaded;
    return loaded;
  }
  if (binding.kind == Binding::Kind::Field) {
    auto selectedKind = selectedBindingKind(value);
    if (selectedKind && *selectedKind == Binding::Kind::ScalarTuple) {
      Binding owner = bindings.lookup(binding.field.owner);
      if (owner.kind == Binding::Kind::Slice) {
        mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
        if (mlir::failed(record))
          return mlir::failure();
        owner = std::move(*record);
      }
      auto field = fieldFor(binding);
      auto valueType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(value);
      const int64_t registerParts = registerPartCount(value);
      llvm::SmallVector<std::string, 4> registerStrides;
      llvm::SmallVector<int64_t, 4> indexAxes;
      llvm::SmallVector<int64_t, 4> directFieldAxes;
      llvm::SmallVector<std::string, 4> tupleLogicalIndices;
      for (int64_t registerAxis : registerAxes) {
        std::string stride;
        for (const auto &[axis, byteStride] : owner.recordByteStrides)
          if (axis == registerAxis) {
            stride = byteStride;
            break;
          }
        registerStrides.push_back(std::move(stride));
      }
      std::string logicalIndex = "0";
      if (binding.field.index) {
        const Binding &index = bindings.lookup(*binding.field.index);
        if (index.kind == Binding::Kind::Scalar) {
          logicalIndex = index.scalar;
        } else if (index.kind == Binding::Kind::Point &&
                   owner.recordElements > 0) {
          const int64_t elements = owner.recordElements;
          logicalIndex =
              binding.field.selector == "group_index"
                  ? "((" + index.point.base + " % " +
                        std::to_string(elements) + ") / " +
                        std::to_string(index.point.physicalExtent) + ")"
                  : "(" + index.point.base + " % " +
                        std::to_string(elements) + ")";
        } else if (index.kind == Binding::Kind::ScalarTuple) {
          auto indexType = mlir::dyn_cast<riscv::ValueType>(
              binding.field.index->getType());
          if (!indexType ||
              static_cast<int64_t>(index.parts.size()) !=
                  registerPartCount(*binding.field.index))
            return value.getDefiningOp()->emitError(
                       "selected scalar tuple field index has no closed register mapping"),
                   mlir::failure();
          indexAxes.assign(indexType.getAxisIds().asArrayRef().begin(),
                           indexType.getAxisIds().asArrayRef().end());
          tupleLogicalIndices.reserve(registerParts);
          for (int64_t part = 0; part < registerParts; ++part) {
            auto projected = projectRegisterPart(*binding.field.index, value, part);
            if (!projected || *projected >= index.parts.size())
              return value.getDefiningOp()->emitError(
                         "selected scalar tuple field index cannot project across result register axes"),
                     mlir::failure();
            tupleLogicalIndices.push_back(index.parts[*projected]);
          }
        } else {
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field has no usable logical index"),
                 mlir::failure();
        }
      }
      for (mlir::Value relativeValue : binding.field.relativeIndices) {
        const Binding &relative = bindings.lookup(relativeValue);
        if (relative.kind != Binding::Kind::Scalar)
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field has a non-scalar relative index"),
                 mlir::failure();
        if (tupleLogicalIndices.empty()) {
          logicalIndex =
              "((" + logicalIndex + ") + (" + relative.scalar + "))";
        } else {
          for (std::string &index : tupleLogicalIndices)
            index = "((" + index + ") + (" + relative.scalar + "))";
        }
      }
      unsigned logicalWidth =
          field ? riscv_internal::logicalBitWidth(field->type) : 0;
      const bool joined = field && field->access.getMapping() == "joined";
      if (!binding.field.index && valueType && binding.field.logicalRank > 0 &&
          binding.field.logicalRank <=
              static_cast<int64_t>(valueType.getAxisIds().size())) {
        const size_t first = valueType.getAxisIds().size() -
                             static_cast<size_t>(binding.field.logicalRank);
        directFieldAxes.append(valueType.getAxisIds().asArrayRef().begin() + first,
                               valueType.getAxisIds().asArrayRef().end());
      }
      if (owner.kind != Binding::Kind::Record || !field || registerAxes.empty() ||
          registerParts <= 0 || registerAxes.size() != registerStrides.size() ||
          field->bitOffset % 8)
        return value.getDefiningOp()->emitError(
                   "selected scalar tuple has no closed encoded-field mapping"),
               mlir::failure();
      for (auto [axis, stride] : llvm::zip(registerAxes, registerStrides))
        if (stride.empty() && !llvm::is_contained(indexAxes, axis) &&
            !llvm::is_contained(directFieldAxes, axis))
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple axis is represented by neither a record stride nor its typed gather index"),
                 mlir::failure();
      Binding tuple;
      tuple.kind = Binding::Kind::ScalarTuple;
      for (int64_t part = 0; part < registerParts; ++part) {
        auto coordinates = registerCoordinates(value, part);
        if (!coordinates || coordinates->size() != registerAxes.size())
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple has no register-coordinate mapping"),
                 mlir::failure();
        std::string record = "(" + owner.recordPointer;
        for (auto [coordinate, stride] :
             llvm::zip(*coordinates, registerStrides))
          if (!stride.empty())
            record += " + " + std::to_string(coordinate) + " * " + stride;
        record += ")";
        std::string partLogicalIndex =
            tupleLogicalIndices.empty()
                ? logicalIndex
                : tupleLogicalIndices[static_cast<size_t>(part)];
        // The selected scalar-replica layout already fixes the coordinates of
        // a whole encoded field value.  Project its trailing logical field
        // axes to the field's linear element index; this is a mechanical
        // spelling of the typed layout, not a terminal layout choice.
        if (tupleLogicalIndices.empty() && !directFieldAxes.empty()) {
          if (!valueType || binding.field.logicalRank < 0 ||
              binding.field.logicalRank >
                  static_cast<int64_t>(binding.field.shape.size()))
            return value.getDefiningOp()->emitError(
                       "selected scalar tuple field has no typed logical rank"),
                   mlir::failure();
          const size_t logicalRank =
              static_cast<size_t>(binding.field.logicalRank);
          const size_t valueOuterRank = valueType.getAxisIds().size() - logicalRank;
          const size_t shapeOuterRank = binding.field.shape.size() - logicalRank;
          int64_t linear = 0;
          for (size_t dimension = 0; dimension < logicalRank; ++dimension) {
            const int64_t axis = valueType.getAxisIds()[valueOuterRank + dimension];
            const int64_t extent = binding.field.shape[shapeOuterRank + dimension];
            auto registerAxis = llvm::find(registerAxes, axis);
            int64_t coordinate = 0;
            if (registerAxis != registerAxes.end())
              coordinate = (*coordinates)[static_cast<size_t>(
                  std::distance(registerAxes.begin(), registerAxis))];
            auto valueAxis = llvm::find(valueType.getAxisIds().asArrayRef(), axis);
            const size_t valuePosition = static_cast<size_t>(std::distance(
                valueType.getAxisIds().asArrayRef().begin(), valueAxis));
            if (extent <= 0 || valueAxis == valueType.getAxisIds().asArrayRef().end() ||
                coordinate < 0 || coordinate >= extent ||
                (registerAxis == registerAxes.end() &&
                 valueType.getLayout().getReplicaFactors()[valuePosition] != 1))
              return value.getDefiningOp()->emitError(
                         "selected scalar tuple field has no closed logical-axis projection"),
                     mlir::failure();
            linear = linear * extent + coordinate;
          }
          partLogicalIndex = "((" + logicalIndex + ") + " +
                             std::to_string(linear) + ")";
        }
        auto fragment =
            singleStorageFragment(*field, partLogicalIndex, logicalWidth);
        if (!fragment && !joined)
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field part has no encoded storage fragment"),
                 mlir::failure();
        const int64_t byteStride = std::max<int64_t>(1, owner.interleaveRows);
        if (joined) {
          int64_t group = field->access.getGroupSize();
          int64_t fields = field->access.getJoinFields();
          int64_t lowBits = field->access.getJoinLowBits();
          int64_t role = field->access.getJoinRole();
          llvm::StringRef order = field->access.getOrder();
          int64_t physicalRole =
              order == "lo_first" ? role : fields - 1 - role;
          std::string headByte =
              "(" + std::to_string(field->bitOffset / 8 + role * group) +
              " + (" + partLogicalIndex + "))";
          std::string tail =
              "((" + partLogicalIndex + ") - " + std::to_string(group) + ")";
          std::string lowByte =
              "(" + std::to_string(field->bitOffset / 8 + fields * group) +
              " + " + tail + ")";
          std::string highByte =
              "(" + std::to_string(field->bitOffset / 8 + role * group) +
              " + " + tail + ")";
          auto load = [&](llvm::StringRef byte) {
            return "*(const uint8_t *)(" + record + " + (" + byte.str() +
                   ") * " + std::to_string(byteStride) + ")";
          };
          std::string head = "(" + load(headByte) + " & " +
                             std::to_string((1u << logicalWidth) - 1) + ")";
          std::string low = "((" + load(lowByte) + " >> " +
                            std::to_string(physicalRole * lowBits) + ") & " +
                            std::to_string((1u << lowBits) - 1) + ")";
          std::string high = "((" + load(highByte) + " >> " +
                             std::to_string(logicalWidth) + ") & " +
                             std::to_string(
                                 (1u << (logicalWidth - lowBits)) - 1) +
                             ")";
          tuple.parts.push_back("((" + partLogicalIndex + ") < " +
                                std::to_string(group) + " ? " + head + " : (" +
                                low + " | (" + high + " << " +
                                std::to_string(lowBits) + ")))" );
          continue;
        }
        const std::string byteAddress =
            record + " + (" + fragment->byte + ") * " +
            std::to_string(byteStride);
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
            integer && integer.getWidth() < 8) {
          const unsigned width = integer.getWidth();
          const unsigned mask = (1u << width) - 1;
          std::string raw = "((*(const uint8_t *)(" + byteAddress + ") >> (" +
                            fragment->shift + ")) & " +
                            std::to_string(mask) + ")";
          if (integer.isUnsigned()) {
            tuple.parts.push_back(std::move(raw));
          } else {
            const unsigned sign = 1u << (width - 1);
            tuple.parts.push_back("((int32_t)((" + raw + " ^ " +
                                  std::to_string(sign) + ") - " +
                                  std::to_string(sign) + "))");
          }
        } else if (field->type.isF32())
          tuple.parts.push_back("weft_load_f32_le(" + byteAddress + ")");
        else if (field->type.isF16())
          tuple.parts.push_back("weft_load_f16_le(" + byteAddress + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 16)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "weft_load_u16_le(" :
                                                 "weft_load_i16_le(") +
              byteAddress + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 8)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "*(const uint8_t *)(" :
                                                 "*(const int8_t *)(") +
              byteAddress + ")");
        else
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field type has no intrinsic-C load"),
                 mlir::failure();
      }
      bindings[value] = tuple;
      return tuple;
    }
    auto expected = selectedBindingKind(value);
    const int64_t expectedParts =
        expected && *expected == Binding::Kind::Vector
            ? vectorPartCount(value)
            : expected && *expected == Binding::Kind::ScalarTuple
                  ? registerPartCount(value)
                  : 1;
    Binding owner = bindings.lookup(binding.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
      if (mlir::failed(record))
        return mlir::failure();
      owner = std::move(*record);
    }
    const bool materializeStreams =
        expected && *expected == Binding::Kind::Vector &&
        owner.kind == Binding::Kind::Record && owner.interleaveRows > 0 &&
        expectedParts > 1 && registerPartCount(value) == 1 &&
        expectedParts == streamPartCount(value);
    if (materializeStreams) {
      Binding streamed;
      streamed.kind = Binding::Kind::Vector;
      for (int64_t stream = 0; stream < expectedParts; ++stream) {
        Binding part =
            emitInterleavedField(value, binding, std::to_string(stream));
        if (part.kind != Binding::Kind::Vector || part.parts.size() != 1) {
          value.getDefiningOp()->emitError(
              "encoded field stream has no single selected RVV value");
          return mlir::failure();
        }
        streamed.parts.push_back(std::move(part.parts.front()));
      }
      bindings[value] = streamed;
      return streamed;
    }
    Binding loaded = emitInterleavedField(value, binding, "0");
    if (loaded.kind == Binding::Kind::None) {
      auto diagnostic = value.getDefiningOp()->emitError(
          "encoded field has no selected numeric load realization");
      diagnostic << " (field=" << binding.field.name;
      if (binding.field.storageAccess)
        diagnostic << ", mapping="
                   << binding.field.storageAccess.getMapping();
      if (binding.field.useAccess)
        diagnostic << ", form=" << binding.field.useAccess.getForm();
      diagnostic << ", value_type=" << value.getType()
                 << ", owner_type=" << binding.field.owner.getType();
      diagnostic << ", selected-carrier="
                 << static_cast<int>(expected.value_or(Binding::Kind::None))
                 << ")";
      return mlir::failure();
    }
    if (expected && loaded.kind != *expected) {
      value.getDefiningOp()->emitError(
          "encoded field load does not implement its selected carrier");
      return mlir::failure();
    }
    if ((loaded.kind == Binding::Kind::Vector ||
         loaded.kind == Binding::Kind::ScalarTuple) &&
        static_cast<int64_t>(loaded.parts.size()) != expectedParts) {
      value.getDefiningOp()->emitError(
          "encoded field load does not materialize every selected physical part");
      return mlir::failure();
    }
    bindings[value] = loaded;
    return loaded;
  }
  return binding;
}

mlir::LogicalResult Emitter::compileOperation(mlir::Operation &operation) {
  auto bindConstant = [&](mlir::Attribute value,
                          mlir::Value result) -> mlir::LogicalResult {
    std::string expression;
    mlir::Type element = riscv_internal::logicalElement(result.getType());
    if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(value)) {
      const llvm::APFloat &number = floating.getValue();
      if (number.isInfinity())
        expression = number.isNegative() ? "(-INFINITY)" : "INFINITY";
      else if (number.isNaN())
        expression = "NAN";
      else {
        llvm::SmallString<32> text;
        number.toString(text);
        expression = text.str().str();
        if (expression.find_first_of(".eEpP") == std::string::npos)
          expression += ".0";
        if (element.isF32())
          expression += "f";
        else if (element.isF16())
          expression = "((_Float16)(" + expression + "f))";
      }
    } else {
      llvm::raw_string_ostream stream(expression);
      value.print(stream);
      stream.flush();
      size_t typeMarker = expression.find(" : ");
      if (typeMarker != std::string::npos)
        expression.resize(typeMarker);
    }
    bindings[result] = scalar(expression);
    return mlir::success();
  };
  if (auto symbol = mlir::dyn_cast<riscv::SymbolOp>(operation)) {
    if (auto binding = autoBindings.find(symbol.getName());
        binding != autoBindings.end()) {
      bindings[symbol.getResult()] = scalar(std::to_string(binding->second));
    } else {
      bindings[symbol.getResult()] = scalar(identifier(symbol.getName()));
    }
    return mlir::success();
  }
  if (auto root = mlir::dyn_cast<riscv::RootDomainOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.domainType = root.getResult().getType();
    bindings[root.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto root = mlir::dyn_cast<riscv::RootPointOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Point;
    binding.point.axis = 0;
    binding.point.base = "0";
    binding.point.active = "1";
    binding.point.physicalExtent = 1;
    bindings[root.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto domain = mlir::dyn_cast<riscv::DomainOp>(operation)) {
    Binding extent = bindings.lookup(domain.getExtent());
    Binding partition = bindings.lookup(domain.getPartition());
    Binding multiplicity = bindings.lookup(domain.getMultiplicity());
    if (extent.kind != Binding::Kind::Scalar ||
        partition.kind != Binding::Kind::Scalar ||
        multiplicity.kind != Binding::Kind::Scalar)
      return fail(domain,
                  "domain extent, partition, and multiplicity require scalar index values");
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.parentDomain = domain.getParent();
    binding.domainType = domain.getResult().getType();
    binding.domainExtent = std::move(extent.scalar);
    binding.domainPartition = std::move(partition.scalar);
    binding.domainMultiplicity = std::move(multiplicity.scalar);
    bindings[domain.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto point = mlir::dyn_cast<riscv::PhysicalPointOp>(operation))
    return compilePhysicalPoint(point);
  if (auto constant = mlir::dyn_cast<riscv::ConstantOp>(operation))
    return bindConstant(constant.getValue(), constant.getResult());
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation))
    return bindConstant(constant.getValue(), constant.getResult());
  if (auto ceil = mlir::dyn_cast<mlir::arith::CeilDivUIOp>(operation)) {
    Binding lhs = bindings.lookup(ceil.getLhs());
    Binding rhs = bindings.lookup(ceil.getRhs());
    if (lhs.kind != Binding::Kind::Scalar || rhs.kind != Binding::Kind::Scalar)
      return fail(ceil, "ceildiv operands require scalar index values");
    bindings[ceil.getResult()] = scalar(
        "((" + lhs.scalar + " + " + rhs.scalar + " - 1) / " + rhs.scalar + ")");
    return mlir::success();
  }
  auto bindIndexBinary = [&](mlir::Value lhsValue, mlir::Value rhsValue,
                             mlir::Value result,
                             llvm::StringRef symbol) -> mlir::LogicalResult {
    Binding lhs = bindings.lookup(lhsValue);
    Binding rhs = bindings.lookup(rhsValue);
    if (lhs.kind != Binding::Kind::Scalar || rhs.kind != Binding::Kind::Scalar)
      return fail(&operation, "physical index arithmetic requires scalar values");
    bindings[result] =
        scalar("(" + lhs.scalar + " " + symbol.str() + " " + rhs.scalar + ")");
    return mlir::success();
  };
  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
    return bindIndexBinary(add.getLhs(), add.getRhs(), add.getResult(), "+");
  if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(operation))
    return bindIndexBinary(sub.getLhs(), sub.getRhs(), sub.getResult(), "-");
  if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
    return bindIndexBinary(mul.getLhs(), mul.getRhs(), mul.getResult(), "*");
  if (auto divide = mlir::dyn_cast<mlir::arith::DivUIOp>(operation))
    return bindIndexBinary(divide.getLhs(), divide.getRhs(), divide.getResult(), "/");
  if (auto remainder = mlir::dyn_cast<mlir::arith::RemUIOp>(operation))
    return bindIndexBinary(remainder.getLhs(), remainder.getRhs(),
                           remainder.getResult(), "%");
  if (auto remainder = mlir::dyn_cast<mlir::arith::RemSIOp>(operation))
    return bindIndexBinary(remainder.getLhs(), remainder.getRhs(),
                           remainder.getResult(), "%");
  if (auto logicalOr = mlir::dyn_cast<mlir::arith::OrIOp>(operation))
    return bindIndexBinary(logicalOr.getLhs(), logicalOr.getRhs(),
                           logicalOr.getResult(), "||");
  if (auto minimum = mlir::dyn_cast<mlir::arith::MinUIOp>(operation)) {
    Binding lhs = bindings.lookup(minimum.getLhs());
    Binding rhs = bindings.lookup(minimum.getRhs());
    if (lhs.kind != Binding::Kind::Scalar || rhs.kind != Binding::Kind::Scalar)
      return fail(minimum, "physical index minimum requires scalar values");
    bindings[minimum.getResult()] =
        scalar("((" + lhs.scalar + " < " + rhs.scalar + ") ? " + lhs.scalar +
               " : " + rhs.scalar + ")");
    return mlir::success();
  }
  if (auto compare = mlir::dyn_cast<mlir::arith::CmpIOp>(operation)) {
    Binding lhs = bindings.lookup(compare.getLhs());
    Binding rhs = bindings.lookup(compare.getRhs());
    if (lhs.kind != Binding::Kind::Scalar || rhs.kind != Binding::Kind::Scalar)
      return fail(compare, "physical index comparison requires scalar values");
    llvm::StringRef symbol;
    switch (compare.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      symbol = "==";
      break;
    case mlir::arith::CmpIPredicate::ne:
      symbol = "!=";
      break;
    case mlir::arith::CmpIPredicate::ult:
    case mlir::arith::CmpIPredicate::slt:
      symbol = "<";
      break;
    case mlir::arith::CmpIPredicate::ule:
    case mlir::arith::CmpIPredicate::sle:
      symbol = "<=";
      break;
    case mlir::arith::CmpIPredicate::ugt:
    case mlir::arith::CmpIPredicate::sgt:
      symbol = ">";
      break;
    case mlir::arith::CmpIPredicate::uge:
    case mlir::arith::CmpIPredicate::sge:
      symbol = ">=";
      break;
    }
    bindings[compare.getResult()] =
        scalar("(" + lhs.scalar + " " + symbol.str() + " " + rhs.scalar + ")");
    return mlir::success();
  }
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation))
    return compileFor(loop);
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation))
    return compileIf(branch);
  if (auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(operation))
    return compileWhile(loop);
  if (auto view = mlir::dyn_cast<riscv::MemoryViewOp>(operation))
    return compileMemoryView(view);
  if (auto load = mlir::dyn_cast<riscv::StorageLoadOp>(operation))
    return compileStorageLoad(load);
  if (auto store = mlir::dyn_cast<riscv::StorageStoreOp>(operation))
    return compileStorageStore(store);
  if (auto slice = mlir::dyn_cast<riscv::SliceOp>(operation))
    return compileSlice(slice);
  if (auto admit = mlir::dyn_cast<riscv::LoadOp>(operation))
    return compileAdmit(admit);
  if (auto staged = mlir::dyn_cast<riscv::StagedViewOp>(operation))
    return compileStagedView(staged);
  if (auto materialize =
          mlir::dyn_cast<riscv::RegisterMaterializeOp>(operation))
    return compileRegisterMaterialize(materialize);
  if (auto allocation = mlir::dyn_cast<riscv::LocalAllocOp>(operation))
    return compileLocalAlloc(allocation);
  if (auto guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(operation))
    return compileLocalCapacityGuard(guard);
  if (auto binding = mlir::dyn_cast<riscv::LocalBindOp>(operation))
    return compileLocalBind(binding);
  if (auto load = mlir::dyn_cast<riscv::LocalLoadOp>(operation))
    return compileLocalLoad(load);
  if (auto store = mlir::dyn_cast<riscv::LocalStoreOp>(operation))
    return compileLocalStore(store);
  if (auto materialize =
          mlir::dyn_cast<riscv::RVVLocalMaterializeOp>(operation))
    return compileRVVLocalMaterialize(materialize);
  if (auto spill = mlir::dyn_cast<riscv::SpillOp>(operation))
    return compileSpill(spill);
  if (auto reload = mlir::dyn_cast<riscv::ReloadOp>(operation))
    return compileReload(reload);
  if (auto pack = mlir::dyn_cast<riscv::IMEPackOp>(operation))
    return compileIMEPack(pack);
  if (auto mma = mlir::dyn_cast<riscv::IMEFragmentMMAOp>(operation))
    return compileIMEFragmentMMA(mma);
  if (auto unpack = mlir::dyn_cast<riscv::IMEUnpackOp>(operation))
    return compileIMEUnpack(unpack);
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation))
    return compileConvertLayout(conversion);
  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation))
    return compileIota(iota);
  if (auto broadcast = mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(operation))
    return compileRVVAxisBroadcast(broadcast);
  if (auto freshValue = mlir::dyn_cast<riscv::NewOp>(operation))
    return compileNew(freshValue);
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation))
    return compileField(field);
  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation))
    return compileExtract(extract);
  if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation))
    return compileUpdate(update);
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation))
    return compileUnary(unary);
  if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation))
    return compileCompare(compare);
  if (auto cast = mlir::dyn_cast<riscv::CastOp>(operation))
    return compileCast(cast);
  if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation))
    return compileNarrow(narrow);
  if (auto widen = mlir::dyn_cast<riscv::WidenOp>(operation))
    return compileWiden(widen);
  if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation))
    return compileReduce(reduce);
  if (auto fold = mlir::dyn_cast<riscv::Fold2Op>(operation))
    return compileFold2(fold);
  if (auto merge = mlir::dyn_cast<riscv::RVVBitplaneMergeOp>(operation))
    return compileRVVBitplaneMerge(merge);
  if (auto merge = mlir::dyn_cast<riscv::PackedPlaneMergeOp>(operation))
    return compilePackedPlaneMerge(merge);
  if (auto decode = mlir::dyn_cast<riscv::RVVBitmaskDecodeOp>(operation))
    return compileRVVBitmaskDecode(decode);
  if (auto reduce = mlir::dyn_cast<riscv::RVVGroupedMacReduceOp>(operation))
    return compileGroupedMacReduce(reduce);
  if (auto load = mlir::dyn_cast<riscv::RVVGroupedMacLoadOp>(operation))
    return compileGroupedMacLoad(load);
  if (auto step = mlir::dyn_cast<riscv::RVVGroupedMacStepOp>(operation))
    return compileGroupedMacStep(step);
  if (auto load = mlir::dyn_cast<riscv::RVVEncodedDotLoadOp>(operation))
    return compileEncodedDotLoad(load);
  if (auto step = mlir::dyn_cast<riscv::RVVEncodedDotStepOp>(operation))
    return compileEncodedDotStep(step);
  if (auto dot = mlir::dyn_cast<riscv::RVVWidenDotOp>(operation))
    return compileRVVWidenDot(dot);
  if (auto multiply = mlir::dyn_cast<riscv::RVVWidenMultiplyOp>(operation))
    return compileRVVWidenMultiply(multiply);
  if (auto multiply =
          mlir::dyn_cast<riscv::RVVWidenScalarMultiplyOp>(operation))
    return compileRVVWidenScalarMultiply(multiply);
  if (auto index =
          mlir::dyn_cast<riscv::RVVRegularRepeatIndexOp>(operation))
    return compileRVVRegularRepeatIndex(index);
  if (auto gather =
          mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(operation))
    return compileRVVRegularRepeatGather(gather);
  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(operation))
    return compileRVVStorageWindow(window);
  if (auto load = mlir::dyn_cast<riscv::RVVLayeredStorageLoadOp>(operation))
    return compileRVVLayeredStorageLoad(load);
  if (auto decode =
          mlir::dyn_cast<riscv::RVVLayeredStorageDecodeOp>(operation))
    return compileRVVLayeredStorageDecode(decode);
  if (auto load = mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(operation))
    return compileRVVReplicaStorageLoad(load);
  if (auto accumulate =
          mlir::dyn_cast<riscv::RVVWidenAccumulateOp>(operation))
    return compileRVVWidenAccumulate(accumulate);
  if (auto finalize =
          mlir::dyn_cast<riscv::RVVFinalizeWidenDotOp>(operation))
    return compileRVVFinalizeWidenDot(finalize);
  if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
    return compileRVVPartialSet(partial);
  if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
    return compileRVVPartialCapture(capture);
  if (auto repack = mlir::dyn_cast<riscv::RVVPartialRepackOp>(operation))
    return compileRVVPartialRepack(repack);
  if (auto merge = mlir::dyn_cast<riscv::RVVPartialMergeOp>(operation))
    return compileRVVPartialMerge(merge);
  if (auto reduce = mlir::dyn_cast<riscv::RVVPartialReduceOp>(operation))
    return compileRVVPartialReduce(reduce);
  if (auto combine =
          mlir::dyn_cast<riscv::RVVPartialScaleCombineOp>(operation))
    return compileRVVPartialScaleCombine(combine);
  if (auto scale =
          mlir::dyn_cast<riscv::RVVPartialWidenScaleOp>(operation))
    return compileRVVPartialWidenScale(scale);
  if (auto combine = mlir::dyn_cast<riscv::RVVPartialCombineOp>(operation))
    return compileRVVPartialCombine(combine);
  if (auto finalize = mlir::dyn_cast<riscv::RVVPartialFinalizeOp>(operation))
    return compileRVVPartialFinalize(finalize);
  if (auto assemble =
          mlir::dyn_cast<riscv::RVVAssembleReplicasOp>(operation))
    return compileRVVAssembleReplicas(assemble);
  if (auto reduce = mlir::dyn_cast<riscv::RVVWidenReduceOp>(operation))
    return compileRVVWidenReduce(reduce);
  if (auto reduce =
          mlir::dyn_cast<riscv::RVVPartitionedWidenReduceStoreOp>(operation))
    return compileRVVPartitionedWidenReduceStore(reduce);
  if (auto window = mlir::dyn_cast<riscv::RVVLayeredWindowOp>(operation))
    return compileRVVLayeredWindow(window);
  if (auto stream = mlir::dyn_cast<riscv::RVVLayeredStreamOp>(operation))
    return compileRVVLayeredStream(stream);
  if (auto stream =
          mlir::dyn_cast<riscv::RVVProjectedLayeredStreamOp>(operation))
    return compileRVVProjectedLayeredStream(stream);
  if (auto reduce = mlir::dyn_cast<riscv::RVVStreamReduceOp>(operation))
    return compileRVVStreamReduce(reduce);
  if (auto dot = mlir::dyn_cast<riscv::RVVStreamDotOp>(operation))
    return compileRVVStreamDot(dot);
  if (auto contract = mlir::dyn_cast<riscv::RVVStreamContractOp>(operation))
    return compileRVVStreamContract(contract);
  if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(operation))
    return compileLookup(lookup);
  if (auto splat = mlir::dyn_cast<riscv::RVVSplatOp>(operation))
    return compileRVVSplat(splat);
  if (auto projected =
          mlir::dyn_cast<riscv::ProjectReductionOperandOp>(operation))
    return compileProjectReductionOperand(projected);
  if (auto step = mlir::dyn_cast<riscv::RVVContractStepOp>(operation))
    return compileRVVContractStep(step);
  if (auto step =
          mlir::dyn_cast<riscv::RVVEncodedContractStepOp>(operation))
    return compileRVVEncodedContractStep(step);
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation))
    return compileBinary(binary);
  if (auto commit = mlir::dyn_cast<riscv::StoreOp>(operation))
    return compileCommit(commit);
  return fail(&operation, llvm::Twine("intrinsic-C emission has no rule for physical operation '") +
                              operation.getName().getStringRef() + "'");
}

mlir::LogicalResult
Emitter::compilePhysicalPoint(riscv::PhysicalPointOp point) {
  Binding base = bindings.lookup(point.getBase());
  Binding active = bindings.lookup(point.getActive());
  Binding partition = bindings.lookup(point.getPartition());
  if (base.kind != Binding::Kind::Scalar ||
      active.kind != Binding::Kind::Scalar ||
      partition.kind != Binding::Kind::Scalar)
    return fail(point, "physical point coordinates require scalar index values");
  int64_t extent = 0;
  if (llvm::StringRef(partition.scalar).getAsInteger(10, extent) || extent <= 0)
    return fail(point,
                "physical point partition must be a positive compile-time integer");
  Binding binding;
  binding.kind = Binding::Kind::Point;
  binding.point.axis = point.getResult().getType().getDomain().getAxisId();
  binding.point.base = std::move(base.scalar);
  binding.point.active = std::move(active.scalar);
  binding.point.physicalExtent = extent;
  bindings[point.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFor(mlir::scf::ForOp operation) {
  Binding lower = bindings.lookup(operation.getLowerBound());
  Binding upper = bindings.lookup(operation.getUpperBound());
  Binding step = bindings.lookup(operation.getStep());
  if (lower.kind != Binding::Kind::Scalar ||
      upper.kind != Binding::Kind::Scalar || step.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered for bounds require selected scalar values");
  llvm::SmallVector<Binding, 0> carried;
  for (mlir::Value value : operation.getInitArgs()) {
    Binding source = bindings.lookup(value);
    if (source.kind == Binding::Kind::Scalar) {
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return fail(operation, "ordered for carry has no scalar C type");
      Binding copy = source;
      copy.scalar = fresh("for_carry");
      line(*type + " " + copy.scalar + " = " + source.scalar + ";");
      carried.push_back(std::move(copy));
    } else if (source.kind == Binding::Kind::Vector) {
      mlir::FailureOr<Binding> copy =
          makeVector(value, "for_carry", source, value);
      if (mlir::failed(copy))
        return mlir::failure();
      carried.push_back(std::move(*copy));
    } else if (source.kind == Binding::Kind::ScalarTuple) {
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return fail(operation,
                    "ordered for tuple carry has no scalar C type");
      Binding copy;
      copy.kind = Binding::Kind::ScalarTuple;
      for (llvm::StringRef expression : source.parts) {
        std::string name = fresh("for_carry");
        line(*type + " " + name + " = " + expression.str() + ";");
        copy.parts.push_back(std::move(name));
      }
      carried.push_back(std::move(copy));
    } else if (source.kind == Binding::Kind::LocalArray) {
      carried.push_back(std::move(source));
    } else if (source.kind == Binding::Kind::Window) {
      Binding copy;
      copy.kind = Binding::Kind::Window;
      copy.windowFamily = source.windowFamily;
      auto declare = [&](llvm::StringRef expression, llvm::StringRef prefix) {
        std::string name = fresh(prefix);
        line("__auto_type " + name + " = " + expression.str() + ";");
        return name;
      };
      for (llvm::StringRef expression : source.windowLhs)
        copy.windowLhs.push_back(declare(expression, "for_window_lhs"));
      for (llvm::StringRef expression : source.windowRhs)
        copy.windowRhs.push_back(declare(expression, "for_window_rhs"));
      for (llvm::StringRef expression : source.windowValidity)
        copy.windowValidity.push_back(
            declare(expression, "for_window_valid"));
      carried.push_back(std::move(copy));
    } else {
      return fail(operation,
                  "ordered for carry has no selected scalar/vector/tuple/window/local handoff");
    }
  }
  std::string iterator = fresh("for_index");
  auto level =
      operation->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
  auto direction =
      operation->getAttrOfType<mlir::StringAttr>("weft.riscv.direction");
  llvm::StringRef ordered =
      level ? level.getDirection()
            : direction ? direction.getValue() : llvm::StringRef();
  if (ordered.empty())
    return fail(operation, "ordered for has no selected physical direction");
  const bool descending = ordered == "descending";
  const std::string indexType = descending ? "ptrdiff_t" : "size_t";
  const std::string comparison = descending ? " > " : " < ";
  auto systemUnroll = operation->getAttrOfType<mlir::StringAttr>(
      "weft.riscv.system_unroll");
  if (systemUnroll && systemUnroll.getValue() == "disable")
    line("#pragma GCC unroll 1");
  line("for (" + indexType + " " + iterator + " = (" + indexType + ")(" +
       lower.scalar + "); " + iterator + comparison + "(" + indexType + ")(" +
       upper.scalar + "); " + iterator + " += " + step.scalar + ") {");
  ++indent;
  llvm::SmallVector<Binding, 0> arguments{scalar(iterator)};
  mlir::Block &body = *operation.getBody();
  for (auto [index, binding] : llvm::enumerate(carried)) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInitArgs()[index], body.getArgument(index + 1), binding);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered for argument requires an explicit physical layout conversion");
    arguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(body, arguments)))
    return mlir::failure();
  auto yield = mlir::cast<mlir::scf::YieldOp>(body.getTerminator());
  if (yield.getResults().size() != carried.size())
    return fail(operation,
                "ordered for yield count does not match its carried state");
  llvm::SmallVector<Binding, 0> staged;
  for (mlir::Value nextValue : yield.getResults()) {
    Binding next = bindings.lookup(nextValue);
    if (next.kind == Binding::Kind::LocalArray) {
      staged.push_back(std::move(next));
      continue;
    }
    if (next.kind == Binding::Kind::Window) {
      Binding temporary;
      temporary.kind = Binding::Kind::Window;
      temporary.windowFamily = next.windowFamily;
      auto stageWindow = [&](llvm::ArrayRef<std::string> expressions,
                             llvm::SmallVectorImpl<std::string> &results) {
        for (llvm::StringRef expression : expressions) {
          std::string name = fresh("for_next_window");
          line("__auto_type " + name + " = " + expression.str() + ";");
          results.push_back(std::move(name));
        }
      };
      stageWindow(next.windowLhs, temporary.windowLhs);
      stageWindow(next.windowRhs, temporary.windowRhs);
      stageWindow(next.windowValidity, temporary.windowValidity);
      staged.push_back(std::move(temporary));
      continue;
    }
    mlir::FailureOr<Binding> temporary =
        declareMutableBinding(nextValue, "for_next");
    if (mlir::failed(temporary) ||
        mlir::failed(assignBinding(
            operation, nextValue, *temporary, nextValue, next,
            "ordered for body requires an explicit physical layout conversion")))
      return mlir::failure();
    staged.push_back(std::move(*temporary));
  }
  for (auto [index, nextValue] : llvm::enumerate(yield.getResults())) {
    if (mlir::failed(assignBinding(
            operation, operation.getInitArgs()[index], carried[index], nextValue,
            staged[index],
            "ordered for body requires an explicit physical layout conversion")))
      return mlir::failure();
  }
  --indent;
  line("}");
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInitArgs()[index], result, carried[index]);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered for result requires an explicit physical layout conversion");
    bindings[result] = std::move(*projected);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIf(mlir::scf::IfOp operation) {
  Binding condition = bindings.lookup(operation.getCondition());
  if (condition.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered if requires a selected scalar condition");

  llvm::SmallVector<Binding, 0> results;
  for (mlir::Value resultValue : operation.getResults()) {
    if (auto value = mlir::dyn_cast<riscv::ValueType>(resultValue.getType());
        value && value.getLayout().getCarrier() == "local") {
      Binding result;
      result.kind = Binding::Kind::None;
      results.push_back(std::move(result));
      continue;
    }
    auto selectedKind = selectedBindingKind(resultValue);
    if (selectedKind && *selectedKind == Binding::Kind::Vector) {
      mlir::FailureOr<Binding> result =
          makeVector(resultValue, "if_result", scalar("0"));
      if (mlir::failed(result))
        return mlir::failure();
      results.push_back(std::move(*result));
      continue;
    }
    auto type = scalarCType(riscv_internal::logicalElement(resultValue.getType()));
    if (!type)
      return fail(operation, "ordered if result has no intrinsic-C scalar type");
    Binding result;
    if (selectedKind && *selectedKind == Binding::Kind::ScalarTuple) {
      result.kind = Binding::Kind::ScalarTuple;
      for (int64_t part = 0; part < registerPartCount(resultValue); ++part) {
        std::string name = fresh("if_result");
        line(*type + " " + name + ";");
        result.parts.push_back(std::move(name));
      }
    } else {
      result.kind = Binding::Kind::Scalar;
      result.scalar = fresh("if_result");
      line(*type + " " + result.scalar + ";");
    }
    results.push_back(std::move(result));
  }

  auto compileBranch = [&](mlir::Region &region) -> mlir::LogicalResult {
    mlir::Block &block = region.front();
    if (mlir::failed(compileBlock(block, {})))
      return mlir::failure();
    auto yield = mlir::cast<mlir::scf::YieldOp>(block.getTerminator());
    if (yield.getResults().size() != results.size())
      return fail(operation,
                  "ordered if yield count does not match its results");
    for (auto [index, value] : llvm::enumerate(yield.getResults())) {
      Binding &target = results[index];
      Binding source = bindings.lookup(value);
      if (mlir::failed(assignBinding(
              operation, operation.getResult(index), target, value, source,
              "ordered if branch requires an explicit physical layout conversion")))
        return mlir::failure();
    }
    return mlir::success();
  };

  line("if (" + condition.scalar + ") {");
  ++indent;
  if (mlir::failed(compileBranch(operation.getThenRegion())))
    return mlir::failure();
  --indent;
  line("} else {");
  ++indent;
  if (mlir::failed(compileBranch(operation.getElseRegion())))
    return mlir::failure();
  --indent;
  line("}");
  for (auto [resultValue, binding] : llvm::zip(operation.getResults(), results))
    bindings[resultValue] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileWhile(mlir::scf::WhileOp operation) {
  llvm::SmallVector<Binding, 0> carried;
  for (mlir::Value value : operation.getInits()) {
    mlir::FailureOr<Binding> storage = declareMutableBinding(value, "while_carry");
    if (mlir::failed(storage))
      return fail(operation,
                  "ordered while carry has no mutable scalar/vector representation");
    Binding source = bindings.lookup(value);
    if (mlir::failed(assignBinding(
            operation, value, *storage, value, source,
            "ordered while initial carry requires an explicit physical conversion")))
      return mlir::failure();
    carried.push_back(std::move(*storage));
  }

  mlir::Block &before = operation.getBefore().front();
  mlir::Block &after = operation.getAfter().front();
  line("while (1) {");
  ++indent;
  llvm::SmallVector<Binding, 0> beforeArguments;
  for (auto [index, binding] : llvm::enumerate(carried)) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInits()[index], before.getArgument(index), binding);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered while condition carry requires an explicit physical conversion");
    beforeArguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(before, beforeArguments)))
    return mlir::failure();
  auto condition = mlir::cast<mlir::scf::ConditionOp>(before.getTerminator());
  Binding predicate = bindings.lookup(condition.getCondition());
  if (predicate.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered while condition requires a scalar predicate");
  line("if (!(" + predicate.scalar + ")) break;");

  llvm::SmallVector<Binding, 0> afterArguments;
  for (auto [index, value] : llvm::enumerate(condition.getArgs())) {
    Binding source = bindings.lookup(value);
    mlir::FailureOr<Binding> projected =
        projectBinding(value, after.getArgument(index), source);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered while body carry requires an explicit physical conversion");
    afterArguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(after, afterArguments)))
    return mlir::failure();
  auto yield = mlir::cast<mlir::scf::YieldOp>(after.getTerminator());
  if (yield.getResults().size() != carried.size())
    return fail(operation,
                "ordered while yield count does not match its carried state");
  llvm::SmallVector<Binding, 0> staged;
  for (mlir::Value nextValue : yield.getResults()) {
    Binding next = bindings.lookup(nextValue);
    if (next.kind == Binding::Kind::LocalArray) {
      staged.push_back(std::move(next));
      continue;
    }
    if (next.kind == Binding::Kind::Window) {
      Binding temporary;
      temporary.kind = Binding::Kind::Window;
      temporary.windowFamily = next.windowFamily;
      auto stageWindow = [&](llvm::ArrayRef<std::string> expressions,
                             llvm::SmallVectorImpl<std::string> &results) {
        for (llvm::StringRef expression : expressions) {
          std::string name = fresh("while_next_window");
          line("__auto_type " + name + " = " + expression.str() + ";");
          results.push_back(std::move(name));
        }
      };
      stageWindow(next.windowLhs, temporary.windowLhs);
      stageWindow(next.windowRhs, temporary.windowRhs);
      stageWindow(next.windowValidity, temporary.windowValidity);
      staged.push_back(std::move(temporary));
      continue;
    }
    mlir::FailureOr<Binding> temporary =
        declareMutableBinding(nextValue, "while_next");
    if (mlir::failed(temporary) ||
        mlir::failed(assignBinding(
            operation, nextValue, *temporary, nextValue, next,
            "ordered while body requires an explicit physical layout conversion")))
      return mlir::failure();
    staged.push_back(std::move(*temporary));
  }
  for (auto [index, value] : llvm::enumerate(yield.getResults())) {
    if (mlir::failed(assignBinding(
            operation, operation.getInits()[index], carried[index], value,
            staged[index],
            "ordered while body requires an explicit physical conversion")))
      return mlir::failure();
  }
  --indent;
  line("}");
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInits()[index], result, carried[index]);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered while result requires an explicit physical conversion");
    bindings[result] = std::move(*projected);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileMemoryView(riscv::MemoryViewOp operation) {
  Binding base = bindings.lookup(operation.getBase());
  if (base.kind != Binding::Kind::Memory)
    return fail(operation, "memory view base has no physical pointer binding");
  const size_t rank = operation.getResult().getType().getShape().size();
  if (operation.getExtents().size() != rank ||
      operation.getStrides().size() != rank ||
      operation.getOrigins().size() != rank)
    return fail(operation, "memory view runtime metadata rank is incomplete");
  base.memory.extents.clear();
  base.memory.strides.clear();
  base.memory.origins.clear();
  auto appendScalars = [&](mlir::ValueRange values,
                           llvm::SmallVectorImpl<std::string> &target) {
    for (mlir::Value value : values) {
      Binding scalarValue = bindings.lookup(value);
      if (scalarValue.kind != Binding::Kind::Scalar)
        return false;
      target.push_back(scalarValue.scalar);
    }
    return true;
  };
  if (!appendScalars(operation.getExtents(), base.memory.extents) ||
      !appendScalars(operation.getStrides(), base.memory.strides) ||
      !appendScalars(operation.getOrigins(), base.memory.origins))
    return fail(operation,
                "memory view extent, stride, and origin must be explicit index SSA values");
  auto descriptor = operation.getResult().getType();
  base.memory.recordBytes = descriptor.getStorageBits() / 8;
  base.memory.logicalElements = descriptor.getElements();
  base.memory.interleaveRows = descriptor.getInterleaveRows();
  bindings[operation.getResult()] = std::move(base);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageLoad(
    riscv::StorageLoadOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.load.u8")
    return fail(operation, "artifact storage load has no closed byte-load leaf");
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (source.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage load requires explicit memory and byte-index bindings");
  bindings[operation.getResult()] =
      scalar(source.memory.name + "[(size_t)(" + index.scalar + ")]");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageStore(
    riscv::StorageStoreOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.store.u8")
    return fail(operation, "artifact storage store has no closed byte-store leaf");
  Binding value = bindings.lookup(operation.getValue());
  Binding target = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (value.kind != Binding::Kind::Scalar ||
      target.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage store requires explicit value, memory, and byte index");
  line(target.memory.name + "[(size_t)(" + index.scalar + ")] = " +
       value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSlice(riscv::SliceOp slice) {
  Binding base = bindings.lookup(slice.getBase());
  if (base.kind == Binding::Kind::Field) {
    auto inputType = mlir::dyn_cast<riscv::ValueType>(slice.getBase().getType());
    const int64_t fieldStart =
        inputType ? static_cast<int64_t>(inputType.getShape().size()) -
                        base.field.logicalRank
                  : 0;
    size_t cursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size())
        return fail(slice, "field slice selector has no corresponding index");
      if (base.field.index)
        return fail(slice,
                    "intrinsic-C field projection supports one logical field axis");
      base.field.index = slice.getIndices()[cursor++];
      base.field.selector = selector.str();
      // A domain projection narrows an axis to the current Level partition but
      // keeps that logical axis in the result.  Only selectors that eliminate
      // the source axis reduce the encoded field rank.
      if (inputType && static_cast<int64_t>(position) >= fieldStart &&
          selector != "domain")
        --base.field.logicalRank;
    }
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(slice.getResult().getType()))
      base.field.shape.assign(resultType.getShape().asArrayRef().begin(),
                              resultType.getShape().asArrayRef().end());
    bindings[slice.getResult()] = std::move(base);
    return mlir::success();
  }
  Binding binding;
  binding.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Slice) {
    binding.slice = base.slice;
    auto baseType = mlir::cast<riscv::MemDescType>(slice.getBase().getType());
    const Binding &root = bindings.lookup(binding.slice.base);
    if (root.kind != Binding::Kind::Memory)
      return fail(slice, "nested slice has no root memory descriptor");
    if (binding.slice.selectors.size() > root.memory.axes.size())
      return fail(slice, "nested slice root selector rank exceeds its memory rank");
    binding.slice.selectors.resize(root.memory.axes.size(), "all");
    size_t cursor = 0;
    for (auto [position, selectorAttribute] : llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size() ||
          position >= baseType.getAxisIds().size())
        return fail(slice, "nested slice selector has no logical axis or index");
      int64_t axis = baseType.getAxisIds()[position];
      mlir::Value index = slice.getIndices()[cursor++];
      if (selector == "index") {
        // Plain scalar indices are offsets relative to the selected block.
        binding.slice.localOffsets.emplace_back(axis, index);
        continue;
      }
      if (selector != "domain" && selector != "group_index")
        return fail(slice, "nested slice has an unknown selector");
      auto found = llvm::find(root.memory.axes, axis);
      if (found == root.memory.axes.end())
        return fail(slice, "nested slice logical axis is absent from root memory");
      size_t rootDimension = static_cast<size_t>(
          std::distance(root.memory.axes.begin(), found));
      size_t rootCursor = 0;
      for (size_t dimension = 0; dimension < rootDimension; ++dimension)
        rootCursor += binding.slice.selectors[dimension] != "all";
      if (binding.slice.selectors[rootDimension] == "all")
        binding.slice.indices.insert(binding.slice.indices.begin() + rootCursor,
                                     index);
      else if (rootCursor < binding.slice.indices.size())
        binding.slice.indices[rootCursor] = index;
      else
        return fail(slice, "nested slice root selector has no matching index");
      // Physical points carry absolute coordinates in their canonical domain.
      // Domain and group-index projections therefore replace the root selector
      // for that axis; adding the child point to its parent would count the
      // parent base twice.
      binding.slice.selectors[rootDimension] = selector.str();
    }
  } else {
    binding.slice.base = slice.getBase();
    binding.slice.indices.assign(slice.getIndices().begin(), slice.getIndices().end());
    for (mlir::Attribute selector : slice.getSelectors())
      binding.slice.selectors.push_back(
          mlir::cast<mlir::StringAttr>(selector).getValue().str());
  }
  bindings[slice.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::FailureOr<Binding> Emitter::recordForSlice(mlir::Value value) {
  Binding region = bindings.lookup(value);
  if (region.kind != Binding::Kind::Slice) {
    value.getDefiningOp()->emitError("encoded record requires a canonical slice region");
    return mlir::failure();
  }
  const Binding &base = bindings.lookup(region.slice.base);
  auto sliceType = mlir::dyn_cast<riscv::MemDescType>(value.getType());
  auto encoding = sliceType
                      ? mlir::dyn_cast<kernel::EncodingType>(sliceType.getEncoding())
                      : kernel::EncodingType();
  if (!encoding || encoding.getKind() == "dense") {
    value.getDefiningOp()->emitError("record slice must carry a concrete encoding");
    return mlir::failure();
  }
  if (base.kind != Binding::Kind::Memory) {
    value.getDefiningOp()->emitError("encoded record requires a memory-backed view");
    return mlir::failure();
  }
  std::string family = encoding.getFamily().str();
  if (base.memory.recordBytes <= 0 || base.memory.logicalElements <= 0 ||
      base.memory.strides.size() != base.memory.axes.size() ||
      base.memory.origins.size() != base.memory.axes.size())
    return value.getDefiningOp()->emitError(
               "encoded record descriptor has no explicit storage geometry"),
           mlir::failure();

  llvm::DenseMap<int64_t, PointInfo> points;
  llvm::SmallVector<std::string> coordinates(base.memory.origins.begin(),
                                             base.memory.origins.end());
  size_t cursor = 0;
  for (auto [dimension, selector] : llvm::enumerate(region.slice.selectors)) {
    if (selector == "all")
      continue;
    if (cursor >= region.slice.indices.size())
      return value.getDefiningOp()->emitError(
                 "encoded record selector has no index operand"),
             mlir::failure();
    const Binding &index = bindings.lookup(region.slice.indices[cursor++]);
    if (selector == "domain") {
      coordinates[dimension] = index.point.base;
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "group_index") {
      coordinates[dimension] = "(" + index.point.base + " / " +
                               std::to_string(index.point.physicalExtent) + ")";
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "index") {
      coordinates[dimension] = index.scalar;
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record has an unknown slice selector"),
             mlir::failure();
    }
  }
  for (auto [axis, offsetValue] : region.slice.localOffsets) {
    auto dimension = llvm::find(base.memory.axes, axis);
    if (dimension == base.memory.axes.end())
      return value.getDefiningOp()->emitError(
                 "encoded record local offset has no matching memory axis"),
             mlir::failure();
    const size_t position =
        static_cast<size_t>(std::distance(base.memory.axes.begin(), dimension));
    const Binding &offset = bindings.lookup(offsetValue);
    std::string expression;
    if (offset.kind == Binding::Kind::Scalar) {
      expression = offset.scalar;
    } else if (offset.kind == Binding::Kind::Point) {
      expression = offset.point.base;
      PointInfo point = offset.point;
      point.base = "(" + coordinates[position] + " + " + expression + ")";
      points[axis] = std::move(point);
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record local offset is neither scalar nor domain point"),
             mlir::failure();
    }
    coordinates[position] =
        "(" + coordinates[position] + " + " + expression + ")";
    if (auto point = points.find(axis); point != points.end())
      point->second.base = coordinates[position];
  }
  Binding result;
  result.kind = Binding::Kind::Record;
  const int64_t recordBytes = base.memory.recordBytes;
  const int64_t elements = base.memory.logicalElements;
  result.recordAxis = base.memory.axes.empty() ? 0 : base.memory.axes.back();
  result.recordElements = elements;
  result.recordStrideBytes = recordBytes;
  for (auto [dimension, axis] : llvm::enumerate(base.memory.axes))
    if (llvm::is_contained(sliceType.getAxisIds().asArrayRef(), axis))
      result.recordOrigins.emplace_back(axis, coordinates[dimension]);
  if (encoding.getKind() == "derived_instance" ||
      base.memory.interleaveRows > 0) {
    int64_t rows = base.memory.interleaveRows > 0
                       ? base.memory.interleaveRows
                       : sliceType.getInterleaveRows();
    if (rows <= 0 || base.memory.axes.size() != 2)
      return value.getDefiningOp()->emitError(
                 "derived interleave requires a two-axis view and selected rows"),
             mlir::failure();
    const PointInfo row = points.lookup(base.memory.axes[0]);
    const PointInfo block = points.lookup(base.memory.axes[1]);
    const std::string rowBase =
        base.memory.interleaveRows > 0
            ? "(" + row.base + " - " + base.memory.origins[0] + ")"
            : row.base;
    const std::string blockBase =
        base.memory.interleaveRows > 0
            ? "(" + block.base + " - " + base.memory.origins[1] + ")"
            : block.base;
    const std::string blocks = "(" + base.memory.extents[1] + " / " +
                               std::to_string(elements) + ")";
    result.interleaveRows = rows;
    result.recordStrideBytes = recordBytes * rows;
    result.recordByteStrides.emplace_back(base.memory.axes[0], "1");
    result.recordPointer = base.memory.name + " + (((" + rowBase + " / " +
                           std::to_string(rows) + ") * " + blocks +
                           " + (" + blockBase + " / " +
                           std::to_string(elements) + ")) * " +
                           std::to_string(recordBytes * rows) + ")";
  } else {
    if (base.memory.axes.empty())
      return value.getDefiningOp()->emitError(
                 "base encoded view has no logical axis"),
             mlir::failure();
    const size_t recordDimension = base.memory.axes.size() - 1;
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      result.recordByteStrides.emplace_back(
          base.memory.axes[dimension],
          "((" + base.memory.strides[dimension] + " / " +
              std::to_string(elements) + ") * " +
              std::to_string(recordBytes) + ")");
    }
    // Every dimension before the encoded element axis advances by whole
    // records. Keep that divisibility visible in C instead of flattening all
    // logical coordinates and dividing the sum: the latter hides the affine
    // record induction from the system compiler when extents are dynamic.
    std::string record = "0";
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      record = "(" + record + " + ((" + coordinates[dimension] + ") - (" +
               base.memory.origins[dimension] + ")) * ((" +
               base.memory.strides[dimension] + ") / " +
               std::to_string(elements) + "))";
    }
    const size_t elementDimension = recordDimension;
    record = "(" + record + " + (((" + coordinates[elementDimension] +
             ") - (" + base.memory.origins[elementDimension] + ")) * (" +
             base.memory.strides[elementDimension] + ")) / " +
             std::to_string(elements) + ")";
    result.recordPointer = base.memory.name + " + (" + record + ") * " +
                           std::to_string(recordBytes);
  }
  return result;
}

mlir::LogicalResult Emitter::compileAdmit(riscv::LoadOp admit) {
  Binding region = bindings.lookup(admit.getRegion());
  if (region.kind == Binding::Kind::Memory) {
    auto view = mlir::dyn_cast<riscv::MemDescType>(admit.getRegion().getType());
    auto encoding = view ? mlir::dyn_cast<kernel::EncodingType>(view.getEncoding())
                         : kernel::EncodingType();
    if (!encoding || encoding.getKind() != "dense")
      return fail(admit,
                  "whole-view admit is only defined for a dense read-only value");
    Binding slice;
    slice.kind = Binding::Kind::Slice;
    slice.slice.base = admit.getRegion();
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(slice));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(admit, "admit currently requires a canonical slice region");
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(
      mlir::cast<riscv::MemDescType>(admit.getRegion().getType()).getEncoding());
  if (!encoding)
    return fail(admit, "admit slice has no encoding");
  if (encoding.getKind() == "dense") {
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(region));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = recordForSlice(admit.getRegion());
  if (mlir::failed(result))
    return mlir::failure();
  bindings[admit.getResult()] = std::move(*result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIota(riscv::IotaOp operation) {
  llvm::StringRef realization = instructionOf(operation.getOperation());
  mlir::Type element =
      riscv_internal::logicalElement(operation.getResult().getType());
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (!element.isIndex() && (!integer || integer.isSigned()))
    return fail(operation, "iota requires an unsigned integer or index result");
  if (realization == "scalar.iota") {
    auto type = scalarCType(element);
    if (!type || operation.getEnd() - operation.getStart() != 1)
      return fail(operation,
                  "scalar iota requires one selected logical element");
    bindings[operation.getResult()] =
        scalar("((" + *type + ")" + std::to_string(operation.getStart()) + ")");
    return mlir::success();
  }
  if (realization == "register.iota") {
    auto type = scalarCType(element);
    const int64_t parts = registerPartCount(operation.getResult());
    if (!type || parts <= 1)
      return fail(operation,
                  "register iota requires a non-trivial scalar tuple mapping");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part)
      result.parts.push_back("((" + *type + ")" +
                             std::to_string(operation.getStart() + part) + ")");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (realization != "rvv.iota")
    return fail(operation, "iota has no selected local realization");
  const int64_t parts = vectorPartCount(operation.getResult());
  if (parts <= 0)
    return fail(operation, "RVV iota has no physical vector parts");
  const std::string suffix = vectorSuffix(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string type = vectorType(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string expression = "__riscv_vid_v_" + suffix + "(" + vl + ")";
    const std::string offset = "(" + std::to_string(operation.getStart()) +
                               " + " + partOffset(operation.getResult(), part) + ")";
    expression = "__riscv_vadd_vx_" + suffix + "(" + expression + ", " +
                 offset + ", " + vl + ")";
    std::string name = fresh("iota");
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVAxisBroadcast(riscv::RVVAxisBroadcastOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.axis-broadcast")
    return fail(operation, "RVV axis broadcast has no exact selected leaf");
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  auto input =
      materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV axis broadcast requires one materialized vector input");
  const int64_t inputStreams = streamPartCount(operation.getInput());
  const int64_t resultStreams = streamPartCount(operation.getResult());
  const int64_t resultParts = vectorPartCount(operation.getResult());
  if (inputStreams <= 0 || resultStreams <= 0 || resultParts <= 0)
    return fail(operation, "RVV axis broadcast has invalid typed part geometry");

  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string indexSuffix =
      "u" + std::to_string(resultType.getLayout().getSew()) +
      lmulSpelling(resultType.getLayout().getLmulEighths());
  const std::string resultCType = vectorType(operation.getResult());
  const std::string indexCType = "vuint" +
                                 std::to_string(resultType.getLayout().getSew()) +
                                 lmulSpelling(resultType.getLayout().getLmulEighths()) +
                                 "_t";

  auto isPowerOfTwo = [](int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
  };
  auto log2Exact = [](int64_t value) {
    int64_t result = 0;
    while (value > 1) {
      value >>= 1;
      ++result;
    }
    return result;
  };

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < resultParts; ++part) {
    const int64_t resultRegister = part / resultStreams;
    auto sourceRegister = projectRegisterPart(operation.getInput(),
                                              operation.getResult(),
                                              resultRegister);
    if (!sourceRegister)
      return fail(operation,
                  "RVV axis broadcast cannot project register coordinates");
    int64_t sourceStream = 0;
    for (auto [axis, factor] : llvm::zip(
             inputType.getAxisIds().asArrayRef(),
             inputType.getLayout().getTimeFactors().asArrayRef())) {
      if (factor <= 0)
        return fail(operation,
                    "RVV axis broadcast has a non-positive input time factor");
      int64_t coordinate = 0;
      if (factor > 1) {
        auto projected = timeCoordinate(operation.getResult(), part, axis);
        if (!projected || *projected >= factor)
          return fail(operation,
                      "RVV axis broadcast cannot project input time coordinates");
        coordinate = *projected;
      }
      sourceStream = sourceStream * factor + coordinate;
    }
    const int64_t sourcePart = *sourceRegister * inputStreams + sourceStream;
    if (sourcePart < 0 || sourcePart >= static_cast<int64_t>(input->parts.size()))
      return fail(operation,
                  "RVV axis broadcast projected an absent input vector part");

    std::string source = input->parts[sourcePart];
    if (inputSuffix != resultSuffix)
      source = "__riscv_vlmul_ext_v_" + inputSuffix + "_" + resultSuffix +
               "(" + source + ")";

    const std::string vl = partVL(operation.getResult(), part);
    std::string index = fresh("broadcast_index");
    line(indexCType + " " + index + " = __riscv_vmv_v_x_" + indexSuffix +
         "(0, " + vl + ");");
    bool hasCoordinate = false;
    int64_t sourceStride = 1;
    llvm::SmallVector<int64_t> sourceLaneStrides(
        inputType.getAxisIds().size(), 1);
    for (int64_t position =
             static_cast<int64_t>(inputType.getAxisIds().size()) - 1;
         position >= 0; --position) {
      sourceLaneStrides[position] = sourceStride;
      sourceStride *= inputType.getLayout().getLaneFactors()[position];
    }
    int64_t resultStride = 1;
    llvm::DenseMap<int64_t, int64_t> resultLaneStrides;
    for (int64_t position =
             static_cast<int64_t>(resultType.getAxisIds().size()) - 1;
         position >= 0; --position) {
      resultLaneStrides[resultType.getAxisIds()[position]] = resultStride;
      resultStride *= resultType.getLayout().getLaneFactors()[position];
    }
    for (size_t position = 0; position < inputType.getAxisIds().size();
         ++position) {
      const int64_t factor = inputType.getLayout().getLaneFactors()[position];
      if (factor <= 1)
        continue;
      const int64_t axis = inputType.getAxisIds()[position];
      auto resultAxis = llvm::find(resultType.getAxisIds().asArrayRef(), axis);
      if (resultAxis == resultType.getAxisIds().asArrayRef().end())
        return fail(operation,
                    "RVV axis broadcast lost one input lane coordinate");
      const size_t resultPosition = static_cast<size_t>(
          resultAxis - resultType.getAxisIds().asArrayRef().begin());
      if (resultType.getLayout().getLaneFactors()[resultPosition] != factor)
        return fail(operation,
                    "RVV axis broadcast changed a source lane factor");
      std::string coordinate = fresh("broadcast_coordinate");
      const int64_t stride = resultLaneStrides.lookup(axis);
      std::string expression = "__riscv_vid_v_" + indexSuffix + "(" + vl + ")";
      if (stride > 1)
        expression = isPowerOfTwo(stride)
                         ? "__riscv_vsrl_vx_" + indexSuffix + "(" + expression +
                               ", " + std::to_string(log2Exact(stride)) + ", " +
                               vl + ")"
                         : "__riscv_vdivu_vx_" + indexSuffix + "(" + expression +
                               ", " + std::to_string(stride) + ", " + vl + ")";
      if (factor > 1)
        expression = isPowerOfTwo(factor)
                         ? "__riscv_vand_vx_" + indexSuffix + "(" + expression +
                               ", " + std::to_string(factor - 1) + ", " + vl +
                               ")"
                         : "__riscv_vremu_vx_" + indexSuffix + "(" + expression +
                               ", " + std::to_string(factor) + ", " + vl + ")";
      if (sourceLaneStrides[position] > 1)
        expression = "__riscv_vmul_vx_" + indexSuffix + "(" + expression +
                     ", " + std::to_string(sourceLaneStrides[position]) + ", " +
                     vl + ")";
      line(indexCType + " " + coordinate + " = " + expression + ";");
      if (!hasCoordinate) {
        line(index + " = " + coordinate + ";");
        hasCoordinate = true;
      } else {
        line(index + " = __riscv_vadd_vv_" + indexSuffix + "(" + index +
             ", " + coordinate + ", " + vl + ");");
      }
    }
    if (!hasCoordinate)
      return fail(operation,
                  "RVV axis broadcast input has no physical lane coordinate");
    std::string broadcast = fresh("axis_broadcast");
    line(resultCType + " " + broadcast + " = __riscv_vrgather_vv_" +
         resultSuffix + "(" + source + ", " + index + ", " + vl + ");");
    result.parts.push_back(std::move(broadcast));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileNew(riscv::NewOp operation) {
  if (!operation.getInitialized() || !operation.getInitial())
    return fail(operation,
                "physical new has no explicit canonical initializer");
  Binding initial = bindings.lookup(operation.getInitial());
  auto selectedKind = selectedBindingKind(operation.getResult());
  if (selectedKind && *selectedKind == Binding::Kind::Vector) {
    mlir::FailureOr<Binding> result =
        initial.kind == Binding::Kind::Slice
            ? loadDenseBlock(operation.getResult(), initial)
            : makeVector(operation.getResult(), "state", initial,
                         operation.getInitial());
    if (mlir::failed(result))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
    return mlir::success();
  }
  if (selectedKind && *selectedKind == Binding::Kind::ScalarTuple) {
    auto type =
        scalarCType(riscv_internal::logicalElement(operation.getResult().getType()));
    const int64_t parts = registerPartCount(operation.getResult());
    if (!type || parts <= 0)
      return fail(operation,
                  "selected register tuple state has no scalar C representation");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    llvm::SmallVector<int64_t, 4> axes = registerAxesFor(operation.getResult());
    for (int64_t part = 0; part < parts; ++part) {
      std::string expression;
      if (initial.kind == Binding::Kind::Scalar) {
        expression = initial.scalar;
      } else if (initial.kind == Binding::Kind::ScalarTuple) {
        auto projected = projectPart(operation.getInitial(), operation.getResult(), part);
        if (!projected || *projected >= initial.parts.size())
          return fail(operation,
                      "register tuple initializer mapping is not projectable");
        expression = initial.parts[*projected];
      } else if (initial.kind == Binding::Kind::Slice) {
        auto coordinates = registerCoordinates(operation.getResult(), part);
        if (!coordinates || coordinates->size() != axes.size())
          return fail(operation,
                      "register tuple initializer has no coordinate projection");
        llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
        std::string active = "1";
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
          offsets.push_back({axis, std::to_string(coordinate)});
          auto scope = axisScopes.find(axis);
          if (scope != axisScopes.end() && !scope->second.empty())
            active += " && " + std::to_string(coordinate) + " < " +
                      scope->second.back().active;
        }
        auto address = denseAddress(initial.slice, offsets);
        if (!address)
          return fail(operation,
                      "register tuple initializer has no dense address relation");
        expression = "((" + active + ") ? *(const " + *type + " *)(" +
                     *address + ") : (" + *type + ")0)";
      } else {
        return fail(operation,
                    "register tuple new requires scalar, tuple, or dense initializer");
      }
      std::string name = fresh("state");
      line(*type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (!selectedKind || *selectedKind != Binding::Kind::Scalar)
    return fail(operation, "intrinsic-C emission has no realization for selected new value");
  if (initial.kind != Binding::Kind::Scalar)
    return fail(operation, "scalar new requires a scalar initializer");
  Binding result = initial;
  result.scalar = fresh("state");
  auto type = scalarCType(riscv_internal::logicalElement(operation.getResult().getType()));
  if (!type)
    return fail(operation, "intrinsic-C emission does not support selected scalar state type");
  line(*type + " " + result.scalar + " = " + initial.scalar + ";");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileField(riscv::FieldOp operation) {
  if (!operation.getAccess() || operation.getAccess().getMapping() == "opaque")
    return fail(operation,
                "field emission requires one selected memory-edge mapping");
  Binding binding;
  binding.kind = Binding::Kind::Field;
  binding.field.owner = operation.getOwner();
  binding.field.name = operation.getName().str();
  binding.field.storageAccess = operation.getAccess();
  binding.field.useAccess = operation.getAccess();
  binding.field.elementType =
      riscv_internal::logicalElement(operation.getResult().getType());
  if (mlir::isa<kernel::EncodingType>(binding.field.elementType)) {
    auto dense = denseElementType(binding.field.elementType);
    if (!dense)
      return fail(operation,
                  "encoded field result has no scalar storage element type");
    binding.field.elementType = *dense;
  }
  if (auto value =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType()))
    binding.field.shape.assign(value.getShape().asArrayRef().begin(),
                               value.getShape().asArrayRef().end());
  binding.field.logicalRank = riscv_internal::fieldFacts(operation).logicalRank;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileExtract(riscv::ExtractOp operation) {
  Binding binding = bindings.lookup(operation.getInput());
  if (instructionOf(operation.getOperation()) == "rvv.extract.vrgather") {
    mlir::FailureOr<Binding> source =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(source))
      return mlir::failure();
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size())
        return fail(operation, "register extract selector has no index");
      if (selector == "gather") {
        if (gatherIndex)
          return fail(operation,
                      "register extract supports one shaped gather selector");
        gatherIndex = operation.getIndices()[cursor];
      }
      ++cursor;
    }
    if (!gatherIndex || source->kind != Binding::Kind::Vector ||
        source->parts.size() != 1)
      return fail(operation,
                  "register extract requires one complete source vector");
    mlir::FailureOr<Binding> indices = materializeNumeric(
        gatherIndex, bindings.lookup(gatherIndex));
    if (mlir::failed(indices) || indices->kind != Binding::Kind::Vector)
      return fail(operation, "register extract requires vector indices");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto indexPart = projectPart(gatherIndex, operation.getResult(), part);
      if (!indexPart || *indexPart >= indices->parts.size())
        return fail(operation,
                    "register extract index mapping requires an explicit layout conversion");
      std::string name = fresh("register_gather");
      line(type + " " + name + " = __riscv_vrgather_vv_" + suffix + "(" +
           source->parts.front() + ", " + indices->parts[*indexPart] + ", " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (instructionOf(operation.getOperation()) ==
      "rvv.extract.lane-to-replica") {
    mlir::FailureOr<Binding> source =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(source) || source->kind != Binding::Kind::Vector)
      return fail(operation,
                  "lane-to-replica extract requires one materialized RVV source");
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size() || gatherIndex ||
          selector != "gather")
        return fail(operation,
                    "lane-to-replica extract requires one typed gather index");
      gatherIndex = operation.getIndices()[cursor++];
    }
    auto laneCount = operation->getAttrOfType<mlir::IntegerAttr>(
        "lane_gather_count");
    auto sourceParts = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "lane_gather_source_parts");
    auto indexParts = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "lane_gather_index_parts");
    const int64_t resultParts = registerPartCount(operation.getResult());
    const Binding &indices = bindings.lookup(gatherIndex);
    mlir::Type element =
        riscv_internal::logicalElement(operation.getResult().getType());
    auto scalarType = scalarCType(element);
    if (!gatherIndex || cursor != operation.getIndices().size() || !laneCount ||
        laneCount.getInt() <= 1 || !sourceParts || !indexParts ||
        resultParts <= 0 ||
        sourceParts.size() != static_cast<size_t>(resultParts) ||
        indexParts.size() != static_cast<size_t>(resultParts) || !scalarType ||
        (indices.kind != Binding::Kind::Scalar &&
         indices.kind != Binding::Kind::ScalarTuple))
      return fail(operation,
                  "lane-to-replica extract has no complete pass-selected mapping");

    const std::string suffix = vectorSuffix(operation.getInput());
    Binding result;
    result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                   : Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < resultParts; ++part) {
      const int64_t sourcePart = sourceParts[part];
      const int64_t indexPart = indexParts[part];
      if (sourcePart < 0 ||
          sourcePart >= static_cast<int64_t>(source->parts.size()))
        return fail(operation,
                    "lane-to-replica extract source part exceeds its RVV value");
      std::string selected;
      if (indices.kind == Binding::Kind::Scalar) {
        if (indexPart != 0)
          return fail(operation,
                      "lane-to-replica scalar index has a nonzero selected part");
        selected = indices.scalar;
      } else {
        if (indexPart < 0 ||
            indexPart >= static_cast<int64_t>(indices.parts.size()))
          return fail(operation,
                      "lane-to-replica index part exceeds its scalar tuple");
        selected = indices.parts[static_cast<size_t>(indexPart)];
      }
      const std::string shifted =
          "__riscv_vslidedown_vx_" + suffix + "(" +
          source->parts[static_cast<size_t>(sourcePart)] + ", " + selected +
          ", " + std::to_string(laneCount.getInt()) + ")";
      std::string expression;
      if (mlir::isa<mlir::FloatType>(element)) {
        expression = "__riscv_vfmv_f_s_" + suffix + "_f" +
                     std::to_string(riscv_internal::logicalBitWidth(element)) +
                     "(" + shifted + ")";
      } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
        expression = "__riscv_vmv_x_s_" + suffix + "_" +
                     std::string(integer.isUnsigned() ? "u" : "i") +
                     std::to_string(integer.getWidth()) + "(" + shifted + ")";
      } else {
        return fail(operation,
                    "lane-to-replica extract has no intrinsic scalar spelling");
      }
      std::string name = fresh("lane_gather");
      line(*scalarType + " " + name + " = " + expression + ";");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(name);
      else
        result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Scalar &&
      !mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
    auto input = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    int64_t elements = 1;
    if (!input)
      return fail(operation,
                  "scalar physical extract has no shaped source value");
    for (int64_t extent : input.getShape().asArrayRef())
      elements *= extent;
    if (elements != 1)
      return fail(operation,
                  "scalar physical extract requires one logical source element");
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Record) {
    auto inputType = operation.getInput().getType();
    if (operation.getSelectors().size() != inputType.getAxisIds().size())
      return fail(operation,
                  "encoded record extract selector rank does not match its logical axes");
    size_t cursor = 0;
    llvm::SmallVector<std::pair<int64_t, std::string>> remainingStrides;
    llvm::SmallVector<std::pair<int64_t, std::string>> remainingOrigins;
    std::string pointer = binding.recordPointer;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      const int64_t axis = inputType.getAxisIds()[dimension];
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      if (!resultType)
        return fail(operation,
                    "encoded record extract result has no physical value type");
      const bool retainsAxis = llvm::is_contained(
          resultType.getAxisIds().asArrayRef(), axis);
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      auto stride = llvm::find_if(
          binding.recordByteStrides,
          [&](const auto &entry) { return entry.first == axis; });
      if (selector == "all") {
        if (stride != binding.recordByteStrides.end())
          remainingStrides.push_back(*stride);
        else if (axis != binding.recordAxis)
          return fail(operation,
                      "encoded record extract has no typed byte stride for one axis");
        if (auto origin = llvm::find_if(
                binding.recordOrigins,
                [&](const auto &entry) { return entry.first == axis; });
            origin != binding.recordOrigins.end())
          remainingOrigins.push_back(*origin);
        continue;
      }
      if (cursor >= operation.getIndices().size())
        return fail(operation,
                    "encoded record extract selector has no coordinate value");
      Binding coordinate = bindings.lookup(operation.getIndices()[cursor++]);
      std::string absoluteExpression;
      if (selector == "domain" && coordinate.kind == Binding::Kind::Point)
        absoluteExpression = coordinate.point.base;
      else if (selector == "group_index" &&
               coordinate.kind == Binding::Kind::Point)
        absoluteExpression = "(" + coordinate.point.base + " / " +
                             std::to_string(coordinate.point.physicalExtent) + ")";
      else if (selector == "index" && coordinate.kind == Binding::Kind::Scalar)
        absoluteExpression = coordinate.scalar;
      else
        return fail(operation,
                    "encoded record extract coordinate does not match its selector");
      std::string expression = absoluteExpression;
      if (coordinate.kind == Binding::Kind::Point)
        if (auto origin = llvm::find_if(
                binding.recordOrigins,
                [&](const auto &entry) { return entry.first == axis; });
            origin != binding.recordOrigins.end())
          expression = "((" + absoluteExpression + ") - (" +
                       origin->second + "))";
      if (stride != binding.recordByteStrides.end()) {
        pointer = "(" + pointer + " + (" + expression + ") * (" +
                  stride->second + "))";
        if (retainsAxis)
          remainingStrides.push_back(*stride);
      } else if (axis == binding.recordAxis && binding.recordElements > 0 &&
                 binding.recordStrideBytes > 0) {
        pointer = "(" + pointer + " + ((" + expression + ") / " +
                  std::to_string(binding.recordElements) + ") * " +
                  std::to_string(binding.recordStrideBytes) + ")";
        if (!retainsAxis)
          binding.recordAxis = 0;
      } else {
        return fail(operation,
                    "encoded record extract has no typed record coordinate relation");
      }
      if (retainsAxis && coordinate.kind == Binding::Kind::Point)
        remainingOrigins.emplace_back(axis, absoluteExpression);
    }
    if (cursor != operation.getIndices().size())
      return fail(operation,
                  "encoded record extract has unused coordinate operands");
    binding.recordPointer = std::move(pointer);
    binding.recordByteStrides = std::move(remainingStrides);
    binding.recordOrigins = std::move(remainingOrigins);
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::LocalArray) {
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType())) {
      if (operation.getLeaf().getInstruction() != "rvv.local-gather" ||
          operation.getAccess().getForm() != "indexed" ||
          resultType.getLayout().getCarrier() != "rvv")
        return fail(operation,
                    "shaped local extract has no selected RVV gather leaf");
      auto inputType = operation.getInput().getType();
      if (operation.getSelectors().size() != inputType.getShape().size())
        return fail(operation,
                    "local gather selector rank does not match its local value");
      size_t cursor = 0;
      size_t gatherDimension = inputType.getShape().size();
      mlir::Value indexValue;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(operation.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= operation.getIndices().size() || selector != "gather" ||
            indexValue)
          return fail(operation,
                      "local gather requires one shaped gather selector");
        gatherDimension = dimension;
        indexValue = operation.getIndices()[cursor++];
      }
      if (!indexValue || cursor != operation.getIndices().size())
        return fail(operation,
                    "local gather has no unique shaped index value");
      Binding indices = bindings.lookup(indexValue);
      auto indexLayout = layoutOf(indexValue);
      auto indexInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(indexValue.getType()));
      auto elementType = scalarCType(inputType.getElementType());
      if (indices.kind != Binding::Kind::Vector || !indexLayout ||
          !indexInteger || !indexInteger.isUnsigned() || !elementType)
        return fail(operation,
                    "local gather index and element types have no RVV spelling");

      llvm::SmallVector<std::string> elementStrides(inputType.getShape().size(),
                                                     "1");
      for (int64_t dimension =
               static_cast<int64_t>(inputType.getShape().size()) - 2;
           dimension >= 0; --dimension) {
        auto extent = extentForShape(inputType.getShape()[dimension + 1],
                                     inputType.getAxisIds()[dimension + 1]);
        if (!extent)
          return fail(operation,
                      "local gather source has no explicit physical extent");
        elementStrides[dimension] = "(" + elementStrides[dimension + 1] +
                                    " * (" + *extent + "))";
      }

      const std::string indexSuffix = vectorSuffix(indexValue);
      const std::string resultSuffix = vectorSuffix(operation.getResult());
      const std::string resultCType = vectorType(operation.getResult());
      const int64_t elementBytes =
          std::max<int64_t>(1, riscv_internal::logicalBitWidth(inputType) / 8);
      Binding gathered;
      gathered.kind = Binding::Kind::Vector;
      llvm::SmallVector<int64_t, 4> registerAxes =
          registerAxesFor(operation.getResult());
      const int64_t streams = streamPartCount(operation.getResult());
      for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
        auto indexPart = projectPart(indexValue, operation.getResult(), part);
        auto coordinates = registerCoordinates(operation.getResult(), part / streams);
        if (!indexPart || *indexPart >= indices.parts.size() || !coordinates ||
            coordinates->size() != registerAxes.size())
          return fail(operation,
                      "local gather index cannot be projected to its result layout");
        std::string baseElements = "0";
        for (size_t dimension = 0; dimension < inputType.getShape().size();
             ++dimension) {
          if (dimension == gatherDimension)
            continue;
          const int64_t axis = inputType.getAxisIds()[dimension];
          std::string coordinate = "0";
          auto registerAxis = llvm::find(registerAxes, axis);
          if (registerAxis != registerAxes.end())
            coordinate = std::to_string(
                (*coordinates)[registerAxis - registerAxes.begin()]);
          auto scope = axisScopes.find(axis);
          if (scope != axisScopes.end() && !scope->second.empty())
            coordinate = "(" + scope->second.back().base + " + " +
                         coordinate + ")";
          baseElements = "(" + baseElements + " + (" + coordinate +
                         ") * (" + elementStrides[dimension] + "))";
        }
        std::string offsets = indices.parts[*indexPart];
        const std::string byteScale =
            "(" + elementStrides[gatherDimension] + " * " +
            std::to_string(elementBytes) + ")";
        if (byteScale != "(1 * 1)")
          offsets = "__riscv_vmul_vx_" + indexSuffix + "(" + offsets +
                    ", " + byteScale + ", " +
                    partVL(operation.getResult(), part) + ")";
        std::string name = fresh("local_gather");
        line(resultCType + " " + name + " = __riscv_vluxei" +
             std::to_string(indexLayout.getSew()) + "_v_" + resultSuffix +
             "((const " + *elementType + " *)(" + binding.scalar + ") + (" +
             baseElements + "), " + offsets + ", " +
             partVL(operation.getResult(), part) + ");");
        gathered.parts.push_back(std::move(name));
      }
      bindings[operation.getResult()] = std::move(gathered);
      return mlir::success();
    }
    auto inputType = operation.getInput().getType();
    if (operation.getSelectors().size() != inputType.getShape().size())
      return fail(operation,
                  "local array extract selector rank does not match its value");
    std::string linear = "0";
    size_t cursor = 0;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        return fail(operation,
                    "scalar local array extract cannot retain a logical axis");
      if (selector != "index" || cursor >= operation.getIndices().size())
        return fail(operation,
                    "local array extract requires one scalar index per axis");
      Binding index = bindings.lookup(operation.getIndices()[cursor++]);
      if (index.kind != Binding::Kind::Scalar)
        return fail(operation, "local array index has no scalar representation");
      auto extent = extentForShape(inputType.getShape()[dimension],
                                   inputType.getAxisIds()[dimension]);
      if (!extent)
        return fail(operation, "local array axis has no physical extent");
      linear = "((" + linear + ") * (" + *extent + ") + (" + index.scalar + "))";
    }
    bindings[operation.getResult()] =
        scalar(binding.scalar + "[" + linear + "]");
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Field) {
    if (!operation.getAccess())
      return fail(operation,
                  "encoded extract has no pass-selected per-use memory edge");
    binding.field.useAccess = operation.getAccess();
    auto inputType =
        mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    const int64_t fieldStart =
        inputType ? static_cast<int64_t>(inputType.getShape().size()) -
                        binding.field.logicalRank
                  : 0;
    size_t cursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (selector == "regular") {
        auto pattern = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "index_pattern");
        if (!pattern || pattern.size() != 3 || binding.field.regularRepeat)
          return fail(operation,
                      "regular encoded extract has no unique index pattern");
        binding.field.regularBase = pattern[0];
        binding.field.regularStride = pattern[1];
        binding.field.regularRepeat = pattern[2];
        binding.field.selector = "regular";
        continue;
      }
      if (cursor >= operation.getIndices().size())
        return fail(operation, "extract selector has no corresponding index");
      mlir::Value index = operation.getIndices()[cursor++];
      if (binding.field.index) {
        if (selector != "index")
          return fail(operation,
                      "nested encoded extract requires a relative scalar index");
        binding.field.relativeIndices.push_back(index);
      } else {
        binding.field.index = index;
        binding.field.selector = selector.str();
      }
      // Keep a domain-selected field axis available to the physical layout.
      // Its per-part time coordinate is needed when the selected sub-domain is
      // materialized as multiple sequential RVV values.
      if (inputType && static_cast<int64_t>(position) >= fieldStart &&
          selector != "domain")
        --binding.field.logicalRank;
    }
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType()))
      binding.field.shape.assign(resultType.getShape().asArrayRef().begin(),
                                 resultType.getShape().asArrayRef().end());
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }

  if (binding.kind == Binding::Kind::Slice) {
    if (auto inputType =
            mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
        inputType && mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
      // A shaped extract of a reloadable dense value is still an addressable
      // memory region.  Preserve that relation and attach the consumed logical
      // coordinates; the first numerical consumer materializes the result in
      // its own selected layout.  Loading the entire parent panel here would
      // destroy both reuse and the child operation's lane/register mapping.
      size_t cursor = 0;
      for (auto [position, selectorAttribute] :
           llvm::enumerate(operation.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= operation.getIndices().size() ||
            position >= inputType.getAxisIds().size())
          return fail(operation,
                      "local materialized extract has no logical axis or index");
        binding.slice.localOffsets.emplace_back(
            inputType.getAxisIds()[position], operation.getIndices()[cursor++]);
      }
      bindings[operation.getResult()] = std::move(binding);
      return mlir::success();
    }
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(materialized))
      return mlir::failure();
    auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
    int64_t elements = 1;
    for (int64_t extent : inputType.getShape().asArrayRef())
      elements *= extent;
    if (materialized->kind == Binding::Kind::Scalar && elements == 1 &&
        !mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
      bindings[operation.getResult()] = std::move(*materialized);
      return mlir::success();
    }
    binding = std::move(*materialized);
  }

  if (binding.kind == Binding::Kind::ScalarTuple) {
    auto resultType =
        mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    const int64_t resultParts = registerPartCount(operation.getResult());
    auto arity = operation->getAttrOfType<mlir::IntegerAttr>(
        "replica_gather_arity");
    auto candidates = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "replica_gather_candidates");
    auto keys = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "replica_gather_keys");
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size() || gatherIndex ||
          selector != "gather")
        return fail(operation,
                    "scalar tuple extract requires exactly one typed gather index");
      gatherIndex = operation.getIndices()[cursor++];
    }
    if (instructionOf(operation.getOperation()) != "scalar.replica-gather" ||
        operation.getAccess().getForm() != "register" || !arity ||
        arity.getInt() <= 0 || !candidates || !keys ||
        keys.size() != candidates.size() || !resultType || resultParts <= 0 ||
        static_cast<int64_t>(binding.parts.size()) !=
            registerPartCount(operation.getInput()) ||
        candidates.size() !=
            static_cast<size_t>(resultParts * arity.getInt()) || !gatherIndex ||
        cursor != operation.getIndices().size())
      return fail(operation,
                  "scalar tuple extract has no pass-selected replica candidate mapping");
    const Binding &index = bindings.lookup(gatherIndex);
    Binding tuple;
    tuple.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < resultParts; ++part) {
      std::string selected;
      if (index.kind == Binding::Kind::Scalar) {
        selected = index.scalar;
      } else if (index.kind == Binding::Kind::ScalarTuple) {
        auto indexPart = projectRegisterPart(gatherIndex, operation.getResult(), part);
        if (!indexPart || *indexPart >= index.parts.size())
          return fail(operation,
                      "scalar tuple extract index does not project to its result replica");
        selected = index.parts[*indexPart];
      } else {
        return fail(operation,
                    "scalar tuple extract index has no selected scalar representation");
      }
      const size_t row = static_cast<size_t>(part * arity.getInt());
      int64_t last = candidates[row + static_cast<size_t>(arity.getInt() - 1)];
      if (last < 0 || static_cast<size_t>(last) >= binding.parts.size())
        return fail(operation,
                    "scalar tuple extract candidate exceeds its source tuple");
      std::string expression = binding.parts[static_cast<size_t>(last)];
      if (arity.getInt() == 1)
        expression = "((void)(" + selected + "), " + expression + ")";
      for (int64_t candidate = arity.getInt() - 2; candidate >= 0; --candidate) {
        int64_t source = candidates[row + static_cast<size_t>(candidate)];
        if (source < 0 || static_cast<size_t>(source) >= binding.parts.size())
          return fail(operation,
                      "scalar tuple extract candidate exceeds its source tuple");
        const int64_t key = keys[row + static_cast<size_t>(candidate)];
        expression = "((" + selected + ") == " + std::to_string(key) +
                     " ? " + binding.parts[static_cast<size_t>(source)] + " : " +
                     expression + ")";
      }
      tuple.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(tuple);
    return mlir::success();
  }

  if (binding.kind != Binding::Kind::Vector)
    return fail(operation,
                "extract has no selected local vector or encoded-field input");
  mlir::Type extractElement =
      riscv_internal::logicalElement(operation.getInput().getType());
  if ((!mlir::isa<mlir::IntegerType>(extractElement) &&
       !mlir::isa<mlir::FloatType>(extractElement)) ||
      extractElement !=
          riscv_internal::logicalElement(operation.getResult().getType()))
    return fail(operation,
                "current local vector extract requires one unchanged numeric element type");
  std::optional<mlir::Value> pointValue;
  size_t cursor = 0;
  for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (cursor >= operation.getIndices().size() || pointValue)
      return fail(operation,
                  "current local vector extract supports one domain selector");
    if (selector != "domain")
      return fail(operation,
                  "current local vector extract requires a domain selector");
    pointValue = operation.getIndices()[cursor++];
  }
  if (!pointValue)
    return fail(operation, "local vector extract has no domain point");
  const Binding &point = bindings.lookup(*pointValue);
  const int64_t laneAxis = laneAxisFor(operation.getInput());
  if (point.kind != Binding::Kind::Point || point.point.axis != laneAxis)
    return fail(operation,
                "local vector extract point does not own the selected lane axis");

  auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
  int64_t logicalExtent = 0;
  for (auto [extent, axis] : llvm::zip(inputType.getShape().asArrayRef(),
                                       inputType.getAxisIds().asArrayRef()))
    if (axis == laneAxis)
      logicalExtent = extent;
  const int64_t inputLanes = physicalLanes(operation.getInput());
  const int64_t resultLanes = physicalLanes(operation.getResult());
  const int64_t inputStreams = streamPartCount(operation.getInput());
  const int64_t resultStreams = streamPartCount(operation.getResult());
  const int64_t inputRegisters = registerPartCount(operation.getInput());
  const int64_t resultRegisters = registerPartCount(operation.getResult());
  if (logicalExtent <= 0 || inputLanes <= 0 || resultLanes <= 0 ||
      inputLanes != resultLanes || inputStreams <= 0 || resultStreams <= 0 ||
      inputRegisters <= 0 || resultRegisters <= 0 ||
      point.point.physicalExtent <= 0 ||
      point.point.physicalExtent % resultLanes ||
      static_cast<int64_t>(binding.parts.size()) !=
          inputRegisters * inputStreams)
    return fail(operation,
                "selected local vector subview has incompatible strip geometry");

  llvm::SmallVector<size_t> sourceRegisters;
  for (int64_t resultRegister = 0; resultRegister < resultRegisters;
       ++resultRegister) {
    auto sourceRegister = projectRegisterPart(operation.getInput(),
                                              operation.getResult(),
                                              resultRegister);
    if (!sourceRegister)
      return fail(operation,
                  "local vector subview cannot project one register replica");
    sourceRegisters.push_back(*sourceRegister);
  }

  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const std::string relative = "(" + point.point.base + " % " +
                               std::to_string(logicalExtent) + ")";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t resultRegister = 0; resultRegister < resultRegisters;
       ++resultRegister) {
    const size_t sourceRegister = sourceRegisters[resultRegister];
    for (int64_t resultStream = 0; resultStream < resultStreams;
         ++resultStream) {
      const int64_t resultPart = resultRegister * resultStreams + resultStream;
      const std::string resultName = fresh("extract");
      const std::string splat =
          mlir::isa<mlir::FloatType>(extractElement) ? "__riscv_vfmv_v_f_"
                                                     : "__riscv_vmv_v_x_";
      line(resultType + " " + resultName + " = " + splat + resultSuffix +
           "(" +
           (mlir::isa<mlir::FloatType>(extractElement) ? "0.0" : "0") +
           ", " + partVL(operation.getResult(), resultPart) + ");");
      const std::string selectedStream =
          "(" + relative + " / " + std::to_string(inputLanes) + " + " +
          std::to_string(resultStream) + ")";
      for (int64_t sourceStream = 0; sourceStream < inputStreams;
           ++sourceStream) {
        const size_t sourcePart = sourceRegister * inputStreams + sourceStream;
        line("if (" + selectedStream + " == " +
             std::to_string(sourceStream) + ") " + resultName + " = " +
             binding.parts[sourcePart] + ";");
      }
      result.parts.push_back(resultName);
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileUpdate(riscv::UpdateOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  mlir::FailureOr<Binding> materialized = materializeNumeric(
      operation.getValue(), bindings.lookup(operation.getValue()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding value = std::move(*materialized);
  if (input.kind != Binding::Kind::LocalArray ||
      value.kind != Binding::Kind::Scalar)
    return fail(operation,
                "physical update currently requires a local array and scalar value");
  auto inputType = operation.getInput().getType();
  if (operation.getSelectors().size() != inputType.getShape().size())
    return fail(operation,
                "local array update selector rank does not match its value");
  std::string linear = "0";
  size_t cursor = 0;
  for (auto [dimension, selectorAttribute] :
       llvm::enumerate(operation.getSelectors())) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector != "index" || cursor >= operation.getIndices().size())
      return fail(operation,
                  "local array update requires one scalar index per axis");
    Binding index = bindings.lookup(operation.getIndices()[cursor++]);
    if (index.kind != Binding::Kind::Scalar)
      return fail(operation, "local array update index has no scalar representation");
    auto extent = extentForShape(inputType.getShape()[dimension],
                                 inputType.getAxisIds()[dimension]);
    if (!extent)
      return fail(operation, "local array update axis has no physical extent");
    linear = "((" + linear + ") * (" + *extent + ") + (" + index.scalar + "))";
  }
  line(input.scalar + "[" + linear + "] = " + value.scalar + ";");
  bindings[operation.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileUnary(riscv::UnaryOp operation) {
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type element =
      riscv_internal::logicalElement(operation.getResult().getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto scalarExpression = [&](llvm::StringRef source)
      -> std::optional<std::string> {
    std::string expression;
    if (selected == "scalar.neg")
      expression = "(-(" + source.str() + "))";
    else if (selected == "scalar.abs" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "fabs(" + source.str() + ")"
                                   : "fabsf(" + source.str() + ")";
    else if (selected == "scalar.abs")
      expression = "((" + source.str() + ") < 0 ? -(" + source.str() +
                   ") : (" + source.str() + "))";
    else if (selected == "scalar.exp" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "exp(" + source.str() + ")"
                                   : "expf(" + source.str() + ")";
    else
      return std::nullopt;
    return expression;
  };
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "unary result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : scalarPartCount(operation.getResult());
    Binding result;
    result.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> source;
      if (input->kind == Binding::Kind::Scalar)
        source = input->scalar;
      else if (input->kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input->parts.size())
          source = input->parts[*sourcePart];
      }
      auto expression = source ? scalarExpression(*source) : std::nullopt;
      if (!expression)
        return fail(operation,
                    "selected scalar unary relation or conversion has no C spelling");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(*expression);
      else
        result.parts.push_back(std::move(*expression));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (input->kind != Binding::Kind::Vector)
    return fail(operation, "unary operation has no selected numeric handoff");
  if (selected == "rvv.exp-approx-f32") {
    if (!element.isF32())
      return fail(operation, "selected RVV exp leaf requires f32 elements");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    auto layout = mlir::cast<riscv::ValueType>(operation.getResult().getType())
                      .getLayout();
    const int64_t maskBits = 32 * 8 / layout.getLmulEighths();
    if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
        maskBits != 16 && maskBits != 32 && maskBits != 64)
      return fail(operation, "selected RVV exp leaf has no legal mask ratio");
    const std::string unsignedSuffix =
        "u32" + lmulSpelling(layout.getLmulEighths());
    const std::string unsignedType =
        "vuint32" + lmulSpelling(layout.getLmulEighths()) + "_t";
    const std::string maskType = "vbool" + std::to_string(maskBits) + "_t";
    for (int64_t partIndex = 0;
         partIndex < vectorPartCount(operation.getResult()); ++partIndex) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, partIndex);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "RVV exp relation requires its typed value-use conversion");
      const std::string vl = partVL(operation.getResult(), partIndex);
      const std::string &x = input->parts[*sourcePart];
      std::string r = fresh("exp_r");
      std::string z = fresh("exp_z");
      std::string n = fresh("exp_n");
      std::string b = fresh("exp_b");
      std::string e = fresh("exp_e");
      std::string k = fresh("exp_k");
      std::string c = fresh("exp_c");
      std::string u = fresh("exp_u");
      std::string j = fresh("exp_j");
      std::string out = fresh("exp");
      line(type + " " + r + " = __riscv_vfmv_v_f_" + suffix +
           "(0x1.8p23f, " + vl + ");");
      line(type + " " + z + " = __riscv_vfmacc_vf_" + suffix + "(" + r +
           ", 0x1.715476p+0f, " + x + ", " + vl + ");");
      line(type + " " + n + " = __riscv_vfsub_vv_" + suffix + "(" + z +
           ", " + r + ", " + vl + ");");
      line(type + " " + b + " = __riscv_vfnmsac_vf_" + suffix +
           "(__riscv_vfnmsac_vf_" + suffix + "(" + x +
           ", 0x1.62e4p-1f, " + n + ", " + vl +
           "), 0x1.7f7d1cp-20f, " + n + ", " + vl + ");");
      line(unsignedType + " " + e + " = __riscv_vsll_vx_" + unsignedSuffix +
           "(__riscv_vreinterpret_v_" + suffix + "_" + unsignedSuffix + "(" +
           z + "), 23, " + vl + ");");
      line(type + " " + k + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vadd_vx_" + unsignedSuffix + "(" + e +
           ", 0x3f800000, " + vl + "));" );
      line(maskType + " " + c + " = __riscv_vmfgt_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(__riscv_vfabs_v_" + suffix + "(" + n +
           ", " + vl + "), 126.0f, " + vl + ");");
      line(type + " " + u + " = __riscv_vfmul_vv_" + suffix + "(" + b +
           ", " + b + ", " + vl + ");");
      line(type + " " + j + " = __riscv_vfmacc_vv_" + suffix +
           "(__riscv_vfmul_vf_" + suffix + "(" + b +
           ", 0x1.ffffecp-1f, " + vl + "), __riscv_vfmacc_vv_" + suffix +
           "(__riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.fffdb6p-2f, " + vl + "), 0x1.555e66p-3f, " + b + ", " + vl +
           "), __riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.573e2ep-5f, " + vl + "), 0x1.0e4020p-7f, " + b + ", " + vl +
           "), " + u + ", " + vl + "), " + u + ", " + vl + ");");
      line(type + " " + out + ";");
      line("if (!__riscv_vcpop_m_b" + std::to_string(maskBits) + "(" + c +
           ", " + vl + ")) {");
      ++indent;
      line(out + " = __riscv_vfmacc_vv_" + suffix + "(" + k + ", " + j +
           ", " + k + ", " + vl + ");");
      --indent;
      line("} else {");
      ++indent;
      std::string dm = fresh("exp_dm");
      std::string d = fresh("exp_d");
      std::string s1 = fresh("exp_s1");
      std::string s2 = fresh("exp_s2");
      std::string r1 = fresh("exp_r1");
      line(maskType + " " + dm + " = __riscv_vmfle_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(" + n + ", 0.0f, " + vl + ");");
      line(unsignedType + " " + d + " = __riscv_vmerge_vxm_" +
           unsignedSuffix + "(__riscv_vmv_v_x_" + unsignedSuffix + "(0, " +
           vl + "), 0x82000000, " + dm + ", " + vl + ");");
      line(type + " " + s1 + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vadd_vx_" + unsignedSuffix + "(" + d +
           ", 0x7f000000, " + vl + "));" );
      line(type + " " + s2 + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vsub_vv_" + unsignedSuffix + "(" + e +
           ", " + d + ", " + vl + "));" );
      line(type + " " + r1 + " = __riscv_vmerge_vvm_" + suffix +
           "(__riscv_vfmacc_vv_" + suffix + "(" + k + ", " + k + ", " + j +
           ", " + vl + "), __riscv_vfmul_vv_" + suffix +
           "(__riscv_vfmacc_vv_" + suffix + "(" + s2 + ", " + s2 + ", " + j +
           ", " + vl + "), " + s1 + ", " + vl + "), " + c + ", " + vl +
           ");");
      line(out + " = __riscv_vmerge_vvm_" + suffix + "(" + r1 +
           ", __riscv_vfmul_vv_" + suffix + "(" + s1 + ", " + s1 + ", " +
           vl + "), __riscv_vmfgt_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(__riscv_vfabs_v_" + suffix + "(" + n +
           ", " + vl + "), 192.0f, " + vl + "), " + vl + ");");
      --indent;
      line("}");
      result.parts.push_back(std::move(out));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const bool floating = mlir::isa<mlir::FloatType>(element);
  for (int64_t index = 0; index < vectorPartCount(operation.getResult());
       ++index) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, index);
    if (!sourcePart || *sourcePart >= input->parts.size())
      return fail(operation,
                  "vector unary relation requires its typed value-use conversion");
    const std::string &part = input->parts[*sourcePart];
    std::string expression;
    if (selected == "rvv.vfneg.v" || selected == "rvv.vneg.v")
      expression = "__riscv_" + std::string(floating ? "vfneg_v_" : "vneg_v_") +
                   suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else if (selected == "rvv.vfabs.v" && floating)
      expression = "__riscv_vfabs_v_" + suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else
      return fail(operation,
                  "selected integer vector absolute value is not implemented");
    if (streamPartCount(operation.getResult()) > 1) {
      result.parts.push_back(std::move(expression));
    } else {
      std::string name = fresh("unary");
      line(type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCompare(riscv::CompareOp operation) {
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
  llvm::StringRef predicate = operation.getLeaf().getInstruction();
  auto scalarSpelling = [](llvm::StringRef kind) -> llvm::StringRef {
    return kind == "eq" ? "=="
         : kind == "ne" ? "!="
         : kind == "lt" ? "<"
         : kind == "le" ? "<="
         : kind == "gt" ? ">"
         : kind == "ge" ? ">="
                          : "";
  };
  auto selectedKind = selectedBindingKind(operation.getResult());
  if (!selectedKind)
    return fail(operation, "comparison result has no selected physical carrier");
  if (*selectedKind == Binding::Kind::Scalar ||
      *selectedKind == Binding::Kind::ScalarTuple) {
    if (!predicate.consume_front("scalar.cmp."))
      return fail(operation,
                  "scalar comparison has no selected scalar predicate spelling");
    llvm::StringRef spelling = scalarSpelling(predicate);
    if (spelling.empty())
      return fail(operation, "unknown selected scalar comparison predicate");
    const int64_t parts = *selectedKind == Binding::Kind::Scalar
                              ? 1
                              : registerPartCount(operation.getResult());
    Binding result;
    result.kind = *selectedKind;
    for (int64_t part = 0; part < parts; ++part) {
      auto expressionFor = [&](size_t operand, const Binding &binding)
          -> std::optional<std::string> {
        if (binding.kind == Binding::Kind::Scalar)
          return binding.scalar;
        if (binding.kind != Binding::Kind::ScalarTuple)
          return std::nullopt;
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      };
      auto left = expressionFor(0, lhs);
      auto right = expressionFor(1, rhs);
      if (!left || !right)
        return fail(operation,
                    "scalar comparison operands do not project to the result tuple");
      std::string expression =
          "(" + *left + " " + spelling.str() + " " + *right + ")";
      if (parts == 1)
        result.scalar = std::move(expression);
      else
        result.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (*selectedKind != Binding::Kind::Vector ||
      !predicate.consume_front("rvv.cmp."))
    return fail(operation,
                "comparison leaf disagrees with the selected result carrier");
  if (lhs.kind != Binding::Kind::Vector && rhs.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV comparison requires at least one selected vector operand");
  mlir::Value referenceValue = lhs.kind == Binding::Kind::Vector
                                   ? operation.getLhs()
                                   : operation.getRhs();
  auto referenceLayout = layoutOf(referenceValue);
  if (!referenceLayout || referenceLayout.getCarrier() != "rvv" ||
      referenceLayout.getLmulEighths() <= 0 ||
      (referenceLayout.getSew() * 8) % referenceLayout.getLmulEighths())
    return fail(operation, "RVV comparison has no legal mask ratio");
  const int64_t maskBits =
      referenceLayout.getSew() * 8 / referenceLayout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation, "RVV comparison mask type is outside the ISA domain");
  mlir::Type operandElement =
      riscv_internal::logicalElement(operation.getLhs().getType());
  const bool floating = mlir::isa<mlir::FloatType>(operandElement);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(operandElement);
  if (!floating && (!integer || integer.isSignless()))
    return fail(operation,
                "RVV integer comparison requires explicit signedness");
  const bool reverse = predicate == "gt" || predicate == "ge";
  llvm::StringRef normalized = predicate == "gt" ? "lt"
                              : predicate == "ge" ? "le"
                                                   : predicate;
  std::string stem;
  if (floating)
    stem = normalized == "eq" ? "vmfeq"
         : normalized == "ne" ? "vmfne"
         : normalized == "lt" ? "vmflt"
         : normalized == "le" ? "vmfle"
                                : std::string();
  else
    stem = normalized == "eq" ? "vmseq"
         : normalized == "ne" ? "vmsne"
         : normalized == "lt" ? (integer.isUnsigned() ? "vmsltu" : "vmslt")
         : normalized == "le" ? (integer.isUnsigned() ? "vmsleu" : "vmsle")
                                : std::string();
  if (stem.empty())
    return fail(operation, "RVV comparison predicate has no exact instruction");

  const int64_t resultParts = vectorPartCount(operation.getResult());
  const std::string operandSuffix = vectorSuffix(referenceValue);
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < resultParts; ++part) {
    auto vectorExpression = [&](size_t operand, mlir::Value value,
                                const Binding &binding)
        -> std::optional<std::string> {
      if (binding.kind == Binding::Kind::Vector) {
        if (vectorSuffix(value) != operandSuffix)
          return std::nullopt;
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      }
      std::optional<std::string> scalarValue;
      if (binding.kind == Binding::Kind::Scalar)
        scalarValue = binding.scalar;
      else if (binding.kind == Binding::Kind::ScalarTuple) {
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (projected && *projected < binding.parts.size())
          scalarValue = binding.parts[*projected];
      }
      if (!scalarValue)
        return std::nullopt;
      const std::string vl = partVL(operation.getResult(), part);
      return "__riscv_" +
             std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
             operandSuffix + "(" + *scalarValue + ", " + vl + ")";
    };
    auto left = vectorExpression(0, operation.getLhs(), lhs);
    auto right = vectorExpression(1, operation.getRhs(), rhs);
    if (!left || !right)
      return fail(operation,
                  "RVV comparison operands do not project to one physical part");
    if (reverse)
      std::swap(left, right);
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = "__riscv_" + stem + "_vv_" + operandSuffix + "_b" +
                       std::to_string(maskBits) + "(" + *left + ", " +
                       *right + ", " + vl + ")";
    std::string zero = "__riscv_vmv_v_x_" + resultSuffix + "(0, " + vl + ")";
    std::string expression = "__riscv_vmerge_vxm_" + resultSuffix + "(" +
                             zero + ", 1, " + mask + ", " + vl + ")";
    std::string name = fresh("compare");
    line(resultType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

template <typename ConversionOp>
mlir::LogicalResult Emitter::compileCastImpl(ConversionOp operation) {
  mlir::Operation *rawOperation = operation.getOperation();
  mlir::Value inputValue = operation.getInput();
  mlir::Value resultValue = operation.getResult();
  mlir::FailureOr<Binding> input = materializeNumeric(
      inputValue, bindings.lookup(inputValue));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type source =
      riscv_internal::logicalElement(inputValue.getType());
  mlir::Type target =
      riscv_internal::logicalElement(resultValue.getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto targetCType = scalarCType(target);
  if (!targetCType)
    return fail(operation, "cast target has no intrinsic-C type");
  auto rounding = rawOperation->getAttrOfType<mlir::StringAttr>("rounding");
  auto saturate = rawOperation->getAttrOfType<mlir::BoolAttr>("saturate");
  auto scalarExpression = [&](llvm::StringRef inputExpression)
      -> mlir::FailureOr<std::string> {
    std::string expression = inputExpression.str();
    if (rounding) {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(target);
      if (!integer) {
        if (!mlir::isa<mlir::FloatType>(target) ||
            (saturate && saturate.getValue()))
          return mlir::failure();
        return "((" + *targetCType + ")(" + expression + "))";
      }
      const unsigned width = integer.getWidth();
      if (width == 0 || width >= 64)
        return mlir::failure();
      const bool isUnsigned = integer.isUnsigned();
      const double minimum = isUnsigned ? 0.0 : -static_cast<double>(uint64_t{1} << (width - 1));
      const double maximum = isUnsigned
                                 ? static_cast<double>((uint64_t{1} << width) - 1)
                                 : static_cast<double>((uint64_t{1} << (width - 1)) - 1);
      if (saturate && saturate.getValue())
        expression = "fmax(" + std::to_string(minimum) + ", fmin(" +
                     std::to_string(maximum) + ", " + expression + "))";
      llvm::StringRef roundFunction =
                                      rounding.getValue() == "dynamic" ? "nearbyint"
                                      : rounding.getValue() == "rne" ? "nearbyint"
                                      : rounding.getValue() == "rtz" ? "trunc"
                                      : rounding.getValue() == "rdn" ? "floor"
                                                                        : "ceil";
      expression = roundFunction.str() + "(" + expression + ")";
    }
    return "((" + *targetCType + ")(" + expression + "))";
  };
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "cast result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    if (!selected.starts_with("scalar."))
      return fail(operation,
                  "scalar cast has no exact selected scalar instruction");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : registerPartCount(operation.getResult());
    Binding result;
    result.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> sourceExpression;
      if (input->kind == Binding::Kind::Scalar)
        sourceExpression = input->scalar;
      else if (input->kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input->parts.size())
          sourceExpression = input->parts[*sourcePart];
      }
      if (!sourceExpression)
        return fail(operation,
                    "cast scalar result requires its typed value-use conversion");
      mlir::FailureOr<std::string> expression =
          scalarExpression(*sourceExpression);
      if (mlir::failed(expression))
        return fail(operation,
                    "rounded scalar or tuple narrowing requires an integer target");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(*expression);
      else
        result.parts.push_back(std::move(*expression));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (input->kind != Binding::Kind::Vector)
    return fail(operation, "cast has no selected numeric handoff");
  const bool identity = source == target;
  auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
  auto targetIntegerForReinterpret = mlir::dyn_cast<mlir::IntegerType>(target);
  const bool integerReinterpret =
      sourceInteger && targetIntegerForReinterpret &&
      sourceInteger.getWidth() == targetIntegerForReinterpret.getWidth() &&
      sourceInteger.getSignedness() != targetIntegerForReinterpret.getSignedness();
  if (selected == "rvv.identity" && !rounding && identity) {
    Binding projected;
    projected.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "identity cast requires its typed value-use conversion");
      projected.parts.push_back(input->parts[*sourcePart]);
    }
    bindings[operation.getResult()] = std::move(projected);
    return mlir::success();
  }
  if (selected == "rvv.reinterpret" && !rounding && integerReinterpret) {
    const std::string sourceSuffix = vectorSuffix(operation.getInput());
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "integer reinterpret requires an explicit physical layout conversion");
      std::string value = fresh("reinterpret");
      line(targetType + " " + value + " = __riscv_vreinterpret_v_" +
           sourceSuffix + "_" + targetSuffix + "(" +
           input->parts[*sourcePart] + ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if ((selected == "rvv.fnarrow.f32-f16" ||
       selected == "rvv.fwiden.f16-f32") &&
      (!rounding || (saturate && !saturate.getValue())) &&
      ((source.isF32() && target.isF16()) ||
       (source.isF16() && target.isF32()))) {
    int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
    int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
    if ((source.isF32() && sourceLMUL != targetLMUL * 2) ||
        (source.isF16() && targetLMUL != sourceLMUL * 2))
      return fail(operation,
                  "selected f16/f32 cast LMUL relation cannot be emitted mechanically");
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "float cast requires an explicit physical layout conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = fresh(source.isF32() ? "narrow_f16" : "widen_f32");
      std::string intrinsic = source.isF32() ? "__riscv_vfncvt_f_f_w_"
                                             : "__riscv_vfwcvt_f_f_v_";
      line(targetType + " " + value + " = " + intrinsic + targetSuffix +
           "(" + input->parts[*sourcePart] + ", " + vl + ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.narrow.int.vf2") {
    auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
    auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
    if (!sourceInteger || !targetInteger ||
        sourceInteger.getWidth() != targetInteger.getWidth() * 2 ||
        targetInteger.getWidth() < 8 || (saturate && saturate.getValue()))
      return fail(operation,
                  "selected integer narrowing has incompatible typed operands");
    const int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
    const int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
    if (sourceLMUL != targetLMUL * 2)
      return fail(operation,
                  "selected integer narrowing has an incompatible LMUL relation");
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "integer narrowing requires its typed value-use conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = fresh("narrow_int");
      line(targetType + " " + value + " = __riscv_vncvt_x_x_w_" +
           targetSuffix + "(" + input->parts[*sourcePart] + ", " + vl +
           ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const bool saturatedF32I8 = selected.starts_with("rvv.f32-i8-saturate.");
  const bool nonsaturatingF32I8 =
      selected.starts_with("rvv.f32-i8-nonsaturating.");
  if ((!saturatedF32I8 && !nonsaturatingF32I8) || !rounding || !saturate ||
      saturate.getValue() != saturatedF32I8 || !source.isF32())
    return fail(operation,
                "current vector cast implements explicit saturated f32 narrowing only");
  auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
  if (!targetInteger || targetInteger.getWidth() != 8)
    return fail(operation,
                "current vector narrowing implements an eight-bit integer target");
  int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
  int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  if (sourceLMUL < 4 || targetLMUL * 4 != sourceLMUL)
    return fail(operation,
                "selected f32-to-i8 LMUL relation cannot be narrowed mechanically");
  const std::string i16Suffix =
      std::string(targetInteger.isUnsigned() ? "u16" : "i16") +
      lmul(sourceLMUL / 2);
  const std::string i16Type =
      std::string(targetInteger.isUnsigned() ? "vuint16" : "vint16") +
      lmul(sourceLMUL / 2) + "_t";
  const std::string i32Suffix =
      std::string(targetInteger.isUnsigned() ? "u32" : "i32") +
      lmul(sourceLMUL);
  const std::string i32Type =
      std::string(targetInteger.isUnsigned() ? "vuint32" : "vint32") +
      lmul(sourceLMUL) + "_t";
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  const std::string targetType = vectorType(operation.getResult());
  llvm::StringRef frm = rounding.getValue() == "rne" ? "__RISCV_FRM_RNE"
                       : rounding.getValue() == "rtz" ? "__RISCV_FRM_RTZ"
                       : rounding.getValue() == "rdn" ? "__RISCV_FRM_RDN"
                       : rounding.getValue() == "rup" ? "__RISCV_FRM_RUP"
                                                       : llvm::StringRef();
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, part);
    if (!sourcePart || *sourcePart >= input->parts.size())
      return fail(operation,
                  "rounded narrowing requires an explicit physical layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    std::string i16Value = fresh("narrow_i16");
    if (saturatedF32I8) {
      std::string i32Value = fresh("narrow_i32");
      const std::string conversion =
          "__riscv_vfcvt_" +
          std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_v_" +
          i32Suffix;
      line(i32Type + " " + i32Value + " = " + conversion +
           (frm.empty() ? "(" + input->parts[*sourcePart] + ", " + vl + ");"
                        : "_rm(" + input->parts[*sourcePart] + ", " + frm.str() +
                              ", " + vl + ");"));
      line(i16Type + " " + i16Value + " = __riscv_" +
           std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
           "_wx_" + i16Suffix + "(" + i32Value +
           ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    } else {
      const std::string conversion =
          "__riscv_vfncvt_" +
          std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_w_" +
          i16Suffix;
      line(i16Type + " " + i16Value + " = " + conversion +
           (frm.empty() ? "(" + input->parts[*sourcePart] + ", " + vl + ");"
                        : "_rm(" + input->parts[*sourcePart] + ", " + frm.str() +
                              ", " + vl + ");"));
    }
    std::string i8Value = fresh("narrow_i8");
    if (saturatedF32I8)
      line(targetType + " " + i8Value + " = __riscv_" +
           std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
           "_wx_" + targetSuffix + "(" + i16Value +
           ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    else
      line(targetType + " " + i8Value + " = __riscv_vncvt_x_x_w_" +
           targetSuffix + "(" + i16Value + ", " + vl + ");");
    result.parts.push_back(std::move(i8Value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCast(riscv::CastOp operation) {
  return compileCastImpl(operation);
}

mlir::LogicalResult Emitter::compileNarrow(riscv::NarrowOp operation) {
  return compileCastImpl(operation);
}

mlir::LogicalResult Emitter::compileWiden(riscv::WidenOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getInput(), std::move(input));
  if (mlir::failed(materialized))
    return mlir::failure();
  input = std::move(*materialized);
  mlir::Type source =
      riscv_internal::logicalElement(operation.getInput().getType());
  mlir::Type target =
      riscv_internal::logicalElement(operation.getResult().getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto targetCType = scalarCType(target);
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "widen result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    if (selected != "scalar.widen")
      return fail(operation,
                  "scalar widening has no exact selected scalar instruction");
    if (!targetCType)
      return fail(operation, "scalar or tuple widening target has no C type");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : scalarPartCount(operation.getResult());
    Binding tuple;
    tuple.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> sourceExpression;
      if (input.kind == Binding::Kind::Scalar)
        sourceExpression = input.scalar;
      else if (input.kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input.parts.size())
          sourceExpression = input.parts[*sourcePart];
      }
      if (!sourceExpression)
        return fail(operation,
                    "scalar or tuple widening requires its typed value-use conversion");
      std::string expression = "((" + *targetCType + ")(" +
                               *sourceExpression + "))";
      if (tuple.kind == Binding::Kind::Scalar)
        tuple.scalar = std::move(expression);
      else
        tuple.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(tuple);
    return mlir::success();
  }
  if (input.kind != Binding::Kind::Vector)
    return fail(operation, "widen requires a selected vector input");
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string targetType = vectorType(operation.getResult());
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, part);
    if (!sourcePart || *sourcePart >= input.parts.size())
      return fail(operation,
                  "widen requires an explicit physical layout conversion");
    const std::string &sourceExpression = input.parts[*sourcePart];
    const std::string vl = partVL(operation.getResult(), part);
    std::string expression;
    if (selected == "rvv.fwiden.f16-f32" && source.isF16() && target.isF32())
      expression = "__riscv_vfwcvt_f_f_v_" + targetSuffix + "(" +
                   sourceExpression + ", " + vl + ")";
    else if (selected == "rvv.convert.i32-f32" && source.isInteger(32) &&
             target.isF32()) {
      auto integer = mlir::cast<mlir::IntegerType>(source);
      expression = std::string(integer.isSigned() ? "__riscv_vfcvt_f_x_v_"
                                                   : "__riscv_vfcvt_f_xu_v_") +
                   targetSuffix + "(" + sourceExpression + ", " + vl + ")";
    }
    else if (selected == "rvv.identity")
      expression = sourceExpression;
    else if (selected == "rvv.reinterpret") {
      const std::string sourceSuffix = vectorSuffix(operation.getInput());
      expression = "__riscv_vreinterpret_v_" + sourceSuffix + "_" +
                   targetSuffix + "(" + sourceExpression + ")";
    }
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(source)) {
      auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
      unsigned sourceWidth = std::max(8u, integer.getWidth());
      unsigned targetWidth = targetInteger ? targetInteger.getWidth() : 0;
      if (!targetInteger || targetWidth <= sourceWidth ||
          targetWidth % sourceWidth)
        return fail(operation, "unsupported integer widening width relation");
      unsigned factor = targetWidth / sourceWidth;
      if (factor != 2 && factor != 4 && factor != 8)
        return fail(operation, "unsupported RVV integer widening factor");
      std::string unsignedSuffix = targetSuffix;
      unsignedSuffix[0] = 'u';
      const std::string expected =
          std::string(integer.isSigned() ? "rvv.sext.vf" : "rvv.zext.vf") +
          std::to_string(factor);
      if (selected != expected)
        return fail(operation,
                    "selected integer widening leaf disagrees with width/sign");
      if (integer.isSigned()) {
        expression = "__riscv_vsext_vf" + std::to_string(factor) + "_" +
                     targetSuffix + "(" + sourceExpression + ", " + vl +
                     ")";
      } else {
        std::string widened =
            "__riscv_vzext_vf" + std::to_string(factor) + "_" + unsignedSuffix +
            "(" + sourceExpression + ", " + vl + ")";
        expression = targetSuffix[0] == 'i'
                         ? "__riscv_vreinterpret_v_" + unsignedSuffix + "_" +
                               targetSuffix + "(" + widened + ")"
                         : widened;
      }
    } else {
      return fail(operation, "unsupported vector widening relation");
    }
    std::string name = fresh("widen");
    line(targetType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReduce(riscv::ReduceOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  int64_t eliminatedAxis =
      inputType && operation.getAxis() >= 0 &&
              operation.getAxis() < static_cast<int64_t>(inputType.getAxisIds().size())
          ? inputType.getAxisIds()[operation.getAxis()]
          : 0;
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  llvm::StringRef selectedKind = selected;
  const bool selectedRegister =
      selectedKind.consume_front("rvv.register-reduce.");
  if (!selectedRegister &&
      !selectedKind.consume_front("rvv.lane-reduce."))
    return fail(operation, "reduction has no exact selected RVV reduction form");
  const bool registerAxis =
      llvm::is_contained(registerAxesFor(operation.getInput()), eliminatedAxis);
  if (selectedRegister != registerAxis)
    return fail(operation,
                "selected reduction form disagrees with the input axis mapping");
  if (selectedRegister) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    if (materialized->kind != Binding::Kind::Vector)
      return fail(operation,
                  "register-axis reduction requires a selected vector input");
    llvm::SmallVector<int64_t, 4> inputAxes =
        registerAxesFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> inputExtents =
        registerExtentsFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> resultAxes =
        registerAxesFor(operation.getResult());
    auto eliminatedPosition = llvm::find(inputAxes, eliminatedAxis);
    if (inputAxes.size() != inputExtents.size() ||
        eliminatedPosition == inputAxes.end())
      return fail(operation,
                  "register-axis reduction input mapping is incomplete");
    const size_t eliminatedOrdinal = eliminatedPosition - inputAxes.begin();
    const int64_t eliminatedExtent = inputExtents[eliminatedOrdinal];
    const int64_t inputStreams = streamPartCount(operation.getInput());
    const int64_t resultStreams = streamPartCount(operation.getResult());
    if (eliminatedExtent <= 0 || inputStreams <= 0 ||
        inputStreams != resultStreams ||
        laneAxisFor(operation.getInput()) != laneAxisFor(operation.getResult()))
      return fail(operation,
                  "register-axis reduction does not preserve its SIMD mapping");

    mlir::Type element =
        riscv_internal::logicalElement(operation.getInput().getType());
    const bool floating = mlir::isa<mlir::FloatType>(element);
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    llvm::StringRef kind = selectedKind;
    std::string instruction;
    if (kind == "add")
      instruction = floating ? "vfadd" : "vadd";
    else if ((kind == "max" || kind == "min") && floating)
      instruction = kind == "max" ? "vfmax" : "vfmin";
    else
      return fail(operation,
                  "register-axis reduction implements add and floating max/min");
    if (!floating && !integer)
      return fail(operation,
                  "register-axis reduction has no numeric vector element type");

    const std::string type = vectorType(operation.getResult());
    const std::string suffix = vectorSuffix(operation.getResult());
    const int64_t resultRegisters = registerPartCount(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto resultCoordinates =
          registerCoordinates(operation.getResult(), resultRegister);
      if (!resultCoordinates || resultCoordinates->size() != resultAxes.size())
        return fail(operation,
                    "register-axis reduction result mapping is incomplete");
      for (int64_t stream = 0; stream < resultStreams; ++stream) {
        std::string accumulator;
        for (int64_t reduced = 0; reduced < eliminatedExtent; ++reduced) {
          int64_t inputRegister = 0;
          for (auto [axis, extent] : llvm::zip(inputAxes, inputExtents)) {
            if (extent <= 0)
              return fail(operation,
                          "register-axis reduction has an invalid input extent");
            int64_t coordinate = reduced;
            if (axis != eliminatedAxis) {
              auto found = llvm::find(resultAxes, axis);
              if (found == resultAxes.end())
                return fail(operation,
                            "register-axis reduction cannot project a free axis");
              coordinate =
                  (*resultCoordinates)[found - resultAxes.begin()];
            }
            if (coordinate < 0 || coordinate >= extent)
              return fail(operation,
                          "register-axis reduction coordinate is out of range");
            inputRegister = inputRegister * extent + coordinate;
          }
          const int64_t inputPart = inputRegister * inputStreams + stream;
          if (inputPart < 0 ||
              inputPart >= static_cast<int64_t>(materialized->parts.size()))
            return fail(operation,
                        "register-axis reduction source mapping is incomplete");
          if (accumulator.empty()) {
            accumulator = fresh("register_reduce");
            line(type + " " + accumulator + " = " +
                 materialized->parts[inputPart] + ";");
            continue;
          }
          const int64_t resultPart = resultRegister * resultStreams + stream;
          line(accumulator + " = __riscv_" + instruction + "_vv_" + suffix +
               "(" + accumulator + ", " + materialized->parts[inputPart] +
               ", " + partVL(operation.getResult(), resultPart) + ");");
        }
        result.parts.push_back(std::move(accumulator));
      }
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getInput(), std::move(input));
  if (mlir::failed(materialized))
    return mlir::failure();
  if (materialized->kind != Binding::Kind::Vector)
    return fail(operation, "wide reduce requires a selected vector input");
  mlir::Type element =
      riscv_internal::logicalElement(operation.getInput().getType());
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "reduce element has no intrinsic-C scalar type");
  const std::string sourceSuffix = vectorSuffix(operation.getInput());
  unsigned sew = riscv_internal::logicalBitWidth(operation.getInput().getType());
  const bool floating = mlir::isa<mlir::FloatType>(element);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  const bool unsignedInteger = integer && integer.isUnsigned();
  const std::string baseSuffix =
      std::string(floating ? "f" : unsignedInteger ? "u" : "i") +
      std::to_string(sew) + "m1";
  const std::string baseType =
      std::string(floating ? "vfloat" : unsignedInteger ? "vuint" : "vint") +
      std::to_string(sew) + "m1_t";
  llvm::StringRef kind = selectedKind;
  std::string initial;
  if (kind == "add")
    initial = "0";
  else if (kind == "max" && floating)
    initial = "-INFINITY";
  else if (kind == "min" && floating)
    initial = "INFINITY";
  else
    return fail(operation,
                "current wide reduce implements add and floating max/min");
  const int64_t resultParts = registerPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getInput());
  if (resultParts <= 0 || streams <= 0)
    return fail(operation,
                "reduce input/result mappings do not form an eliminated-axis slice");
  Binding result;
  result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                 : Binding::Kind::ScalarTuple;
  const bool combineFullStreams =
      streams > 1 && inputType.getLayout().getValidity() == "full";
  std::string horizontalStem;
  if (floating)
    horizontalStem = kind == "add"   ? "vfredusum"
                     : kind == "max" ? "vfredmax"
                                       : "vfredmin";
  else
    horizontalStem = "vredsum";
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    auto sourceRegister = projectRegisterPart(
        operation.getInput(), operation.getResult(), resultPart);
    if (!sourceRegister)
      return fail(operation,
                  "reduce free-axis mapping is not projectable");
    std::string accumulator = fresh("reduce_scalar");
    line(*scalarType + " " + accumulator + " = (" + *scalarType + ")(" +
         initial + ");");
    auto emitHorizontal = [&](llvm::StringRef source, size_t index) {
      const std::string seed = fresh("reduce_seed");
      line(baseType + " " + seed + " = __riscv_" +
           std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + baseSuffix +
           "(" + accumulator + ", 1);");
      const std::string reduced = fresh("reduce_vector");
      line(baseType + " " + reduced + " = __riscv_" + horizontalStem +
           "_vs_" + sourceSuffix + "_" + baseSuffix + "(" + source.str() +
           ", " + seed + ", " + partVL(operation.getInput(), index) + ");");
      if (floating)
        line(accumulator + " = __riscv_vfmv_f_s_" + baseSuffix + "_f" +
             std::to_string(sew) + "(" + reduced + ");");
      else
        line(accumulator + " = __riscv_vmv_x_s_" + baseSuffix + "_" +
             std::string(unsignedInteger ? "u" : "i") +
             std::to_string(sew) + "(" + reduced + ");");
    };
    if (combineFullStreams) {
      const size_t first = *sourceRegister * streams;
      if (first + streams > materialized->parts.size())
        return fail(operation, "reduce source mapping is incomplete");
      const std::string vectorTypeName = vectorType(operation.getInput());
      const std::string combineStem =
          floating ? (kind == "add" ? "vfadd" : kind == "max" ? "vfmax"
                                                              : "vfmin")
                   : "vadd";
      std::string combined = fresh("reduce_streams");
      line(vectorTypeName + " " + combined + " = " +
           materialized->parts[first] + ";");
      for (int64_t stream = 1; stream < streams; ++stream)
        line(combined + " = __riscv_" + combineStem + "_vv_" + sourceSuffix +
             "(" + combined + ", " + materialized->parts[first + stream] +
             ", " + std::to_string(physicalLanes(operation.getInput())) +
             ");");
      emitHorizontal(combined, first);
    } else for (int64_t stream = 0; stream < streams; ++stream) {
      const size_t index = *sourceRegister * streams + stream;
      if (index >= materialized->parts.size())
        return fail(operation, "reduce source mapping is incomplete");
      emitHorizontal(materialized->parts[index], index);
    }
    if (resultParts == 1)
      result.scalar = accumulator;
    else
      result.parts.push_back(std::move(accumulator));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFold2(riscv::Fold2Op operation) {
  if (operation.getLeaf().getInstruction() != "rvv.pair-fold-add")
    return fail(operation, "fold2 has no exact selected RVV pair-fold leaf");
  Binding fieldBinding = bindings.lookup(operation.getInput());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation,
                "pair-fold leaf requires one explicit encoded-field operand");
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto input = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto inputInteger = input
                          ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
                          : mlir::IntegerType();
  auto resultInteger = result
                           ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                           : mlir::IntegerType();
  auto resultLayout = result ? result.getLayout() : riscv::LayoutAttr();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || field->access != operation.getAccess() ||
      operation.getAccess().getMapping() != "natural" ||
      operation.getAccess().getBitOffset() % 8 ||
      !inputInteger || inputInteger.getWidth() != 16 ||
      !inputInteger.isSigned() || !resultInteger ||
      resultInteger.getWidth() != 32 || !resultInteger.isSigned() ||
      !resultLayout || resultLayout.getCarrier() != "rvv" ||
      resultLayout.getLmulEighths() <= 0 ||
      resultLayout.getLmulEighths() % 2)
    return fail(operation,
                "pair-fold leaf has no closed natural i16-to-i32 RVV representation");

  const int64_t inputLMUL = resultLayout.getLmulEighths() / 2;
  const std::string inputSuffix = "i16" + lmulSpelling(inputLMUL);
  const std::string inputType = "vint16" + lmulSpelling(inputLMUL) + "_t";
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  Binding folded;
  folded.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates ||
        coordinates->size() != registerAxes.size())
      return fail(operation,
                  "pair-fold result has no complete register coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates))
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis)
          record += " + " + std::to_string(coordinate) + " * " + stride;
    record += " + " + std::to_string(field->bitOffset / 8) + ")";
    const std::string outputOffset = partOffset(operation.getResult(), part);
    const std::string evenAddress =
        record + " + 4 * (" + outputOffset + ")";
    const std::string oddAddress = evenAddress + " + 2";
    const std::string vl = partVL(operation.getResult(), part);
    std::string even = fresh("pair_even");
    std::string odd = fresh("pair_odd");
    line(inputType + " " + even + " = __riscv_vlse16_v_" + inputSuffix +
         "((const int16_t *)(" + evenAddress + "), (ptrdiff_t)4, " + vl +
         ");");
    line(inputType + " " + odd + " = __riscv_vlse16_v_" + inputSuffix +
         "((const int16_t *)(" + oddAddress + "), (ptrdiff_t)4, " + vl +
         ");");
    std::string sum = fresh("pair_fold");
    line(resultType + " " + sum + " = __riscv_vwadd_vv_" + resultSuffix +
         "(" + even + ", " + odd + ", " + vl + ");");
    folded.parts.push_back(std::move(sum));
  }
  bindings[operation.getResult()] = std::move(folded);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLookup(riscv::LookupOp operation) {
  llvm::StringRef realization = instructionOf(operation.getOperation());
  Binding indices = bindings.lookup(operation.getIndices());
  mlir::FailureOr<Binding> materializedIndices =
      materializeNumeric(operation.getIndices(), indices);
  if (mlir::failed(materializedIndices))
    return mlir::failure();
  indices = std::move(*materializedIndices);

  if (realization == "rvv.vrgather") {
    Binding table = bindings.lookup(operation.getTable());
    mlir::FailureOr<Binding> materializedTable =
        materializeNumeric(operation.getTable(), table);
    if (mlir::failed(materializedTable))
      return mlir::failure();
    table = std::move(*materializedTable);
    if (table.kind != Binding::Kind::Vector || table.parts.size() != 1 ||
        indices.kind != Binding::Kind::Vector)
      return fail(operation,
                  "RVV register lookup requires one complete table vector and vector indices");
    const int64_t parts = vectorPartCount(operation.getResult());
    const std::string resultSuffix = vectorSuffix(operation.getResult());
    const std::string resultType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < parts; ++part) {
      auto indexPart = mappedPart(operation.getOperation(), 1, part);
      if (!indexPart || *indexPart >= indices.parts.size())
        return fail(operation,
                    "register lookup index mapping requires an explicit layout conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string expression = "__riscv_vrgather_vv_" + resultSuffix + "(" +
                               table.parts.front() + ", " +
                               indices.parts[*indexPart] + ", " + vl + ")";
      std::string name = fresh("lookup");
      line(resultType + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }

  Binding table = bindings.lookup(operation.getTable());
  if (table.kind != Binding::Kind::Slice && table.kind != Binding::Kind::Memory)
    return fail(operation,
                "selected lookup requires one explicit dense memory descriptor");
  std::optional<std::string> address;
  if (table.kind == Binding::Kind::Slice)
    address = denseAddress(table.slice, {});
  else
    address = table.memory.name;
  if (!address)
    return fail(operation, "selected lookup table has no dense base address");

  mlir::Type tableElement =
      riscv_internal::logicalElement(operation.getTable().getType());
  if (auto encoding = mlir::dyn_cast<kernel::EncodingType>(tableElement);
      encoding && encoding.getKind() == "dense")
    tableElement = riscv_internal::logicalElement(operation.getResult().getType());
  auto tableCType = scalarCType(tableElement);
  const unsigned tableBits = riscv_internal::logicalBitWidth(tableElement);
  if (!tableCType || !tableBits || tableBits % 8)
    return fail(operation,
                "selected lookup requires a byte-addressable numeric table");
  const unsigned tableBytes = tableBits / 8;
  const std::string base = "((const " + *tableCType + " *)(" + *address + "))";

  if (realization == "scalar.lookup") {
    if (indices.kind == Binding::Kind::Scalar) {
      std::string loaded = fresh("lookup_scalar");
      line(*tableCType + " " + loaded + " = " + base + "[(size_t)(" +
           indices.scalar + ")];");
      bindings[operation.getResult()] = scalar(std::move(loaded));
      return mlir::success();
    }
    const int64_t resultParts = registerPartCount(operation.getResult());
    if (indices.kind != Binding::Kind::ScalarTuple || resultParts <= 1 ||
        static_cast<int64_t>(indices.parts.size()) != resultParts)
      return fail(operation,
                  "scalar lookup requires scalar or replica-tuple unsigned indices");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    for (llvm::StringRef part : indices.parts) {
      std::string loaded = fresh("lookup_scalar");
      line(*tableCType + " " + loaded + " = " + base + "[(size_t)(" +
           part.str() + ")];");
      result.parts.push_back(std::move(loaded));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (realization != "rvv.vluxei")
    return fail(operation, "lookup has an unknown selected realization");
  if (indices.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV indexed lookup requires vector unsigned indices");

  const int64_t parts = vectorPartCount(operation.getResult());
  auto indexLayout = layoutOf(operation.getIndices());
  if (!indexLayout)
    return fail(operation,
                "RVV indexed lookup has no selected index SEW/LMUL");
  const std::string indexSuffix =
      "u" + std::to_string(indexLayout.getSew()) +
      lmulSpelling(indexLayout.getLmulEighths());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto indexPart = mappedPart(operation.getOperation(), 1, part);
    if (!indexPart || *indexPart >= indices.parts.size())
      return fail(operation,
                  "lookup index mapping requires an explicit layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    std::string byteOffsets = indices.parts[*indexPart];
    if (tableBytes != 1)
      byteOffsets = "__riscv_vmul_vx_" + indexSuffix + "(" + byteOffsets +
                    ", " + std::to_string(tableBytes) + ", " + vl + ")";
    std::string expression =
        "__riscv_vluxei" + std::to_string(indexLayout.getSew()) + "_v_" +
        resultSuffix + "(" + base + ", " + byteOffsets + ", " + vl + ")";
    std::string name = fresh("lookup");
    line(resultType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileBinary(riscv::BinaryOp operation) {
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  const bool lhsLaneToRegister = false;
  const bool rhsLaneToRegister = false;
  if (lhs.kind == Binding::Kind::Scalar && rhs.kind == Binding::Kind::Scalar) {
    llvm::StringRef kind = selected;
    if (!kind.consume_front("scalar."))
      return fail(operation,
                  "scalar binary has no exact selected scalar instruction");
    llvm::StringRef spelling = kind == "add" ? "+"
                              : kind == "sub" ? "-"
                              : kind == "mul" ? "*"
                              : kind == "div" ? "/"
                              : kind == "mod" ? "%"
                              : kind == "and" ? "&"
                              : kind == "or"  ? "|"
                              : kind == "xor" ? "^"
                              : kind == "shl" ? "<<"
                              : kind == "shr" ? ">>"
                                                : "";
    if (!spelling.empty()) {
      auto resultType = scalarCType(
          riscv_internal::logicalElement(operation.getResult().getType()));
      if (!resultType)
        return fail(operation,
                    "scalar binary result has no exact intrinsic-C type");
      bindings[operation.getResult()] = scalar(
          "((" + *resultType + ")(" + lhs.scalar + " " + spelling.str() +
          " " + rhs.scalar + "))");
      return mlir::success();
    }
    if (kind == "max" || kind == "min") {
      bindings[operation.getResult()] = scalar(
          "((" + lhs.scalar + ") " + (kind == "max" ? ">" : "<") + " (" +
          rhs.scalar + ") ? (" + lhs.scalar + ") : (" + rhs.scalar + "))");
      return mlir::success();
    }
    return fail(operation, "unsupported selected scalar pointwise binary kind");
  }
  auto scalarTupleCompatible = [](const Binding &binding,
                                  bool laneToRegister) {
    return binding.kind == Binding::Kind::Scalar ||
           binding.kind == Binding::Kind::ScalarTuple ||
           (laneToRegister && binding.kind == Binding::Kind::Vector);
  };
  if (scalarTupleCompatible(lhs, lhsLaneToRegister) &&
      scalarTupleCompatible(rhs, rhsLaneToRegister)) {
    llvm::StringRef kind = selected;
    if (!kind.consume_front("scalar."))
      return fail(operation,
                  "scalar-tuple binary has no exact selected scalar instruction");
    llvm::StringRef spelling = kind == "add" ? "+"
                              : kind == "sub" ? "-"
                              : kind == "mul" ? "*"
                              : kind == "div" ? "/"
                              : kind == "mod" ? "%"
                              : kind == "and" ? "&"
                              : kind == "or"  ? "|"
                              : kind == "xor" ? "^"
                              : kind == "shl" ? "<<"
                              : kind == "shr" ? ">>"
                                                : "";
    if (spelling.empty() && kind != "max" && kind != "min")
      return fail(operation,
                  "unsupported selected scalar-tuple pointwise binary kind");
    const int64_t parts = registerPartCount(operation.getResult());
    auto scalarResultType = scalarCType(
        riscv_internal::logicalElement(operation.getResult().getType()));
    if (parts > 1 && !scalarResultType)
      return fail(operation,
                  "scalar-tuple binary result has no exact intrinsic-C type");
    Binding result;
    result.kind = parts == 1 ? Binding::Kind::Scalar
                             : Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part) {
      auto expressionFor = [&](size_t operand, mlir::Value value,
                               const Binding &binding, bool laneToRegister)
          -> std::optional<std::string> {
        if (binding.kind == Binding::Kind::Scalar)
          return binding.scalar;
        if (laneToRegister)
          return extractLaneForRegisterBroadcast(value, operation.getResult(),
                                                 part, binding);
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      };
      auto lhsExpression =
          expressionFor(0, operation.getLhs(), lhs, lhsLaneToRegister);
      auto rhsExpression =
          expressionFor(1, operation.getRhs(), rhs, rhsLaneToRegister);
      if (!lhsExpression || !rhsExpression)
        return fail(operation,
                    "scalar-tuple operand projection is not representable");
      std::string expression;
      if (!spelling.empty())
        expression = "(" + *lhsExpression + " " + spelling.str() + " " +
                     *rhsExpression + ")";
      else
        expression = "((" + *lhsExpression + ") " +
                     std::string(kind == "max" ? ">" : "<") + " (" +
                     *rhsExpression + ") ? (" + *lhsExpression + ") : (" +
                     *rhsExpression + "))";
      if (parts == 1)
        result.scalar = std::move(expression);
      else {
        // A physical scalar tuple is SSA state, not a macro expression.  Name
        // every selected replica once so consumers in descendant Levels reuse
        // the value instead of duplicating its encoded loads and decode chain.
        std::string name = fresh("scalar_tuple");
        line(*scalarResultType + " " + name + " = " + expression + ";");
        result.parts.push_back(std::move(name));
      }
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (lhsLaneToRegister && rhsLaneToRegister)
    return fail(operation,
                "pointwise operation cannot convert both lane operands to register broadcasts");
  const bool lhsIsVectorOperand =
      !lhsLaneToRegister && lhs.kind == Binding::Kind::Vector;
  const Binding &vector = lhsIsVectorOperand ? lhs : rhs;
  const Binding &other = lhsIsVectorOperand ? rhs : lhs;
  mlir::Value vectorValue = lhsIsVectorOperand ? operation.getLhs()
                                               : operation.getRhs();
  mlir::Value otherValue = lhsIsVectorOperand ? operation.getRhs()
                                              : operation.getLhs();
  const bool otherLaneToRegister =
      lhsIsVectorOperand ? rhsLaneToRegister : lhsLaneToRegister;
  if (vector.kind != Binding::Kind::Vector ||
      (other.kind != Binding::Kind::Vector &&
       other.kind != Binding::Kind::Scalar &&
       other.kind != Binding::Kind::ScalarTuple))
    return fail(operation, "pointwise binary has no selected vector/scalar handoff");
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string type = vectorType(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  bool floating = suffix.front() == 'f';
  llvm::StringRef leaf = selected;
  if (!leaf.consume_front("rvv."))
    return fail(operation, "vector binary has no exact selected RVV instruction");
  size_t separator = leaf.find('.');
  if (separator == llvm::StringRef::npos)
    return fail(operation, "selected RVV binary instruction has no operand form");
  std::string stem = leaf.take_front(separator).str();
  llvm::StringRef form = leaf.drop_front(separator + 1);
  const bool leafVV = form == "vv";
  const bool leafSplatLhs = form == "vv.splat-lhs";
  const bool leafScalar = form == (floating ? "vf" : "vx") ||
                          form == (floating ? "vf.swap" : "vx.swap");
  const bool leafSwap = form.ends_with(".swap");
  const int64_t resultParts = vectorPartCount(operation.getResult());
  auto compatibleParts = [&](size_t operand, mlir::Value value,
                             const Binding &binding, bool laneToRegister) {
    if (binding.kind == Binding::Kind::Scalar)
      return true;
    if (laneToRegister) {
      if (binding.kind != Binding::Kind::Vector)
        return false;
      for (int64_t part = 0; part < resultParts; ++part)
        if (!extractLaneForRegisterBroadcast(value, operation.getResult(), part,
                                             binding))
          return false;
      return true;
    }
    if (binding.kind != Binding::Kind::Vector &&
        binding.kind != Binding::Kind::ScalarTuple)
      return false;
    for (int64_t part = 0; part < resultParts; ++part) {
      auto projected = mappedPart(operation.getOperation(), operand, part);
      if (!projected || *projected >= binding.parts.size())
        return false;
    }
    return true;
  };
  if (resultParts <= 0 ||
      !compatibleParts(0, operation.getLhs(), lhs, lhsLaneToRegister) ||
      !compatibleParts(1, operation.getRhs(), rhs, rhsLaneToRegister))
    return fail(operation,
                "pointwise operand mappings cannot broadcast to the selected result mapping");
  for (int64_t index = 0; index < resultParts; ++index) {
    std::string expression;
    if (otherLaneToRegister || other.kind == Binding::Kind::Scalar ||
        other.kind == Binding::Kind::ScalarTuple) {
      const size_t vectorOperand = lhsIsVectorOperand ? 0 : 1;
      const size_t otherOperand = lhsIsVectorOperand ? 1 : 0;
      auto vectorPartIndex =
          mappedPart(operation.getOperation(), vectorOperand, index);
      auto scalarPartIndex =
          otherLaneToRegister || other.kind == Binding::Kind::Scalar
              ? std::optional<size_t>(0)
              : mappedPart(operation.getOperation(), otherOperand, index);
      auto convertedScalar =
          otherLaneToRegister
              ? extractLaneForRegisterBroadcast(otherValue, operation.getResult(),
                                                index, other)
              : std::optional<std::string>();
      if (!vectorPartIndex || !scalarPartIndex ||
          (otherLaneToRegister && !convertedScalar))
        return fail(operation,
                    "pointwise operand projection is not representable");
      std::string vectorPart = vector.parts[*vectorPartIndex];
      const std::string scalarPart =
          otherLaneToRegister
              ? *convertedScalar
          : other.kind == Binding::Kind::Scalar
              ? other.scalar
              : other.parts[*scalarPartIndex];
      bool scalarOnLeft = !lhsIsVectorOperand;
      if (leafSplatLhs) {
        if (!scalarOnLeft)
          return fail(operation,
                      "selected scalar-left splat form disagrees with operands");
        std::string splat = "__riscv_" +
                            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
                            suffix + "(" + scalarPart + ", " +
                            partVL(operation.getResult(), index) + ")";
        expression = "__riscv_" + stem + "_vv_" + suffix + "(" + splat +
                     ", " + vectorPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      } else {
        if (!leafScalar || leafSwap != scalarOnLeft)
          return fail(operation,
                      "selected vector-scalar form disagrees with operands");
        expression = "__riscv_" + stem + (floating ? "_vf_" : "_vx_") +
                     suffix + "(" + vectorPart + ", " + scalarPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      }
    } else {
      if (!leafVV)
        return fail(operation,
                    "selected vector-vector form disagrees with operands");
      auto lhsPartIndex = mappedPart(operation.getOperation(), 0, index);
      auto rhsPartIndex = mappedPart(operation.getOperation(), 1, index);
      if (!lhsPartIndex || !rhsPartIndex)
        return fail(operation,
                    "pointwise vector operand projection is not representable");
      const std::string &lhsPart = lhs.parts[*lhsPartIndex];
      const std::string &rhsPart = rhs.parts[*rhsPartIndex];
      expression = "__riscv_" + stem + "_vv_" + suffix + "(" + lhsPart +
                   ", " + rhsPart + ", " +
                   partVL(operation.getResult(), index) + ")";
    }
    // Preserve the physical SSA operation as one C value.  Inlining the
    // expression into each downstream use would duplicate loop-invariant
    // coordinate work after LICM and would no longer spell the selected
    // physical program mechanically.
    std::string name = fresh("pointwise");
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

std::optional<EncodingField>
Emitter::fieldFor(const Binding &fieldBinding) const {
  if (fieldBinding.kind != Binding::Kind::Field ||
      !fieldBinding.field.storageAccess || !fieldBinding.field.elementType)
    return std::nullopt;
  EncodingField field;
  field.name = fieldBinding.field.name;
  field.type = fieldBinding.field.elementType;
  field.shape = fieldBinding.field.shape;
  field.logicalRank = fieldBinding.field.logicalRank;
  field.access = fieldBinding.field.storageAccess;
  field.bitOffset = field.access.getBitOffset();
  field.storageBits = field.access.getStorageBits();
  return field;
}

Binding Emitter::emitInterleavedField(mlir::Value result,
                                      const Binding &fieldBinding,
                                      llvm::StringRef requestedIndex) {
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return {};
    owner = std::move(*record);
  }
  if (owner.kind != Binding::Kind::Record)
    return {};
  auto field = fieldFor(fieldBinding);
  if (!field)
    return {};
  unsigned logicalWidth = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type))
    logicalWidth = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(field->type))
    logicalWidth = floating.getWidth();
  if (!logicalWidth)
    return {};
  std::string logicalIndex = requestedIndex.str();
  int64_t logicalPartition = 0;
  if (fieldBinding.field.regularRepeat > 0) {
    logicalIndex = std::to_string(fieldBinding.field.regularBase);
    if (fieldBinding.field.regularRepeat == 1)
      logicalPartition = physicalLanes(result);
  } else if (fieldBinding.field.index) {
    const Binding &index = bindings.lookup(*fieldBinding.field.index);
    if (index.kind == Binding::Kind::Vector) {
      llvm::StringRef form = fieldBinding.field.useAccess
                                 ? fieldBinding.field.useAccess.getForm()
                                 : llvm::StringRef();
      if (form != "indexed" || owner.interleaveRows != 0 ||
          field->bitOffset % 8)
        return {};
      mlir::Value indexValue = *fieldBinding.field.index;
      auto indexLayout = layoutOf(indexValue);
      auto elementType = scalarCType(field->type);
      if (!indexLayout)
        return {};
      const std::string indexSuffix =
          "u" + std::to_string(indexLayout.getSew()) +
          lmulSpelling(indexLayout.getLmulEighths());
      const std::string indexType =
          "vuint" + std::to_string(indexLayout.getSew()) +
          lmulSpelling(indexLayout.getLmulEighths()) + "_t";
      const std::string resultSuffix = vectorSuffix(result);
      const std::string resultType = vectorType(result);
      llvm::StringRef mapping = field->access.getMapping();
      if (mapping == "grouped_layered") {
        auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
        auto resultLayout = layoutOf(result);
        const int64_t indexSEW = indexLayout.getSew();
        const int64_t group = field->access.getGroupSize();
        const int64_t layer = field->access.getLayerSize();
        const int64_t layers = layer > 0 ? group / layer : 0;
        llvm::StringRef order = field->access.getOrder();
        const uint64_t indexRange = uint64_t(1) << indexSEW;
        if (!integer || integer.isSigned() || !resultLayout ||
            (indexSEW != 8 && indexSEW != 16 && indexSEW != 32) ||
            resultLayout.getSew() != 8 ||
            logicalWidth == 0 || logicalWidth > 8 || group <= 0 ||
            layer <= 0 || group % layer || layers * logicalWidth > 8 ||
            static_cast<uint64_t>(group) > indexRange ||
            physicalLanes(indexValue) != physicalLanes(result))
          return {};

        const int64_t indexLMUL = indexLayout.getLmulEighths();
        const int64_t widthRatio = indexSEW / 8;
        if (indexLMUL != resultLayout.getLmulEighths() * widthRatio)
          return {};
        const uint64_t mask = (uint64_t(1) << logicalWidth) - 1;
        Binding gathered;
        gathered.kind = Binding::Kind::Vector;
        const int64_t resultStreams = streamPartCount(result);
        for (int64_t part = 0; part < vectorPartCount(result); ++part) {
          auto indexPart = projectPart(indexValue, result, part);
          auto coordinates =
              registerCoordinates(result, part / resultStreams);
          llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
          if (!indexPart || *indexPart >= index.parts.size() || !coordinates ||
              coordinates->size() != axes.size())
            return {};
          std::string record = "(" + owner.recordPointer;
          for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
            for (const auto &[recordAxis, stride] : owner.recordByteStrides)
              if (recordAxis == axis)
                record += " + " + std::to_string(coordinate) + " * " + stride;
          record += " + " + std::to_string(field->bitOffset / 8) + ")";
          const std::string vl = partVL(result, part);
          const std::string &indices = index.parts[*indexPart];
          std::string within = fresh("encoded_within");
          const bool oneRepresentableGroup =
              static_cast<uint64_t>(group) == indexRange;
          line(indexType + " " + within + " = " +
               (oneRepresentableGroup
                    ? indices
                    : "__riscv_vremu_vx_" + indexSuffix + "(" + indices +
                          ", " + std::to_string(group) + ", " + vl + ")") +
               ";");
          std::string bytes = fresh("encoded_bytes");
          std::string byteExpression =
              "__riscv_vremu_vx_" + indexSuffix + "(" + within + ", " +
              std::to_string(layer) + ", " + vl + ")";
          if (!oneRepresentableGroup) {
            std::string groups = fresh("encoded_group");
            line(indexType + " " + groups +
                 " = __riscv_vdivu_vx_" + indexSuffix + "(" + indices + ", " +
                 std::to_string(group) + ", " + vl + ");");
            byteExpression =
                "__riscv_vadd_vv_" + indexSuffix + "(__riscv_vmul_vx_" +
                indexSuffix + "(" + groups + ", " + std::to_string(layer) +
                ", " + vl + "), " + byteExpression + ", " + vl + ")";
          }
          line(indexType + " " + bytes + " = " + byteExpression + ";");
          std::string loaded = fresh("encoded_gather");
          line(resultType + " " + loaded + " = __riscv_vluxei" +
               std::to_string(indexSEW) + "_v_" +
               resultSuffix + "((const uint8_t *)(" + record + "), " + bytes +
               ", " + vl + ");");
          std::string layerWide = fresh("encoded_layer");
          std::string layerExpression =
              "__riscv_vdivu_vx_" + indexSuffix + "(" + within + ", " +
              std::to_string(layer) + ", " + vl + ")";
          if (order == "hi_first")
            layerExpression = "__riscv_vrsub_vx_" + indexSuffix + "(" +
                              layerExpression + ", " +
                              std::to_string(layers - 1) + ", " + vl + ")";
          line("vuint" + std::to_string(indexSEW) +
               lmulSpelling(indexLMUL) + "_t " + layerWide +
               " = __riscv_vmul_vx_" + indexSuffix + "(" + layerExpression +
               ", " + std::to_string(logicalWidth) + ", " + vl + ");");
          std::string layer8 = layerWide;
          if (indexSEW == 16) {
            layer8 = fresh("encoded_shift8");
            line(resultType + " " + layer8 + " = __riscv_vnsrl_wx_" +
                 resultSuffix + "(" + layerWide + ", 0, " + vl + ");");
          } else if (indexSEW == 32) {
            const int64_t shift16LMUL = indexLMUL / 2;
            const std::string shift16Suffix =
                "u16" + lmulSpelling(shift16LMUL);
            const std::string shift16Type =
                "vuint16" + lmulSpelling(shift16LMUL) + "_t";
            std::string layer16 = fresh("encoded_shift16");
            line(shift16Type + " " + layer16 + " = __riscv_vnsrl_wx_" +
                 shift16Suffix + "(" + layerWide + ", 0, " + vl + ");");
            layer8 = fresh("encoded_shift8");
            line(resultType + " " + layer8 + " = __riscv_vnsrl_wx_" +
                 resultSuffix + "(" + layer16 + ", 0, " + vl + ");");
          }
          std::string decoded = fresh("encoded_value");
          line(resultType + " " + decoded + " = __riscv_vand_vx_" +
               resultSuffix + "(__riscv_vsrl_vv_" + resultSuffix + "(" +
               loaded + ", " + layer8 + ", " + vl + "), " +
               std::to_string(mask) + ", " + vl + ");");
          gathered.parts.push_back(std::move(decoded));
        }
        return gathered;
      }
      if (mapping != "natural" || !elementType ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32))
        return {};
      Binding gathered;
      gathered.kind = Binding::Kind::Vector;
      const int64_t resultStreams = streamPartCount(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto indexPart = projectPart(indexValue, result, part);
        auto coordinates =
            registerCoordinates(result, part / resultStreams);
        llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
        if (!indexPart || *indexPart >= index.parts.size() || !coordinates ||
            coordinates->size() != axes.size())
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + " + std::to_string(field->bitOffset / 8) + ")";
        std::string offsets = index.parts[*indexPart];
        if (logicalWidth != 8)
          offsets = "__riscv_vmul_vx_" + indexSuffix + "(" + offsets +
                    ", " + std::to_string(logicalWidth / 8) + ", " +
                    partVL(result, part) + ")";
        std::string expression =
        "__riscv_vluxei" + std::to_string(indexLayout.getSew()) + "_v_" +
            resultSuffix + "((const " + *elementType + " *)(" + record +
            "), " + offsets + ", " + partVL(result, part) + ")";
        std::string loaded = fresh("gather");
        line(resultType + " " + loaded + " = " + expression + ";");
        gathered.parts.push_back(std::move(loaded));
      }
      return gathered;
    } else if (index.kind == Binding::Kind::Scalar) {
      logicalIndex = index.scalar;
    } else if (index.kind == Binding::Kind::Point) {
      const int64_t elements = owner.recordElements;
      if (elements <= 0)
        return {};
      logicalPartition = index.point.physicalExtent;
      if (fieldBinding.field.selector == "group_index")
        logicalIndex = "((" + index.point.base + " % " +
                       std::to_string(elements) + ") / " +
                       std::to_string(index.point.physicalExtent) + ")";
      else
        logicalIndex = "(" + index.point.base + " % " +
                       std::to_string(elements) + ")";
    } else {
      return {};
    }
  }
  for (mlir::Value relativeValue : fieldBinding.field.relativeIndices) {
    const Binding &relative = bindings.lookup(relativeValue);
    if (relative.kind != Binding::Kind::Scalar)
      return {};
    logicalIndex = "((" + logicalIndex + ") + (" + relative.scalar + "))";
  }
  if (owner.interleaveRows == 0) {
    auto resultLayout = layoutOf(result);
    const bool vectorResult = resultLayout && resultLayout.getCarrier() == "rvv";
    llvm::StringRef selectedForm = fieldBinding.field.useAccess
                                       ? fieldBinding.field.useAccess.getForm()
                                       : llvm::StringRef();
    std::string logicalProjectionFailure;
    auto logicalIndexForPart = [&](int64_t part)
        -> std::optional<std::string> {
      if (field->logicalRank == 0)
        return logicalIndex;
      auto value = mlir::dyn_cast<riscv::ValueType>(result.getType());
      if (!value || !resultLayout || field->logicalRank < 0 ||
          field->logicalRank > static_cast<int64_t>(field->shape.size()) ||
          field->logicalRank > static_cast<int64_t>(value.getAxisIds().size())) {
        logicalProjectionFailure = "field rank exceeds the physical value rank";
        return std::nullopt;
      }
      const size_t logicalRank = static_cast<size_t>(field->logicalRank);
      const size_t valueOuterRank = value.getAxisIds().size() - logicalRank;
      const size_t shapeOuterRank = field->shape.size() - logicalRank;
      int64_t linear = 0;
      for (size_t dimension = 0; dimension < logicalRank; ++dimension) {
        const int64_t axis = value.getAxisIds()[valueOuterRank + dimension];
        auto position = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
        if (position == resultLayout.getAxisIds().asArrayRef().end()) {
          logicalProjectionFailure = "field axis is absent from the physical layout";
          return std::nullopt;
        }
        const size_t layoutPosition = static_cast<size_t>(
            std::distance(resultLayout.getAxisIds().asArrayRef().begin(), position));
        if (resultLayout.getLaneFactors()[layoutPosition] != 1 ||
            resultLayout.getReplicaFactors()[layoutPosition] != 1 ||
            resultLayout.getFragmentFactors()[layoutPosition] != 1 ||
            resultLayout.getLocalFactors()[layoutPosition] != 1) {
          logicalProjectionFailure =
              "field axis is not represented solely by sequential time";
          return std::nullopt;
        }
        auto coordinate = timeCoordinate(result, part, axis);
        const int64_t logicalExtent = field->shape[shapeOuterRank + dimension];
        if (!coordinate || *coordinate < 0 ||
            *coordinate >= logicalExtent) {
          logicalProjectionFailure =
              "field time coordinate is incomplete or outside its logical extent";
          return std::nullopt;
        }
        linear = linear * logicalExtent + *coordinate;
      }
      return "((" + logicalIndex + ") + " + std::to_string(linear) + ")";
    };
    if (vectorResult && fieldBinding.field.regularRepeat > 1 &&
        selectedForm == "indexed" &&
        field->access.getMapping() == "natural") {
      auto elementType = scalarCType(field->type);
      const int64_t repeat = fieldBinding.field.regularRepeat;
      const int64_t stride = fieldBinding.field.regularStride;
      const int64_t lanes = physicalLanes(result);
      if (!elementType || field->bitOffset % 8 ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32) ||
          repeat <= 1 || stride <= 0 || lanes <= 0 ||
          (repeat % lanes && lanes % repeat))
        return {};
      Binding repeated;
      repeated.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        int64_t offset = 0;
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            llvm::StringRef(partOffset(result, part)).getAsInteger(10, offset) ||
            offset % lanes ||
            (repeat >= lanes ? offset % repeat + lanes > repeat
                             : offset % repeat))
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, byteStride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " +
                        byteStride;
        record += ")";
        auto address = [&](int64_t logicalOffset) {
          const int64_t sourceIndex =
              fieldBinding.field.regularBase +
              (logicalOffset / repeat) * stride;
          return "((const " + *elementType + " *)((const uint8_t *)(" +
                 record + ") + " + std::to_string(field->bitOffset / 8) +
                 "))[" + std::to_string(sourceIndex) + "]";
        };
        std::string name = fresh("field_repeat");
        const bool floating = mlir::isa<mlir::FloatType>(field->type);
        const std::string broadcast =
            "__riscv_" +
            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix;
        const std::string vl = partVL(result, part);
        std::string expression = broadcast + "(" + address(offset) + ", " +
                                 vl + ")";
        if (lanes > repeat)
          for (int64_t piece = 1; piece < lanes / repeat; ++piece) {
            const std::string next =
                broadcast + "(" + address(offset + piece * repeat) + ", " +
                vl + ")";
            expression = "__riscv_vslideup_vx_" + suffix + "(" + expression +
                         ", " + next + ", " + std::to_string(piece * repeat) +
                         ", " + vl + ")";
          }
        line(type + " " + name + " = " + expression + ";");
        repeated.parts.push_back(std::move(name));
      }
      return repeated;
    }
    if (vectorResult && field->access.getMapping() == "joined" &&
        !field->shape.empty()) {
      const int64_t laneAxis = laneAxisFor(result);
      auto laneStride = llvm::find_if(
          owner.recordByteStrides,
          [&](const auto &entry) { return entry.first == laneAxis; });
      const int64_t group = field->access.getGroupSize();
      const int64_t fields = field->access.getJoinFields();
      const int64_t lowBits = field->access.getJoinLowBits();
      const int64_t role = field->access.getJoinRole();
      llvm::StringRef order = field->access.getOrder();
      if (!laneAxis || laneStride == owner.recordByteStrides.end() || group <= 0 ||
          fields <= 1 || lowBits <= 0 || role < 0 || role >= fields ||
          logicalWidth <= lowBits || logicalWidth > 8)
        return {};
      const int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
      const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
      const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
      const uint64_t highMask =
          (uint64_t(1) << (logicalWidth - lowBits)) - 1;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      Binding decoded;
      decoded.kind = Binding::Kind::Vector;
      auto byteLoad = [&](llvm::StringRef record, llvm::StringRef offset,
                          llvm::StringRef stem, llvm::StringRef vl) {
        std::string name = fresh(stem);
        std::string statement =
            type + " " + name + " = __riscv_vlse8_v_" + suffix +
            "((const uint8_t *)(" + record.str() + " + (" + offset.str() +
            ")), (ptrdiff_t)(" + laneStride->second + "), " + vl.str() + ");";
        line(statement);
        return name;
      };
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        auto partLogical = logicalIndexForPart(part);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            !partLogical) {
          if (!partLogical)
            result.getDefiningOp()->emitError(
                "joined field has no closed per-part logical coordinate: ")
                << logicalProjectionFailure;
          return {};
        }
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + (" + partOffset(result, part) + ") * (" +
                  laneStride->second + "))";
        const std::string logical = "(" + *partLogical + ")";
        const std::string tail = "(" + logical + " - " +
                                 std::to_string(group) + ")";
        const std::string headOffset =
            std::to_string(field->bitOffset / 8 + role * group) + " + " + logical;
        const std::string lowOffset =
            std::to_string(field->bitOffset / 8 + fields * group) + " + " + tail;
        const std::string highOffset =
            std::to_string(field->bitOffset / 8 + role * group) + " + " + tail;
        const std::string vl = partVL(result, part);
        std::string head = byteLoad(record, headOffset, "joined_head", vl);
        head = "__riscv_vand_vx_" + suffix + "(" + head + ", " +
               std::to_string(logicalMask) + ", " + vl + ")";
        std::string low = byteLoad(record, lowOffset, "joined_low", vl);
        low = "__riscv_vand_vx_" + suffix + "(__riscv_vsrl_vx_" + suffix +
              "(" + low + ", " + std::to_string(physicalRole * lowBits) +
              ", " + vl + "), " + std::to_string(lowMask) + ", " + vl + ")";
        std::string high = byteLoad(record, highOffset, "joined_high", vl);
        high = "__riscv_vand_vx_" + suffix + "(__riscv_vsrl_vx_" + suffix +
               "(" + high + ", " + std::to_string(logicalWidth) +
               ", " + vl + "), " + std::to_string(highMask) + ", " + vl + ")";
        std::string assembled =
            order == "lo_first"
                ? "__riscv_vor_vv_" + suffix + "(" + low +
                      ", __riscv_vsll_vx_" + suffix + "(" + high + ", " +
                      std::to_string(lowBits) + ", " + vl + "), " + vl + ")"
                : "__riscv_vor_vv_" + suffix + "(" + high +
                      ", __riscv_vsll_vx_" + suffix + "(" + low + ", " +
                      std::to_string(logicalWidth - lowBits) + ", " + vl +
                      "), " + vl + ")";
        std::string name = fresh("joined_field");
        line(type + " " + name + " = (" + logical + " < " +
             std::to_string(group) + " ? " + head + " : " + assembled +
             ");");
        decoded.parts.push_back(std::move(name));
      }
      return decoded;
    }
    if (vectorResult && selectedForm == "strided" &&
        field->access.getMapping() == "natural" &&
        field->bitOffset % 8 == 0 &&
        (logicalWidth == 8 || logicalWidth == 16 || logicalWidth == 32)) {
      const int64_t laneAxis = laneAxisFor(result);
      auto laneStride = llvm::find_if(
          owner.recordByteStrides,
          [&](const auto &entry) { return entry.first == laneAxis; });
      auto elementType = scalarCType(field->type);
      if (!laneAxis || laneStride == owner.recordByteStrides.end() || !elementType)
        return {};
      Binding vectorResult;
      vectorResult.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        auto partLogical = logicalIndexForPart(part);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            !partLogical) {
          if (!partLogical)
            result.getDefiningOp()->emitError(
                "natural field has no closed per-part logical coordinate: ")
                << logicalProjectionFailure;
          return {};
        }
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + (" + partOffset(result, part) + ") * (" +
                  laneStride->second + ") + " +
                  std::to_string(field->bitOffset / 8) + " + (" +
                  *partLogical + ") * " +
                  std::to_string(logicalWidth / 8) + ")";
        std::string loaded = fresh("field_strided");
        line(type + " " + loaded + " = __riscv_vlse" +
             std::to_string(logicalWidth) + "_v_" + suffix + "((const " +
             *elementType + " *)(" + record + "), (ptrdiff_t)(" +
             laneStride->second + "), " + partVL(result, part) + ");");
        vectorResult.parts.push_back(std::move(loaded));
      }
      return vectorResult;
    }
    if (vectorResult && field->access.getMapping() == "grouped_layered") {
      result.getDefiningOp()->emitError(
          "grouped/layered vector access reached emission without a typed physical window/stream operation");
      return {};
    }
    if (vectorResult && !field->shape.empty()) {
      if (field->bitOffset % 8 || field->access.getMapping() != "natural" ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32))
        return {};
      Binding vectorResult;
      vectorResult.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      auto elementType = scalarCType(field->type);
      if (!elementType)
        return {};
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        const int64_t streams = streamPartCount(result);
        auto coordinates = registerCoordinates(result, part / streams);
        llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += ")";
        const std::string address =
            record + " + " + std::to_string(field->bitOffset / 8) +
            " + ((" + logicalIndex + ") + " + partOffset(result, part) + ") * " +
            std::to_string(logicalWidth / 8);
        std::string loaded = fresh("field");
        line(type + " " + loaded + " = __riscv_vle" +
             std::to_string(logicalWidth) + "_v_" + suffix + "((const " +
             *elementType + " *)(" + address + ")" +
             ", " + partVL(result, part) + ");");
        vectorResult.parts.push_back(std::move(loaded));
      }
      return vectorResult;
    }
    Binding scalarResult;
    scalarResult.kind = Binding::Kind::Scalar;
    llvm::StringRef kind = field->access.getMapping();
    if (kind == "joined") {
      int64_t group = field->access.getGroupSize();
      int64_t fields = field->access.getJoinFields();
      int64_t lowBits = field->access.getJoinLowBits();
      int64_t role = field->access.getJoinRole();
      llvm::StringRef order = field->access.getOrder();
      int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
      const std::string headByte =
          "(" + std::to_string(field->bitOffset / 8 + role * group) +
          " + (" + logicalIndex + "))";
      const std::string tail = "((" + logicalIndex + ") - " +
                               std::to_string(group) + ")";
      const std::string lowByte =
          "(" + std::to_string(field->bitOffset / 8 + fields * group) +
          " + " + tail + ")";
      const std::string highByte =
          "(" + std::to_string(field->bitOffset / 8 + role * group) +
          " + " + tail + ")";
      const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
      const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
      const uint64_t highMask =
          (uint64_t(1) << (logicalWidth - lowBits)) - 1;
      const std::string head =
          "((uint8_t)((" + owner.recordPointer + ")[" + headByte + "] & " +
          std::to_string(logicalMask) + "))";
      const std::string low =
          "((uint8_t)(((" + owner.recordPointer + ")[" + lowByte + "] >> " +
          std::to_string(physicalRole * lowBits) + ") & " +
          std::to_string(lowMask) + "))";
      const std::string high =
          "((uint8_t)(((" + owner.recordPointer + ")[" + highByte + "] >> " +
          std::to_string(logicalWidth) + ") & " +
          std::to_string(highMask) + "))";
      scalarResult.scalar = "((" + logicalIndex + ") < " +
                            std::to_string(group) + " ? " + head + " : (" +
                            low + " | (" + high + " << " +
                            std::to_string(lowBits) + ")))";
      return scalarResult;
    }
    auto fragment = singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    const std::string address = owner.recordPointer + " + (" + fragment->byte + ")";
    const bool naturallyByteAligned =
        kind == "natural" && field->bitOffset % 8 == 0 && logicalWidth % 8 == 0;
    if (field->type.isF32() && logicalWidth == 32 && naturallyByteAligned)
      scalarResult.scalar = field->access.getAlignment() >= 4
                                ? "*(const float *)(" + address + ")"
                                : "weft_load_f32_le(" + owner.recordPointer +
                                      " + " + fragment->byte + ")";
    else if (field->type.isF16() && logicalWidth == 16 && naturallyByteAligned)
      scalarResult.scalar = field->access.getAlignment() >= 2
                                ? "*(const _Float16 *)(" + address + ")"
                                : "weft_load_f16_le(" + owner.recordPointer +
                                      " + " + fragment->byte + ")";
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
             integer && integer.getWidth() <= 8) {
      const uint64_t mask = (uint64_t(1) << integer.getWidth()) - 1;
      std::string value = "((" + address + ")[0] >> (" + fragment->shift + ")) & " +
                          std::to_string(mask);
      if (integer.isSigned() && integer.getWidth() < 8) {
        const uint64_t sign = uint64_t(1) << (integer.getWidth() - 1);
        value = "((int8_t)((((" + value + ") ^ " + std::to_string(sign) +
                ") - " + std::to_string(sign) + ")))";
      } else if (integer.isSigned()) {
        value = "((int8_t)(" + value + "))";
      } else {
        value = "((uint8_t)(" + value + "))";
      }
      scalarResult.scalar = std::move(value);
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
               integer && integer.getWidth() == 16 && naturallyByteAligned) {
      scalarResult.scalar =
          std::string(integer.isUnsigned() ? "weft_load_u16_le(" :
                                             "weft_load_i16_le(") +
          owner.recordPointer + " + " + fragment->byte + ")";
    } else {
      return {};
    }
    return scalarResult;
  }

  // Byte assembly preserves the selected result lane count.  Its raw LMUL is
  // therefore the result LMUL scaled by the SEW ratio; this is a mechanical
  // projection of the selected value layout, not a second layout choice.
  auto resultLayout = riscv_internal::layoutOf(result.getType());
  const int64_t carrierWidth = std::max<int64_t>(8, logicalWidth);
  const int64_t rawNumerator =
      resultLayout ? resultLayout.getLmulEighths() * 8 : 0;
  int64_t rawLMUL = rawNumerator > 0 && rawNumerator % carrierWidth == 0
                        ? rawNumerator / carrierWidth
                        : 0;
  if (rawLMUL != 1 && rawLMUL != 2 && rawLMUL != 4 && rawLMUL != 8 &&
      rawLMUL != 16 && rawLMUL != 32 && rawLMUL != 64)
    return {};
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  const std::string rawSuffix = "u8" + lmul(rawLMUL);
  const std::string rawType = "vuint8" + lmul(rawLMUL) + "_t";
  auto loadByte = [&](llvm::StringRef byte, llvm::StringRef prefix) {
    std::string loaded = fresh(prefix);
    line(rawType + " " + loaded + " = __riscv_vle8_v_" + rawSuffix + "(" +
         owner.recordPointer + " + (" + byte.str() + ") * " +
         std::to_string(owner.interleaveRows) + ", " + laneVL() + ");");
    return loaded;
  };

  std::string raw;
  llvm::StringRef kind = field->access.getMapping();
  if (kind == "joined") {
    int64_t group = field->access.getGroupSize();
    int64_t fields = field->access.getJoinFields();
    int64_t lowBits = field->access.getJoinLowBits();
    int64_t role = field->access.getJoinRole();
    llvm::StringRef order = field->access.getOrder();
    int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
    raw = fresh("joined_value");
    line(rawType + " " + raw + ";");
    line("if ((" + logicalIndex + ") < " + std::to_string(group) + ") {");
    ++indent;
    std::string headByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                           std::to_string(role * group) + " + (" +
                           logicalIndex + "))";
    std::string head = loadByte(headByte, "joined_head");
    line(raw + " = __riscv_vand_vx_" + rawSuffix + "(" + head + ", " +
         std::to_string((1u << logicalWidth) - 1) + ", " + laneVL() + ");");
    --indent;
    line("} else {");
    ++indent;
    std::string tail = "((" + logicalIndex + ") - " + std::to_string(group) + ")";
    std::string lowByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                          std::to_string(fields * group) + " + " + tail + ")";
    std::string highByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                           std::to_string(role * group) + " + " + tail + ")";
    std::string low = loadByte(lowByte, "joined_low");
    std::string high = loadByte(highByte, "joined_high");
    std::string lowPart = fresh("joined_low_bits");
    line(rawType + " " + lowPart + " = __riscv_vand_vx_" + rawSuffix +
         "(__riscv_vsrl_vx_" + rawSuffix + "(" + low + ", " +
         std::to_string(physicalRole * lowBits) + ", " + laneVL() + "), " +
         std::to_string((1u << lowBits) - 1) + ", " + laneVL() + ");");
    std::string highPart = fresh("joined_high_bits");
    line(rawType + " " + highPart + " = __riscv_vsll_vx_" + rawSuffix +
         "(__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" + rawSuffix +
         "(" + high + ", " + std::to_string(logicalWidth) + ", " + laneVL() +
         "), " + std::to_string((1u << (logicalWidth - lowBits)) - 1) + ", " +
         laneVL() + "), " + std::to_string(lowBits) + ", " + laneVL() + ");");
    line(raw + " = __riscv_vor_vv_" + rawSuffix + "(" + lowPart + ", " +
         highPart + ", " + laneVL() + ");");
    --indent;
    line("}");
  } else {
    auto fragment =
        singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    std::string loaded = loadByte(fragment->byte, "packed_byte");
    if (logicalWidth <= 8) {
      raw = fresh("packed_value");
      line(rawType + " " + raw + " = __riscv_vand_vx_" + rawSuffix +
           "(__riscv_vsrl_vx_" + rawSuffix + "(" + loaded + ", " +
           fragment->shift + ", " + laneVL() + "), " +
           std::to_string((1u << logicalWidth) - 1) + ", " + laneVL() + ");");
    } else {
      raw = std::move(loaded);
    }
  }
  Binding resultBinding;
  resultBinding.kind = Binding::Kind::Vector;
  if (logicalWidth <= 8) {
    resultBinding.parts.push_back(raw);
    return resultBinding;
  }
  if (logicalWidth == 16 && kind == "natural") {
    auto fragment =
        singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    const std::string u16Suffix = "u16" + lmul(rawLMUL * 2);
    const std::string u16Type = "vuint16" + lmul(rawLMUL * 2) + "_t";
    std::string high = fresh("word_high");
    line(rawType + " " + high + " = __riscv_vle8_v_" + rawSuffix + "(" +
         owner.recordPointer + " + (" + fragment->byte + " + 1) * " +
         std::to_string(owner.interleaveRows) + ", " + laneVL() + ");");
    std::string low16 = fresh("word_low16");
    std::string high16 = fresh("word_high16");
    line(u16Type + " " + low16 + " = __riscv_vzext_vf2_" + u16Suffix + "(" +
         raw + ", " + laneVL() + ");");
    line(u16Type + " " + high16 + " = __riscv_vsll_vx_" + u16Suffix + "(" +
         "__riscv_vzext_vf2_" + u16Suffix + "(" + high + ", " + laneVL() +
         "), 8, " + laneVL() + ");");
    std::string word = fresh("word");
    line(u16Type + " " + word + " = __riscv_vor_vv_" + u16Suffix + "(" +
         low16 + ", " + high16 + ", " + laneVL() + ");");
    const std::string resultSuffix = vectorSuffix(result);
    if (resultSuffix == u16Suffix) {
      resultBinding.parts.push_back(std::move(word));
      return resultBinding;
    }
    std::string converted = fresh("field");
    line(vectorType(result) + " " + converted + " = __riscv_vreinterpret_v_" +
         u16Suffix + "_" + resultSuffix + "(" + word + ");");
    resultBinding.parts.push_back(std::move(converted));
    return resultBinding;
  }
  return {};
}

mlir::LogicalResult Emitter::compileRegisterMaterialize(
    riscv::RegisterMaterializeOp materialize) {
  Binding input = bindings.lookup(materialize.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(materialize, "register materialize input has no emitted value");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> value =
        materializeNumeric(materialize.getResult(), std::move(input));
    if (mlir::failed(value))
      return mlir::failure();
    bindings[materialize.getResult()] = std::move(*value);
    return mlir::success();
  }
  bindings[materialize.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStagedView(riscv::StagedViewOp staged) {
  Binding source = bindings.lookup(staged.getSource());
  if (source.kind != Binding::Kind::Memory &&
      source.kind != Binding::Kind::Slice)
    return fail(staged, "reload staged view requires one emitted memory descriptor");
  bindings[staged.getResult()] = std::move(source);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalCapacityGuard(
    riscv::LocalCapacityGuardOp operation) {
  Binding bytes = bindings.lookup(operation.getSizeBytes());
  if (bytes.kind != Binding::Kind::Scalar ||
      operation.getLeaf().getInstruction() != "scalar.local-capacity-guard")
    return fail(operation, "dynamic local allocation has no closed capacity guard");
  line("if ((uint64_t)(" + bytes.scalar + ") > " +
       std::to_string(operation.getLimitBytes()) +
       "ULL) __builtin_trap();");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalAlloc(riscv::LocalAllocOp operation) {
  riscv::LocalType type = operation.getResult().getType();
  Binding size = bindings.lookup(operation.getSizeBytes());
  if (size.kind != Binding::Kind::Scalar || type.getAlignment() <= 0)
    return fail(operation, "local allocation has no closed byte size/alignment");
  std::string name = fresh(type.getPurpose() == "spill" ? "spill_slot"
                                                         : "local_object");
  line("_Alignas(" + std::to_string(type.getAlignment()) + ") uint8_t " + name +
       "[" + size.scalar + "];");
  Binding binding;
  binding.kind = Binding::Kind::LocalArray;
  binding.scalar = name;
  binding.localElements = size.scalar;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalBind(riscv::LocalBindOp operation) {
  Binding storage = bindings.lookup(operation.getStorage());
  Binding elements = bindings.lookup(operation.getElements());
  if (storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      elements.kind != Binding::Kind::Scalar)
    return fail(operation,
                "local bind requires one allocation and one explicit element extent");
  auto element = scalarCType(operation.getResult().getType().getElementType());
  if (!element)
    return fail(operation, "local bind element has no intrinsic-C type");
  storage.scalar = "((" + *element + " *)" + storage.scalar + ")";
  storage.localElements = elements.scalar;
  bindings[operation.getResult()] = std::move(storage);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalLoad(riscv::LocalLoadOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getResult().getType());
  if (source.kind != Binding::Kind::LocalArray || source.scalar.empty() ||
      index.kind != Binding::Kind::Scalar || !element ||
      operation.getLeaf().getInstruction() != "local.load.element")
    return fail(operation, "local element load has no closed physical binding");
  bindings[operation.getResult()] =
      scalar("((const " + *element + " *)" + source.scalar + ")[" +
             index.scalar + "]");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalStore(riscv::LocalStoreOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding destination = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getValue().getType());
  if (value.kind != Binding::Kind::Scalar ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || index.kind != Binding::Kind::Scalar ||
      !element || operation.getLeaf().getInstruction() != "local.store.element")
    return fail(operation, "local element store has no closed physical binding");
  line("((" + *element + " *)" + destination.scalar + ")[" + index.scalar +
       "] = " + value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLocalMaterialize(
    riscv::RVVLocalMaterializeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.local-materialize")
    return fail(operation,
                "RVV local materialize has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  Binding destination = bindings.lookup(operation.getDestination());
  auto element = scalarCType(operation.getInput().getType().getElementType());
  const int64_t parts = vectorPartCount(operation.getInput());
  const int64_t lanes = physicalLanes(operation.getInput());
  const unsigned bits = operation.getInput().getType().getLayout().getSew();
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || !element || parts <= 0 || lanes <= 0 ||
      input->parts.size() != static_cast<size_t>(parts))
    return fail(operation,
                "RVV local materialize has no complete vector/local binding");
  const std::string suffix = vectorSuffix(operation.getInput());
  for (int64_t part = 0; part < parts; ++part)
    line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix + "((" +
         *element + " *)" + destination.scalar + " + " +
         std::to_string(part * lanes) + ", " + input->parts[part] + ", " +
         partVL(operation.getInput(), part) + ");");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSpill(riscv::SpillOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  Binding slot = bindings.lookup(operation.getSlot());
  const bool rvvSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "rvv.spill";
  const bool scalarTupleSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "scalar.tuple-spill";
  if ((!rvvSpill && !scalarTupleSpill) ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "spill destination has no typed local allocation");
  riscv::LocalType slotType = operation.getSlot().getType();
  mlir::Type element = riscv_internal::logicalElement(operation.getInput().getType());
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "spill value has no C element type");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (input.kind == Binding::Kind::Vector) {
    if (!rvvSpill)
      return fail(operation, "vector spill requires the selected RVV spill leaf");
    auto inputType =
        mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    const int64_t parts = vectorPartCount(operation.getInput());
    if (!inputType || parts <= 0 ||
        input.parts.size() != static_cast<size_t>(parts) ||
        slotType.getSizeBytes() % parts)
      return fail(operation, "vector spill byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = inputType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getInput());
    for (int64_t part = 0; part < parts; ++part)
      line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix +
           "((" + *scalarType + " *)(" + slot.scalar + " + " +
           std::to_string(part * bytes) + "), " + input.parts[part] + ", " +
           partVL(operation.getInput(), part) + ");");
    return mlir::success();
  }
  if (input.kind == Binding::Kind::Scalar ||
      input.kind == Binding::Kind::ScalarTuple) {
    if (!scalarTupleSpill)
      return fail(operation,
                  "scalar register spill requires the selected tuple-spill leaf");
    llvm::SmallVector<std::string> values;
    if (input.kind == Binding::Kind::Scalar)
      values.push_back(input.scalar);
    else
      values.assign(input.parts.begin(), input.parts.end());
    for (auto [index, value] : llvm::enumerate(values))
      line("((" + *scalarType + " *)" + slot.scalar + ")[" +
           std::to_string(index) + "] = " + value + ";");
    return mlir::success();
  }
  return fail(operation, "spill supports only typed RVV/scalar register values");
}

mlir::LogicalResult Emitter::compileReload(riscv::ReloadOp operation) {
  Binding slot = bindings.lookup(operation.getSlot());
  if (operation.getLeaf().getEngine() != "transfer" ||
      operation.getLeaf().getFamily() != "reload" ||
      operation.getLeaf().getInstruction() != "rvv.reload" ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "reload source has no typed local allocation");
  auto resultType = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!resultType)
    return fail(operation, "reload result has no typed physical value");
  auto kind = selectedBindingKind(operation.getResult());
  auto scalarType = scalarCType(resultType.getElementType());
  if (!kind || !scalarType)
    return fail(operation, "reload result has no terminal representation");
  Binding result;
  result.kind = *kind;
  if (*kind == Binding::Kind::Vector) {
    riscv::LocalType slotType = operation.getSlot().getType();
    const int64_t parts = vectorPartCount(operation.getResult());
    if (parts <= 0 || slotType.getSizeBytes() % parts)
      return fail(operation, "vector reload byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = resultType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    auto partIsUsed = [&](int64_t part) {
      for (mlir::OpOperand &use : operation.getResult().getUses()) {
        auto set = mlir::dyn_cast<riscv::RVVPartialSetOp>(use.getOwner());
        if (!set)
          return true;
        const size_t operand = use.getOperandNumber();
        const size_t lhsCount = set.getLhs().size();
        if (operand >= lhsCount + set.getRhs().size())
          return true;
        for (int64_t slot = 0; slot < set.getResult().getType().getSlots(); ++slot) {
          if (operand < lhsCount) {
            if (set.getLhsOperands()[slot] == static_cast<int64_t>(operand) &&
                set.getLhsParts()[slot] == part)
              return true;
          } else if (set.getRhsOperands()[slot] ==
                         static_cast<int64_t>(operand - lhsCount) &&
                     set.getRhsParts()[slot] == part) {
            return true;
          }
        }
      }
      return false;
    };
    for (int64_t part = 0; part < parts; ++part) {
      if (!partIsUsed(part)) {
        result.parts.emplace_back();
        continue;
      }
      std::string loaded = fresh("reload");
      line(type + " " + loaded + " = __riscv_vle" + std::to_string(bits) +
           "_v_" + suffix + "((const " + *scalarType + " *)(" + slot.scalar +
           " + " + std::to_string(part * bytes) + "), " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(loaded));
    }
  } else if (*kind == Binding::Kind::Scalar) {
    result.scalar = "((const " + *scalarType + " *)" + slot.scalar + ")[0]";
  } else if (*kind == Binding::Kind::ScalarTuple) {
    for (int64_t part = 0; part < registerPartCount(operation.getResult()); ++part)
      result.parts.push_back("((const " + *scalarType + " *)" + slot.scalar +
                             ")[" + std::to_string(part) + "]");
  } else {
    return fail(operation, "reload supports only typed RVV/scalar values");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEPack(riscv::IMEPackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto inputType =
      mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  auto fragment = operation.getResult().getType();
  if (!inputType)
    return fail(operation, "IME pack input has no typed physical value");
  auto integer = mlir::dyn_cast<mlir::IntegerType>(inputType.getElementType());
  if (!integer || integer.getWidth() > 8)
    return fail(operation,
                "IME pack currently requires an at-most-eight-bit integer value");
  if (input.kind == Binding::Kind::Field || input.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (input.kind != Binding::Kind::Vector ||
      static_cast<int64_t>(input.parts.size()) !=
          vectorPartCount(operation.getInput()))
    return fail(operation,
                "IME pack input does not materialize its complete selected RVV value");

  auto layout = inputType.getLayout();
  auto shape = fragment.getShape().asArrayRef();
  auto time = layout.getTimeFactors().asArrayRef();
  auto lanes = layout.getLaneFactors().asArrayRef();
  auto replicas = layout.getReplicaFactors().asArrayRef();
  if (shape.size() != time.size() || shape.size() != lanes.size() ||
      shape.size() != replicas.size())
    return fail(operation, "IME pack input layout rank is incomplete");
  int64_t laneDimension = -1;
  int64_t elements = 1;
  llvm::SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
       dimension >= 0; --dimension) {
    if (shape[dimension] <= 0 || time[dimension] <= 0 ||
        lanes[dimension] <= 0 || replicas[dimension] <= 0 ||
        time[dimension] * lanes[dimension] * replicas[dimension] !=
            shape[dimension])
      return fail(operation,
                  "IME pack requires a closed time/lane/register decomposition");
    if (lanes[dimension] > 1) {
      if (laneDimension >= 0)
        return fail(operation,
                    "IME pack currently accepts one SIMD lane axis");
      laneDimension = dimension;
    }
    strides[dimension] = elements;
    elements *= shape[dimension];
  }
  if (laneDimension < 0 || layout.getSew() != 8)
    return fail(operation,
                "IME pack requires one selected eight-bit RVV lane mapping");
  auto scalarType = scalarCType(inputType.getElementType());
  if (!scalarType)
    return fail(operation, "IME pack input has no intrinsic-C scalar type");

  const std::string logical = fresh("ime_pack_logical");
  line("_Alignas(32) " + *scalarType + " " + logical + "[" +
       std::to_string(elements) + "] = {0};");
  const int64_t streams = streamPartCount(operation.getInput());
  const int64_t registers = registerPartCount(operation.getInput());
  for (int64_t reg = 0; reg < registers; ++reg) {
    int64_t registerCursor = reg;
    llvm::SmallVector<int64_t> registerCoordinates(shape.size(), 0);
    for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
         dimension >= 0; --dimension) {
      registerCoordinates[dimension] = registerCursor % replicas[dimension];
      registerCursor /= replicas[dimension];
    }
    if (registerCursor)
      return fail(operation, "IME pack register coordinates are incomplete");
    for (int64_t stream = 0; stream < streams; ++stream) {
      int64_t streamCursor = stream;
      llvm::SmallVector<int64_t> timeCoordinates(shape.size(), 0);
      for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
           dimension >= 0; --dimension) {
        timeCoordinates[dimension] = streamCursor % time[dimension];
        streamCursor /= time[dimension];
      }
      if (streamCursor)
        return fail(operation, "IME pack time coordinates are incomplete");
      int64_t linearBase = 0;
      for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        int64_t coordinate =
            (timeCoordinates[dimension] * replicas[dimension] +
             registerCoordinates[dimension]) *
            lanes[dimension];
        linearBase += coordinate * strides[dimension];
      }
      const size_t part = static_cast<size_t>(reg * streams + stream);
      line("__riscv_vsse8_v_" + vectorSuffix(operation.getInput()) + "(" +
           logical + " + " + std::to_string(linearBase) + ", (ptrdiff_t)" +
           std::to_string(strides[laneDimension] *
                          std::max<int64_t>(1, integer.getWidth() / 8)) +
           ", " + input.parts[part] + ", " +
           partVL(operation.getInput(), part) + ");");
    }
  }

  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();

  if (operation.getRole() != "lhs" && operation.getRole() != "rhs")
    return fail(operation, "IME pack role must be lhs or rhs");
  auto packing = operation.getPacking();
  const bool lhs = operation.getRole() == "lhs";
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  if (shape.size() != 2 || axisOrder.size() != 2)
    return fail(operation, "IME pack requires one rank-two physical axis order");
  const int64_t rows = shape[axisOrder[0]];
  const int64_t columns = shape[axisOrder[1]];
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() != 8 || integer.getWidth() > 8)
    return fail(operation,
                "selected IME fragment has no closed tiled packing geometry");
  const int64_t tileBytes = packing.getRowsPerTile() *
                            packing.getColumnsPerTile() *
                            packing.getStorageBits() / 8;
  const int64_t tileRows = rows / packing.getRowsPerTile();
  const int64_t tileColumns = columns / packing.getColumnsPerTile();
  const int64_t storageBytes = rows * columns * packing.getStorageBits() / 8;
  const std::string name = fresh(lhs ? "ime_lhs_fragment" : "ime_rhs_fragment");
  line("_Alignas(" + std::to_string(packing.getAlignment()) + ") uint8_t " +
       name + "[" + std::to_string(storageBytes) + "];");
  const uint64_t mask = integer.getWidth() == 64
                            ? ~uint64_t(0)
                            : ((uint64_t(1) << integer.getWidth()) - 1);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column) {
      const int64_t tileRow = row / packing.getRowsPerTile();
      const int64_t tileColumn = column / packing.getColumnsPerTile();
      const int64_t tile = packing.getTileOrder() == "row_major"
                               ? tileRow * tileColumns + tileColumn
                               : tileColumn * tileRows + tileRow;
      const int64_t innerRow = row % packing.getRowsPerTile();
      const int64_t innerColumn = column % packing.getColumnsPerTile();
      const int64_t inner = packing.getElementOrder() == "row_major"
                                ? innerRow * packing.getColumnsPerTile() +
                                      innerColumn
                                : innerColumn * packing.getRowsPerTile() +
                                      innerRow;
      const int64_t index = tile * tileBytes + inner;
      llvm::SmallVector<int64_t, 2> logicalCoordinates(2, 0);
      logicalCoordinates[axisOrder[0]] = row;
      logicalCoordinates[axisOrder[1]] = column;
      const int64_t logicalIndex =
          logicalCoordinates[0] * shape[1] + logicalCoordinates[1];
      line(name + "[" + std::to_string(index) + "] = (uint8_t)(" + logical +
           "[" + std::to_string(logicalIndex) + "] & " +
           std::to_string(mask) + ");");
    }
  result.parts.push_back(name);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentMMA(
    riscv::IMEFragmentMMAOp operation) {
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  if (lhs.kind != Binding::Kind::Fragment ||
      rhs.kind != Binding::Kind::Fragment || lhs.parts.size() != 1 ||
      rhs.parts.size() != 1 ||
      lhs.fragmentFamily != operation.getResult().getType().getFamily() ||
      rhs.fragmentFamily != operation.getResult().getType().getFamily() ||
      operation.getLeaf().getInstruction() != "spacemit-ime1-i8-mma" ||
      !operation.getVolatileAsm() || !operation.getMemoryClobber() ||
      operation.getAsmClobbers().empty())
    return fail(operation, "IME MMA operands do not match the selected fragment leaf");
  auto shape = operation.getResult().getType().getShape().asArrayRef();
  if (shape.size() != 2 || shape[0] != 4 || shape[1] != 4 ||
      operation.getGroups() != 1 || operation.getChunks() != 1)
    return fail(operation, "IME MMA spelling requires one selected M4xN4xK8 fragment");
  const std::string output = fresh("ime_acc_fragment");
  const std::string lhsPointer = fresh("ime_lhs");
  const std::string rhsPointer = fresh("ime_rhs");
  const std::string lowPointer = fresh("ime_acc_low");
  const std::string highPointer = fresh("ime_acc_high");
  line("_Alignas(32) int32_t " + output + "[16] = {0};");
  line("const int8_t *" + lhsPointer + " = (const int8_t *)" +
       lhs.parts.front() + ";");
  line("const int8_t *" + rhsPointer + " = (const int8_t *)" +
       rhs.parts.front() + ";");
  line("int32_t *" + lowPointer + " = " + output + ";");
  line("int32_t *" + highPointer + " = " + output + " + 8;");
  line("__asm__ volatile(");
  ++indent;
  line("\"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
  line("\"vle8.v v0, (%[lhs])\\n\\t\"");
  line("\"vle8.v v1, (%[rhs])\\n\\t\"");
  line("\"vmv.v.i v2, 0\\n\\t\"");
  line("\"vmv.v.i v3, 0\\n\\t\"");
  line("\"vmadot v2, v0, v1\\n\\t\"");
  line("\"vsetvli t0, zero, e32, m1, ta, ma\\n\\t\"");
  line("\"vse32.v v2, (%[low])\\n\\t\"");
  line("\"vse32.v v3, (%[high])\"");
  line(":");
  line(": [lhs] \"r\"(" + lhsPointer + "), [rhs] \"r\"(" + rhsPointer +
       "), [low] \"r\"(" + lowPointer + "), [high] \"r\"(" +
       highPointer + ")");
  std::string clobbers = "\"memory\"";
  for (mlir::Attribute attribute : operation.getAsmClobbers())
    clobbers += ", \"" +
                mlir::cast<mlir::StringAttr>(attribute).getValue().str() +
                "\"";
  line(": " + clobbers + ");");
  --indent;

  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = operation.getResult().getType().getFamily().str();
  result.parts.push_back(output);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEUnpack(riscv::IMEUnpackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  if (input.kind != Binding::Kind::Fragment || input.parts.size() != 1 ||
      input.fragmentFamily != operation.getInput().getType().getFamily() ||
      operation.getLeaf().getEngine() != "ime" ||
      operation.getLeaf().getFamily() != "fragment-unpack" ||
      operation.getLeaf().getInstruction() != "ime.unpack.rvv" ||
      operation.getConversion().getKind() != "fragment_to_rvv" ||
      operation.getConversion().getEffect() != "handoff")
    return fail(operation, "IME unpack input has no typed fragment binding");
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t lanes = physicalLanes(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    const std::string name = fresh("ime_unpack");
    line(vectorType(operation.getResult()) + " " + name +
         " = __riscv_vle32_v_" + suffix + "(" + input.parts.front() +
         " + " + std::to_string(part * lanes) + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(name);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileConvertLayout(riscv::ConvertLayoutOp conversion) {
  Binding input = bindings.lookup(conversion.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(conversion, "layout conversion input has no emitted value");
  if (auto access = conversion->getAttrOfType<riscv::AccessAttr>("source_access");
      access && input.kind == Binding::Kind::Field)
    input.field.useAccess = access;
  llvm::StringRef kind = conversion.getConversion().getKind();
  if (kind == "local_load") {
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(conversion.getResult(), std::move(input));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[conversion.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  if (kind == "splat") {
    if (input.kind != Binding::Kind::Scalar)
      return fail(conversion, "RVV splat requires one scalar input");
    mlir::FailureOr<Binding> result = makeVector(
        conversion.getResult(), "layout_splat", input, conversion.getInput());
    if (mlir::failed(result))
      return mlir::failure();
    bindings[conversion.getResult()] = std::move(*result);
    return mlir::success();
  }
  if (kind == "register_to_lane" &&
      (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field)) {
    // The conversion's source layout is an explicit register tuple.  Load that
    // source representation first, then perform the typed tuple-to-lane pack.
    // Loading the field directly as the result layout would bypass a genuine
    // physical conversion and loses tuple-valued gather coordinates.
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(conversion.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (kind == "register_to_lane" && input.kind == Binding::Kind::Vector) {
    auto sourceType = conversion.getInput().getType();
    auto resultType = conversion.getResult().getType();
    auto sourceLayout = sourceType.getLayout();
    auto resultLayout = resultType.getLayout();
    const auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    const auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    const auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    const auto targetTime = resultLayout.getTimeFactors().asArrayRef();
    const auto targetLane = resultLayout.getLaneFactors().asArrayRef();
    const auto targetReplica = resultLayout.getReplicaFactors().asArrayRef();
    if (sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getShape() != resultType.getShape() ||
        sourceType.getAxisIds() != resultType.getAxisIds() ||
        sourceLayout.getSew() != resultLayout.getSew() ||
        sourceTime.size() != targetTime.size())
      return fail(conversion,
                  "vector register-to-lane conversion changes its logical domain");

    llvm::SmallVector<size_t> movedAxes;
    llvm::SmallVector<int64_t> pieceFactors;
    int64_t pieces = 1;
    for (size_t position = 0; position < sourceTime.size(); ++position) {
      const bool moved = sourceReplica[position] > targetReplica[position] &&
                         targetLane[position] > sourceLane[position];
      if (moved) {
        const int64_t sourceExtent =
            sourceTime[position] * sourceLane[position] *
            sourceReplica[position];
        const int64_t targetExtent =
            targetTime[position] * targetLane[position] *
            targetReplica[position];
        if (sourceLane[position] <= 0 ||
            targetLane[position] % sourceLane[position] ||
            sourceExtent != targetExtent ||
            pieces > std::numeric_limits<int64_t>::max() /
                         (targetLane[position] / sourceLane[position]))
          return fail(conversion,
                      "vector register-to-lane conversion has no closed moved-axis geometry");
        movedAxes.push_back(position);
        pieceFactors.push_back(targetLane[position] / sourceLane[position]);
        pieces *= pieceFactors.back();
        continue;
      }
      if (sourceLane[position] != targetLane[position] ||
          sourceTime[position] != targetTime[position] ||
          sourceReplica[position] != targetReplica[position])
        return fail(conversion,
                    "vector register-to-lane conversion remaps an unrelated axis");
    }
    const int64_t sourceLanes = physicalLanes(conversion.getInput());
    const int64_t resultLanes = physicalLanes(conversion.getResult());
    const int64_t sourceStreams = streamPartCount(conversion.getInput());
    const int64_t resultStreams = streamPartCount(conversion.getResult());
    const int64_t resultRegisters = registerPartCount(conversion.getResult());
    if (movedAxes.empty() || pieces <= 1 || sourceLanes <= 0 ||
        resultLanes <= 0 ||
        resultLanes != sourceLanes * pieces || sourceStreams <= 0 ||
        resultStreams <= 0 || resultRegisters <= 0 ||
        resultLayout.getLmulEighths() !=
            sourceLayout.getLmulEighths() * pieces ||
        (pieces & (pieces - 1)))
      return fail(conversion,
                  "vector register-to-lane conversion has incomplete pack geometry");

    auto decode = [](int64_t linear, llvm::ArrayRef<int64_t> factors) {
      llvm::SmallVector<int64_t> coordinates(factors.size(), 0);
      for (int64_t position = static_cast<int64_t>(factors.size()) - 1;
           position >= 0; --position) {
        coordinates[static_cast<size_t>(position)] =
            linear % factors[static_cast<size_t>(position)];
        linear /= factors[static_cast<size_t>(position)];
      }
      return std::pair{std::move(coordinates), linear};
    };
    auto encode = [](llvm::ArrayRef<int64_t> coordinates,
                     llvm::ArrayRef<int64_t> factors) {
      int64_t linear = 0;
      for (auto [coordinate, factor] : llvm::zip(coordinates, factors))
        linear = linear * factor + coordinate;
      return linear;
    };
    const std::string sourceSuffix = vectorSuffix(conversion.getInput());
    const std::string resultSuffix = vectorSuffix(conversion.getResult());
    const char category = resultSuffix.front();
    auto suffixFor = [&](int64_t lmulEighths) {
      return std::string(1, category) +
             std::to_string(resultLayout.getSew()) +
             lmulSpelling(lmulEighths);
    };
    auto typeFor = [&](int64_t lmulEighths) {
      const std::string suffix = suffixFor(lmulEighths);
      const std::string prefix =
          category == 'f' ? "vfloat" : category == 'u' ? "vuint" : "vint";
      return prefix + suffix.substr(1) + "_t";
    };
    if (sourceSuffix != suffixFor(sourceLayout.getLmulEighths()) ||
        resultSuffix != suffixFor(resultLayout.getLmulEighths()))
      return fail(conversion,
                  "vector register-to-lane conversion has inconsistent RVV spelling");

    Binding packed;
    packed.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto [targetRegisters, remainingRegister] =
          decode(resultRegister, targetReplica);
      if (remainingRegister)
        return fail(conversion,
                    "vector register-to-lane target register is out of range");
      for (int64_t resultStream = 0; resultStream < resultStreams;
           ++resultStream) {
        auto [targetTimes, remainingStream] = decode(resultStream, targetTime);
        if (remainingStream)
          return fail(conversion,
                      "vector register-to-lane target stream is out of range");
        llvm::SmallVector<std::string> level;
        for (int64_t piece = 0; piece < pieces; ++piece) {
          auto [pieceCoordinates, remainingPiece] = decode(piece, pieceFactors);
          if (remainingPiece)
            return fail(conversion,
                        "vector register-to-lane moved-axis coordinate is out of range");
          llvm::SmallVector<int64_t> sourceTimes(sourceTime.size(), 0);
          llvm::SmallVector<int64_t> sourceRegisters(sourceReplica.size(), 0);
          for (size_t position = 0; position < sourceTime.size(); ++position) {
            int64_t logicalBase =
                (targetTimes[position] * targetReplica[position] +
                 targetRegisters[position]) *
                targetLane[position];
            auto moved = llvm::find(movedAxes, position);
            if (moved != movedAxes.end()) {
              const size_t movedPosition = static_cast<size_t>(
                  moved - movedAxes.begin());
              logicalBase += pieceCoordinates[movedPosition] *
                             sourceLane[position];
            }
            const int64_t sourceTile =
                sourceLane[position] * sourceReplica[position];
            if (sourceTile <= 0 || logicalBase < 0 ||
                logicalBase % sourceLane[position] ||
                logicalBase / sourceTile >= sourceTime[position])
              return fail(conversion,
                          "vector register-to-lane source coordinate is not aligned");
            sourceTimes[position] = logicalBase / sourceTile;
            sourceRegisters[position] =
                (logicalBase % sourceTile) / sourceLane[position];
          }
          const int64_t sourcePart =
              encode(sourceRegisters, sourceReplica) * sourceStreams +
              encode(sourceTimes, sourceTime);
          if (sourcePart < 0 ||
              sourcePart >= static_cast<int64_t>(input.parts.size()))
            return fail(conversion,
                        "vector register-to-lane source part is out of range");
          level.push_back(input.parts[static_cast<size_t>(sourcePart)]);
        }
        int64_t currentLMUL = sourceLayout.getLmulEighths();
        int64_t currentLanes = sourceLayout.getVl();
        while (level.size() > 1) {
          llvm::SmallVector<std::string> next;
          const int64_t nextLMUL = currentLMUL * 2;
          const int64_t nextLanes = currentLanes * 2;
          for (size_t index = 0; index < level.size(); index += 2) {
            std::string name = fresh("layout_pack");
            std::string expression;
            if (currentLMUL >= 8) {
              expression = "__riscv_vcreate_v_" + suffixFor(currentLMUL) +
                           "_" + suffixFor(nextLMUL) + "(" + level[index] +
                           ", " + level[index + 1] + ")";
            } else {
              const std::string extend =
                  "__riscv_vlmul_ext_v_" + suffixFor(currentLMUL) + "_" +
                  suffixFor(nextLMUL);
              expression = "__riscv_vslideup_vx_" + suffixFor(nextLMUL) +
                           "(" + extend + "(" + level[index] + "), " +
                           extend + "(" + level[index + 1] + "), " +
                           std::to_string(currentLanes) + ", " +
                           std::to_string(nextLanes) + ")";
            }
            line(typeFor(nextLMUL) + " " + name + " = " + expression + ";");
            next.push_back(std::move(name));
          }
          level = std::move(next);
          currentLMUL = nextLMUL;
          currentLanes = nextLanes;
        }
        if (level.size() != 1 || currentLMUL != resultLayout.getLmulEighths())
          return fail(conversion,
                      "vector register-to-lane pack did not reach its target LMUL");
        packed.parts.push_back(std::move(level.front()));
      }
    }
    bindings[conversion.getResult()] = std::move(packed);
    return mlir::success();
  }
  if (kind == "register_to_lane" &&
      input.kind == Binding::Kind::ScalarTuple) {
    auto sourceLayout = layoutOf(conversion.getInput());
    auto resultLayout = layoutOf(conversion.getResult());
    const int64_t laneAxis = laneAxisFor(conversion.getResult());
    const int64_t lanes = physicalLanes(conversion.getResult());
    const int64_t streams = streamPartCount(conversion.getResult());
    const int64_t resultRegisters =
        registerPartCount(conversion.getResult());
    auto sourceAxes = registerAxesFor(conversion.getInput());
    auto sourceExtents = registerExtentsFor(conversion.getInput());
    auto resultAxes = registerAxesFor(conversion.getResult());
    if (!sourceLayout || !resultLayout || laneAxis <= 0 || lanes <= 1 ||
        streams <= 0 || resultRegisters <= 0 ||
        streamPartCount(conversion.getInput()) != 1 ||
        sourceAxes.size() != sourceExtents.size() ||
        !llvm::is_contained(sourceAxes, laneAxis))
      return fail(conversion,
                  "register-to-lane conversion has incomplete typed factors");

    mlir::Type element =
        riscv_internal::logicalElement(conversion.getResult().getType());
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    const bool floating = mlir::isa<mlir::FloatType>(element);
    if (!floating && !integer)
      return fail(conversion,
                  "register-to-lane conversion has no scalar element type");
    const std::string suffix = vectorSuffix(conversion.getResult());
    const std::string vectorCType = vectorType(conversion.getResult());
    const std::string broadcast =
        floating ? "__riscv_vfmv_v_f_" : "__riscv_vmv_v_x_";
    const std::string slide =
        floating ? "__riscv_vfslide1up_vf_" : "__riscv_vslide1up_vx_";

    Binding packed;
    packed.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto resultCoordinates =
          registerCoordinates(conversion.getResult(), resultRegister);
      if (!resultCoordinates || resultCoordinates->size() != resultAxes.size())
        return fail(conversion,
                    "register-to-lane result has no register coordinates");
      for (int64_t stream = 0; stream < streams; ++stream) {
        llvm::SmallVector<size_t, 16> sourceParts;
        for (int64_t lane = 0; lane < lanes; ++lane) {
          const int64_t laneCoordinate = stream * lanes + lane;
          int64_t sourcePart = 0;
          for (auto [axis, extent] : llvm::zip(sourceAxes, sourceExtents)) {
            int64_t coordinate = laneCoordinate;
            if (axis != laneAxis) {
              auto found = llvm::find(resultAxes, axis);
              if (found == resultAxes.end())
                return fail(
                    conversion,
                    "register-to-lane source axis is absent from result registers");
              coordinate = (*resultCoordinates)[static_cast<size_t>(
                  found - resultAxes.begin())];
            }
            if (extent <= 0 || coordinate < 0 || coordinate >= extent)
              return fail(conversion,
                          "register-to-lane coordinate exceeds source tuple");
            sourcePart = sourcePart * extent + coordinate;
          }
          if (sourcePart < 0 ||
              sourcePart >= static_cast<int64_t>(input.parts.size()))
            return fail(conversion,
                        "register-to-lane source part is outside its tuple");
          sourceParts.push_back(static_cast<size_t>(sourcePart));
        }
        const int64_t resultPart = resultRegister * streams + stream;
        const std::string vl = partVL(conversion.getResult(), resultPart);
        std::string expression = broadcast + suffix + "(" +
                                 input.parts[sourceParts.back()] + ", " + vl +
                                 ")";
        for (int64_t lane = lanes - 2; lane >= 0; --lane)
          expression = slide + suffix + "(" + expression + ", " +
                       input.parts[sourceParts[static_cast<size_t>(lane)]] +
                       ", " + vl + ")";
        std::string name = fresh("layout_pack");
        line(vectorCType + " " + name + " = " + expression + ";");
        packed.parts.push_back(std::move(name));
      }
    }
    bindings[conversion.getResult()] = std::move(packed);
    return mlir::success();
  }
  if (kind == "time_to_lane" &&
      (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field)) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(conversion.getResult(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    bindings[conversion.getResult()] = std::move(*materialized);
    return mlir::success();
  }
  if (kind == "tuple" || kind == "reshape" ||
      kind == "register_to_lane") {
    if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
      mlir::FailureOr<Binding> materialized =
          materializeNumeric(conversion.getInput(), std::move(input));
      if (mlir::failed(materialized))
        return mlir::failure();
      input = std::move(*materialized);
    }
    auto sourceType = conversion.getInput().getType();
    auto resultType = conversion.getResult().getType();
    auto sourceLayout = sourceType.getLayout();
    auto resultLayout = resultType.getLayout();
    if (input.kind == Binding::Kind::Vector &&
        sourceLayout.getCarrier() == "rvv" &&
        resultLayout.getCarrier() == "rvv" &&
        sourceType.getElementType() == resultType.getElementType() &&
        laneAxisFor(conversion.getInput()) == laneAxisFor(conversion.getResult())) {
      const int64_t sourceLanes = physicalLanes(conversion.getInput());
      const int64_t resultLanes = physicalLanes(conversion.getResult());
      const int64_t sourceStreams = streamPartCount(conversion.getInput());
      const int64_t resultStreams = streamPartCount(conversion.getResult());
      const int64_t resultRegisters = registerPartCount(conversion.getResult());
      const std::string sourceSuffix = vectorSuffix(conversion.getInput());
      const std::string resultSuffix = vectorSuffix(conversion.getResult());
      const std::string resultCType = vectorType(conversion.getResult());
      if (sourceLanes <= 0 || resultLanes <= 0 || sourceStreams <= 0 ||
          resultStreams <= 0 || resultRegisters <= 0)
        return fail(conversion, "RVV repartition has incomplete typed factors");

      Binding repartitioned;
      repartitioned.kind = Binding::Kind::Vector;
      if (sourceLanes == resultLanes && sourceSuffix == resultSuffix) {
        for (int64_t resultPart = 0;
             resultPart < resultRegisters * resultStreams; ++resultPart) {
          auto sourcePart = projectPart(conversion.getInput(),
                                        conversion.getResult(), resultPart);
          if (!sourcePart || *sourcePart >= input.parts.size())
            return fail(conversion,
                        "RVV repartition cannot project its typed coordinates");
          repartitioned.parts.push_back(input.parts[*sourcePart]);
        }
        bindings[conversion.getResult()] = std::move(repartitioned);
        return mlir::success();
      }
      for (int64_t resultRegister = 0; resultRegister < resultRegisters;
           ++resultRegister) {
        auto sourceRegister = projectRegisterPart(
            conversion.getInput(), conversion.getResult(), resultRegister);
        if (!sourceRegister)
          return fail(conversion,
                      "RVV repartition cannot preserve register-axis coordinates");
        for (int64_t resultStream = 0; resultStream < resultStreams;
             ++resultStream) {
          const int64_t logicalOffset = resultStream * resultLanes;
          std::string expression;
          if (resultLanes <= sourceLanes) {
            const int64_t sourceStream = logicalOffset / sourceLanes;
            const int64_t laneOffset = logicalOffset % sourceLanes;
            const size_t sourcePart =
                *sourceRegister * sourceStreams + sourceStream;
            if (sourceStream >= sourceStreams || sourcePart >= input.parts.size())
              return fail(conversion,
                          "RVV split repartition exceeds its source streams");
            expression = input.parts[sourcePart];
            if (laneOffset)
              expression = "__riscv_vslidedown_vx_" + sourceSuffix + "(" +
                           expression + ", " + std::to_string(laneOffset) +
                           ", " + std::to_string(sourceLanes) + ")";
            if (sourceSuffix != resultSuffix)
              expression = "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" +
                           resultSuffix + "(" + expression + ")";
          } else {
            if (resultLanes % sourceLanes != 0)
              return fail(conversion,
                          "RVV merge repartition requires integral lane groups");
            const int64_t pieces = resultLanes / sourceLanes;
            auto sourceLayout = layoutOf(conversion.getInput());
            auto resultLayout = layoutOf(conversion.getResult());
            if (pieces == 2 && sourceLayout && resultLayout &&
                sourceLayout.getSew() == resultLayout.getSew() &&
                sourceLayout.getLmulEighths() >= 8 &&
                sourceLayout.getLmulEighths() * 2 ==
                    resultLayout.getLmulEighths()) {
              const int64_t firstStream = resultStream * pieces;
              const size_t firstPart =
                  *sourceRegister * sourceStreams + firstStream;
              const size_t secondPart = firstPart + 1;
              if (firstStream + 1 >= sourceStreams ||
                  secondPart >= input.parts.size())
                return fail(conversion,
                            "RVV create repartition exceeds its source streams");
              expression = "__riscv_vcreate_v_" + sourceSuffix + "_" +
                           resultSuffix + "(" + input.parts[firstPart] + ", " +
                           input.parts[secondPart] + ")";
              std::string name = fresh("layout_repartition");
              line(resultCType + " " + name + " = " + expression + ";");
              repartitioned.parts.push_back(std::move(name));
              continue;
            }
            mlir::Type element = resultType.getElementType();
            std::string merged =
                std::string(mlir::isa<mlir::FloatType>(element)
                                ? "__riscv_vfmv_v_f_"
                                : "__riscv_vmv_v_x_") +
                resultSuffix + "(" +
                (mlir::isa<mlir::FloatType>(element) ? "0.0" : "0") + ", " +
                partVL(conversion.getResult(),
                       resultRegister * resultStreams + resultStream) +
                ")";
            for (int64_t piece = 0; piece < pieces; ++piece) {
              const int64_t sourceStream =
                  resultStream * pieces + piece;
              const size_t sourcePart =
                  *sourceRegister * sourceStreams + sourceStream;
              if (sourceStream >= sourceStreams || sourcePart >= input.parts.size())
                return fail(conversion,
                            "RVV merge repartition exceeds its source streams");
              std::string extended = input.parts[sourcePart];
              if (sourceSuffix != resultSuffix)
                extended = "__riscv_vlmul_ext_v_" + sourceSuffix + "_" +
                           resultSuffix + "(" + extended + ")";
              merged = "__riscv_vslideup_vx_" + resultSuffix + "(" + merged +
                       ", " + extended + ", " +
                       std::to_string(piece * sourceLanes) + ", " +
                       partVL(conversion.getResult(),
                              resultRegister * resultStreams + resultStream) +
                       ")";
            }
            expression = std::move(merged);
          }
          std::string name = fresh("layout_repartition");
          line(resultCType + " " + name + " = " + expression + ";");
          repartitioned.parts.push_back(std::move(name));
        }
      }
      bindings[conversion.getResult()] = std::move(repartitioned);
      return mlir::success();
    }
    mlir::FailureOr<Binding> projected =
        projectBinding(conversion.getInput(), conversion.getResult(), input);
    if (mlir::failed(projected))
      return fail(conversion,
                  "layout conversion is not a projection of its typed layouts");
    bindings[conversion.getResult()] = std::move(*projected);
    return mlir::success();
  }
  if (kind == "extract" || kind == "lane_to_register") {
    if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
      mlir::FailureOr<Binding> materialized =
          materializeNumeric(conversion.getInput(), std::move(input));
      if (mlir::failed(materialized))
        return mlir::failure();
      input = std::move(*materialized);
    }
    if (input.kind != Binding::Kind::Vector || input.parts.empty())
      return fail(conversion, "RVV extract requires one vector input");
    mlir::Type element =
        riscv_internal::logicalElement(conversion.getResult().getType());
    auto scalarType = scalarCType(element);
    if (!scalarType)
      return fail(conversion, "RVV extract result has no scalar C type");
    int64_t parts = scalarPartCount(conversion.getResult());
    if (parts == 1) {
      std::string failureReason;
      auto extracted = extractLaneForRegisterBroadcast(
          conversion.getInput(), conversion.getResult(), 0, input,
          &failureReason);
      if (!extracted) {
        std::string operationText;
        llvm::raw_string_ostream operationStream(operationText);
        conversion.print(operationStream);
        return fail(conversion,
                    llvm::Twine("RVV extract cannot preserve its typed register coordinates; part=0") +
                        ", source_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getInput())) +
                        ", result_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getResult())) +
                        ", source_register_parts=" +
                        llvm::Twine(registerPartCount(conversion.getInput())) +
                        ", source_stream_parts=" +
                        llvm::Twine(streamPartCount(conversion.getInput())) +
                        ", input_binding_parts=" +
                        llvm::Twine(input.parts.size()) + ", reason=" +
                        failureReason + ", operation=" + operationText);
      }
      bindings[conversion.getResult()] = scalar(*extracted);
      return mlir::success();
    }
    Binding tuple;
    tuple.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part) {
      std::string failureReason;
      auto extracted = extractLaneForRegisterBroadcast(
          conversion.getInput(), conversion.getResult(), part, input,
          &failureReason);
      if (!extracted) {
        std::string operationText;
        llvm::raw_string_ostream operationStream(operationText);
        conversion.print(operationStream);
        return fail(conversion,
                    llvm::Twine("RVV extract cannot preserve its typed register coordinates; part=") +
                        llvm::Twine(part) + ", source_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getInput())) +
                        ", result_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getResult())) +
                        ", source_register_parts=" +
                        llvm::Twine(registerPartCount(conversion.getInput())) +
                        ", source_stream_parts=" +
                        llvm::Twine(streamPartCount(conversion.getInput())) +
                        ", input_binding_parts=" +
                        llvm::Twine(input.parts.size()) + ", reason=" +
                        failureReason + ", operation=" + operationText);
      }
      tuple.parts.push_back(*extracted);
    }
    bindings[conversion.getResult()] = std::move(tuple);
    return mlir::success();
  }
  return fail(conversion,
              "layout conversion kind is not a terminal RVV representation operation");
}

mlir::LogicalResult
Emitter::compilePackedPlaneMerge(riscv::PackedPlaneMergeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "scalar.packed-plane-merge.words")
    return fail(operation, "packed plane merge has no exact selected leaf");
  Binding low = bindings.lookup(operation.getLowField());
  Binding high = bindings.lookup(operation.getHighField());
  if (low.kind != Binding::Kind::Field || high.kind != Binding::Kind::Field ||
      low.field.owner != high.field.owner)
    return fail(operation,
                "packed plane merge requires two fields from one encoded record");
  Binding owner = bindings.lookup(low.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(low.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto lowField = fieldFor(low);
  auto highField = fieldFor(high);
  auto plan = operation.getPlan();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !lowField || !highField || lowField->bitOffset % 8 ||
      highField->bitOffset % 8 ||
      lowField->bitOffset / 8 != plan.getLowByteOffset() ||
      highField->bitOffset / 8 != plan.getHighByteOffset() ||
      registerPartCount(operation.getResult()) != plan.getLogicalElements())
    return fail(operation,
                "packed plane merge has no closed record and scalar-tuple mapping");

  llvm::DenseMap<int64_t, std::string> loadedWords;
  auto loadWord = [&](int64_t byteOffset) -> std::string {
    auto found = loadedWords.find(byteOffset);
    if (found != loadedWords.end())
      return found->second;
    std::string name = fresh("packed_plane_word");
    line("uint32_t " + name + " = weft_load_u32_le((const uint8_t *)(" +
         owner.recordPointer + " + " + std::to_string(byteOffset) + "));");
    loadedWords[byteOffset] = name;
    return name;
  };
  auto repeatedByteMask = [](int64_t bits) -> uint32_t {
    const uint32_t byte = (uint32_t{1} << bits) - 1;
    return byte * uint32_t{0x01010101};
  };

  Binding result;
  result.kind = Binding::Kind::ScalarTuple;
  const int64_t words = plan.getLogicalElements() / 4;
  for (int64_t word = 0; word < words; ++word) {
    const int64_t logical = word * 4;
    const int64_t lowLayer = logical / plan.getLowLayerBytes();
    const int64_t highLayer = logical / plan.getHighLayerBytes();
    const int64_t lowOffset =
        plan.getLowByteOffset() + logical % plan.getLowLayerBytes();
    const int64_t highOffset =
        plan.getHighByteOffset() + logical % plan.getHighLayerBytes();
    const std::string lowWord = loadWord(lowOffset);
    const std::string highWord = loadWord(highOffset);
    const uint32_t lowMask = repeatedByteMask(plan.getLowBits());
    const uint32_t highMask = repeatedByteMask(plan.getHighBits());
    std::string merged = fresh("packed_plane_merge");
    line("uint32_t " + merged + " = ((" + lowWord + " >> " +
         std::to_string(lowLayer * plan.getLowBits()) + ") & " +
         std::to_string(lowMask) + "u) | (((" + highWord + " >> " +
         std::to_string(highLayer * plan.getHighBits()) + ") & " +
         std::to_string(highMask) + "u) << " +
         std::to_string(plan.getInsertBit()) + ");");
    for (int64_t byte = 0; byte < 4; ++byte)
      result.parts.push_back("((uint8_t)((" + merged + " >> " +
                             std::to_string(byte * 8) + ") & 255u))");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVBitplaneMerge(riscv::RVVBitplaneMergeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.bitplane-merge.mask" &&
      instruction != "rvv.bitplane-merge.strided")
    return fail(operation, "RVV bitplane merge has no exact selected leaf");
  mlir::FailureOr<Binding> low = materializeNumeric(
      operation.getLow(), bindings.lookup(operation.getLow()));
  Binding plane = bindings.lookup(operation.getPlane());
  if (mlir::failed(low) || low->kind != Binding::Kind::Vector ||
      plane.kind != Binding::Kind::Field)
    return fail(operation,
                "RVV bitplane merge requires one vector value and one encoded field edge");
  Binding owner = bindings.lookup(plane.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(plane.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(plane);
  auto planeInteger = field
                          ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                          : mlir::IntegerType();
  const bool logicalPlane =
      planeInteger && planeInteger.isUnsigned() && planeInteger.getWidth() == 1 &&
      field->access.getForm() == "indexed" &&
      field->access.getMapping() == "grouped_layered" &&
      field->access.getGroupSize() == 8 && field->access.getLayerSize() == 1;
  const bool bytePlane =
      planeInteger && planeInteger.isUnsigned() && planeInteger.getWidth() == 8 &&
      field->access.getMapping() == "natural";
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || (!logicalPlane && !bytePlane) || field->bitOffset % 8)
    return fail(operation,
                "RVV bitplane merge field has no direct byte-plane address");

  auto layout = operation.getResult().getType().getLayout();
  if (instruction == "rvv.bitplane-merge.strided") {
    if (!bytePlane)
      return fail(operation,
                  "strided RVV bitplane merge requires a byte-plane field");
    auto lowType = operation.getResult().getType();
    auto planeType = operation.getPlane().getType();
    int64_t bitplaneAxis = 0;
    const int64_t laneAxis = laneAxisFor(operation.getResult());
    for (auto [axis, lowExtent, planeExtent] :
         llvm::zip(lowType.getAxisIds().asArrayRef(),
                   lowType.getShape().asArrayRef(),
                   planeType.getShape().asArrayRef()))
      if (lowExtent != planeExtent) {
        if (bitplaneAxis || planeExtent <= 0 || lowExtent != planeExtent * 8)
          return fail(operation,
                      "strided bitplane merge has no unique packed logical axis");
        bitplaneAxis = axis;
      }
    auto laneStride = llvm::find_if(owner.recordByteStrides, [&](const auto &entry) {
      return entry.first == laneAxis;
    });
    auto packed = llvm::find(lowType.getAxisIds().asArrayRef(), bitplaneAxis);
    if (!bitplaneAxis || laneAxis <= 0 || laneAxis == bitplaneAxis ||
        laneStride == owner.recordByteStrides.end() ||
        packed == lowType.getAxisIds().asArrayRef().end())
      return fail(operation,
                  "strided bitplane merge has no lane stride or packed-axis mapping");
    const size_t packedPosition = static_cast<size_t>(
        packed - lowType.getAxisIds().asArrayRef().begin());
    const int64_t streams = streamPartCount(operation.getResult());
    const int64_t parts = vectorPartCount(operation.getResult());
    llvm::SmallVector<int64_t, 4> registerAxes =
        registerAxesFor(operation.getResult());
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < parts; ++part) {
      auto lowPart = mappedPart(operation.getOperation(), 0, part);
      auto coordinates =
          registerCoordinates(operation.getResult(), part / streams);
      if (!lowPart || *lowPart >= low->parts.size() || streams <= 0 ||
          !coordinates || coordinates->size() != registerAxes.size())
        return fail(operation,
                    "strided bitplane merge lacks a typed value/register mapping");
      int64_t stream = part % streams;
      int64_t bitCoordinate = 0;
      for (int64_t dimension =
               static_cast<int64_t>(layout.getAxisIds().size()) - 1;
           dimension >= 0; --dimension) {
        const int64_t factor = layout.getTimeFactors()[dimension];
        if (factor <= 0)
          return fail(operation,
                      "strided bitplane merge has an invalid time factor");
        const int64_t coordinate = stream % factor;
        stream /= factor;
        if (static_cast<size_t>(dimension) == packedPosition)
          bitCoordinate = coordinate;
      }
      if (stream != 0)
        return fail(operation,
                    "strided bitplane merge time coordinates are incomplete");
      std::string record = "(" + owner.recordPointer;
      for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
        auto stride = llvm::find_if(owner.recordByteStrides,
                                    [&](const auto &entry) {
                                      return entry.first == axis;
                                    });
        if (stride == owner.recordByteStrides.end())
          return fail(operation,
                      "strided bitplane register axis has no record-byte stride");
        record += " + " + std::to_string(coordinate) + " * " + stride->second;
      }
      record += " + (" + partOffset(operation.getResult(), part) + ") * " +
                laneStride->second + " + " +
                std::to_string(field->bitOffset / 8 + bitCoordinate / 8) + ")";
      const std::string vl = partVL(operation.getResult(), part);
      std::string loaded = fresh("bitplane_bytes");
      line(type + " " + loaded + " = __riscv_vlse8_v_" + suffix +
           "((const uint8_t *)(" + record + "), (ptrdiff_t)(" +
           laneStride->second + "), " + vl + ");");
      std::string bits = "__riscv_vsrl_vx_" + suffix + "(" + loaded + ", " +
                         std::to_string(bitCoordinate % 8) + ", " + vl + ")";
      bits = "__riscv_vand_vx_" + suffix + "(" + bits + ", 1, " + vl + ")";
      bits = "__riscv_vsll_vx_" + suffix + "(" + bits + ", " +
             std::to_string(operation.getInsertBit()) + ", " + vl + ")";
      std::string merged = fresh("bitplane_merge");
      line(type + " " + merged + " = __riscv_vor_vv_" + suffix + "(" +
           low->parts[*lowPart] + ", " + bits + ", " + vl + ");");
      result.parts.push_back(std::move(merged));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if ((maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
       maskBits != 16 && maskBits != 32 && maskBits != 64) ||
      layout.getVl() % 8)
    return fail(operation,
                "RVV bitplane merge selected an illegal mask ratio or byte boundary");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const std::string maskType =
      "vbool" + std::to_string(maskBits) + "_t";
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto lowPart = mappedPart(operation.getOperation(), 0, part);
    auto coordinates = registerCoordinates(operation.getResult(), part / streams);
    auto axes = registerAxesFor(operation.getResult());
    if (!lowPart || *lowPart >= low->parts.size() || !coordinates ||
        coordinates->size() != axes.size())
      return fail(operation,
                  "RVV bitplane merge lacks one typed value-use or register mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides, [&](const auto &entry) {
        return entry.first == axis;
      });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV bitplane merge register axis has no record-byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += " + " + std::to_string(field->bitOffset / 8) + " + (" +
              partOffset(operation.getResult(), part) + " / 8))";
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("bitplane_mask");
    line(maskType + " " + mask + " = __riscv_vlm_v_b" +
         std::to_string(maskBits) + "((const uint8_t *)(" + record + "), " +
         vl + ");");
    std::string merged = fresh("bitplane_merge");
    line(type + " " + merged + " = __riscv_vor_vx_" + suffix + "_mu(" +
         mask + ", " + low->parts[*lowPart] + ", " + low->parts[*lowPart] +
         ", " + std::to_string(int64_t(1) << operation.getInsertBit()) + ", " +
         vl + ");");
    result.parts.push_back(std::move(merged));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVBitmaskDecode(riscv::RVVBitmaskDecodeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.bitmask-decode")
    return fail(operation, "RVV bitmask decode has no exact selected leaf");
  Binding packed = bindings.lookup(operation.getField());
  Binding origin = bindings.lookup(operation.getOrigin());
  Binding point = bindings.lookup(operation.getPoint());
  if (packed.kind != Binding::Kind::Field ||
      origin.kind != Binding::Kind::Point || point.kind != Binding::Kind::Point)
    return fail(operation,
                "RVV bitmask decode requires one encoded field and two typed points");
  Binding owner = bindings.lookup(packed.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(packed.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(packed);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || !integer || !integer.isUnsigned() || integer.getWidth() != 1 ||
      field->access.getMapping() != "grouped_layered" ||
      field->access.getOrder() != "lo_first" || field->bitOffset % 8 ||
      field->access.getGroupSize() != field->access.getLayerSize() * 8)
    return fail(operation,
                "RVV bitmask decode field has no direct contiguous mask address");

  auto layout = operation.getResult().getType().getLayout();
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation, "RVV bitmask decode selected an illegal mask ratio");
  const int64_t group = field->access.getGroupSize();
  const int64_t layer = field->access.getLayerSize();
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const std::string maskType =
      "vbool" + std::to_string(maskBits) + "_t";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "RVV bitmask decode lacks a typed register mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV bitmask decode register axis has no record-byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    const std::string relative =
        "((" + point.point.base + ") - (" + origin.point.base + ") + " +
        partOffset(operation.getResult(), part) + ")";
    record += " + " + std::to_string(field->bitOffset / 8) + " + ((" +
              relative + ") / " + std::to_string(group) + ") * " +
              std::to_string(layer) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("bitmask");
    line(maskType + " " + mask + " = __riscv_vlm_v_b" +
         std::to_string(maskBits) + "((const uint8_t *)(" + record + "), " +
         vl + ");");
    std::string decoded = fresh("bitmask_decode");
    line(type + " " + decoded + " = __riscv_vmerge_vxm_" + suffix +
         "(__riscv_vmv_v_x_" + suffix + "(0, " + vl + "), 1, " + mask +
         ", " + vl + ");");
    result.parts.push_back(std::move(decoded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileGroupedMacReduce(
    riscv::RVVGroupedMacReduceOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.grouped-mac-reduce.u8-s8")
    return fail(operation,
                "grouped MAC reduction has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding active = bindings.lookup(operation.getActiveTerms());
  if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field ||
      active.kind != Binding::Kind::Scalar)
    return fail(operation,
                "grouped MAC reduction requires two fields and one active extent");
  auto lhsField = fieldFor(lhs);
  auto rhsField = fieldFor(rhs);
  Binding lhsOwner = bindings.lookup(lhs.field.owner);
  Binding rhsOwner = bindings.lookup(rhs.field.owner);
  if (lhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(lhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    lhsOwner = std::move(*record);
  }
  if (rhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(rhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    rhsOwner = std::move(*record);
  }
  auto lhsInteger = lhsField ? mlir::dyn_cast<mlir::IntegerType>(lhsField->type)
                             : mlir::IntegerType();
  auto rhsInteger = rhsField ? mlir::dyn_cast<mlir::IntegerType>(rhsField->type)
                             : mlir::IntegerType();
  auto lhsType = mlir::dyn_cast<riscv::ValueType>(operation.getLhs().getType());
  const int64_t outputParts = vectorPartCount(operation.getResult());
  const int64_t outputStreams = streamPartCount(operation.getResult());
  if (!lhsField || !rhsField || lhsOwner.kind != Binding::Kind::Record ||
      rhsOwner.kind != Binding::Kind::Record || !lhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 || !rhsInteger ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8 || !lhsType ||
      lhsType.getLayout().getCarrier() != "rvv" || outputParts <= 0 ||
      outputStreams <= 0 || outputParts % outputStreams)
    return fail(operation,
                "grouped MAC reduction fields do not match its selected mixed-width vector form");
  Binding point = lhs.field.index ? bindings.lookup(*lhs.field.index) : Binding();
  Binding rhsPoint = rhs.field.index ? bindings.lookup(*rhs.field.index) : Binding();
  if (point.kind != Binding::Kind::Point || rhsPoint.kind != Binding::Kind::Point ||
      point.point.axis != rhsPoint.point.axis ||
      point.point.base != rhsPoint.point.base || lhsOwner.recordElements <= 0)
    return fail(operation,
                "grouped MAC reduction operands do not share one typed reduction point");

  riscv::GroupedMacPlanAttr plan = operation.getPlan();
  const int64_t group = plan.getGroup();
  const int64_t unroll = plan.getUnroll();
  const int64_t storageGroup = plan.getStorageGroup();
  const int64_t storageLayer = plan.getStorageLayer();
  std::string rowStride;
  for (const auto &[axis, stride] : lhsOwner.recordByteStrides)
    if (axis == plan.getRowStrideAxis())
      rowStride = stride;
  if ((plan.getLoadForm() == "unit" && plan.getInterleaveRows() <= 0) ||
      (plan.getLoadForm() == "strided" && rowStride.empty()))
    return fail(operation,
                "grouped MAC reduction plan has no bound load form");

  const std::string outSuffix = vectorSuffix(operation.getResult());
  const std::string outType = vectorType(operation.getResult());
  const std::string vl = partVL(operation.getResult(), 0);
  const std::string partialSuffix =
      "i16" + lmulSpelling(operation.getPartialLayout().getLmulEighths());
  const std::string partialType =
      "vint16" + lmulSpelling(operation.getPartialLayout().getLmulEighths()) +
      "_t";
  const std::string rawSuffix =
      "u8" + lmulSpelling(operation.getLoadLayout().getLmulEighths());
  const std::string rawType =
      "vuint8" + lmulSpelling(operation.getLoadLayout().getLmulEighths()) +
      "_t";
  const std::string signedRawSuffix =
      "i8" + lmulSpelling(operation.getLoadLayout().getLmulEighths());
  const std::string signedRawType =
      "vint8" + lmulSpelling(operation.getLoadLayout().getLmulEighths()) +
      "_t";
  if (outSuffix.empty() || partialSuffix == "i16" || rawSuffix == "u8")
    return fail(operation, "grouped MAC reduction has an invalid RVV type spelling");

  llvm::SmallVector<int64_t, 4> resultAxes =
      registerAxesFor(operation.getResult());
  llvm::SmallVector<std::string> lhsRecords;
  llvm::SmallVector<std::string> rhsRecords;
  auto recordForPart = [&](const Binding &owner, riscv::ValueType operand,
                           int64_t part) -> std::optional<std::string> {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / outputStreams);
    if (!coordinates || coordinates->size() != resultAxes.size())
      return std::nullopt;
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(resultAxes, *coordinates)) {
      if (!llvm::is_contained(operand.getAxisIds().asArrayRef(), axis))
        continue;
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return std::nullopt;
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    const int64_t laneAxis = laneAxisFor(operation.getResult());
    if (llvm::is_contained(operand.getAxisIds().asArrayRef(), laneAxis)) {
      const std::string issueOffset = partOffset(operation.getResult(), part);
      if (plan.getLoadForm() == "unit")
        record += " + " + issueOffset;
      else
        record += " + (" + issueOffset + ") * (" + rowStride + ")";
    }
    record += ")";
    return record;
  };
  auto rhsType = mlir::dyn_cast<riscv::ValueType>(operation.getRhs().getType());
  if (!rhsType)
    return fail(operation, "grouped MAC reduction rhs has no physical value type");
  for (int64_t part = 0; part < outputParts; ++part) {
    auto lhsRecord = recordForPart(lhsOwner, lhsType, part);
    auto rhsRecord = recordForPart(rhsOwner, rhsType, part);
    if (!lhsRecord || !rhsRecord)
      return fail(operation,
                  "grouped MAC reduction output replica has no record-coordinate mapping");
    lhsRecords.push_back(std::move(*lhsRecord));
    rhsRecords.push_back(std::move(*rhsRecord));
  }

  llvm::SmallVector<std::string> results;
  for (int64_t part = 0; part < outputParts; ++part) {
    std::string result = fresh("grouped_reduce");
    line(outType + " " + result + " = __riscv_vmv_v_x_" + outSuffix +
         "(0, " + partVL(operation.getResult(), part) + ");");
    results.push_back(std::move(result));
  }
  std::string logical = fresh("group_logical_base");
  std::string within = fresh("group_layer_within");
  std::string byte = fresh("group_storage_byte");
  std::string shift = fresh("group_shift");
  line("const size_t " + logical + " = " + point.point.base + " % " +
       std::to_string(plan.getRecordElements()) + ";");
  line("const size_t " + within + " = " + logical + " % " +
       std::to_string(storageGroup) + ";");
  line("const size_t " + byte + " = " +
       std::to_string(plan.getLhsBitOffsetBytes()) + " + (" + logical + " / " +
       std::to_string(storageGroup) + ") * " +
       std::to_string(storageLayer) + " + " + within + " % " +
       std::to_string(storageLayer) + ";");
  line("const ptrdiff_t " + shift + " = " +
       std::to_string(plan.getShiftBase()) + " + (" + within + " / " +
       std::to_string(storageLayer) + ") * " +
       std::to_string(plan.getShiftStep()) + ";");
  const int64_t qTermStride = plan.getTermByteStride();
  llvm::SmallVector<std::string> qWindows;
  llvm::SmallVector<std::string> xWindows;
  for (int64_t part = 0; part < outputParts; ++part) {
    std::string qWindow = fresh("group_q_window");
    std::string xWindow = fresh("group_x_window");
    line("const uint8_t *" + qWindow + " = " + lhsRecords[part] + " + " +
         byte + " * " + std::to_string(qTermStride) + ";");
    line("const int8_t *" + xWindow + " = (const int8_t *)(" +
         rhsRecords[part] + " + " +
         std::to_string(plan.getRhsBitOffsetBytes()) + " + " + logical + ");");
    qWindows.push_back(std::move(qWindow));
    xWindows.push_back(std::move(xWindow));
  }

  auto qLoad = [&](llvm::StringRef window, llvm::StringRef term,
                   llvm::StringRef partVL) {
    std::string address = window.str() + " + (" + term.str() + ") * " +
                          std::to_string(qTermStride);
    std::string loaded;
    if (plan.getLoadForm() == "unit")
      loaded = "__riscv_vle8_v_" + rawSuffix +
               "((const uint8_t *)(" + address + "), " + partVL.str() + ")";
    else
      loaded = "__riscv_vlse8_v_" + rawSuffix +
               "((const uint8_t *)(" + address + "), (ptrdiff_t)(" +
               rowStride + "), " + partVL.str() + ")";
    return "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
           rawSuffix + "(" + loaded + ", " + shift + ", " + partVL.str() +
           "), " + std::to_string(plan.getMaskValue()) + ", " +
           partVL.str() + ")";
  };
  auto accumulateGroup = [&](llvm::StringRef groupIndex, int64_t termBase,
                             llvm::StringRef termLimit) {
    llvm::SmallVector<std::string> partials;
    for (int64_t part = 0; part < outputParts; ++part) {
      std::string partial = fresh("group_partial");
      line(partialType + " " + partial + ";");
      partials.push_back(std::move(partial));
    }
    for (int64_t term = 0; term < group; ++term) {
      const int64_t plannedTerm =
          plan.getTermOrder()[static_cast<size_t>(termBase + term)] % group;
      std::string linear = "((" + groupIndex.str() + ") * " +
                           std::to_string(group) + " + " +
                           std::to_string(plannedTerm) + ")";
      if (!termLimit.empty() && term != 0) {
        line("if (" + std::to_string(plannedTerm) + " < " + termLimit.str() + ") {");
        ++indent;
      }
      for (int64_t part = 0; part < outputParts; ++part) {
        const std::string partVl = partVL(operation.getResult(), part);
        std::string q = fresh("group_q");
        line(rawType + " " + q + " = " +
             qLoad(qWindows[part], linear, partVl) + ";");
        std::string signedQ = fresh("group_q_signed");
        line(signedRawType + " " + signedQ + " = __riscv_vreinterpret_v_" +
             rawSuffix + "_" + signedRawSuffix + "(" + q + ");");
        const std::string x = "*(" + xWindows[part] + " + " + linear + ")";
        if (term == 0)
          line(partials[part] + " = __riscv_vwmul_vx_" + partialSuffix +
               "(" + signedQ + ", " + x + ", " + partVl + ");");
        else
          line(partials[part] + " = __riscv_vwmacc_vx_" + partialSuffix +
               "(" + partials[part] + ", " + x + ", " + signedQ + ", " +
               partVl + ");");
      }
      if (!termLimit.empty() && term != 0) {
        --indent;
        line("}");
      }
    }
    for (int64_t part = 0; part < outputParts; ++part) {
      const std::string partVl = partVL(operation.getResult(), part);
      std::string widened = fresh("group_wide");
      line(outType + " " + widened + " = __riscv_vwcvt_x_x_v_" + outSuffix +
           "(" + partials[part] + ", " + partVl + ");");
      line(results[part] + " = __riscv_vadd_vv_" + outSuffix + "(" +
           results[part] + ", " + widened + ", " + partVl + ");");
    }
  };

  std::string fullGroups = fresh("group_full_count");
  std::string tail = fresh("group_tail_count");
  std::string fullChunks = fresh("group_full_chunks");
  line("const size_t " + fullGroups + " = " + active.scalar + " / " +
       std::to_string(group) + ";");
  line("const size_t " + tail + " = " + active.scalar + " % " +
       std::to_string(group) + ";");
  line("const size_t " + fullChunks + " = " + fullGroups + " / " +
       std::to_string(unroll) + ";");
  std::string chunk = fresh("group_chunk");
  line("#pragma GCC unroll 1");
  line("for (size_t " + chunk + " = 0; " + chunk + " < " + fullChunks +
       "; ++" + chunk + ") {");
  ++indent;
  for (int64_t slot = 0; slot < unroll; ++slot)
    accumulateGroup("(" + chunk + " * " + std::to_string(unroll) + " + " +
                        std::to_string(slot) + ")",
                    slot * group,
                    "");
  --indent;
  line("}");
  std::string remainder = fresh("group_remainder");
  line("for (size_t " + remainder + " = " + fullChunks + " * " +
       std::to_string(unroll) + "; " + remainder + " < " + fullGroups +
       "; ++" + remainder + ") {");
  ++indent;
  accumulateGroup(remainder, 0, "");
  --indent;
  line("}");
  line("if (" + tail + " != 0) {");
  ++indent;
  accumulateGroup(fullGroups, 0, tail);
  --indent;
  line("}");

  Binding output;
  output.kind = Binding::Kind::Vector;
  output.parts.assign(std::make_move_iterator(results.begin()),
                      std::make_move_iterator(results.end()));
  bindings[operation.getResult()] = std::move(output);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileGroupedMacLoad(riscv::RVVGroupedMacLoadOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.grouped-mac-load.u8-s8")
    return fail(operation, "grouped MAC load has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding groupIndex = bindings.lookup(operation.getGroupIndex());
  Binding activeTerms = bindings.lookup(operation.getActiveTerms());
  if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field ||
      groupIndex.kind != Binding::Kind::Scalar ||
      activeTerms.kind != Binding::Kind::Scalar)
    return fail(operation,
                "grouped MAC load requires two fields and explicit scalar window bounds");
  auto lhsField = fieldFor(lhs);
  auto rhsField = fieldFor(rhs);
  Binding lhsOwner = bindings.lookup(lhs.field.owner);
  Binding rhsOwner = bindings.lookup(rhs.field.owner);
  if (lhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(lhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    lhsOwner = std::move(*record);
  }
  if (rhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(rhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    rhsOwner = std::move(*record);
  }
  auto lhsInteger = lhsField ? mlir::dyn_cast<mlir::IntegerType>(lhsField->type)
                             : mlir::IntegerType();
  auto rhsInteger = rhsField ? mlir::dyn_cast<mlir::IntegerType>(rhsField->type)
                             : mlir::IntegerType();
  if (!lhsField || !rhsField || lhsOwner.kind != Binding::Kind::Record ||
      rhsOwner.kind != Binding::Kind::Record || !lhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 || !rhsInteger ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8)
    return fail(operation,
                "grouped MAC load fields do not match the selected narrow/signed window");

  auto lhsType = mlir::dyn_cast<riscv::ValueType>(operation.getLhs().getType());
  if (!lhsType || lhsType.getLayout().getCarrier() != "rvv" ||
      product(operation.getResult().getType().getResultLayout().getTimeFactors()) *
              product(operation.getResult().getType().getResultLayout().getReplicaFactors()) !=
          1)
    return fail(operation,
                "grouped MAC load currently requires one closed output RVV part");
  const std::string rawSuffix =
      "u8" + lmulSpelling(operation.getLoadLayout().getLmulEighths());
  const std::string rawType =
      "vuint8" + lmulSpelling(operation.getLoadLayout().getLmulEighths()) +
      "_t";
  if (rawSuffix == "u8" || rawType == "vuint8_t")
    return fail(operation, "grouped MAC load has no legal raw RVV shape");

  Binding point = lhs.field.index ? bindings.lookup(*lhs.field.index) : Binding();
  Binding rhsPoint = rhs.field.index ? bindings.lookup(*rhs.field.index) : Binding();
  if (point.kind != Binding::Kind::Point || rhsPoint.kind != Binding::Kind::Point ||
      point.point.axis != rhsPoint.point.axis ||
      point.point.base != rhsPoint.point.base)
    return fail(operation,
                "grouped MAC load operands do not share one reduction point");
  riscv::GroupedMacPlanAttr plan = operation.getPlan();
  const std::string logicalBase = "((" + point.point.base + " % " +
                                  std::to_string(plan.getRecordElements()) +
                                  ") + " + groupIndex.scalar + " * " +
                                  std::to_string(plan.getGroup()) + ")";
  std::string rowStride;
  for (const auto &[axis, stride] : lhsOwner.recordByteStrides)
    if (axis == plan.getRowStrideAxis())
      rowStride = stride;
  if ((plan.getLoadForm() == "unit" && plan.getInterleaveRows() <= 0) ||
      (plan.getLoadForm() == "strided" && rowStride.empty()))
    return fail(operation, "grouped MAC load plan has no bound load form");

  const int64_t storageGroup = plan.getStorageGroup();
  const int64_t storageLayer = plan.getStorageLayer();
  const bool compactWindow = plan.getKind() == "compact";
  std::string compactQBase;
  std::string compactXBase;
  std::string compactShift;
  if (compactWindow) {
    std::string logical = fresh("group_logical_base");
    std::string within = fresh("group_layer_within");
    std::string byte = fresh("group_storage_byte");
    compactShift = fresh("group_shift");
    compactQBase = fresh("group_q_window");
    compactXBase = fresh("group_x_window");
    line("const size_t " + logical + " = " + logicalBase + ";");
    line("const size_t " + within + " = " + logical + " % " +
         std::to_string(storageGroup) + ";");
    line("const size_t " + byte + " = " +
         std::to_string(plan.getLhsBitOffsetBytes()) + " + (" + logical + " / " +
         std::to_string(storageGroup) + ") * " +
         std::to_string(storageLayer) + " + " + within + " % " +
         std::to_string(storageLayer) + ";");
    line("const ptrdiff_t " + compactShift + " = " +
         std::to_string(plan.getShiftBase()) + " + (" + within + " / " +
         std::to_string(storageLayer) + ") * " +
         std::to_string(plan.getShiftStep()) + ";");
    const std::string qScale = std::to_string(plan.getTermByteStride());
    line("const uint8_t *" + compactQBase + " = " + lhsOwner.recordPointer +
         " + " + byte + " * " + qScale + ";");
    line("const int8_t *" + compactXBase + " = (const int8_t *)(" +
         rhsOwner.recordPointer + " + " +
         std::to_string(plan.getRhsBitOffsetBytes()) + " + " + logical + ");");
  }

  Binding window;
  window.kind = Binding::Kind::Window;
  window.windowFamily = "grouped-mac";
  const bool exactWindow = operation.getLeaf().getTail() == "exact";
  for (int64_t slot = 0; slot < plan.getUnroll(); ++slot) {
    for (int64_t term = 0; term < plan.getGroup(); ++term) {
      const int64_t ordinal = slot * plan.getGroup() + term;
      const int64_t linear = plan.getTermOrder()[ordinal];
      const std::string logical = "(" + logicalBase + " + " +
                                  std::to_string(linear) + ")";
      const std::string valid =
          exactWindow ? "1"
                      : "(" + groupIndex.scalar + " * " +
                            std::to_string(plan.getGroup()) + " + " +
                            std::to_string(linear) + " < " +
                            activeTerms.scalar + ")";
      std::string qAddress;
      std::string qStride;
      std::string shift;
      std::string xAddress;
      if (compactWindow) {
        const int64_t qOffset =
            linear * plan.getTermByteStride();
        qAddress = compactQBase + " + " + std::to_string(qOffset);
        if (plan.getLoadForm() == "strided")
          qStride = rowStride;
        shift = compactShift;
        xAddress = compactXBase + " + " + std::to_string(linear);
      } else {
        const std::string within = "((" + logical + ") % " +
                                   std::to_string(storageGroup) + ")";
        const std::string physicalLayer =
            "(" + std::to_string(plan.getPhysicalLayerBase()) + " + ((" +
            within + ") / " + std::to_string(storageLayer) + ") * " +
            std::to_string(plan.getPhysicalLayerStep()) + ")";
        const std::string byte =
            "(" + std::to_string(plan.getLhsBitOffsetBytes()) + " + ((" +
            logical + ") / " + std::to_string(storageGroup) + ") * " +
            std::to_string(storageLayer) + " + (" + within + ") % " +
            std::to_string(storageLayer) + ")";
        if (plan.getLoadForm() == "unit") {
          qAddress = lhsOwner.recordPointer + " + (" + byte + ") * " +
                     std::to_string(plan.getInterleaveRows());
        } else {
          qAddress = lhsOwner.recordPointer + " + (" + byte + ")";
          qStride = rowStride;
        }
        shift = "(" + std::to_string(plan.getShiftBase()) + " + ((" + within +
                ") / " + std::to_string(storageLayer) + ") * " +
                std::to_string(plan.getShiftStep()) + ")";
        xAddress = rhsOwner.recordPointer + " + " +
                   std::to_string(plan.getRhsBitOffsetBytes()) + " + " + logical;
      }
      auto loadExpression = [&]() {
        if (qStride.empty())
          return "__riscv_vle8_v_" + rawSuffix +
                 "((const uint8_t *)(" + qAddress + "), " + laneVL() + ")";
        return "__riscv_vlse8_v_" + rawSuffix +
               "((const uint8_t *)(" + qAddress + "), (ptrdiff_t)(" + qStride +
               "), " + laneVL() + ")";
      };
      std::string loaded = fresh("group_q_raw");
      std::string q = fresh("group_q");
      if (exactWindow) {
        line(rawType + " " + loaded + " = " + loadExpression() + ";");
      } else {
        line(rawType + " " + q + " = __riscv_vmv_v_x_" + rawSuffix +
             "(0, " + laneVL() + ");");
        line("if (" + valid + ") {");
        ++indent;
        line(rawType + " " + loaded + " = " + loadExpression() + ";");
      }
      std::string decoded = loaded;
      if (shift != "0")
        decoded = "__riscv_vsrl_vx_" + rawSuffix + "(" + decoded + ", " +
                  shift + ", " + laneVL() + ")";
      if (plan.getLogicalWidth() < 8)
        decoded = "__riscv_vand_vx_" + rawSuffix + "(" + decoded + ", " +
                  std::to_string(plan.getMaskValue()) + ", " +
                  laneVL() + ")";
      if (exactWindow) {
        line(rawType + " " + q + " = " + decoded + ";");
      } else {
        line(q + " = " + decoded + ";");
        --indent;
        line("}");
      }
      std::string x = fresh("group_x");
      if (exactWindow)
        line("int8_t " + x + " = *(const int8_t *)(" + xAddress + ");");
      else {
        line("int8_t " + x + " = 0;");
        line("if (" + valid + ") " + x + " = *(const int8_t *)(" +
             xAddress + ");");
      }
      window.windowLhs.push_back(std::move(q));
      window.windowRhs.push_back(std::move(x));
      window.windowValidity.push_back(valid);
    }
  }
  bindings[operation.getResult()] = std::move(window);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileGroupedMacStep(riscv::RVVGroupedMacStepOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.vwmaccsu.vx.grouped")
    return fail(operation, "grouped MAC step has no exact selected leaf");
  Binding window = bindings.lookup(operation.getWindow());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  auto windowType = operation.getWindow().getType();
  if (window.kind != Binding::Kind::Window ||
      window.windowFamily != "grouped-mac" ||
      accumulator.kind != Binding::Kind::Vector || accumulator.parts.size() != 1 ||
      window.windowLhs.size() != window.windowRhs.size() ||
      window.windowLhs.size() != static_cast<size_t>(windowType.getSlots() *
                                                     windowType.getTermsPerSlot()))
    return fail(operation, "grouped MAC step has an incomplete physical window");
  const std::string partialSuffix =
      "i16" + lmulSpelling(windowType.getPartialLayout().getLmulEighths());
  const std::string partialType =
      "vint16" + lmulSpelling(windowType.getPartialLayout().getLmulEighths()) +
      "_t";
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  std::string result = fresh("grouped_mac");
  line(vectorType(operation.getResult()) + " " + result + " = " +
       accumulator.parts.front() + ";");
  for (int64_t slot = 0; slot < windowType.getSlots(); ++slot) {
    std::string partial = fresh("group_partial");
    line(partialType + " " + partial + " = __riscv_vmv_v_x_" +
         partialSuffix + "(0, " + laneVL() + ");");
    for (int64_t term = 0; term < windowType.getTermsPerSlot(); ++term) {
      size_t index = static_cast<size_t>(slot * windowType.getTermsPerSlot() + term);
      line(partial + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
           partial + ", " + window.windowRhs[index] + ", " +
           window.windowLhs[index] + ", " + laneVL() + ");");
    }
    std::string widened = fresh("group_partial_wide");
    line(vectorType(operation.getResult()) + " " + widened +
         " = __riscv_vwcvt_x_x_v_" + resultSuffix + "(" + partial + ", " +
         laneVL() + ");");
    line(result + " = __riscv_vadd_vv_" + resultSuffix + "(" + result + ", " +
         widened + ", " + laneVL() + ");");
  }
  Binding output;
  output.kind = Binding::Kind::Vector;
  output.parts.push_back(std::move(result));
  bindings[operation.getResult()] = std::move(output);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileEncodedDotLoad(riscv::RVVEncodedDotLoadOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.encoded-dot-load.i16-pairs")
    return fail(operation, "encoded dot load has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding index = bindings.lookup(operation.getGroupIndex());
  Binding count = bindings.lookup(operation.getGroupCount());
  if (lhs.kind != Binding::Kind::Vector || rhs.kind != Binding::Kind::Field ||
      index.kind != Binding::Kind::Scalar || count.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded dot load requires decoded RVV groups, one encoded field, and explicit bounds");
  Binding owner = bindings.lookup(rhs.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(rhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(rhs);
  const int64_t resultParts = operation.getResult().getType().getResultParts();
  if (!field || owner.kind != Binding::Kind::Record || resultParts <= 0 ||
      static_cast<int64_t>(lhs.parts.size()) % resultParts)
    return fail(operation,
                "encoded dot load has no closed output-part to decoded-group mapping");
  const int64_t groups =
      static_cast<int64_t>(lhs.parts.size()) / std::max<int64_t>(1, resultParts);
  if (groups <= 0)
    return fail(operation, "encoded dot lhs has no physical group parts");
  Binding window;
  window.kind = Binding::Kind::Window;
  window.windowFamily = "encoded-dot";
  for (int64_t slot = 0; slot < operation.getUnroll(); ++slot) {
    const std::string logical = "(" + index.scalar + " + " +
                                std::to_string(slot) + ")";
    const std::string valid = "(" + logical + " < " + count.scalar + ")";
    for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
      std::string selected = fresh("encoded_group");
      line(vectorType(operation.getLhs()) + " " + selected +
           " = __riscv_vmv_v_x_" + vectorSuffix(operation.getLhs()) +
           "(0, " + laneVL() + ");");
      line("if (" + valid + ") {");
      ++indent;
      for (int64_t group = 0; group < groups; ++group)
        line(std::string(group == 0 ? "if" : "else if") + " (" + logical +
             " == " + std::to_string(group) + ") " + selected + " = " +
             lhs.parts[static_cast<size_t>(resultPart * groups + group)] + ";");
      --indent;
      line("}");
      window.windowLhs.push_back(std::move(selected));
    }
    std::string sum = fresh("encoded_pair_sum");
    line("int32_t " + sum + " = 0;");
    line("if (" + valid + ") {");
    ++indent;
    for (int64_t pair = 0; pair < operation.getRhsPairsPerGroup(); ++pair) {
      std::string address = owner.recordPointer + " + " +
                            std::to_string(field->bitOffset / 8) + " + 2 * (" +
                            std::to_string(operation.getRhsPairsPerGroup()) +
                            " * " + logical + " + " + std::to_string(pair) + ")";
      line(sum + " += (int32_t)weft_load_i16_le(" + address + ");");
    }
    --indent;
    line("}");
    window.windowRhs.push_back(std::move(sum));
    window.windowValidity.push_back(valid);
  }
  bindings[operation.getResult()] = std::move(window);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileEncodedDotStep(riscv::RVVEncodedDotStepOp operation) {
  if (operation.getLeaf().getInstruction() != "rvv.vzext-vmacc.vx")
    return fail(operation, "encoded dot step has no exact selected leaf");
  Binding window = bindings.lookup(operation.getWindow());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  auto windowType = operation.getWindow().getType();
  if (window.kind != Binding::Kind::Window ||
      window.windowFamily != "encoded-dot" ||
      accumulator.kind != Binding::Kind::Vector ||
      accumulator.parts.size() != static_cast<size_t>(windowType.getResultParts()) ||
      window.windowLhs.size() != static_cast<size_t>(
          windowType.getSlots() * windowType.getResultParts()) ||
      window.windowRhs.size() != static_cast<size_t>(windowType.getSlots()))
    return fail(operation, "encoded dot step has an incomplete physical window");
  const std::string suffix = vectorSuffix(operation.getResult());
  std::string unsignedSuffix = suffix;
  unsignedSuffix[0] = 'u';
  Binding output;
  output.kind = Binding::Kind::Vector;
  for (int64_t resultPart = 0; resultPart < windowType.getResultParts();
       ++resultPart) {
    std::string result = fresh("encoded_dot");
    line(vectorType(operation.getResult()) + " " + result + " = " +
         accumulator.parts[static_cast<size_t>(resultPart)] + ";");
    for (int64_t slot = 0; slot < windowType.getSlots(); ++slot) {
      const size_t lhsIndex = static_cast<size_t>(
          slot * windowType.getResultParts() + resultPart);
      std::string widened = fresh("encoded_group_wide");
      line("vuint32" + lmulSpelling(
               operation.getResult().getType().getLayout().getLmulEighths()) +
           "_t " + widened + " = __riscv_vzext_vf4_" + unsignedSuffix + "(" +
           window.windowLhs[lhsIndex] + ", " + laneVL() + ");");
      const std::string lane =
          "__riscv_vreinterpret_v_" + unsignedSuffix + "_" + suffix + "(" +
          widened + ")";
      line(result + " = __riscv_vmacc_vx_" + suffix + "(" + result + ", " +
           window.windowRhs[static_cast<size_t>(slot)] + ", " + lane + ", " +
           laneVL() + ");");
    }
    output.parts.push_back(std::move(result));
  }
  bindings[operation.getResult()] = std::move(output);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVWidenDot(riscv::RVVWidenDotOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vwmul-vwredsum")
    return fail(operation, "RVV widening dot has no exact selected leaf");
  mlir::FailureOr<Binding> lhs = materializeNumeric(
      operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhs = materializeNumeric(
      operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();
  if (lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Vector || lhs->parts.empty() ||
      rhs->parts.empty())
    return fail(operation,
                "RVV widening dot requires materialized vector operands");
  const int64_t lhsStreams = streamPartCount(operation.getLhs());
  const int64_t rhsStreams = streamPartCount(operation.getRhs());
  if (lhsStreams <= 0 || rhsStreams <= 0 ||
      lhs->parts.size() !=
          static_cast<size_t>(lhsStreams * registerPartCount(operation.getLhs())) ||
      rhs->parts.size() !=
          static_cast<size_t>(rhsStreams * registerPartCount(operation.getRhs())))
    return fail(operation,
                "RVV widening dot operand streams disagree with their layouts");
  const int64_t sliceLMUL = operation.getSliceLmulEighths();
  const int64_t partialLMUL = sliceLMUL * 2;
  const int64_t inputSEW = operation.getLhs().getType().getLayout().getSew();
  const int64_t partialSEW = inputSEW * 2;
  const std::string partialSuffix =
      "i" + std::to_string(partialSEW) + lmulSpelling(partialLMUL);
  const std::string partialType =
      "vint" + std::to_string(partialSEW) + lmulSpelling(partialLMUL) + "_t";
  auto lhsElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getLhs().getType()));
  auto rhsElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getRhs().getType()));
  const bool mixedSignedness = lhsElement.isSigned() != rhsElement.isSigned();
  std::string seed = fresh("widen_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  auto resultValue = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  const int64_t outputParts =
      resultValue ? registerPartCount(operation.getResult()) : 1;
  Binding output;
  output.kind = resultValue ? Binding::Kind::ScalarTuple : Binding::Kind::Scalar;
  if (operation.getReductionLanes() <= 0 || sliceLMUL <= 0 ||
      operation.getReductionStreams() <= 0 ||
      operation.getLhsLaneOffsets().size() !=
          static_cast<size_t>(outputParts) ||
      operation.getRhsLaneOffsets().size() !=
          static_cast<size_t>(outputParts) ||
      operation.getLhsParts().size() !=
          static_cast<size_t>(outputParts * operation.getReductionStreams()) ||
      operation.getRhsParts().size() !=
          static_cast<size_t>(outputParts * operation.getReductionStreams()))
    return fail(operation,
                "RVV widening dot is missing its typed lane-slice plan");
  for (int64_t outputPart = 0; outputPart < outputParts; ++outputPart) {
    std::string total = fresh("widen_dot");
    line("int32_t " + total + " = 0;");
    llvm::SmallVector<std::string> products;
    llvm::SmallVector<std::string> productVLs;
    const bool combineStreams =
        operation.getReductionStreams() > 1 &&
        operation.getPartialTopology().getKind() == "sequential_fused";
    for (int64_t stream = 0; stream < operation.getReductionStreams(); ++stream) {
      const size_t planned = static_cast<size_t>(
          outputPart * operation.getReductionStreams() + stream);
      const int64_t lhsPartValue = operation.getLhsParts()[planned];
      const int64_t rhsPartValue = operation.getRhsParts()[planned];
      if (lhsPartValue < 0 || rhsPartValue < 0 ||
          lhsPartValue >= static_cast<int64_t>(lhs->parts.size()) ||
          rhsPartValue >= static_cast<int64_t>(rhs->parts.size()))
        return fail(operation,
                    "RVV widening dot typed operand-part plan is out of bounds");
      const size_t lhsPart = static_cast<size_t>(lhsPartValue);
      const size_t rhsPart = static_cast<size_t>(rhsPartValue);
      auto sliceOperand = [&](mlir::Value value, llvm::StringRef source,
                              int64_t laneOffset) -> std::optional<std::string> {
        const int64_t sourceLMUL =
            mlir::cast<riscv::ValueType>(value.getType())
                .getLayout()
                .getLmulEighths();
        if (sliceLMUL <= 0 || sourceLMUL < sliceLMUL ||
            sourceLMUL % sliceLMUL || laneOffset < 0 ||
            laneOffset % operation.getReductionLanes())
          return std::nullopt;
        const std::string sourceSuffix = vectorSuffix(value);
        if (sourceSuffix.empty())
          return std::nullopt;
        const char category = sourceSuffix.front();
        const std::string sliceSuffix =
            std::string(1, category) + std::to_string(inputSEW) +
            lmulSpelling(sliceLMUL);
        const std::string sliceType =
            std::string(category == 'u' ? "vuint" : "vint") +
            sliceSuffix.substr(1) + "_t";
        auto kernel = operation->getParentOfType<riscv::KernelOp>();
        if (!kernel)
          return std::nullopt;
        const int64_t sliceSpan =
            kernel.getTarget().getVlenBits() * sliceLMUL / (8 * inputSEW);
        if (sliceSpan <= 0)
          return std::nullopt;
        const int64_t group = laneOffset / sliceSpan;
        const int64_t intra = laneOffset % sliceSpan;
        std::string sliced = source.str();
        if (sourceLMUL != sliceLMUL) {
          sliced = fresh("widen_dot_slice");
          line(sliceType + " " + sliced + " = __riscv_vget_v_" +
               sourceSuffix + "_" + sliceSuffix + "(" + source.str() + ", " +
               std::to_string(group) + ");");
        } else if (group != 0) {
          return std::nullopt;
        }
        if (intra != 0) {
          std::string shifted = fresh("widen_dot_slide");
          line(sliceType + " " + shifted + " = __riscv_vslidedown_vx_" +
               sliceSuffix + "(" + sliced + ", " + std::to_string(intra) +
               ", " + std::to_string(operation.getReductionLanes()) + ");");
          sliced = std::move(shifted);
        }
        return sliced;
      };
      auto lhsSlice = sliceOperand(operation.getLhs(), lhs->parts[lhsPart],
                                   operation.getLhsLaneOffsets()[outputPart]);
      auto rhsSlice = sliceOperand(operation.getRhs(), rhs->parts[rhsPart],
                                   operation.getRhsLaneOffsets()[outputPart]);
      if (!lhsSlice || !rhsSlice)
        return fail(operation,
                    "RVV widening dot cannot materialize its selected lane slice");
      const std::string vl = std::to_string(operation.getReductionLanes());
      const std::string &signedOperand =
          lhsElement.isSigned() ? *lhsSlice : *rhsSlice;
      const std::string &unsignedOperand =
          lhsElement.isSigned() ? *rhsSlice : *lhsSlice;
      const std::string operands =
          mixedSignedness
              ? signedOperand + ", " + unsignedOperand
              : *lhsSlice + ", " + *rhsSlice;
      if (combineStreams && !products.empty()) {
        line(products.front() + " = __riscv_" +
             std::string(mixedSignedness ? "vwmaccsu" : "vwmacc") + "_vv_" +
             partialSuffix + "(" + products.front() + ", " + operands + ", " +
             vl + ");");
        continue;
      }
      std::string product = fresh("widen_product");
      line(partialType + " " + product + " = __riscv_" +
           std::string(mixedSignedness ? "vwmulsu" : "vwmul") + "_vv_" +
           partialSuffix + "(" + operands + ", " + vl + ");");
      products.push_back(std::move(product));
      productVLs.push_back(vl);
    }
    for (size_t part = 0; part < products.size(); ++part) {
      std::string reduced = fresh("widen_sum");
      const std::string reduction = partialSEW == 16 ? "vwredsum" : "vredsum";
      line("vint32m1_t " + reduced + " = __riscv_" + reduction + "_vs_" +
           partialSuffix + "_i32m1(" + products[part] + ", " + seed + ", " +
           productVLs[part] + ");");
      line(total + " += __riscv_vmv_x_s_i32m1_i32(" + reduced + ");");
    }
    if (resultValue)
      output.parts.push_back(std::move(total));
    else
      output.scalar = std::move(total);
  }
  bindings[operation.getResult()] = std::move(output);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenMultiply(
    riscv::RVVWidenMultiplyOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV widening multiply requires two selected vector operands");
  const std::string instruction = instructionOf(operation.getOperation()).str();
  if (instruction != "rvv.vwmulu.vv" && instruction != "rvv.vwmul.vv" &&
      instruction != "rvv.vwmulsu.vv" &&
      instruction != "rvv.vwmulsu.vv.swap")
    return fail(operation,
                "RVV widening multiply has no exact selected instruction");
  Binding result;
  result.kind = Binding::Kind::Vector;
  std::string suffix = vectorSuffix(operation.getResult());
  std::string unsignedSuffix = suffix;
  unsignedSuffix[0] = 'u';
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto lhsPart = mappedPart(operation.getOperation(), 0, part);
    auto rhsPart = mappedPart(operation.getOperation(), 1, part);
    if (!lhsPart || !rhsPart || *lhsPart >= lhs->parts.size() ||
        *rhsPart >= rhs->parts.size())
      return fail(operation,
                  "RVV widening multiply has no closed operand-part mapping");
    std::string intrinsic;
    std::string left = lhs->parts[*lhsPart];
    std::string right = rhs->parts[*rhsPart];
    if (instruction == "rvv.vwmulu.vv")
      intrinsic = "__riscv_vwmulu_vv_" + unsignedSuffix;
    else {
      intrinsic = "__riscv_" +
                  std::string(instruction == "rvv.vwmul.vv"
                                  ? "vwmul_vv_"
                                  : "vwmulsu_vv_") +
                  suffix;
      if (instruction == "rvv.vwmulsu.vv.swap")
        std::swap(left, right);
    }
    std::string expression = intrinsic + "(" + left + ", " + right + ", " +
                             partVL(operation.getResult(), part) + ")";
    if (instruction == "rvv.vwmulu.vv" && suffix[0] == 'i')
      expression = "__riscv_vreinterpret_v_" + unsignedSuffix + "_" + suffix +
                   "(" + expression + ")";
    std::string name = fresh("widen_multiply");
    line(vectorType(operation.getResult()) + " " + name + " = " + expression +
         ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenScalarMultiply(
    riscv::RVVWidenScalarMultiplyOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV widening scalar multiply requires one vector and one scalar operand");
  const std::string instruction = instructionOf(operation.getOperation()).str();
  if (instruction != "rvv.vwmulu.vx" && instruction != "rvv.vwmul.vx" &&
      instruction != "rvv.vwmulsu.vx")
    return fail(operation,
                "RVV widening scalar multiply has no exact selected instruction");
  Binding result;
  result.kind = Binding::Kind::Vector;
  std::string suffix = vectorSuffix(operation.getResult());
  std::string unsignedSuffix = suffix;
  unsignedSuffix[0] = 'u';
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto lhsPart = mappedPart(operation.getOperation(), 0, part);
    if (!lhsPart || *lhsPart >= lhs->parts.size())
      return fail(operation,
                  "RVV widening scalar multiply has no closed operand-part mapping");
    std::string intrinsic;
    if (instruction == "rvv.vwmulu.vx")
      intrinsic = "__riscv_vwmulu_vx_" + unsignedSuffix;
    else
      intrinsic = "__riscv_" +
                  std::string(instruction == "rvv.vwmul.vx"
                                  ? "vwmul_vx_"
                                  : "vwmulsu_vx_") +
                  suffix;
    std::string expression =
        intrinsic + "(" + lhs->parts[*lhsPart] + ", " + rhs->scalar + ", " +
        partVL(operation.getResult(), part) + ")";
    if (instruction == "rvv.vwmulu.vx" && suffix[0] == 'i')
      expression = "__riscv_vreinterpret_v_" + unsignedSuffix + "_" + suffix +
                   "(" + expression + ")";
    std::string name = fresh("widen_scalar_multiply");
    line(vectorType(operation.getResult()) + " " + name + " = " + expression +
         ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRegularRepeatIndex(
    riscv::RVVRegularRepeatIndexOp operation) {
  const int64_t repeat = operation.getRepeat();
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool powerOfTwo = instruction == "rvv.regular-repeat-index.pow2";
  if ((!powerOfTwo && instruction != "rvv.regular-repeat-index.div") ||
      operation.getResults().size() != operation.getPartBases().size())
    return fail(operation,
                "regular-repeat index has no exact selected physical spelling");
  mlir::Value firstResult = operation.getResults().front();
  const std::string indexSuffix = vectorSuffix(firstResult);
  const int64_t lanes = physicalLanes(firstResult);
  std::string baseExpression = "__riscv_vid_v_" + indexSuffix + "(" +
                               std::to_string(lanes) + ")";
  if (powerOfTwo) {
    int64_t shift = 0;
    for (int64_t value = repeat; value > 1; value >>= 1)
      ++shift;
    baseExpression = "__riscv_vsrl_vx_" + indexSuffix + "(" +
                     baseExpression + ", " + std::to_string(shift) + ", " +
                     std::to_string(lanes) + ")";
  } else {
    baseExpression = "__riscv_vdivu_vx_" + indexSuffix + "(" +
                     baseExpression + ", " + std::to_string(repeat) + ", " +
                     std::to_string(lanes) + ")";
  }
  std::string baseIndex = fresh("repeat_index_base");
  line(vectorType(firstResult) + " " + baseIndex + " = " + baseExpression +
       ";");
  for (auto [resultValue, basesAttribute] :
       llvm::zip(operation.getResults(), operation.getPartBases())) {
    auto bases = mlir::cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    const int64_t streams = streamPartCount(resultValue);
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      const int64_t stream = part % streams;
      if (stream < 0 || stream >= static_cast<int64_t>(bases.size()))
        return fail(operation,
                    "regular-repeat index result part exceeds its typed bases");
      const std::string suffix = vectorSuffix(resultValue);
      std::string expression = baseIndex;
      if (bases[stream] != 0)
        expression = "__riscv_vadd_vx_" + suffix + "(" + expression + ", " +
                     std::to_string(bases[stream]) + ", " +
                     partVL(resultValue, part) + ")";
      std::string name = fresh("repeat_index");
      line(vectorType(resultValue) + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[resultValue] = std::move(result);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRegularRepeatGather(
    riscv::RVVRegularRepeatGatherOp operation) {
  mlir::Value firstValue = operation.getResults().front();
  const int64_t lanes = physicalLanes(firstValue);
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool broadcastOnly = instruction == "rvv.regular-repeat-broadcast";
  const bool powerOfTwo = instruction == "rvv.regular-repeat-gather.pow2";
  if (!broadcastOnly && !powerOfTwo &&
      instruction != "rvv.regular-repeat-gather.div")
    return fail(operation,
                "regular-repeat gather has no exact selected index spelling");
  Binding field = bindings.lookup(operation.getField());
  if (field.kind != Binding::Kind::Field || field.field.index ||
      operation.getResults().empty())
    return fail(operation,
                "regular-repeat gather requires one unprojected encoded field");
  Binding owner = bindings.lookup(field.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(field.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = fieldFor(field);
  auto elementType = storage ? scalarCType(storage->type) : std::nullopt;
  auto integer = storage
                     ? mlir::dyn_cast<mlir::IntegerType>(storage->type)
                     : mlir::IntegerType();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !storage || !elementType || !integer || storage->bitOffset % 8)
    return fail(operation,
                "regular-repeat gather has incomplete record storage geometry");
  const std::string suffix = vectorSuffix(firstValue);
  const std::string type = vectorType(firstValue);
  auto layout = layoutOf(firstValue);
  if (!layout)
    return fail(operation, "regular-repeat gather result has no RVV layout");
  if (lanes <= 1)
    return fail(operation, "regular-repeat gather has no lane extent");

  const int64_t registerParts = registerPartCount(firstValue);
  llvm::SmallVector<std::string> sourceWindows;
  llvm::SmallVector<std::string> sourceRecords;
  llvm::SmallVector<int64_t, 4> axes = registerAxesFor(firstValue);
  for (int64_t replica = 0; replica < registerParts; ++replica) {
    auto coordinates = registerCoordinates(firstValue, replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "regular-repeat gather has no free-axis register coordinates");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto byteStride = llvm::find_if(owner.recordByteStrides,
                                      [&](const auto &entry) {
                                        return entry.first == axis;
                                      });
      if (byteStride == owner.recordByteStrides.end())
        return fail(operation,
                    "regular-repeat gather free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " +
                byteStride->second;
    }
    record += ")";
    sourceRecords.push_back(record);
    if (broadcastOnly)
      continue;
    const std::string pointer =
        "((const " + *elementType + " *)((const uint8_t *)(" + record +
        ") + " + std::to_string(storage->bitOffset / 8) + ") + " +
        std::to_string(operation.getSourceBase()) + ")";
    std::string loaded = fresh("repeat_source");
    line(type + " " + loaded + " = __riscv_vle" +
         std::to_string(integer.getWidth()) + "_v_" + suffix + "(" + pointer +
         ", " + std::to_string(operation.getSourceCount()) + ");");
    sourceWindows.push_back(std::move(loaded));
  }

  for (auto [resultNumber, values] : llvm::enumerate(
           llvm::zip(operation.getResults(), operation.getPartBases()))) {
    auto [resultValue, basesAttribute] = values;
    auto bases = mlir::cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    const int64_t streams = streamPartCount(resultValue);
    Binding indexBinding;
    if (!broadcastOnly) {
      if (resultNumber >= operation.getIndices().size())
        return fail(operation,
                    "regular-repeat gather lost its typed index operand");
      indexBinding = bindings.lookup(operation.getIndices()[resultNumber]);
      if (indexBinding.kind != Binding::Kind::Vector ||
          indexBinding.parts.size() !=
              static_cast<size_t>(vectorPartCount(resultValue)))
        return fail(operation,
                    "regular-repeat gather typed index is not materialized");
    }
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      const int64_t replica = part / streams;
      const int64_t stream = part % streams;
      const int64_t availableSources = static_cast<int64_t>(
          broadcastOnly ? sourceRecords.size() : sourceWindows.size());
      if (replica < 0 || replica >= availableSources ||
          stream < 0 || stream >= static_cast<int64_t>(bases.size()))
        return fail(operation,
                    "regular-repeat gather result parts exceed its selected window");
      if (broadcastOnly) {
        const int64_t sourceIndex = operation.getSourceBase() + bases[stream];
        const std::string scalarValue =
            "((const " + *elementType + " *)((const uint8_t *)(" +
            sourceRecords[replica] + ") + " +
            std::to_string(storage->bitOffset / 8) + "))[" +
            std::to_string(sourceIndex) + "]";
        std::string gathered = fresh("regular_broadcast");
        const bool floating = mlir::isa<mlir::FloatType>(storage->type);
        line(vectorType(resultValue) + " " + gathered + " = __riscv_" +
             std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
             vectorSuffix(resultValue) + "(" + scalarValue + ", " +
             partVL(resultValue, part) + ");");
        result.parts.push_back(std::move(gathered));
        continue;
      }
      std::string gathered = fresh("regular_gather");
      line(vectorType(resultValue) + " " + gathered +
           " = __riscv_vrgather_vv_" + vectorSuffix(resultValue) + "(" +
           sourceWindows[replica] + ", " + indexBinding.parts[part] + ", " +
           partVL(resultValue, part) + ");");
      result.parts.push_back(std::move(gathered));
    }
    bindings[resultValue] = std::move(result);
  }
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVStorageWindow(riscv::RVVStorageWindowOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.storage-window.natural" &&
      instruction != "rvv.storage-window.layered")
    return fail(operation, "RVV storage window has no exact selected leaf");
  Binding field = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  Binding offset = bindings.lookup(operation.getLogicalOffset());
  if (field.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point ||
      offset.kind != Binding::Kind::Scalar || field.field.index)
    return fail(operation,
                "RVV storage window requires one unprojected field, point, and scalar offset");
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  field.field.useAccess = operation.getAccess();
  if (instruction == "rvv.storage-window.natural") {
    Binding owner = bindings.lookup(field.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      auto record = recordForSlice(field.field.owner);
      if (mlir::failed(record))
        return mlir::failure();
      owner = std::move(*record);
    }
    auto storage = fieldFor(field);
    auto resultType = operation.getResult().getType();
    auto elementType = storage ? scalarCType(storage->type) : std::nullopt;
    auto axis = llvm::find(resultType.getAxisIds().asArrayRef(),
                           plan.getReductionAxis());
    const int64_t lanes =
        axis == resultType.getAxisIds().asArrayRef().end()
            ? 0
            : resultType.getLayout().getLaneFactors()[static_cast<size_t>(
                  axis - resultType.getAxisIds().asArrayRef().begin())];
    unsigned width = 0;
    if (storage)
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(storage->type))
        width = integer.getWidth();
      else if (auto floating = mlir::dyn_cast<mlir::FloatType>(storage->type))
        width = floating.getWidth();
    const int64_t bytes = width / 8;
    const int64_t repeat = plan.getProjectionRepeat();
    const int64_t stride = plan.getProjectionStride();
    if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
        !storage || !elementType ||
        owner.recordElements != plan.getRecordElements() ||
        storage->bitOffset / 8 != plan.getByteOffset() ||
        width != static_cast<unsigned>(plan.getElementBits()) ||
        (plan.getKind() != "unit" && plan.getKind() != "strided" &&
         plan.getKind() != "repeat") ||
        (width != 8 && width != 16 && width != 32) ||
        bytes <= 0 || lanes <= 0 || repeat <= 0 || stride <= 0 ||
        (repeat % lanes && lanes % repeat))
      return fail(operation,
                  "RVV projected storage window has incomplete natural-field geometry");
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    const bool floating = mlir::isa<mlir::FloatType>(storage->type);
    const std::string broadcast =
        "__riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix;
    const int64_t streams = streamPartCount(operation.getResult());
    llvm::SmallVector<int64_t, 4> axes =
        registerAxesFor(operation.getResult());
    Binding loaded;
    loaded.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto coordinates =
          registerCoordinates(operation.getResult(), part / streams);
      if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
        return fail(operation,
                    "RVV projected storage window has no register-coordinate mapping");
      std::string record = "(" + owner.recordPointer;
      for (auto [freeAxis, coordinate] : llvm::zip(axes, *coordinates)) {
        auto byteStride = llvm::find_if(owner.recordByteStrides,
                                        [&](const auto &entry) {
                                          return entry.first == freeAxis;
                                        });
        if (byteStride == owner.recordByteStrides.end())
          return fail(operation,
                      "RVV projected storage window free axis has no record stride");
        record += " + " + std::to_string(coordinate) + " * " + byteStride->second;
      }
      record += ")";
      const std::string logical =
          "((" + point.point.base + ") % " +
          std::to_string(plan.getRecordElements()) + " + (" + offset.scalar + "))";
      auto address = [&](llvm::StringRef sourceIndex) {
        return "((const " + *elementType + " *)((const uint8_t *)(" + record +
               ") + " + std::to_string(plan.getByteOffset()) + "))[(" +
               sourceIndex.str() + ")]";
      };
      const std::string vl = partVL(operation.getResult(), part);
      std::string value;
      if (plan.getKind() == "unit" || plan.getKind() == "strided") {
        const std::string sourceIndex =
            std::to_string(plan.getProjectionBase()) + " + (" + logical +
            ") * " + std::to_string(stride);
        std::string name = fresh("projected_window");
        const std::string pointer = "&(" + address(sourceIndex) + ")";
        const std::string expression =
            plan.getKind() == "unit"
                ? "__riscv_vle" + std::to_string(width) + "_v_" + suffix +
                      "(" + pointer + ", " + vl + ")"
                : "__riscv_vlse" + std::to_string(width) + "_v_" + suffix +
                      "(" + pointer + ", (ptrdiff_t)" +
                      std::to_string(stride * bytes) + ", " + vl + ")";
        line(type + " " + name + " = " + expression + ";");
        value = std::move(name);
      } else {
        const int64_t pieceWidth = std::min<int64_t>(repeat, lanes);
        const int64_t pieces = lanes / pieceWidth;
        for (int64_t piece = 0; piece < pieces; ++piece) {
          const std::string sourceIndex =
              std::to_string(plan.getProjectionBase()) + " + ((" + logical +
              " + " + std::to_string(piece * pieceWidth) + ") / " +
              std::to_string(repeat) + ") * " + std::to_string(stride);
          const std::string next =
              broadcast + "(" + address(sourceIndex) + ", " + vl + ")";
          if (piece == 0) {
            value = next;
          } else {
            value = "__riscv_vslideup_vx_" + suffix + "(" + value + ", " +
                    next + ", " + std::to_string(piece * pieceWidth) + ", " + vl +
                    ")";
          }
        }
        std::string name = fresh("projected_repeat");
        line(type + " " + name + " = " + value + ";");
        value = std::move(name);
      }
      loaded.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(loaded);
    return mlir::success();
  }
  Binding owner = bindings.lookup(field.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(field.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = fieldFor(field);
  auto integer = storage
                     ? mlir::dyn_cast<mlir::IntegerType>(storage->type)
                     : mlir::IntegerType();
  auto resultType = operation.getResult().getType();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  const int64_t width = plan.getElementBits();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      owner.recordElements != plan.getRecordElements() || !storage || !integer ||
      integer.isSigned() || integer.getWidth() != width ||
      storage->bitOffset / 8 != plan.getByteOffset() ||
      plan.getKind() != "layered" ||
      width <= 0 || width >= 8 || group <= 0 || layer <= 0 || group % layer ||
      plan.getProjectionBase() != 0 || plan.getProjectionStride() != 1 ||
      plan.getProjectionRepeat() != 1)
    return fail(operation,
                "layered RVV storage window has incomplete typed record geometry");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  Binding loaded;
  loaded.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "layered RVV storage window has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto byteStride = llvm::find_if(owner.recordByteStrides,
                                      [&](const auto &entry) {
                                        return entry.first == axis;
                                      });
      if (byteStride == owner.recordByteStrides.end())
        return fail(operation,
                    "layered RVV storage window free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " + byteStride->second;
    }
    record += ")";
    const std::string logical =
        "((" + point.point.base + ") % " +
        std::to_string(plan.getRecordElements()) + " + (" + offset.scalar +
        ") + " + std::to_string(plan.getProjectionBase()) + " + " +
        partOffset(operation.getResult(), part) + ")";
    const std::string within = "((" + logical + ") % " +
                               std::to_string(group) + ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + ((" + logical +
        ") / " + std::to_string(group) + ") * " + std::to_string(layer) +
        " + (" + within + ") % " + std::to_string(layer) + ")";
    const std::string logicalLayer =
        "((" + within + ") / " + std::to_string(layer) + ")";
    const std::string shift =
        "(" + std::to_string(plan.getShiftBase()) + " + (" + logicalLayer +
        ") * " + std::to_string(plan.getShiftStep()) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string raw = fresh("layered_window_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " + vl + ");");
    std::string decoded = "__riscv_vsrl_vx_" + suffix + "(" + raw + ", " +
                          shift + ", " + vl + ")";
    decoded = "__riscv_vand_vx_" + suffix + "(" + decoded + ", " +
              std::to_string(plan.getMaskValue()) + ", " + vl + ")";
    std::string name = fresh("layered_window");
    line(type + " " + name + " = " + decoded + ";");
    loaded.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(loaded);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLayeredStorageLoad(
    riscv::RVVLayeredStorageLoadOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-storage-load")
    return fail(operation,
                "RVV layered storage load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  Binding windowIndex = bindings.lookup(operation.getWindowIndex());
  if (fieldBinding.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point ||
      windowIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV layered storage load requires a field, point, and scalar window index");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto windowType = operation.getResult().getType();
  auto valueType = windowType.getResultType();
  auto layout = valueType.getLayout();
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  auto reduction = llvm::find(valueType.getAxisIds().asArrayRef(),
                              plan.getReductionAxis());
  const int64_t lanes =
      reduction == valueType.getAxisIds().asArrayRef().end()
          ? 0
          : layout.getLaneFactors()[static_cast<size_t>(
                reduction - valueType.getAxisIds().asArrayRef().begin())];
  const int64_t windowsPerLayer = windowType.getWindowsPerLayer();
  const int64_t replicas = product(layout.getReplicaFactors());
  if (owner.kind != Binding::Kind::Record || !field || !integer ||
      integer.isSigned() || field->bitOffset % 8 || owner.recordElements <= 0 ||
      plan.getKind() != "layered" ||
      owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      integer.getWidth() != plan.getElementBits() ||
      group <= 0 || layer <= 0 || lanes <= 0 || windowsPerLayer <= 0 ||
      replicas <= 0 || layout.getSew() != 8)
    return fail(operation,
                "RVV layered storage load has incomplete record or vector geometry");

  const std::string suffix =
      "u8" + lmulSpelling(layout.getLmulEighths());
  const std::string type =
      "vuint8" + lmulSpelling(layout.getLmulEighths()) + "_t";
  llvm::SmallVector<int64_t, 4> axes;
  for (auto [axis, factor] :
       llvm::zip(valueType.getAxisIds().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef()))
    if (factor > 1)
      axes.push_back(axis);
  const int64_t sourceStreams = streamPartCount(operation.getField());
  Binding result;
  result.kind = Binding::Kind::Window;
  result.windowFamily = "layered-storage";
  for (int64_t replica = 0; replica < replicas; ++replica) {
    llvm::SmallVector<int64_t, 4> allCoordinates(
        layout.getReplicaFactors().size(), 0);
    int64_t remaining = replica;
    for (int64_t position =
             static_cast<int64_t>(layout.getReplicaFactors().size()) - 1;
         position >= 0; --position) {
      const int64_t factor = layout.getReplicaFactors()[position];
      if (factor <= 0)
        return fail(operation,
                    "RVV layered storage load has an invalid replica factor");
      allCoordinates[static_cast<size_t>(position)] = remaining % factor;
      remaining /= factor;
    }
    llvm::SmallVector<int64_t, 4> coordinates;
    for (auto [coordinate, factor] :
         llvm::zip(allCoordinates, layout.getReplicaFactors().asArrayRef()))
      if (factor > 1)
        coordinates.push_back(coordinate);
    if (sourceStreams <= 0 || coordinates.size() != axes.size())
      return fail(operation,
                  "RVV layered storage load has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV layered storage load register axis has no byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    const std::string logicalBase =
        "(((" + point.point.base + ") % " +
        std::to_string(plan.getRecordElements()) + ") + " +
        std::to_string(plan.getProjectionBase()) + ")";
    const std::string groupIndex =
        "((" + logicalBase + " / " + std::to_string(group) + ") + (" +
        windowIndex.scalar + " / " + std::to_string(windowsPerLayer) + "))";
    const std::string within =
        "((" + logicalBase + " % " + std::to_string(layer) + ") + (" +
        windowIndex.scalar + " % " + std::to_string(windowsPerLayer) + ") * " +
        std::to_string(lanes) + ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + " + groupIndex +
        " * " + std::to_string(layer) + " + " + within + ")";
    std::string raw = fresh("layered_window_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " +
         std::to_string(lanes) + ");");
    result.windowLhs.push_back(std::move(raw));
    result.windowValidity.push_back(std::to_string(lanes));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLayeredStorageDecode(
    riscv::RVVLayeredStorageDecodeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool emitsShift =
      instruction == "rvv.layered-storage-decode.shift" ||
      instruction == "rvv.layered-storage-decode.shift-mask";
  const bool emitsMask =
      instruction == "rvv.layered-storage-decode.mask" ||
      instruction == "rvv.layered-storage-decode.shift-mask";
  if (instruction != "rvv.layered-storage-decode.identity" &&
      instruction != "rvv.layered-storage-decode.mask" &&
      instruction != "rvv.layered-storage-decode.shift" &&
      instruction != "rvv.layered-storage-decode.shift-mask")
    return fail(operation,
                "RVV layered storage decode has no exact selected leaf");
  Binding window = bindings.lookup(operation.getWindow());
  auto load =
      operation.getWindow().getDefiningOp<riscv::RVVLayeredStorageLoadOp>();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      operation.getResult().getType().getElementType());
  if (window.kind != Binding::Kind::Window ||
      window.windowFamily != "layered-storage" || !load || !integer ||
      integer.isSigned() || window.windowLhs.empty())
    return fail(operation,
                "RVV layered storage decode has no typed raw storage window");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (auto [part, raw] : llvm::enumerate(window.windowLhs)) {
    std::string value = raw;
    const std::string vl = part < window.windowValidity.size()
                               ? window.windowValidity[part]
                               : std::to_string(physicalLanes(operation.getResult()));
    if (emitsShift) {
      std::string shifted = fresh("layered_window_shift");
      line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
           value + ", " + std::to_string(operation.getShiftAmount()) + ", " + vl +
           ");");
      value = std::move(shifted);
    }
    if (emitsMask) {
      std::string masked = fresh("layered_window_value");
      line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
           value + ", " + std::to_string(operation.getMaskValue()) + ", " + vl +
           ");");
      value = std::move(masked);
    }
    result.parts.push_back(std::move(value));
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV layered storage decode did not produce every register replica");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVReplicaStorageLoad(
    riscv::RVVReplicaStorageLoadOp operation) {
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  const bool layered = plan.getKind() == "layered";
  const bool natural = plan.getKind() == "unit";
  const bool interleavedNatural = plan.getKind() == "interleaved_natural";
  const bool interleavedJoined = plan.getKind() == "interleaved_joined";
  const bool interleaved = interleavedNatural || interleavedJoined;
  if (!natural && !layered && !interleaved)
    return fail(operation,
                "RVV replica storage load has no selected storage-window form");
  const std::string expected =
      layered ? "rvv.replica-storage-load.layered"
      : natural ? "rvv.replica-storage-load.natural"
      : interleavedNatural
          ? "rvv.replica-storage-load.interleaved-natural"
          : "rvv.replica-storage-load.interleaved-joined";
  if (instructionOf(operation.getOperation()) != expected)
    return fail(operation,
                "RVV replica storage load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding logicalBase = bindings.lookup(operation.getLogicalBase());
  if (fieldBinding.kind != Binding::Kind::Field ||
      logicalBase.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV replica storage load requires a field and scalar logical base");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto fieldInteger = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                            : mlir::IntegerType();
  auto fieldFloat = field ? mlir::dyn_cast<mlir::FloatType>(field->type)
                          : mlir::FloatType();
  const unsigned fieldWidth =
      fieldInteger ? fieldInteger.getWidth() : fieldFloat ? fieldFloat.getWidth() : 0;
  auto resultType = operation.getResult().getType();
  auto layout = resultType.getLayout();
  const int64_t lanes = physicalLanes(operation.getResult());
  const int64_t recordRank = operation.getRecordRank();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !field || !fieldWidth || field->bitOffset % 8 || lanes <= 0 ||
      owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      static_cast<int64_t>(fieldWidth) != plan.getElementBits() ||
      operation.getWindowOffsets().empty() ||
      recordRank < 0 ||
      recordRank > static_cast<int64_t>(operation.getField().getType().getShape().size()) ||
      operation.getRecordCoordinatesForWindow().size() !=
          operation.getWindowOffsets().size() * static_cast<size_t>(recordRank) ||
      operation.getWindowForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getLayerForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getPhysicalLayerForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getShiftOffsetForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getShiftBaseFactorForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getMaskValueForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV replica storage load has incomplete record or part geometry");

  const std::string vectorSuffixValue = vectorSuffix(operation.getResult());
  const std::string vectorTypeValue = vectorType(operation.getResult());
  const std::string vl = std::to_string(lanes);
  const char category = vectorSuffixValue.empty() ? '\0' : vectorSuffixValue.front();
  const std::string loadSuffix =
      category == '\0'
          ? std::string()
          : std::string(1, category) + std::to_string(layout.getSew()) +
                lmulSpelling(layout.getLmulEighths());
  const std::string loadType =
      category == 'u' ? "vuint" + loadSuffix.substr(1) + "_t"
      : category == 'i' ? "vint" + loadSuffix.substr(1) + "_t"
      : category == 'f' ? "vfloat" + loadSuffix.substr(1) + "_t"
                        : std::string();
  const std::string loadVL = std::to_string(layout.getVl());
  auto recordForWindow = [&](size_t window) -> std::optional<std::string> {
    if (window >= operation.getWindowOffsets().size())
      return std::nullopt;
    std::string record = "(" + owner.recordPointer;
    auto fieldType = operation.getField().getType();
    for (int64_t position = 0; position < recordRank; ++position) {
      const int64_t axis = fieldType.getAxisIds()[static_cast<size_t>(position)];
      const int64_t coordinate = operation.getRecordCoordinatesForWindow()[
          window * static_cast<size_t>(recordRank) + static_cast<size_t>(position)];
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end()) {
        if (coordinate != 0)
          return std::nullopt;
        continue;
      }
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    return record;
  };
  llvm::SmallVector<std::string> windows;
  windows.reserve(operation.getWindowOffsets().size());
  if (interleaved) {
    if (owner.interleaveRows <= 0 || layout.getSew() <= 0 ||
        layout.getLmulEighths() <= 0)
      return fail(operation,
                  "interleaved replica storage load has no selected byte-plane geometry");
    const int64_t rawNumerator = layout.getLmulEighths() * 8;
    if (rawNumerator % layout.getSew())
      return fail(operation,
                  "interleaved replica storage load cannot preserve the selected lane count");
    const int64_t rawLMUL = rawNumerator / layout.getSew();
    const std::string rawLMULName = lmulSpelling(rawLMUL);
    if (rawLMULName.empty())
      return fail(operation,
                  "interleaved replica storage load has an illegal raw-byte LMUL");
    const std::string rawSuffix = "u8" + rawLMULName;
    const std::string rawType = "vuint8" + rawLMULName + "_t";
    auto byteLoad = [&](llvm::StringRef record, const std::string &byte,
                        llvm::StringRef stem) {
      std::string loaded = fresh(stem);
      line(rawType + " " + loaded + " = __riscv_vle8_v_" + rawSuffix +
           "((const uint8_t *)(" + record.str() + " + (" + byte + ") * " +
           std::to_string(owner.interleaveRows) + "), " + vl + ");");
      return loaded;
    };
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "interleaved replica storage window has no typed record coordinate");
      const std::string logical = "((" + logicalBase.scalar + ") + " +
                                  std::to_string(offset) + ")";
      if (interleavedJoined) {
        if (!fieldInteger || !fieldInteger.isUnsigned())
          return fail(operation,
                      "joined interleaved storage requires an unsigned logical field");
        const int64_t group = operation.getAccess().getGroupSize();
        const int64_t fields = operation.getAccess().getJoinFields();
        const int64_t lowBits = operation.getAccess().getJoinLowBits();
        const int64_t role = operation.getAccess().getJoinRole();
        const int64_t physicalRole = operation.getAccess().getOrder() == "lo_first"
                                         ? role
                                         : fields - 1 - role;
        const uint64_t logicalMask = (uint64_t{1} << fieldWidth) - 1;
        const uint64_t lowMask = (uint64_t{1} << lowBits) - 1;
        const uint64_t highMask =
            (uint64_t{1} << (fieldWidth - lowBits)) - 1;
        const std::string tail = "(" + logical + " - " +
                                 std::to_string(group) + ")";
        const std::string headByte =
            std::to_string(plan.getByteOffset() + role * group) + " + " + logical;
        const std::string lowByte =
            std::to_string(plan.getByteOffset() + fields * group) + " + " + tail;
        const std::string highByte =
            std::to_string(plan.getByteOffset() + role * group) + " + " + tail;
        std::string head = byteLoad(*record, headByte, "replica_joined_head");
        head = "__riscv_vand_vx_" + rawSuffix + "(" + head + ", " +
               std::to_string(logicalMask) + ", " + vl + ")";
        std::string low = byteLoad(*record, lowByte, "replica_joined_low");
        low = "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
              rawSuffix + "(" + low + ", " +
              std::to_string(physicalRole * lowBits) + ", " + vl + "), " +
              std::to_string(lowMask) + ", " + vl + ")";
        std::string high = byteLoad(*record, highByte, "replica_joined_high");
        high = "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
               rawSuffix + "(" + high + ", " + std::to_string(fieldWidth) +
               ", " + vl + "), " + std::to_string(highMask) + ", " + vl +
               ")";
        const std::string assembled =
            operation.getAccess().getOrder() == "lo_first"
                ? "__riscv_vor_vv_" + rawSuffix + "(" + low +
                      ", __riscv_vsll_vx_" + rawSuffix + "(" + high + ", " +
                      std::to_string(lowBits) + ", " + vl + "), " + vl + ")"
                : "__riscv_vor_vv_" + rawSuffix + "(" + high +
                      ", __riscv_vsll_vx_" + rawSuffix + "(" + low + ", " +
                      std::to_string(fieldWidth - lowBits) + ", " + vl +
                      "), " + vl + ")";
        std::string value = fresh("replica_joined_value");
        line(rawType + " " + value + " = (" + logical + " < " +
             std::to_string(group) + " ? " + head + " : " + assembled +
             ");");
        windows.push_back(std::move(value));
        continue;
      }

      const unsigned bytes = fieldWidth / 8;
      if (!bytes || (fieldWidth != 8 && fieldWidth != 16 && fieldWidth != 32))
        return fail(operation,
                    "natural interleaved storage requires one byte-addressable logical element");
      const std::string unsignedSuffix =
          "u" + std::to_string(fieldWidth) +
          lmulSpelling(layout.getLmulEighths());
      const std::string unsignedType =
          "vuint" + std::to_string(fieldWidth) +
          lmulSpelling(layout.getLmulEighths()) + "_t";
      std::string assembled;
      for (unsigned byte = 0; byte < bytes; ++byte) {
        const std::string byteIndex =
            std::to_string(plan.getByteOffset() + byte) + " + (" + logical +
            ") * " + std::to_string(bytes);
        std::string loaded =
            byteLoad(*record, byteIndex, "replica_natural_byte");
        std::string widened = loaded;
        if (bytes > 1) {
          widened = "__riscv_vzext_vf" + std::to_string(bytes) + "_" +
                    unsignedSuffix + "(" + loaded + ", " + vl + ")";
          if (byte)
            widened = "__riscv_vsll_vx_" + unsignedSuffix + "(" + widened +
                      ", " + std::to_string(byte * 8) + ", " + vl + ")";
        }
        assembled = assembled.empty()
                        ? widened
                        : "__riscv_vor_vv_" + unsignedSuffix + "(" +
                              assembled + ", " + widened + ", " + vl + ")";
      }
      std::string value = fresh("replica_natural_value");
      if (vectorSuffixValue == unsignedSuffix) {
        line(vectorTypeValue + " " + value + " = " + assembled + ";");
      } else {
        line(vectorTypeValue + " " + value + " = __riscv_vreinterpret_v_" +
             unsignedSuffix + "_" + vectorSuffixValue + "(" + assembled +
             ");");
      }
      windows.push_back(std::move(value));
    }
  } else if (!layered) {
    const unsigned width = fieldInteger.getWidth();
    auto scalarType = scalarCType(field->type);
    if (!scalarType || (width != 8 && width != 16 && width != 32) ||
        loadSuffix.empty() || loadType.empty())
      return fail(operation,
                  "natural replica storage load requires a byte-addressable integer field");
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "natural replica storage window has no register-coordinate record");
      const std::string index = "((" + logicalBase.scalar + ") + " +
                                std::to_string(offset) + ")";
      const std::string pointer =
          "((const " + *scalarType + " *)((const uint8_t *)(" +
          *record + ") + " + std::to_string(plan.getByteOffset()) +
          ")) + " + index;
      std::string raw = fresh("replica_window");
      line(loadType + " " + raw + " = __riscv_vle" +
           std::to_string(width) + "_v_" + loadSuffix + "(" + pointer +
           ", " + loadVL + ");");
      windows.push_back(std::move(raw));
    }
  } else {
    const int64_t group = plan.getGroupSize();
    const int64_t layer = plan.getLayerSize();
    const int64_t layers = group / layer;
    if (fieldInteger.isSigned() || group <= 0 || layer <= 0 || group % layer ||
        fieldInteger.getWidth() * layers > 8 || layout.getSew() != 8)
      return fail(operation,
                  "layered replica storage load has invalid packed storage geometry");
    const std::string groupIndex = "((" + logicalBase.scalar + ") / " +
                                   std::to_string(group) + ")";
    const std::string withinLayer = "((" + logicalBase.scalar + ") % " +
                                    std::to_string(layer) + ")";
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "layered replica storage window has no register-coordinate record");
      const std::string byte =
          "(" + std::to_string(plan.getByteOffset()) + " + " + groupIndex +
          " * " + std::to_string(layer) + " + " + withinLayer + " + " +
          std::to_string(offset) + ")";
      std::string raw = fresh("replica_layered_raw");
      line(vectorTypeValue + " " + raw + " = __riscv_vle8_v_" +
           vectorSuffixValue + "((const uint8_t *)(" + *record +
           " + " + byte + "), " + vl + ");");
      windows.push_back(std::move(raw));
    }
  }

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < operation.getWindowForPart().size(); ++part) {
    const int64_t window = operation.getWindowForPart()[part];
    if (window < 0 || window >= static_cast<int64_t>(windows.size()))
      return fail(operation,
                  "RVV replica storage load part references an absent source window");
    std::string value = windows[static_cast<size_t>(window)];
    if (layered) {
      const int64_t group = plan.getGroupSize();
      const int64_t layer = plan.getLayerSize();
      const int64_t shiftOffset = operation.getShiftOffsetForPart()[part];
      const int64_t shiftBaseFactor =
          operation.getShiftBaseFactorForPart()[part];
      if (shiftOffset != 0 || shiftBaseFactor != 0) {
        std::string shift = std::to_string(shiftOffset);
        if (shiftBaseFactor != 0)
          shift = "(" + shift + " + " + std::to_string(shiftBaseFactor) +
                  " * (((" + logicalBase.scalar + ") % " +
                  std::to_string(group) + ") / " + std::to_string(layer) +
                  "))";
        std::string shifted = fresh("replica_layered_shift");
        line(vectorTypeValue + " " + shifted + " = __riscv_vsrl_vx_" +
             vectorSuffixValue + "(" + value + ", (size_t)(" +
             shift + "), " + vl + ");");
        value = std::move(shifted);
      }
      const int64_t maskValue = operation.getMaskValueForPart()[part];
      if (maskValue != 0) {
        std::string decoded = fresh("replica_layered_value");
        line(vectorTypeValue + " " + decoded + " = __riscv_vand_vx_" +
             vectorSuffixValue + "(" + value + ", " +
             std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(decoded);
      }
    }
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenAccumulate(
    riscv::RVVWidenAccumulateOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vwmacc.partial")
    return fail(operation,
                "RVV widened partial accumulation has no exact selected leaf");
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  auto accumulator = materializeNumeric(
      operation.getAccumulator(), bindings.lookup(operation.getAccumulator()));
  if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(accumulator) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Vector ||
      accumulator->kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV widened partial accumulation requires materialized vectors");
  auto lhsElement = operation.getLhs().getType().getElementType();
  auto rhsElement = operation.getRhs().getType().getElementType();
  const bool mixed = lhsElement.isSignedInteger() != rhsElement.isSignedInteger();
  const std::string suffix = vectorSuffix(operation.getAccumulator());
  const std::string type = vectorType(operation.getAccumulator());
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t parts = vectorPartCount(operation.getResult());
  if (accumulator->parts.size() != static_cast<size_t>(parts))
    return fail(operation,
                "RVV widened partial accumulator part count is incomplete");
  for (int64_t part = 0; part < parts; ++part) {
    auto lhsPart = projectRegisterPart(operation.getLhs(), operation.getResult(), part);
    auto rhsPart = projectRegisterPart(operation.getRhs(), operation.getResult(), part);
    if (!lhsPart || !rhsPart || *lhsPart >= lhs->parts.size() ||
        *rhsPart >= rhs->parts.size())
      return fail(operation,
                  "RVV widened partial operands do not project to accumulator replicas");
    const std::string &signedOperand =
        lhsElement.isSignedInteger() ? lhs->parts[*lhsPart] : rhs->parts[*rhsPart];
    const std::string &unsignedOperand =
        lhsElement.isSignedInteger() ? rhs->parts[*rhsPart] : lhs->parts[*lhsPart];
    const std::string operands =
        mixed ? signedOperand + ", " + unsignedOperand
              : lhs->parts[*lhsPart] + ", " + rhs->parts[*rhsPart];
    std::string name = fresh("widen_partial");
    line(type + " " + name + " = __riscv_" +
         std::string(mixed ? "vwmaccsu" : "vwmacc") + "_vv_" + suffix + "(" +
         accumulator->parts[part] + ", " + operands + ", " +
         partVL(operation.getLhs(), *lhsPart) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVFinalizeWidenDot(
    riscv::RVVFinalizeWidenDotOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.vwredsum.partial" &&
      instruction != "rvv.vredsum.partial")
    return fail(operation,
                "RVV final widened dot has no exact selected leaf");
  auto partial = materializeNumeric(operation.getPartial(),
                                    bindings.lookup(operation.getPartial()));
  if (mlir::failed(partial) || partial->kind != Binding::Kind::Vector ||
      partial->parts.empty())
    return fail(operation,
                "RVV final widened dot requires a materialized partial tuple");
  auto resultValue = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  const int64_t resultParts =
      resultValue ? registerPartCount(operation.getResult()) : 1;
  if (partial->parts.size() != static_cast<size_t>(resultParts))
    return fail(operation,
                "RVV final widened dot cannot project partials to result replicas");
  std::string seed = fresh("widen_partial_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  Binding result;
  result.kind = resultValue ? Binding::Kind::ScalarTuple : Binding::Kind::Scalar;
  const std::string partialSuffix = vectorSuffix(operation.getPartial());
  auto partialElement = mlir::dyn_cast<mlir::IntegerType>(
      operation.getPartial().getType().getElementType());
  if (!partialElement ||
      (partialElement.getWidth() != 16 && partialElement.getWidth() != 32))
    return fail(operation,
                "RVV final widened dot has no supported partial element width");
  for (int64_t part = 0; part < resultParts; ++part) {
    std::string reduced = fresh("widen_partial_sum");
    line("vint32m1_t " + reduced + " = __riscv_" +
         std::string(partialElement.getWidth() == 16 ? "vwredsum" : "vredsum") +
         "_vs_" +
         partialSuffix + "_i32m1(" + partial->parts[part] + ", " + seed + ", " +
         partVL(operation.getPartial(), part) + ");");
    std::string scalarName = fresh("widen_partial_scalar");
    line("int32_t " + scalarName + " = __riscv_vmv_x_s_i32m1_i32(" +
         reduced + ");");
    if (resultValue)
      result.parts.push_back(std::move(scalarName));
    else
      result.scalar = std::move(scalarName);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialSet(riscv::RVVPartialSetOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-set")
    return fail(operation, "RVV partial set has no exact selected leaf");
  auto setType = operation.getResult().getType();
  auto partialType = setType.getPartialType();
  auto partialAxis = llvm::find(partialType.getAxisIds().asArrayRef(),
                                operation.getReductionAxis());
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  const int64_t sliceLMUL = partialType.getLayout().getLmulEighths() / 2;
  const int64_t sliceLanes =
      partialAxis == partialType.getAxisIds().asArrayRef().end()
          ? 0
          : partialType.getLayout().getLaneFactors()[static_cast<size_t>(
                partialAxis - partialType.getAxisIds().asArrayRef().begin())];
  if (!kernel || sliceLMUL <= 0 || sliceLanes <= 0 ||
      operation.getLhsLaneOffsets().size() !=
          static_cast<size_t>(setType.getSlots()) ||
      operation.getRhsLaneOffsets().size() !=
          static_cast<size_t>(setType.getSlots()))
    return fail(operation,
                "RVV partial set is missing its typed lane-slice plan");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  const llvm::StringRef instruction = operation.getMultiplyInstruction();
  for (int64_t slot = 0; slot < setType.getSlots(); ++slot) {
    const int64_t lhsOperand = operation.getLhsOperands()[slot];
    const int64_t rhsOperand = operation.getRhsOperands()[slot];
    if (lhsOperand < 0 ||
        lhsOperand >= static_cast<int64_t>(operation.getLhs().size()) ||
        rhsOperand < 0 ||
        rhsOperand >= static_cast<int64_t>(operation.getRhs().size()))
      return fail(operation,
                  "RVV partial set slot references an absent operand");
    mlir::Value lhsValue = operation.getLhs()[lhsOperand];
    mlir::Value rhsValue = operation.getRhs()[rhsOperand];
    auto lhs = materializeNumeric(lhsValue, bindings.lookup(lhsValue));
    auto rhs = materializeNumeric(rhsValue, bindings.lookup(rhsValue));
    if (mlir::failed(lhs) || mlir::failed(rhs) ||
        lhs->kind != Binding::Kind::Vector ||
        rhs->kind != Binding::Kind::Vector)
      return fail(operation,
                  "RVV partial set requires materialized vector operands");
    auto lhsElement =
        mlir::cast<riscv::ValueType>(lhsValue.getType()).getElementType();
    auto rhsElement =
        mlir::cast<riscv::ValueType>(rhsValue.getType()).getElementType();
    const int64_t lhsPart = operation.getLhsParts()[slot];
    const int64_t rhsPart = operation.getRhsParts()[slot];
    if (lhsPart < 0 || rhsPart < 0 ||
        lhsPart >= static_cast<int64_t>(lhs->parts.size()) ||
        rhsPart >= static_cast<int64_t>(rhs->parts.size()) ||
        lhs->parts[lhsPart].empty() || rhs->parts[rhsPart].empty())
      return fail(operation,
                  "RVV partial set slot has no selected physical operand part");
    struct OperandSlice {
      std::string value;
      std::string suffix;
    };
    auto sliceOperand = [&](mlir::Value value, llvm::StringRef source,
                            int64_t laneOffset)
        -> std::optional<OperandSlice> {
      auto valueType = mlir::cast<riscv::ValueType>(value.getType());
      const int64_t sourceLMUL = valueType.getLayout().getLmulEighths();
      const int64_t inputSEW = valueType.getLayout().getSew();
      if (sourceLMUL < sliceLMUL || sourceLMUL % sliceLMUL || inputSEW <= 0 ||
          laneOffset < 0 || laneOffset % sliceLanes)
        return std::nullopt;
      const std::string sourceSuffix = vectorSuffix(value);
      if (sourceSuffix.empty())
        return std::nullopt;
      const char category = sourceSuffix.front();
      const std::string sliceSuffix =
          std::string(1, category) + std::to_string(inputSEW) +
          lmulSpelling(sliceLMUL);
      const std::string sliceType =
          std::string(category == 'u' ? "vuint" : "vint") +
          sliceSuffix.substr(1) + "_t";
      const int64_t sliceSpan =
          kernel.getTarget().getVlenBits() * sliceLMUL / (8 * inputSEW);
      if (sliceSpan <= 0)
        return std::nullopt;
      const int64_t group = laneOffset / sliceSpan;
      const int64_t intra = laneOffset % sliceSpan;
      std::string sliced = source.str();
      if (sourceLMUL != sliceLMUL) {
        sliced = fresh("partial_operand_slice");
        line(sliceType + " " + sliced + " = __riscv_vget_v_" +
             sourceSuffix + "_" + sliceSuffix + "(" + source.str() + ", " +
             std::to_string(group) + ");");
      } else if (group != 0) {
        return std::nullopt;
      }
      if (intra != 0) {
        std::string shifted = fresh("partial_operand_slide");
        line(sliceType + " " + shifted + " = __riscv_vslidedown_vx_" +
             sliceSuffix + "(" + sliced + ", " + std::to_string(intra) +
             ", " + std::to_string(sliceLanes) + ");");
        sliced = std::move(shifted);
      }
      return OperandSlice{std::move(sliced), sliceSuffix};
    };
    auto lhsSlice = sliceOperand(lhsValue, lhs->parts[lhsPart],
                                 operation.getLhsLaneOffsets()[slot]);
    auto rhsSlice = sliceOperand(rhsValue, rhs->parts[rhsPart],
                                 operation.getRhsLaneOffsets()[slot]);
    if (!lhsSlice || !rhsSlice)
      return fail(operation,
                  "RVV partial set cannot materialize its selected lane slice");
    std::string left = lhsSlice->value;
    std::string right = rhsSlice->value;
    std::string mnemonic;
    if (instruction == "rvv.vwmul.vv") {
      mnemonic = "vwmul";
    } else if (instruction == "rvv.vwmul.vv.reinterpret-rhs" ||
               instruction == "rvv.vwmul.vv.reinterpret-lhs") {
      mnemonic = "vwmul";
      std::string sourceSuffix =
          instruction == "rvv.vwmul.vv.reinterpret-rhs" ? rhsSlice->suffix
                                                          : lhsSlice->suffix;
      std::string targetSuffix = sourceSuffix;
      if (sourceSuffix.empty() || sourceSuffix.front() != 'u')
        return fail(operation,
                    "narrow unsigned partial operand has no unsigned RVV spelling");
      targetSuffix.front() = 'i';
      std::string &operand =
          instruction == "rvv.vwmul.vv.reinterpret-rhs" ? right : left;
      operand = "__riscv_vreinterpret_v_" + sourceSuffix + "_" +
                targetSuffix + "(" + operand + ")";
    } else if (instruction == "rvv.vwmulsu.vv") {
      mnemonic = "vwmulsu";
    } else if (instruction == "rvv.vwmulsu.vv.swap") {
      mnemonic = "vwmulsu";
      std::swap(left, right);
    } else {
      return fail(operation,
                  "RVV partial set has no exact widening multiply instruction");
    }
    std::string product = fresh("partial_slot");
    line(vectorTypeFor(partialType) + " " + product + " = __riscv_" +
         mnemonic + "_vv_" + vectorSuffixFor(partialType) + "(" + left +
         ", " + right + ", " + std::to_string(sliceLanes) + ");");
    result.parts.push_back(std::move(product));
  }
  if (result.parts.size() != static_cast<size_t>(setType.getSlots()))
    return fail(operation,
                "RVV partial set did not materialize every declared slot");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialCapture(riscv::RVVPartialCaptureOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-capture")
    return fail(operation, "RVV partial capture has no exact selected leaf");
  auto input = materializeNumeric(operation.getInput(),
                                  bindings.lookup(operation.getInput()));
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      input->parts.size() != 1 || operation.getResult().getType().getSlots() != 1)
    return fail(operation,
                "RVV partial capture requires one materialized vector partial");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  result.parts.push_back(input->parts.front());
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialRepack(riscv::RVVPartialRepackOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-repack.split")
    return fail(operation, "RVV partial repack has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  const int64_t split = operation.getSplitFactor();
  if (input.kind != Binding::Kind::PartialSet || split <= 1 ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() != inputType.getSlots() * split)
    return fail(operation,
                "RVV partial repack has no complete typed split input");
  const std::string sourceSuffix =
      vectorSuffixFor(inputType.getPartialType());
  const std::string targetSuffix =
      vectorSuffixFor(resultType.getPartialType());
  const std::string targetType = vectorTypeFor(resultType.getPartialType());
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (llvm::StringRef part : input.parts)
    for (int64_t chunk = 0; chunk < split; ++chunk) {
      std::string splitPart = fresh("partial_split");
      line(targetType + " " + splitPart + " = __riscv_vget_v_" +
           sourceSuffix + "_" + targetSuffix + "(" + part.str() + ", " +
           std::to_string(chunk) + ");");
      result.parts.push_back(std::move(splitPart));
    }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialMerge(riscv::RVVPartialMergeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-merge" ||
      operation.getTopology() != "pairwise")
    return fail(operation, "RVV partial merge has no exact selected topology");
  llvm::SmallVector<std::string> level;
  for (mlir::Value inputValue : operation.getInputs()) {
    Binding input = bindings.lookup(inputValue);
    auto inputType = mlir::cast<riscv::PartialSetType>(inputValue.getType());
    if (input.kind != Binding::Kind::PartialSet ||
        input.parts.size() != static_cast<size_t>(inputType.getSlots()))
      return fail(operation,
                  "RVV partial merge input has no complete typed slot set");
    level.append(input.parts.begin(), input.parts.end());
  }
  if (level.size() < 2 || operation.getResult().getType().getSlots() != 1)
    return fail(operation,
                "RVV partial merge requires multiple slots and one result");
  const std::string type =
      vectorTypeFor(operation.getResult().getType().getPartialType());
  const std::string suffix =
      vectorSuffixFor(operation.getResult().getType().getPartialType());
  const std::string vl = std::to_string(
      physicalLanesFor(operation.getResult().getType().getPartialType()));
  while (level.size() > 1) {
    llvm::SmallVector<std::string> next;
    for (size_t part = 0; part < level.size(); part += 2) {
      if (part + 1 == level.size()) {
        next.push_back(std::move(level[part]));
        continue;
      }
      std::string combined = fresh("partial_merge");
      line(type + " " + combined + " = __riscv_vadd_vv_" + suffix + "(" +
           level[part] + ", " + level[part + 1] + ", " + vl + ");");
      next.push_back(std::move(combined));
    }
    level = std::move(next);
  }
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  result.parts.push_back(std::move(level.front()));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialReduce(riscv::RVVPartialReduceOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-reduce.widen" &&
      instruction != "rvv.partial-reduce")
    return fail(operation, "RVV partial reduction has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  auto inputPartial = inputType.getPartialType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() != inputType.getSlots())
    return fail(operation,
                "RVV partial reduction has no complete typed input set");
  std::string seed = fresh("partial_reduce_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  const std::string reduction = instruction == "rvv.partial-reduce.widen"
                                    ? "vwredsum"
                                    : "vredsum";
  const std::string suffix = vectorSuffixFor(inputPartial);
  const std::string vl = std::to_string(physicalLanesFor(inputPartial));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (llvm::StringRef partial : input.parts) {
    std::string reduced = fresh("partial_reduced");
    line("vint32m1_t " + reduced + " = __riscv_" + reduction + "_vs_" +
         suffix + "_i32m1(" + partial.str() + ", " + seed + ", " + vl +
         ");");
    result.parts.push_back(std::move(reduced));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartialScaleCombine(
    riscv::RVVPartialScaleCombineOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.partial-scale-combine")
    return fail(operation,
                "RVV scaled partial combine has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      operation.getScales().size() != input.parts.size() ||
      operation.getScaleReplicas().size() != input.parts.size() ||
      operation.getSlotOrder().size() != input.parts.size() ||
      resultType.getSlots() <= 0 || inputType.getSlots() % resultType.getSlots())
    return fail(operation,
                "RVV scaled partial combine has no complete typed slot set");
  llvm::SmallVector<std::string> scales;
  for (auto [scaleValue, replica] :
       llvm::zip(operation.getScales(), operation.getScaleReplicas())) {
    auto scale =
        materializeNumeric(scaleValue, bindings.lookup(scaleValue));
    if (mlir::failed(scale))
      return mlir::failure();
    if (scale->kind == Binding::Kind::Scalar && replica == 0) {
      scales.push_back(scale->scalar);
      continue;
    }
    if (scale->kind == Binding::Kind::ScalarTuple && replica >= 0 &&
        static_cast<size_t>(replica) < scale->parts.size()) {
      scales.push_back(scale->parts[replica]);
      continue;
    }
    return fail(operation,
                "RVV scaled partial combine scale replica has no scalar binding");
  }
  const int64_t fanout = inputType.getSlots() / resultType.getSlots();
  const std::string suffix = vectorSuffixFor(inputType.getPartialType());
  const std::string type = vectorTypeFor(inputType.getPartialType());
  const std::string vl =
      std::to_string(physicalLanesFor(inputType.getPartialType()));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  llvm::SmallVector<std::string> combined;
  combined.reserve(resultType.getSlots());
  for (int64_t resultSlot = 0; resultSlot < resultType.getSlots(); ++resultSlot) {
    const int64_t first = operation.getSlotOrder()[resultSlot * fanout];
    std::string seed = fresh("scaled_partial");
    line(type + " " + seed + " = __riscv_vmul_vx_" + suffix + "(" +
         input.parts[first] + ", " + scales[first] + ", " + vl + ");");
    combined.push_back(std::move(seed));
  }
  // Independent chains are already fixed by the typed result slots.  Spell
  // one dependency level from every chain before advancing to the next level
  // so the C compiler can preserve their instruction-level parallelism.
  for (int64_t term = 1; term < fanout; ++term) {
    for (int64_t resultSlot = 0; resultSlot < resultType.getSlots();
         ++resultSlot) {
      const int64_t slot =
          operation.getSlotOrder()[resultSlot * fanout + term];
      std::string next = fresh("scaled_partial_chain");
      line(type + " " + next + " = __riscv_vmacc_vx_" + suffix + "(" +
           combined[resultSlot] + ", " + scales[slot] + ", " +
           input.parts[slot] + ", " + vl + ");");
      combined[resultSlot] = std::move(next);
    }
  }
  result.parts = std::move(combined);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartialWidenScale(
    riscv::RVVPartialWidenScaleOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-widen-scale")
    return fail(operation,
                "RVV partial widen-scale has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      operation.getCombineArity() <= 0 ||
      resultType.getSlots() * operation.getCombineArity() !=
          inputType.getSlots() ||
      operation.getScales().size() != input.parts.size() ||
      operation.getScaleReplicas().size() != input.parts.size())
    return fail(operation,
                "RVV partial widen-scale has no complete typed slot set");
  llvm::SmallVector<std::string> scales;
  for (auto [scaleValue, replica] :
       llvm::zip(operation.getScales(), operation.getScaleReplicas())) {
    auto scale =
        materializeNumeric(scaleValue, bindings.lookup(scaleValue));
    if (mlir::failed(scale))
      return mlir::failure();
    if (scale->kind == Binding::Kind::Scalar && replica == 0) {
      scales.push_back(scale->scalar);
      continue;
    }
    if (scale->kind == Binding::Kind::ScalarTuple && replica >= 0 &&
        static_cast<size_t>(replica) < scale->parts.size()) {
      scales.push_back(scale->parts[replica]);
      continue;
    }
    return fail(
        operation,
        "RVV partial widen-scale scale replica has no scalar binding");
  }
  const std::string type = vectorTypeFor(resultType.getPartialType());
  const std::string suffix = vectorSuffixFor(resultType.getPartialType());
  const std::string vl =
      std::to_string(physicalLanesFor(inputType.getPartialType()));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (int64_t resultSlot = 0; resultSlot < resultType.getSlots();
       ++resultSlot) {
    const size_t first =
        static_cast<size_t>(resultSlot * operation.getCombineArity());
    std::string accumulated = fresh("partial_widen_scale");
    line(type + " " + accumulated + " = __riscv_vwmul_vx_" + suffix + "(" +
         input.parts[first] + ", " + scales[first] + ", " + vl + ");");
    for (int64_t term = 1; term < operation.getCombineArity(); ++term) {
      const size_t slot = first + static_cast<size_t>(term);
      std::string next = fresh("partial_widen_scale_chain");
      line(type + " " + next + " = __riscv_vwmacc_vx_" + suffix + "(" +
           accumulated + ", " + scales[slot] + ", " + input.parts[slot] +
           ", " + vl + ");");
      accumulated = std::move(next);
    }
    result.parts.push_back(std::move(accumulated));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialCombine(riscv::RVVPartialCombineOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-combine")
    return fail(operation, "RVV partial combine has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() * operation.getArity() != inputType.getSlots() ||
      operation.getTopology() != "pairwise")
    return fail(operation,
                "RVV partial combine has no complete input slot set");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  const std::string type = vectorTypeFor(resultType.getPartialType());
  const std::string suffix = vectorSuffixFor(resultType.getPartialType());
  const std::string vl = std::to_string(
      physicalLanesFor(resultType.getPartialType()));
  for (int64_t slot = 0; slot < resultType.getSlots(); ++slot) {
    llvm::SmallVector<std::string> level;
    for (int64_t term = 0; term < operation.getArity(); ++term)
      level.push_back(input.parts[slot * operation.getArity() + term]);
    while (level.size() > 1) {
      llvm::SmallVector<std::string> nextLevel;
      for (size_t term = 0; term < level.size(); term += 2) {
        if (term + 1 == level.size()) {
          nextLevel.push_back(std::move(level[term]));
          continue;
        }
        std::string next = fresh("partial_tree");
        line(type + " " + next + " = __riscv_vadd_vv_" + suffix + "(" +
             level[term] + ", " + level[term + 1] + ", " + vl + ");");
        nextLevel.push_back(std::move(next));
      }
      level = std::move(nextLevel);
    }
    result.parts.push_back(std::move(level.front()));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialFinalize(riscv::RVVPartialFinalizeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-finalize.reduce" &&
      instruction != "rvv.partial-finalize.extract" &&
      instruction != "rvv.partial-finalize.widen")
    return fail(operation, "RVV partial finalize has no exact selected leaf");
  auto setType = operation.getInput().getType();
  auto partialType = setType.getPartialType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partialType.getElementType());
  Binding input = bindings.lookup(operation.getInput());
  if (!partialElement || input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(setType.getSlots()) ||
      input.parts.empty())
    return fail(operation,
                "RVV partial finalize has no complete typed partial set");
  if (instruction == "rvv.partial-finalize.widen") {
    auto resultType = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (!resultType || input.parts.size() != 1 ||
        vectorPartCount(operation.getResult()) != 1)
      return fail(operation,
                  "RVV vector partial finalize requires one typed input and one output part");
    Binding result;
    result.kind = Binding::Kind::Vector;
    std::string widened = fresh("partial_vector");
    line(vectorType(operation.getResult()) + " " + widened +
         " = __riscv_vwcvt_x_x_v_" + vectorSuffix(operation.getResult()) +
         "(" + input.parts.front() + ", " +
         partVL(operation.getResult(), 0) + ");");
    result.parts.push_back(std::move(widened));
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  llvm::SmallVector<std::string> scalars;
  if (instruction == "rvv.partial-finalize.extract") {
    const std::string suffix = vectorSuffixFor(partialType);
    const std::string scalarSuffix =
        partialElement.getWidth() == 16 ? "i16" : "i32";
    for (llvm::StringRef part : input.parts) {
      std::string extracted = fresh("partial_scalar_part");
      line("int32_t " + extracted + " = (int32_t)__riscv_vmv_x_s_" +
           suffix + "_" + scalarSuffix + "(" + part.str() + ");");
      scalars.push_back(std::move(extracted));
    }
  } else {
    std::string seed = fresh("partial_seed");
    line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
    const std::string reduction =
        partialElement.getWidth() == 16 ? "vwredsum" : "vredsum";
    const std::string suffix = vectorSuffixFor(partialType);
    const std::string vl = std::to_string(physicalLanesFor(partialType));
    for (llvm::StringRef part : input.parts) {
      std::string reduced = fresh("partial_reduction");
      line("vint32m1_t " + reduced + " = __riscv_" + reduction + "_vs_" +
           suffix + "_i32m1(" + part.str() + ", " + seed + ", " + vl +
           ");");
      seed = std::move(reduced);
    }
    std::string extracted = fresh("partial_scalar_part");
    line("int32_t " + extracted + " = __riscv_vmv_x_s_i32m1_i32(" + seed +
         ");");
    scalars.push_back(std::move(extracted));
  }
  std::string scalarName = fresh("partial_scalar");
  std::string expression = scalars.front();
  for (size_t index = 1; index < scalars.size(); ++index)
    expression = "(" + expression + " + " + scalars[index] + ")";
  line("int32_t " + scalarName + " = " + expression + ";");
  bindings[operation.getResult()] = scalar(std::move(scalarName));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVAssembleReplicas(
    riscv::RVVAssembleReplicasOp operation) {
  if (instructionOf(operation.getOperation()) != "scalar.assemble-replicas")
    return fail(operation,
                "scalar replica assembly has no exact selected leaf");
  Binding result;
  result.kind = Binding::Kind::ScalarTuple;
  for (mlir::Value value : operation.getValues()) {
    Binding part = bindings.lookup(value);
    if (part.kind != Binding::Kind::Scalar)
      return fail(operation,
                  "scalar replica assembly operand has no scalar binding");
    result.parts.push_back(part.scalar);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVWidenReduce(riscv::RVVWidenReduceOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vwredsum")
    return fail(operation, "RVV widening reduction has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  if (input->kind != Binding::Kind::Vector || input->parts.size() != 1)
    return fail(operation,
                "RVV widening reduction requires one materialized vector part");
  auto inputElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getInput().getType()));
  auto resultElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getResult().getType()));
  const bool isUnsigned = inputElement.isUnsigned();
  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string resultStem =
      std::string(isUnsigned ? "u" : "i") +
      std::to_string(resultElement.getWidth());
  const std::string resultSuffix = resultStem + "m1";
  const std::string resultVectorType =
      std::string(isUnsigned ? "vuint" : "vint") +
      std::to_string(resultElement.getWidth()) + "m1_t";
  auto scalarType = scalarCType(resultElement);
  if (!scalarType)
    return fail(operation, "RVV widening reduction result has no C scalar type");
  const std::string vl = partVL(operation.getInput(), 0);
  std::string seed = fresh("widen_reduce_seed");
  line(resultVectorType + " " + seed + " = __riscv_vmv_v_x_" +
       resultSuffix + "(0, 1);");
  std::string reduced = fresh("widen_reduce");
  line(resultVectorType + " " + reduced + " = __riscv_" +
       std::string(isUnsigned ? "vwredsumu" : "vwredsum") + "_vs_" +
       inputSuffix + "_" + resultSuffix + "(" + input->parts.front() + ", " +
       seed + ", " + vl + ");");
  std::string scalarName = fresh("widen_reduce_scalar");
  line(*scalarType + " " + scalarName + " = __riscv_vmv_x_s_" +
       resultSuffix + "_" + resultStem + "(" + reduced + ");");
  bindings[operation.getResult()] = scalar(scalarName);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartitionedWidenReduceStore(
    riscv::RVVPartitionedWidenReduceStoreOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.partitioned-vwredsum-store")
    return fail(operation,
                "partitioned widening reduction store has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  if (input->kind != Binding::Kind::Vector)
    return fail(operation,
                "partitioned widening reduction store requires an RVV input");
  Binding destination = bindings.lookup(operation.getDestination());
  if (destination.kind != Binding::Kind::Field)
    return fail(operation,
                "partitioned widening reduction store requires one encoded field destination");
  auto field = fieldFor(destination);
  Binding record = bindings.lookup(destination.field.owner);
  if (record.kind == Binding::Kind::Slice) {
    auto materialized = recordForSlice(destination.field.owner);
    if (mlir::failed(materialized))
      return mlir::failure();
    record = std::move(*materialized);
  }
  auto inputElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getInput().getType()));
  auto fieldInteger = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                            : mlir::IntegerType();
  if (!field || record.kind != Binding::Kind::Record ||
      field->access.getMapping() != "natural" || field->bitOffset % 8 ||
      !fieldInteger || !fieldInteger.isSigned() ||
      fieldInteger.getWidth() != 16 ||
      !inputElement.isSigned() || inputElement.getWidth() != 8 ||
      operation.getPartition() != 16)
    return fail(operation,
                "partitioned widening reduction store has no closed i8-to-i16 field realization");
  Binding baseIndex = bindings.lookup(operation.getBaseIndex());
  if (baseIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "partitioned widening reduction store base index is not scalar");
  const int64_t count = operation.getCount();
  const int64_t inputParts = static_cast<int64_t>(input->parts.size());
  if (inputParts <= 0 || count % inputParts)
    return fail(operation,
                "partitioned widening reduction store cannot project its input parts");
  const int64_t partitionsPerInput = count / inputParts;
  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string inputType = vectorType(operation.getInput());
  for (int64_t part = 0; part < count; ++part) {
    const int64_t sourcePart = part / partitionsPerInput;
    const int64_t withinPart = part % partitionsPerInput;
    std::string chunk = input->parts[sourcePart];
    std::string reductionSuffix = inputSuffix;
    if (withinPart != 0) {
      std::string projected = fresh("partition_chunk");
      line(inputType + " " + projected + " = __riscv_vslidedown_vx_" +
           inputSuffix + "(" + chunk + ", " +
           std::to_string(withinPart * operation.getPartition()) + ", " +
           std::to_string(operation.getInput().getType().getLayout().getVl()) +
           ");");
      chunk = std::move(projected);
    }
    std::string seed = fresh("partition_seed");
    line("vint16m1_t " + seed + " = __riscv_vmv_v_x_i16m1(0, 1);");
    std::string reduced = fresh("partition_sum");
    line("vint16m1_t " + reduced + " = __riscv_vwredsum_vs_" +
         reductionSuffix +
         "_i16m1(" + chunk + ", " + seed + ", " +
         std::to_string(operation.getPartition()) + ");");
    std::string scalarName = fresh("partition_scalar");
    line("int16_t " + scalarName + " = __riscv_vmv_x_s_i16m1_i16(" +
         reduced + ");");
    const std::string address =
        record.recordPointer + " + " + std::to_string(field->bitOffset / 8) +
        " + (" + baseIndex.scalar + " + " + std::to_string(part) + ") * 2";
    line(field->access.getAlignment() >= 2
             ? "*(int16_t *)(" + address + ") = " + scalarName + ";"
             : "weft_store_i16_le(" + address + ", " + scalarName + ");");
  }
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVLayeredStream(riscv::RVVLayeredStreamOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-stream")
    return fail(operation, "RVV layered stream has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation, "RVV layered stream requires one encoded field edge");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto geometry = operation.getGeometry();
  const int64_t layer = geometry.getLayerSize();
  const int64_t streams = geometry.getStreamCount();
  const int64_t replicas = geometry.getReplicaCount();
  const int64_t lanes = geometry.getLaneCount();
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  if (owner.kind != Binding::Kind::Record || !field || !integer ||
      integer.isSigned() || field->bitOffset % 8 || layer <= 0 || lanes <= 0 ||
      streams <= 1 || replicas <= 0 ||
      geometry.getWindowForStream().size() != static_cast<size_t>(streams) ||
      geometry.getRepresentativeStreamForWindow().size() !=
          geometry.getGroupForWindow().size() ||
      geometry.getPhysicalLayerForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getShiftAmountForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getMaskValueForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getGroupForWindow().size() !=
          geometry.getWithinForWindow().size())
    return fail(operation,
                "RVV layered stream has incomplete field, record, or layout geometry");

  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t replica = 0; replica < replicas; ++replica) {
    auto coordinates = registerCoordinates(operation.getResult(), replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "RVV layered stream has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      bool foundStride = false;
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis) {
          record += " + " + std::to_string(coordinate) + " * " + stride;
          foundStride = true;
        }
      if (!foundStride)
        return fail(operation,
                    "RVV layered stream register axis has no record-byte stride");
    }
    record += ")";
    llvm::SmallVector<std::string> loadedWindows;
    loadedWindows.reserve(geometry.getRepresentativeStreamForWindow().size());
    for (auto [window, firstStream] :
         llvm::enumerate(
             geometry.getRepresentativeStreamForWindow().asArrayRef())) {
      if (firstStream < 0)
        return fail(operation,
                    "RVV layered stream geometry contains an unused raw window");
      const int64_t byteOffset =
          field->bitOffset / 8 + geometry.getGroupForWindow()[window] * layer +
          geometry.getWithinForWindow()[window];
      const int64_t part = replica * streams + firstStream;
      std::string raw = fresh("layered_stream_raw");
      line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
           "((const uint8_t *)(" + record + " + " +
           std::to_string(byteOffset) + "), " +
           partVL(operation.getResult(), part) + ");");
      loadedWindows.push_back(std::move(raw));
    }
    for (int64_t stream = 0; stream < streams; ++stream) {
      const int64_t part = replica * streams + stream;
      const int64_t window = geometry.getWindowForStream()[stream];
      const int64_t shiftAmount = geometry.getShiftAmountForStream()[stream];
      const int64_t maskValue = geometry.getMaskValueForStream()[stream];
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = loadedWindows[static_cast<size_t>(window)];
      if (shiftAmount != 0) {
        std::string shifted = fresh("layered_stream_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue != 0) {
        std::string masked = fresh("layered_stream_value");
        line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
             value + ", " + std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(masked);
      }
      result.parts.push_back(std::move(value));
    }
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV layered stream did not produce every selected physical part");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVProjectedLayeredStream(
    riscv::RVVProjectedLayeredStreamOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.projected-layered-stream")
    return fail(operation,
                "projected RVV layered stream has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  if (fieldBinding.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point)
    return fail(operation,
                "projected RVV layered stream requires one encoded field and typed point");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto geometry = operation.getGeometry();
  const int64_t group = geometry.getGroupSize();
  const int64_t layer = geometry.getLayerSize();
  const int64_t streams = geometry.getStreamCount();
  const int64_t replicas = geometry.getReplicaCount();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !field || !integer || integer.isSigned() || field->bitOffset % 8 ||
      group <= 0 || layer <= 0 || group % layer || streams <= 1 || replicas <= 0 ||
      operation.getProjectionStride() != 1 ||
      operation.getProjectionRepeat() != 1 ||
      geometry.getWindowForStream().size() != static_cast<size_t>(streams) ||
      geometry.getRepresentativeStreamForWindow().size() !=
          geometry.getGroupForWindow().size() ||
      geometry.getPhysicalLayerForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getShiftAmountForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getMaskValueForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getGroupForWindow().size() !=
          geometry.getWithinForWindow().size())
    return fail(operation,
                "projected RVV layered stream has incomplete field or projection geometry");

  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t replica = 0; replica < replicas; ++replica) {
    auto coordinates = registerCoordinates(operation.getResult(), replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "projected RVV layered stream has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "projected RVV layered stream free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    const std::string recordGroup =
        "(((" + point.point.base + ") % " +
        std::to_string(owner.recordElements) + ") / " +
        std::to_string(group) + ")";
    llvm::SmallVector<std::string> loadedWindows;
    loadedWindows.reserve(geometry.getRepresentativeStreamForWindow().size());
    for (auto [window, firstStream] :
         llvm::enumerate(
             geometry.getRepresentativeStreamForWindow().asArrayRef())) {
      if (firstStream < 0)
        return fail(operation,
                    "projected RVV layered stream geometry contains an unused raw window");
      const std::string byte =
          "(" + std::to_string(field->bitOffset / 8) + " + (" + recordGroup +
          " + " + std::to_string(geometry.getGroupForWindow()[window]) +
          ") * " + std::to_string(layer) + " + " +
          std::to_string(geometry.getWithinForWindow()[window]) + ")";
      const int64_t part = replica * streams + firstStream;
      std::string raw = fresh("projected_layered_raw");
      line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
           "((const uint8_t *)(" + record + " + " + byte + "), " +
           partVL(operation.getResult(), part) + ");");
      loadedWindows.push_back(std::move(raw));
    }
    for (int64_t stream = 0; stream < streams; ++stream) {
      const int64_t part = replica * streams + stream;
      const int64_t window = geometry.getWindowForStream()[stream];
      const int64_t shiftAmount = geometry.getShiftAmountForStream()[stream];
      const int64_t maskValue = geometry.getMaskValueForStream()[stream];
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = loadedWindows[static_cast<size_t>(window)];
      if (shiftAmount != 0) {
        std::string shifted = fresh("projected_layered_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue != 0) {
        std::string masked = fresh("projected_layered_value");
        line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
             value + ", " + std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(masked);
      }
      result.parts.push_back(std::move(value));
    }
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "projected RVV layered stream did not produce every physical part");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVLayeredWindow(riscv::RVVLayeredWindowOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-window")
    return fail(operation, "layered window has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation, "layered window requires one encoded field edge");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  Binding point = bindings.lookup(operation.getPoint());
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto plan = operation.getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  if (owner.kind != Binding::Kind::Record || !field ||
      point.kind != Binding::Kind::Point || !integer || integer.isSigned() ||
      field->bitOffset % 8 || plan.getKind() != "layered" ||
      group != layer * 2 || owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      operation.getPhysicalLayerForResult().size() != 2 ||
      operation.getShiftForResult().size() != 2 ||
      operation.getMaskForResult().size() != 2 ||
      vectorPartCount(operation.getFirst()) !=
          vectorPartCount(operation.getSecond()))
    return fail(operation,
                "layered window has no closed two-layer encoded realization");

  const std::string suffix = vectorSuffix(operation.getFirst());
  const std::string type = vectorType(operation.getFirst());
  const int64_t streams = streamPartCount(operation.getFirst());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getFirst());
  const std::string logical = "(" + point.point.base + " % " +
                              std::to_string(plan.getRecordElements()) + ")";
  Binding first;
  Binding second;
  first.kind = Binding::Kind::Vector;
  second.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getFirst()); ++part) {
    auto coordinates = registerCoordinates(operation.getFirst(), part / streams);
    if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "layered window has no complete register coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis)
          record += " + " + std::to_string(coordinate) + " * " + stride;
    record += ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + (" + logical +
        " / " + std::to_string(group) + ") * " + std::to_string(layer) +
        " + (" + logical + " % " + std::to_string(layer) + ") + " +
        partOffset(operation.getFirst(), part) + ")";
    const std::string vl = partVL(operation.getFirst(), part);
    std::string raw = fresh("layered_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " + vl +
         ");");
    auto decode = [&](size_t resultIndex, llvm::StringRef prefix) {
      const int64_t shiftAmount = operation.getShiftForResult()[resultIndex];
      const int64_t maskValue = operation.getMaskForResult()[resultIndex];
      std::string value = raw;
      if (shiftAmount != 0) {
        std::string shifted = fresh("layered_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue == 0)
        return value;
      std::string masked = fresh(prefix);
      line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
           value + ", " + std::to_string(maskValue) + ", " + vl + ");");
      return masked;
    };
    first.parts.push_back(decode(0, "layered_first"));
    second.parts.push_back(decode(1, "layered_second"));
  }
  bindings[operation.getFirst()] = std::move(first);
  bindings[operation.getSecond()] = std::move(second);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVStreamReduce(riscv::RVVStreamReduceOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.stream-reduce")
    return fail(operation, "RVV stream reduction has no exact selected leaf");
  Binding input = bindings.lookup(operation.getSource());
  riscv::LayoutAttr layout = operation.getInputLayout();
  if (input.kind != Binding::Kind::Slice ||
      product(layout.getReplicaFactors()) != 1 || layout.getAxisIds().size() != 1)
    return fail(operation,
                "RVV stream reduction requires one memory-backed scalar-replica slice");
  const Binding &base = bindings.lookup(input.slice.base);
  if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
      !base.memory.elementType.isF32())
    return fail(operation, "RVV stream reduction input is not a dense f32 slice");
  const int64_t laneAxis = layout.getAxisIds()[0];
  auto lane = llvm::find(base.memory.axes, laneAxis);
  const size_t laneDimension =
      static_cast<size_t>(lane - base.memory.axes.begin());
  if (lane == base.memory.axes.end() || laneDimension >= base.memory.strides.size())
    return fail(operation, "RVV stream reduction has no dense lane stride");
  const std::string suffix = "f32" + lmulSpelling(layout.getLmulEighths());
  const std::string type = "vfloat32" + lmulSpelling(layout.getLmulEighths()) + "_t";
  const std::string fullVL = std::to_string(layout.getVl());
  llvm::SmallVector<std::string> accumulators;
  for (mlir::Attribute kindAttribute : operation.getKinds()) {
    llvm::StringRef kind = mlir::cast<mlir::StringAttr>(kindAttribute).getValue();
    const std::string initial = kind == "max" ? "(-INFINITY)"
                                : kind == "min" ? "INFINITY"
                                                : "0.0f";
    std::string accumulator = fresh("stream_reduce");
    line(type + " " + accumulator + " = __riscv_vfmv_v_f_" + suffix +
         "(" + initial + ", " + fullVL + ");");
    accumulators.push_back(std::move(accumulator));
  }
  const int64_t streams = product(layout.getTimeFactors());
  if (streams <= 1)
    return fail(operation,
                "RVV stream reduction input has an unsupported part mapping");
  std::string streamIndex = fresh("stream_index");
  line("#pragma GCC unroll 1");
  line("for (size_t " + streamIndex + " = 0; " + streamIndex + " < " +
       std::to_string(streams) + "; ++" + streamIndex + ") {");
  ++indent;
  llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
  offsets.push_back({laneAxis, "(" + streamIndex + " * " + fullVL + ")"});
  auto address = denseAddress(input.slice, offsets);
  if (!address)
    return fail(operation, "RVV stream reduction has no dense address relation");
  std::string loaded = fresh("stream_load");
  if (operation.getAccess().getForm() == "unit")
    line(type + " " + loaded + " = __riscv_vle32_v_" + suffix + "(" +
         *address + ", " + fullVL + ");");
  else
    line(type + " " + loaded + " = __riscv_vlse32_v_" + suffix + "(" +
         *address + ", (ptrdiff_t)(" + base.memory.strides[laneDimension] +
         " * (ptrdiff_t)sizeof(float)), " + fullVL + ");");
  for (auto [index, kindAttribute] : llvm::enumerate(operation.getKinds())) {
    llvm::StringRef kind =
        mlir::cast<mlir::StringAttr>(kindAttribute).getValue();
    llvm::StringRef stem = kind == "max" ? "vfmax"
                           : kind == "min" ? "vfmin"
                                           : "vfadd";
    line(accumulators[index] + " = __riscv_" + stem.str() + "_vv_" + suffix +
         "(" + accumulators[index] + ", " + loaded + ", " + fullVL + ");");
  }
  --indent;
  line("}");
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    llvm::StringRef kind =
        mlir::cast<mlir::StringAttr>(operation.getKinds()[index]).getValue();
    const std::string initial = kind == "max" ? "(-INFINITY)"
                                : kind == "min" ? "INFINITY"
                                                : "0.0f";
    std::string seed = fresh("stream_seed");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" + initial +
         ", 1);");
    llvm::StringRef stem = kind == "max" ? "vfredmax"
                           : kind == "min" ? "vfredmin"
                                           : "vfredsum";
    std::string reduced = fresh("stream_scalar");
    line("vfloat32m1_t " + reduced + " = __riscv_" + stem.str() + "_vs_" +
         suffix + "_f32m1(" + accumulators[index] + ", " + seed + ", " +
         fullVL + ");");
    std::string scalarName = fresh("stream_value");
    line("float " + scalarName + " = __riscv_vfmv_f_s_f32m1_f32(" + reduced +
         ");");
    bindings[result] = scalar(scalarName);
  }
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVStreamDot(riscv::RVVStreamDotOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.stream-dot")
    return fail(operation, "RVV stream dot has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding extent = bindings.lookup(operation.getExtent());
  riscv::LayoutAttr layout = operation.getInputLayout();
  if (lhs.kind != Binding::Kind::Slice || rhs.kind != Binding::Kind::Slice ||
      extent.kind != Binding::Kind::Scalar ||
      product(layout.getReplicaFactors()) != 1 ||
      layout.getAxisIds().size() != 1)
    return fail(operation,
                "RVV stream dot requires two memory-backed scalar-replica slices");
  const Binding &lhsBase = bindings.lookup(lhs.slice.base);
  const Binding &rhsBase = bindings.lookup(rhs.slice.base);
  if (lhsBase.kind != Binding::Kind::Memory ||
      rhsBase.kind != Binding::Kind::Memory || !lhsBase.memory.elementType ||
      !rhsBase.memory.elementType || !lhsBase.memory.elementType.isF32() ||
      !rhsBase.memory.elementType.isF32())
    return fail(operation, "RVV stream dot inputs are not dense f32 slices");
  const int64_t laneAxis = layout.getAxisIds()[0];
  auto lhsLane = llvm::find(lhsBase.memory.axes, laneAxis);
  auto rhsLane = llvm::find(rhsBase.memory.axes, laneAxis);
  const size_t lhsDimension =
      static_cast<size_t>(lhsLane - lhsBase.memory.axes.begin());
  const size_t rhsDimension =
      static_cast<size_t>(rhsLane - rhsBase.memory.axes.begin());
  if (lhsLane == lhsBase.memory.axes.end() ||
      rhsLane == rhsBase.memory.axes.end() ||
      lhsDimension >= lhsBase.memory.strides.size() ||
      rhsDimension >= rhsBase.memory.strides.size())
    return fail(operation, "RVV stream dot has no dense lane stride");

  const std::string suffix = "f32" + lmulSpelling(layout.getLmulEighths());
  const std::string type =
      "vfloat32" + lmulSpelling(layout.getLmulEighths()) + "_t";
  const std::string fullVL = std::to_string(layout.getVl());

  std::string accumulator = fresh("stream_dot");
  line(type + " " + accumulator + " = __riscv_vfmv_v_f_" + suffix +
       "(0.0f, " + fullVL + ");");
  std::string streamIndex = fresh("stream_index");
  line("#pragma GCC unroll 1");
  line("for (size_t " + streamIndex + " = 0; " + streamIndex +
       " < (size_t)(" + extent.scalar + ");) {");
  ++indent;
  std::string streamVL = fresh("stream_vl");
  line("size_t " + streamVL + " = __riscv_vsetvl_e32" +
       lmulSpelling(layout.getLmulEighths()) + "((size_t)(" + extent.scalar +
       ") - " + streamIndex + ");");
  llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
  offsets.push_back({laneAxis, streamIndex});
  auto lhsAddress = denseAddress(lhs.slice, offsets);
  auto rhsAddress = denseAddress(rhs.slice, offsets);
  if (!lhsAddress || !rhsAddress)
    return fail(operation, "RVV stream dot has no dense address relation");
  std::string lhsValue = fresh("stream_lhs");
  std::string rhsValue = fresh("stream_rhs");
  if (operation.getLhsAccess().getForm() == "unit")
    line(type + " " + lhsValue + " = __riscv_vle32_v_" + suffix + "(" +
         *lhsAddress + ", " + streamVL + ");");
  else
    line(type + " " + lhsValue + " = __riscv_vlse32_v_" + suffix + "(" +
         *lhsAddress + ", (ptrdiff_t)(" +
         lhsBase.memory.strides[lhsDimension] +
         " * (ptrdiff_t)sizeof(float)), " + streamVL + ");");
  if (operation.getRhsAccess().getForm() == "unit")
    line(type + " " + rhsValue + " = __riscv_vle32_v_" + suffix + "(" +
         *rhsAddress + ", " + streamVL + ");");
  else
    line(type + " " + rhsValue + " = __riscv_vlse32_v_" + suffix + "(" +
         *rhsAddress + ", (ptrdiff_t)(" +
         rhsBase.memory.strides[rhsDimension] +
         " * (ptrdiff_t)sizeof(float)), " + streamVL + ");");
  line(accumulator + " = __riscv_vfmacc_vv_" + suffix + "(" + accumulator +
       ", " + lhsValue + ", " + rhsValue + ", " + streamVL + ");");
  line(streamIndex + " += " + streamVL + ";");
  --indent;
  line("}");

  std::string seed = fresh("stream_seed");
  line("vfloat32m1_t " + seed +
       " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
  std::string reduced = fresh("stream_scalar");
  line("vfloat32m1_t " + reduced + " = __riscv_vfredusum_vs_" + suffix +
       "_f32m1(" + accumulator + ", " + seed + ", " + fullVL + ");");
  std::string scalarName = fresh("stream_value");
  line("float " + scalarName + " = __riscv_vfmv_f_s_f32m1_f32(" + reduced +
       ");");
  bindings[operation.getResult()] = scalar(scalarName);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVStreamContract(
    riscv::RVVStreamContractOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool widening = instruction == "rvv.stream-widen-contract";
  if (instruction != "rvv.stream-contract" && !widening)
    return fail(operation, "RVV stream contract has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding extent = bindings.lookup(operation.getExtent());
  if (lhs.kind != Binding::Kind::Slice || rhs.kind != Binding::Kind::Slice ||
      extent.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV stream contract requires two slices and one reduction extent");
  const Binding &lhsBase = bindings.lookup(lhs.slice.base);
  const Binding &rhsBase = bindings.lookup(rhs.slice.base);
  if (lhsBase.kind != Binding::Kind::Memory ||
      rhsBase.kind != Binding::Kind::Memory || !lhsBase.memory.elementType ||
      !rhsBase.memory.elementType ||
      (widening ? (!lhsBase.memory.elementType.isF16() ||
                   !rhsBase.memory.elementType.isF16())
                : (!lhsBase.memory.elementType.isF32() ||
                   !rhsBase.memory.elementType.isF32())))
    return fail(operation,
                "RVV stream contract inputs do not match its selected floating leaf");

  riscv::LayoutAttr operandLayout = operation.getOperandLayout();
  riscv::LayoutAttr accumulatorLayout = operation.getAccumulatorLayout();
  const int64_t reductionAxis = operation.getReductionAxis();
  auto lhsReduction = llvm::find(lhsBase.memory.axes, reductionAxis);
  auto rhsReduction = llvm::find(rhsBase.memory.axes, reductionAxis);
  if (lhsReduction == lhsBase.memory.axes.end() ||
      rhsReduction == rhsBase.memory.axes.end())
    return fail(operation,
                "RVV stream contract memory descriptors lost the reduction axis");
  const size_t lhsReductionDimension = static_cast<size_t>(
      lhsReduction - lhsBase.memory.axes.begin());
  const size_t rhsReductionDimension = static_cast<size_t>(
      rhsReduction - rhsBase.memory.axes.begin());
  const std::string operandLMUL =
      lmulSpelling(operandLayout.getLmulEighths());
  const std::string accumulatorLMUL =
      lmulSpelling(accumulatorLayout.getLmulEighths());
  if (operandLMUL.empty() || accumulatorLMUL.empty())
    return fail(operation, "RVV stream contract has an invalid LMUL");
  const std::string operandSuffix =
      std::string(widening ? "f16" : "f32") + operandLMUL;
  const std::string accumulatorSuffix = "f32" + accumulatorLMUL;
  const std::string operandType =
      std::string(widening ? "vfloat16" : "vfloat32") + operandLMUL + "_t";
  const std::string accumulatorType = "vfloat32" + accumulatorLMUL + "_t";
  const std::string scalarType = widening ? "_Float16" : "float";
  const unsigned operandBits = widening ? 16 : 32;
  const std::string fullVL = std::to_string(accumulatorLayout.getVl());
  const int64_t parts = registerPartCount(operation.getResult());
  if (parts <= 0)
    return fail(operation,
                "RVV stream contract result has no register-replica coordinates");

  llvm::SmallVector<std::string> accumulators;
  for (int64_t part = 0; part < parts; ++part) {
    std::string accumulator = fresh("stream_contract");
    line(accumulatorType + " " + accumulator + " = __riscv_vfmv_v_f_" +
         accumulatorSuffix +
         "(0.0f, " + fullVL + ");");
    accumulators.push_back(std::move(accumulator));
  }

  llvm::SmallVector<int64_t, 4> resultAxes =
      registerAxesFor(operation.getResult());
  auto emitReductionLoop = [&](bool guarded,
                               bool exactStrips) -> mlir::LogicalResult {
    std::string streamIndex = fresh("stream_index");
    const int64_t exactStep =
        operation.getUnroll() * accumulatorLayout.getVl();
    line("#pragma GCC unroll 1");
    line("for (size_t " + streamIndex + " = 0; " + streamIndex +
         " < (size_t)(" + extent.scalar + "); " + streamIndex + " += " +
         (exactStrips ? std::to_string(exactStep) : "0") + ") {");
    ++indent;
    std::string streamVL = fullVL;
    if (!exactStrips) {
      streamVL = fresh("stream_vl");
      line("size_t " + streamVL + " = __riscv_vsetvl_e" +
           std::to_string(operandBits) + operandLMUL +
           "((size_t)(" + extent.scalar + ") - " + streamIndex + ");");
    }
    const int64_t stripCount = exactStrips ? operation.getUnroll() : 1;
    for (int64_t unrolled = 0; unrolled < stripCount; ++unrolled) {
      const std::string stripIndex =
          unrolled == 0
              ? streamIndex
              : "(" + streamIndex + " + " +
                    std::to_string(unrolled * accumulatorLayout.getVl()) + ")";
      llvm::StringMap<std::string> lhsLoads;
      llvm::StringMap<std::string> rhsLoads;
      auto loadOperand = [&](const Binding &slice, const Binding &base,
                           riscv::AccessAttr access, size_t reductionDimension,
                           llvm::ArrayRef<int64_t> coordinates,
                           llvm::StringRef prefix,
                           llvm::StringMap<std::string> &cache)
        -> std::optional<std::string> {
      llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
      offsets.push_back({reductionAxis, stripIndex});
      std::string key;
      std::string active = "1";
      for (auto [axis, coordinate] : llvm::zip(resultAxes, coordinates)) {
        if (!llvm::is_contained(base.memory.axes, axis))
          continue;
        offsets.push_back({axis, std::to_string(coordinate)});
        key += ":" + std::to_string(axis) + "=" +
               std::to_string(coordinate);
        auto scope = axisScopes.find(axis);
        if (scope != axisScopes.end() && !scope->second.empty())
          active += " && " + std::to_string(coordinate) + " < " +
                    scope->second.back().active;
      }
      if (auto found = cache.find(key); found != cache.end())
        return found->second;
      auto address = denseAddress(slice.slice, offsets);
      if (!address)
        return std::nullopt;
      std::string loaded = fresh(prefix);
      std::string expression;
      if (access.getForm() == "unit")
        expression = "__riscv_vle" + std::to_string(operandBits) + "_v_" +
                     operandSuffix + "(" + *address + ", " + streamVL + ")";
      else
        expression = "__riscv_vlse" + std::to_string(operandBits) + "_v_" +
                     operandSuffix + "(" + *address +
                     ", (ptrdiff_t)(" + base.memory.strides[reductionDimension] +
                     " * (ptrdiff_t)sizeof(" + scalarType + ")), " + streamVL +
                     ")";
      if (guarded) {
        line(operandType + " " + loaded + " = __riscv_vfmv_v_f_" +
             operandSuffix + "(0.0f, " + streamVL + ");");
        line("if (" + active + ") " + loaded + " = " + expression + ";");
      } else {
        line(operandType + " " + loaded + " = " + expression + ";");
      }
      cache[key] = loaded;
      return loaded;
    };

    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> partCoordinates;
    llvm::SmallVector<int64_t> partOrder;
    for (int64_t part = 0; part < parts; ++part) {
      auto coordinates = registerCoordinates(operation.getResult(), part);
      if (!coordinates || coordinates->size() != resultAxes.size())
        return fail(operation,
                    "RVV stream contract has no complete result coordinate mapping");
      partCoordinates.push_back(*coordinates);
      partOrder.push_back(part);
    }
    const bool stationaryLhs = operation.getStationaryOperand() == "lhs";
    const Binding &stationarySlice = stationaryLhs ? lhs : rhs;
    const Binding &stationaryBase = stationaryLhs ? lhsBase : rhsBase;
    riscv::AccessAttr stationaryAccess =
        stationaryLhs ? operation.getLhsAccess() : operation.getRhsAccess();
    const size_t stationaryReductionDimension =
        stationaryLhs ? lhsReductionDimension : rhsReductionDimension;
    llvm::StringMap<std::string> &stationaryCache =
        stationaryLhs ? lhsLoads : rhsLoads;
    for (int64_t part : partOrder)
      if (!loadOperand(stationarySlice, stationaryBase, stationaryAccess,
                       stationaryReductionDimension, partCoordinates[part],
                       stationaryLhs ? "stream_lhs" : "stream_rhs",
                       stationaryCache))
        return fail(operation,
                    "RVV stream contract cannot materialize its stationary operand window");
    auto nonStationaryKey = [&](int64_t part) {
      std::string key;
      const Binding &base = stationaryLhs ? rhsBase : lhsBase;
      for (auto [axis, coordinate] :
           llvm::zip(resultAxes, partCoordinates[part]))
        if (llvm::is_contained(base.memory.axes, axis))
          key += ":" + std::to_string(axis) + "=" +
                 std::to_string(coordinate);
      return key;
    };
    std::stable_sort(partOrder.begin(), partOrder.end(),
                     [&](int64_t lhsPart, int64_t rhsPart) {
                       return nonStationaryKey(lhsPart) <
                              nonStationaryKey(rhsPart);
                     });
    for (int64_t part : partOrder) {
      llvm::ArrayRef<int64_t> coordinates = partCoordinates[part];
      auto lhsValue = loadOperand(
          lhs, lhsBase, operation.getLhsAccess(), lhsReductionDimension,
          coordinates, "stream_lhs", lhsLoads);
      auto rhsValue = loadOperand(
          rhs, rhsBase, operation.getRhsAccess(), rhsReductionDimension,
          coordinates, "stream_rhs", rhsLoads);
      if (!lhsValue || !rhsValue)
        return fail(operation,
                    "RVV stream contract has no selected dense address relation");
      line(accumulators[part] + " = __riscv_" +
           std::string(widening ? "vfwmacc_vv_" : "vfmacc_vv_") +
           accumulatorSuffix + "(" +
           accumulators[part] + ", " + *lhsValue + ", " + *rhsValue + ", " +
           streamVL + ");");
    }
    }
    if (!exactStrips)
      line(streamIndex + " += " + streamVL + ";");
    --indent;
    line("}");
    return mlir::success();
  };

  std::string fullTile = "1";
  for (auto [axis, factor] : llvm::zip(
           operation.getResult().getType().getAxisIds().asArrayRef(),
           operation.getResult().getType().getLayout().getReplicaFactors()
               .asArrayRef())) {
    if (factor <= 1)
      continue;
    auto scope = axisScopes.find(axis);
    if (scope == axisScopes.end() || scope->second.empty()) {
      fullTile.clear();
      break;
    }
    fullTile += " && " + scope->second.back().active + " == " +
                std::to_string(factor);
  }
  if (!fullTile.empty())
    fullTile += " && ((size_t)(" + extent.scalar + ") % " +
                std::to_string(operation.getUnroll() *
                               accumulatorLayout.getVl()) +
                " == 0)";
  if (!fullTile.empty() && fullTile != "1") {
    line("if (" + fullTile + ") {");
    ++indent;
    if (mlir::failed(emitReductionLoop(false, true)))
      return mlir::failure();
    --indent;
    line("} else {");
    ++indent;
    if (mlir::failed(emitReductionLoop(true, false)))
      return mlir::failure();
    --indent;
    line("}");
  } else if (mlir::failed(emitReductionLoop(true, false))) {
    return mlir::failure();
  }

  Binding result;
  result.kind = parts == 1 ? Binding::Kind::Scalar
                           : Binding::Kind::ScalarTuple;
  for (int64_t part = 0; part < parts; ++part) {
    std::string seed = fresh("stream_seed");
    line("vfloat32m1_t " + seed +
         " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
    std::string reduced = fresh("stream_scalar");
    line("vfloat32m1_t " + reduced + " = __riscv_vfredusum_vs_" +
         accumulatorSuffix +
         "_f32m1(" + accumulators[part] + ", " + seed + ", " + fullVL +
         ");");
    std::string scalarName = fresh("stream_value");
    line("float " + scalarName + " = __riscv_vfmv_f_s_f32m1_f32(" +
         reduced + ");");
    if (result.kind == Binding::Kind::Scalar)
      result.scalar = std::move(scalarName);
    else
      result.parts.push_back(std::move(scalarName));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVSplat(riscv::RVVSplatOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.splat")
    return fail(operation, "RVV splat has no exact selected leaf");
  Binding scalarValue = bindings.lookup(operation.getScalar());
  if (scalarValue.kind != Binding::Kind::Scalar)
    return fail(operation, "RVV splat input has no scalar representation");
  mlir::FailureOr<Binding> result = makeVector(
      operation.getResult(), "splat", scalarValue, operation.getScalar());
  if (mlir::failed(result))
    return mlir::failure();
  bindings[operation.getResult()] = std::move(*result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileProjectReductionOperand(
    riscv::ProjectReductionOperandOp operation) {
  mlir::FailureOr<Binding> materialized = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding input = std::move(*materialized);
  Binding index = bindings.lookup(operation.getReductionIndex());
  if (input.kind != Binding::Kind::Vector || index.kind != Binding::Kind::Scalar) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    stream << "reduction operand projection requires an RVV value and scalar "
              "index; input type="
           << operation.getInput().getType() << ", input binding="
           << static_cast<unsigned>(input.kind) << ", index binding="
           << static_cast<unsigned>(index.kind);
    return fail(operation, stream.str());
  }
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  riscv::LayoutAttr inputLayout = inputType.getLayout();
  auto found = llvm::find(inputType.getAxisIds().asArrayRef(),
                          operation.getReductionAxis());
  if (found == inputType.getAxisIds().asArrayRef().end())
    return fail(operation, "reduction projection axis is absent from its input");
  const size_t axisIndex = static_cast<size_t>(
      found - inputType.getAxisIds().asArrayRef().begin());
  const int64_t lanes = inputLayout.getLaneFactors()[axisIndex];
  const int64_t streams = inputLayout.getTimeFactors()[axisIndex];
  if (lanes == 1 && streams > 1 &&
      resultType.getLayout().getCarrier() == "rvv") {
    if (laneAxisFor(operation.getInput()) != laneAxisFor(operation.getResult()) ||
        laneAxisFor(operation.getResult()) == operation.getReductionAxis())
      return fail(operation,
                  "time-axis projection does not preserve its independent SIMD axis");
    const int64_t inputStreams = streamPartCount(operation.getInput());
    const int64_t resultStreams = streamPartCount(operation.getResult());
    if (inputStreams <= 0 || resultStreams <= 0)
      return fail(operation,
                  "time-axis projection has incomplete stream factors");
    auto resultLayout = resultType.getLayout();
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      const int64_t resultRegister = part / resultStreams;
      int64_t resultStream = part % resultStreams;
      auto sourceRegister = projectRegisterPart(
          operation.getInput(), operation.getResult(), resultRegister);
      if (!sourceRegister)
        return fail(operation,
                    "time-axis projection cannot preserve register coordinates");
      llvm::DenseMap<int64_t, int64_t> resultTimeCoordinates;
      for (int64_t dimension =
               static_cast<int64_t>(resultType.getAxisIds().size()) - 1;
           dimension >= 0; --dimension) {
        int64_t factor = resultLayout.getTimeFactors()[dimension];
        if (factor <= 0)
          return fail(operation,
                      "time-axis projection has an invalid result factor");
        resultTimeCoordinates[resultType.getAxisIds()[dimension]] =
            resultStream % factor;
        resultStream /= factor;
      }
      if (resultStream != 0)
        return fail(operation,
                    "time-axis projection result stream is out of range");
      std::string expression;
      for (int64_t reduced = streams - 1; reduced >= 0; --reduced) {
        int64_t sourceStream = 0;
        for (auto [axis, factor] :
             llvm::zip(inputLayout.getAxisIds().asArrayRef(),
                       inputLayout.getTimeFactors().asArrayRef())) {
          if (factor <= 0)
            return fail(operation,
                        "time-axis projection has an invalid input factor");
          int64_t coordinate = reduced;
          if (axis != operation.getReductionAxis()) {
            auto coordinateIt = resultTimeCoordinates.find(axis);
            coordinate = coordinateIt == resultTimeCoordinates.end()
                             ? 0
                             : coordinateIt->second;
          }
          if (coordinate < 0 || coordinate >= factor)
            return fail(operation,
                        "time-axis projection coordinate is out of range");
          sourceStream = sourceStream * factor + coordinate;
        }
        size_t sourcePart = *sourceRegister * inputStreams + sourceStream;
        if (sourcePart >= input.parts.size())
          return fail(operation,
                      "time-axis projection source part exceeds its typed layout");
        expression = expression.empty()
                         ? input.parts[sourcePart]
                         : "(" + index.scalar + " == " +
                               std::to_string(reduced) + " ? " +
                               input.parts[sourcePart] + " : " + expression +
                               ")";
      }
      std::string name = fresh("time_projection");
      line(vectorType(operation.getResult()) + " " + name + " = " +
           expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (lanes <= 1 || streams <= 0 || laneAxisFor(operation.getInput()) !=
                                      operation.getReductionAxis())
    return fail(operation,
                "current reduction projection requires the eliminated axis in RVV lanes");
  if (resultType.getLayout().getCarrier() != "scalar")
    return fail(operation,
                "current reduction projection must produce a register scalar tuple");

  mlir::Type element = resultType.getElementType();
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "reduction projection result has no scalar C type");
  const std::string suffix = vectorSuffix(operation.getInput());
  const std::string streamIndex = "((" + index.scalar + ") / " +
                                  std::to_string(lanes) + ")";
  const std::string laneIndex = "((" + index.scalar + ") % " +
                                std::to_string(lanes) + ")";
  auto extract = [&](size_t sourcePart) {
    const std::string shifted =
        "__riscv_vslidedown_vx_" + suffix + "(" + input.parts[sourcePart] +
        ", " + laneIndex + ", " +
        partVL(operation.getInput(), sourcePart) + ")";
    if (mlir::isa<mlir::FloatType>(element))
      return "__riscv_vfmv_f_s_" + suffix + "_f" +
             std::to_string(riscv_internal::logicalBitWidth(element)) + "(" +
             shifted + ")";
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!integer)
      return std::string();
    return "__riscv_vmv_x_s_" + suffix + "_" +
           std::string(integer.isUnsigned() ? "u" : "i") +
           std::to_string(integer.getWidth()) + "(" + shifted + ")";
  };

  Binding result;
  const int64_t resultParts = registerPartCount(operation.getResult());
  result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                 : Binding::Kind::ScalarTuple;
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    auto sourceRegister = projectRegisterPart(
        operation.getInput(), operation.getResult(), resultPart);
    if (!sourceRegister)
      return fail(operation,
                  "reduction projection cannot preserve the free-axis register mapping");
    std::string expression;
    for (int64_t stream = streams - 1; stream >= 0; --stream) {
      size_t sourcePart = *sourceRegister * streamPartCount(operation.getInput()) +
                          static_cast<size_t>(stream);
      if (sourcePart >= input.parts.size())
        return fail(operation,
                    "reduction projection source part exceeds its typed layout");
      std::string value = extract(sourcePart);
      if (value.empty())
        return fail(operation,
                    "reduction projection has no intrinsic scalar extraction");
      expression = expression.empty()
                       ? value
                       : "(" + streamIndex + " == " + std::to_string(stream) +
                             " ? " + value + " : " + expression + ")";
    }
    std::string name = fresh("reduction_operand");
    line(*scalarType + " " + name + " = " + expression + ";");
    if (result.kind == Binding::Kind::Scalar)
      result.scalar = std::move(name);
    else
      result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVContractStep(
    riscv::RVVContractStepOp operation) {
  llvm::StringRef instruction = operation.getLeaf().getInstruction();
  const bool floating = instruction == "rvv.vfmacc.vf";
  const bool wideningFloat = instruction == "rvv.vfwmacc.vf";
  const bool integer = instruction == "rvv.vmacc.vx";
  if (!floating && !wideningFloat && !integer)
    return fail(operation,
                "RVV contract step has no exact selected FMA operand form");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  Binding reductionIndex = bindings.lookup(operation.getReductionIndex());
  if (accumulator.kind != Binding::Kind::Vector ||
      reductionIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV contract step requires a vector accumulator and scalar index");
  const bool rhsLane = operation.getLaneOperand() == "rhs";
  const Binding &laneSlice = rhsLane ? rhs : lhs;
  const Binding &repeated = rhsLane ? lhs : rhs;
  mlir::Value repeatedValue = rhsLane ? operation.getLhs() : operation.getRhs();
  if (laneSlice.kind == Binding::Kind::Vector) {
    if (repeated.kind != Binding::Kind::Scalar &&
        repeated.kind != Binding::Kind::ScalarTuple)
      return fail(operation,
                  "register contract step requires a scalar/tuple repeated operand");
    Binding result;
    result.kind = Binding::Kind::Vector;
    mlir::Value resultValue = operation.getResult();
    llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(resultValue);
    const int64_t streams = streamPartCount(resultValue);
    const size_t laneOperandIndex = rhsLane ? 1 : 0;
    for (size_t part = 0; part < accumulator.parts.size(); ++part) {
      auto lanePart = mappedPart(operation.getOperation(), laneOperandIndex, part);
      if (!lanePart || *lanePart >= laneSlice.parts.size())
        return fail(operation,
                    "register contract lane value has no typed part mapping");
      std::string repeatedExpression;
      if (repeated.kind == Binding::Kind::Scalar) {
        repeatedExpression = repeated.scalar;
      } else {
        auto repeatedPart = projectRegisterPart(
            repeatedValue, resultValue, part / std::max<int64_t>(1, streams));
        if (!repeatedPart || *repeatedPart >= repeated.parts.size())
          return fail(operation,
                      "register contract repeated value has no free-axis projection");
        repeatedExpression = repeated.parts[*repeatedPart];
      }
      std::string name = fresh("contract_step");
      line(vectorType(resultValue) + " " + name + " = __riscv_" +
           std::string(wideningFloat ? "vfwmacc_vf_"
                                     : (floating ? "vfmacc_vf_" :
                                                   "vmacc_vx_")) +
           vectorSuffix(resultValue) + "(" + accumulator.parts[part] + ", " +
           repeatedExpression + ", " + laneSlice.parts[*lanePart] + ", " +
           partVL(resultValue, part) + ");");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (laneSlice.kind != Binding::Kind::Slice)
    return fail(operation,
                "RVV contract step lane operand requires one physical memory slice");
  const Binding &laneMemory = bindings.lookup(laneSlice.slice.base);
  Binding repeatedMemoryStorage;
  const Binding *repeatedMemory = nullptr;
  if (repeated.kind == Binding::Kind::Slice) {
    repeatedMemoryStorage = bindings.lookup(repeated.slice.base);
    repeatedMemory = &repeatedMemoryStorage;
  }
  if (laneMemory.kind != Binding::Kind::Memory ||
      !laneMemory.memory.elementType ||
      (!laneMemory.memory.elementType.isF16() &&
       !laneMemory.memory.elementType.isF32()) ||
      (repeatedMemory &&
       (repeatedMemory->kind != Binding::Kind::Memory ||
        !repeatedMemory->memory.elementType ||
        (!repeatedMemory->memory.elementType.isF16() &&
         !repeatedMemory->memory.elementType.isF32()))) ||
      (!repeatedMemory && repeated.kind != Binding::Kind::Scalar &&
       repeated.kind != Binding::Kind::ScalarTuple))
    return fail(operation,
                "current RVV contract step requires one f16/f32 lane load and one memory/register repeated operand");
  mlir::Value resultValue = operation.getResult();
  const int64_t laneAxis = laneAxisFor(resultValue);
  const int64_t reductionAxis = operation.getReductionAxis();
  size_t laneDimension = llvm::find(laneMemory.memory.axes, laneAxis) -
                         laneMemory.memory.axes.begin();
  if (laneDimension >= laneMemory.memory.axes.size())
    return fail(operation, "RVV contract step lane operand has no lane axis");
  if (operation.getLaneMemoryForm() != "unit" &&
      operation.getLaneMemoryForm() != "strided")
    return fail(operation,
                "RVV contract step has no unit/strided lane memory form");

  Binding result;
  result.kind = Binding::Kind::Vector;
  llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(resultValue);
  const int64_t streams = streamPartCount(resultValue);
  const std::string suffix = vectorSuffix(resultValue);
  riscv::LayoutAttr laneLoadLayout = operation.getLaneLoadLayout();
  std::string laneLMUL = lmulSpelling(laneLoadLayout.getLmulEighths());
  if (laneLMUL.empty())
    return fail(operation, "RVV contract step has an invalid lane-load LMUL");
  unsigned laneBits = riscv_internal::logicalBitWidth(
      laneMemory.memory.elementType);
  const bool laneFloat =
      mlir::isa<mlir::FloatType>(laneMemory.memory.elementType);
  std::string laneSuffix =
      std::string(laneFloat ? "f" : "i") + std::to_string(laneBits) + laneLMUL;
  std::string laneType =
      std::string(laneFloat ? "vfloat" : "vint") +
      std::to_string(laneBits) + laneLMUL + "_t";
  llvm::StringMap<std::string> loadedLaneValues;
  for (size_t part = 0; part < accumulator.parts.size(); ++part) {
    auto coordinates = registerCoordinates(resultValue, part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "RVV contract step result has no register-coordinate mapping");
    llvm::SmallVector<std::pair<int64_t, std::string>> laneOffsets{
        {reductionAxis, reductionIndex.scalar}};
    llvm::SmallVector<std::pair<int64_t, std::string>> repeatedOffsets{
        {reductionAxis, reductionIndex.scalar}};
    std::string active = "1";
    std::string laneKey = std::to_string(part % streams);
    if (streams > 1)
      laneOffsets.push_back({laneAxis, partOffset(resultValue, part)});
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
      if (llvm::is_contained(laneMemory.memory.axes, axis)) {
        laneOffsets.push_back({axis, std::to_string(coordinate)});
        laneKey += ":" + std::to_string(axis) + "=" +
                   std::to_string(coordinate);
      }
      if (repeatedMemory &&
          llvm::is_contained(repeatedMemory->memory.axes, axis))
        repeatedOffsets.push_back({axis, std::to_string(coordinate)});
    }
    std::string loaded;
    auto existing = loadedLaneValues.find(laneKey);
    if (existing != loadedLaneValues.end()) {
      loaded = existing->second;
    } else {
      auto address = denseAddress(laneSlice.slice, laneOffsets);
      if (!address)
        return fail(operation,
                    "RVV contract step lane operand has no address relation");
      loaded = fresh("operand");
      if (operation.getLaneMemoryForm() == "unit")
        line(laneType + " " + loaded + " = __riscv_vle" +
             std::to_string(laneBits) + "_v_" + laneSuffix + "(" + *address +
             ", " + partVL(resultValue, part) +
             ");");
      else
        line(laneType + " " + loaded + " = __riscv_vlse" +
             std::to_string(laneBits) + "_v_" + laneSuffix + "(" + *address +
             ", " +
             laneMemory.memory.strides[laneDimension] +
             " * (ptrdiff_t)sizeof(" +
             *scalarCType(laneMemory.memory.elementType) + "), " +
             partVL(resultValue, part) +
             ");");
      if (laneMemory.memory.elementType.isF16() && !wideningFloat) {
        std::string widened = fresh("operand_wide");
        line(vectorType(resultValue) + " " + widened +
             " = __riscv_vfwcvt_f_f_v_" + suffix + "(" + loaded + ", " +
             partVL(resultValue, part) + ");");
        loaded = std::move(widened);
      }
      loadedLaneValues[laneKey] = loaded;
    }
    std::string name = fresh("contract_step");
    line(vectorType(resultValue) + " " + name + " = " +
         accumulator.parts[part] + ";");
    std::string repeatedExpression;
    if (repeatedMemory) {
      auto repeatedAddress = denseAddress(repeated.slice, repeatedOffsets);
      if (!repeatedAddress)
        return fail(operation,
                    "RVV contract step repeated operand has no address relation");
      repeatedExpression = "*(" + *repeatedAddress + ")";
      if (repeatedMemory->memory.elementType.isF16() && !wideningFloat)
        repeatedExpression = "((float)(" + repeatedExpression + "))";
    } else if (repeated.kind == Binding::Kind::Scalar) {
      repeatedExpression = repeated.scalar;
    } else {
      auto repeatedPart = projectRegisterPart(
          repeatedValue, resultValue, part / streams);
      if (!repeatedPart || *repeatedPart >= repeated.parts.size())
        return fail(operation,
                    "RVV contract step cannot project the repeated register operand");
      repeatedExpression = repeated.parts[*repeatedPart];
    }
    std::string statement =
        name + " = __riscv_" +
        std::string(wideningFloat ? "vfwmacc_vf_" : "vfmacc_vf_") +
        suffix + "(" + name + ", " +
        repeatedExpression + ", " + loaded + ", " +
        partVL(resultValue, part) + ");";
    if (!registerAxes.empty())
      line("if (" + active + ") " + statement);
    else
      line(statement);
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVEncodedContractStep(
    riscv::RVVEncodedContractStepOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.vmacc.decoded-u8-s8")
    return fail(operation,
                "encoded contract step has no exact selected decode-MAC leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  Binding reductionIndex = bindings.lookup(operation.getReductionIndex());
  if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field ||
      accumulator.kind != Binding::Kind::Vector ||
      reductionIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded contract step requires two field values, an RVV accumulator, and a scalar reduction index");

  const bool rhsLane = operation.getLaneOperand() == "rhs";
  mlir::Value laneValue = rhsLane ? operation.getRhs() : operation.getLhs();
  mlir::Value repeatedValue = rhsLane ? operation.getLhs() : operation.getRhs();
  Binding lane = rhsLane ? rhs : lhs;
  Binding repeated = rhsLane ? lhs : rhs;
  auto laneField = fieldFor(lane);
  auto repeatedField = fieldFor(repeated);
  auto ownerFor = [&](Binding field) -> mlir::FailureOr<Binding> {
    Binding owner = bindings.lookup(field.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      auto materialized = recordForSlice(field.field.owner);
      if (mlir::failed(materialized))
        return mlir::failure();
      owner = std::move(*materialized);
    }
    if (owner.kind != Binding::Kind::Record)
      return mlir::failure();
    return owner;
  };
  auto laneOwnerOr = ownerFor(lane);
  auto repeatedOwnerOr = ownerFor(repeated);
  if (!laneField || !repeatedField || mlir::failed(laneOwnerOr) ||
      mlir::failed(repeatedOwnerOr))
    return fail(operation,
                "encoded contract step fields have no physical record mapping");
  Binding laneOwner = std::move(*laneOwnerOr);
  Binding repeatedOwner = std::move(*repeatedOwnerOr);
  auto laneInteger = mlir::dyn_cast<mlir::IntegerType>(laneField->type);
  auto repeatedInteger = mlir::dyn_cast<mlir::IntegerType>(repeatedField->type);
  if (!laneInteger || !laneInteger.isUnsigned() || laneInteger.getWidth() > 8 ||
      !repeatedInteger || !repeatedInteger.isSigned() ||
      repeatedInteger.getWidth() != 8)
    return fail(operation,
                "encoded contract step currently requires an unsigned narrow lane field and signed i8 repeated field");

  mlir::Value resultValue = operation.getResult();
  const int64_t laneAxis = laneAxisFor(resultValue);
  const int64_t streams = streamPartCount(resultValue);
  llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(resultValue);
  std::string laneStride;
  for (const auto &[axis, stride] : laneOwner.recordByteStrides)
    if (axis == laneAxis) {
      laneStride = stride;
      break;
    }
  if (laneOwner.interleaveRows == 0 && laneStride.empty()) {
    std::string available;
    llvm::raw_string_ostream stream(available);
    for (const auto &[axis, stride] : laneOwner.recordByteStrides)
      stream << (available.empty() ? "" : ",") << axis << ":" << stride;
    stream.flush();
    return fail(operation,
                "canonical encoded lane field has no record-byte stride on lane axis " +
                    std::to_string(laneAxis) + "; available=" + available);
  }

  const std::string resultSuffix = vectorSuffix(resultValue);
  const std::string unsignedResultSuffix = "u" + resultSuffix.substr(1);
  const std::string laneLMUL =
      lmulSpelling(operation.getLaneLoadLayout().getLmulEighths());
  if (laneLMUL.empty())
    return fail(operation, "encoded contract step has no legal lane-load LMUL");
  const std::string laneSuffix = "u8" + laneLMUL;
  const std::string laneType = "vuint8" + laneLMUL + "_t";
  const std::string laneIndex = reductionIndex.scalar;
  llvm::StringMap<std::string> laneVectors;

  auto loadLane = [&](int64_t stream) -> std::optional<std::string> {
    const std::string key = std::to_string(stream);
    if (auto found = laneVectors.find(key); found != laneVectors.end())
      return found->second;
    std::string loaded;
    if (laneOwner.interleaveRows > 0) {
      Binding vector = emitInterleavedField(laneValue, lane, laneIndex);
      if (vector.kind != Binding::Kind::Vector || vector.parts.empty())
        return std::nullopt;
      size_t part = std::min<size_t>(stream, vector.parts.size() - 1);
      loaded = vector.parts[part];
    } else {
      auto fragment = singleStorageFragment(
          *laneField, laneIndex, static_cast<unsigned>(laneInteger.getWidth()));
      if (!fragment)
        return std::nullopt;
      std::string raw = fresh("encoded_lane_byte");
      const std::string streamOffset =
          stream == 0
              ? "0"
              : "(" + std::to_string(stream * physicalLanes(resultValue)) +
                    " * (" + laneStride + "))";
      line(laneType + " " + raw + " = __riscv_vlse8_v_" + laneSuffix +
           "((const uint8_t *)(" + laneOwner.recordPointer + " + (" +
           fragment->byte + ") + " + streamOffset + "), (ptrdiff_t)(" +
           laneStride + "), " +
           partVL(resultValue, stream) + ");");
      loaded = raw;
      if (fragment->shift != "0") {
        std::string shifted = fresh("encoded_lane_shift");
        line(laneType + " " + shifted + " = __riscv_vsrl_vx_" +
             laneSuffix + "(" + loaded + ", " + fragment->shift + ", " +
             partVL(resultValue, stream) + ");");
        loaded = std::move(shifted);
      }
      if (laneInteger.getWidth() < 8) {
        std::string masked = fresh("encoded_lane_value");
        line(laneType + " " + masked + " = __riscv_vand_vx_" + laneSuffix +
             "(" + loaded + ", " +
             std::to_string((1u << laneInteger.getWidth()) - 1) + ", " +
             partVL(resultValue, stream) + ");");
        loaded = std::move(masked);
      }
    }
    laneVectors[key] = loaded;
    return loaded;
  };

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < accumulator.parts.size(); ++part) {
    const int64_t stream = static_cast<int64_t>(part % streams);
    auto laneRaw = loadLane(stream);
    auto coordinates = registerCoordinates(resultValue, part / streams);
    if (!laneRaw || !coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "encoded contract step cannot project its lane/register mapping");
    std::string record = repeatedOwner.recordPointer;
    std::string active = "1";
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
      for (const auto &[recordAxis, stride] : repeatedOwner.recordByteStrides)
        if (recordAxis == axis)
          record = "(" + record + " + " + std::to_string(coordinate) +
                   " * (" + stride + "))";
    }
    auto scalarFragment = singleStorageFragment(
        *repeatedField, laneIndex,
        static_cast<unsigned>(repeatedInteger.getWidth()));
    if (!scalarFragment)
      return fail(operation,
                  "encoded contract repeated field has no storage fragment");
    const std::string repeatedScalar =
        "((int32_t)(int8_t)((" + record + ")[" + scalarFragment->byte + "]))";
    const std::string vl = partVL(resultValue, part);
    std::string widened = fresh("encoded_lane_wide");
    line("vuint32" + lmulSpelling(
             mlir::cast<riscv::ValueType>(resultValue.getType())
                 .getLayout().getLmulEighths()) +
         "_t " + widened + " = __riscv_vzext_vf4_" +
         unsignedResultSuffix + "(" + *laneRaw + ", " + vl + ");");
    const std::string laneI32 = "__riscv_vreinterpret_v_" +
                                unsignedResultSuffix + "_" + resultSuffix +
                                "(" + widened + ")";
    std::string name = fresh("encoded_contract_step");
    line(vectorType(resultValue) + " " + name + " = " +
         accumulator.parts[part] + ";");
    line("if (" + active + ") " + name + " = __riscv_vmacc_vx_" +
         resultSuffix + "(" + name + ", " + repeatedScalar + ", " +
         laneI32 + ", " + vl + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCommit(riscv::StoreOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding region = bindings.lookup(operation.getRegion());
  riscv::AccessAttr selectedEdge = operation.getAccess();
  if (region.kind == Binding::Kind::Field) {
    auto field = fieldFor(region);
    if (!field || field->bitOffset % 8 ||
        field->access.getMapping() != "natural")
      return fail(operation,
                  "encoded field commit currently requires a byte-aligned natural field");
    Binding record = bindings.lookup(region.field.owner);
    if (record.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> materialized = recordForSlice(region.field.owner);
      if (mlir::failed(materialized))
        return mlir::failure();
      record = std::move(*materialized);
    }
    if (record.kind != Binding::Kind::Record)
      return fail(operation,
                  "encoded field commit owner has no selected record address");
    const unsigned width = riscv_internal::logicalBitWidth(field->type);
    std::string logicalIndex = "0";
    if (region.field.index) {
      const Binding &point = bindings.lookup(*region.field.index);
      if (point.kind != Binding::Kind::Point || record.recordElements <= 0)
        return fail(operation,
                    "encoded field commit index has no record-domain relation");
      logicalIndex = "(" + point.point.base + " % " +
                     std::to_string(record.recordElements) + ")";
      if (region.field.selector == "group_index")
        logicalIndex = "(" + logicalIndex + " / " +
                       std::to_string(point.point.physicalExtent) + ")";
    }
    const std::string address = record.recordPointer + " + " +
                                std::to_string(field->bitOffset / 8) + " + (" +
                                logicalIndex + ") * " +
                                std::to_string(width / 8);
    if (value.kind == Binding::Kind::Scalar) {
      mlir::Type element =
          riscv_internal::logicalElement(operation.getValue().getType());
      if (element.isF16())
        line(field->access.getAlignment() >= 2
                 ? "*(_Float16 *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_f16_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isF32())
        line(field->access.getAlignment() >= 4
                 ? "*(float *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_f32_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isInteger(16))
        line(field->access.getAlignment() >= 2
                 ? "*(int16_t *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_i16_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isInteger(8))
        line("*(" + *scalarCType(element) + " *)(" + address + ") = " +
             value.scalar + ";");
      else
        return fail(operation,
                    "encoded scalar field has no intrinsic-C store spelling");
      return mlir::success();
    }
    if (value.kind != Binding::Kind::Vector ||
        (width != 8 && width != 16 && width != 32))
      return fail(operation,
                  "encoded array field commit requires selected RVV values");
    for (auto [part, expression] : llvm::enumerate(value.parts)) {
      const std::string partAddress =
          address + " + " + partOffset(operation.getValue(), part) + " * " +
          std::to_string(width / 8);
      auto elementType = scalarCType(field->type);
      if (!elementType)
        return fail(operation,
                    "encoded array field element has no intrinsic-C type");
      line("__riscv_vse" + std::to_string(width) + "_v_" +
           vectorSuffix(operation.getValue()) + "((" + *elementType + " *)(" +
           partAddress + "), " +
           expression + ", " + partVL(operation.getValue(), part) + ");");
    }
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(operation, "commit requires an explicit View slice");
  const Binding &memory = bindings.lookup(region.slice.base);
  if (memory.kind != Binding::Kind::Memory)
    return fail(operation, "commit slice is not memory-backed");
  mlir::Type element =
      riscv_internal::logicalElement(operation.getValue().getType());
  if (!memory.memory.elementType || memory.memory.elementType != element)
    return fail(operation, "commit value and dense memory element types disagree");
  if (value.kind == Binding::Kind::Scalar) {
    auto address = denseAddress(region.slice, {});
    auto type = scalarCType(element);
    if (!address || !type)
      return fail(operation, "scalar commit has no dense address or C type");
    line("*(" + *type + " *)(" + *address + ") = " + value.scalar + ";");
    return mlir::success();
  }
  if (value.kind == Binding::Kind::ScalarTuple) {
    auto type = scalarCType(element);
    llvm::SmallVector<int64_t, 4> axes =
        registerAxesFor(operation.getValue());
    if (!type || value.parts.size() !=
                     static_cast<size_t>(registerPartCount(operation.getValue())))
      return fail(operation,
                  "dense scalar tuple commit has no complete physical mapping");
    for (size_t part = 0; part < value.parts.size(); ++part) {
      auto coordinates = registerCoordinates(operation.getValue(), part);
      if (!coordinates || coordinates->size() != axes.size())
        return fail(operation,
                    "dense scalar tuple commit has no coordinate projection");
      llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
      std::string active = "1";
      for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
        offsets.push_back({axis, std::to_string(coordinate)});
        auto scope = axisScopes.find(axis);
        if (scope != axisScopes.end() && !scope->second.empty())
          active += " && " + std::to_string(coordinate) + " < " +
                    scope->second.back().active;
      }
      auto address = denseAddress(region.slice, offsets);
      if (!address)
        return fail(operation,
                    "dense scalar tuple commit has no address relation");
      line("if (" + active + ") *(" + *type + " *)(" + *address + ") = " +
           value.parts[part] + ";");
    }
    return mlir::success();
  }
  if (value.kind != Binding::Kind::Vector)
    return fail(operation, "dense commit has no selected numeric handoff");
  size_t laneDimension =
      llvm::find(memory.memory.axes, laneAxisFor(operation.getValue())) -
      memory.memory.axes.begin();
  if (laneDimension >= memory.memory.axes.size())
    return fail(operation, "commit destination has no selected lane axis");
  const int64_t laneAxis = laneAxisFor(operation.getValue());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getValue());
  const int64_t streams = streamPartCount(operation.getValue());
  llvm::StringRef memoryForm =
      selectedEdge ? selectedEdge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided")
    return fail(operation,
                "dense commit has no pass-selected unit or strided form");
  for (size_t part = 0; part < value.parts.size(); ++part) {
    auto coordinates =
        registerCoordinates(operation.getValue(), part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "dense vector commit has no register-coordinate mapping");
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string active = "1";
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      offsets.push_back({axis, std::to_string(coordinate)});
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
    }
    if (streams > 1)
      offsets.push_back({laneAxis, partOffset(operation.getValue(), part)});
    auto address = denseAddress(region.slice, offsets);
    if (!address)
      return fail(operation, "commit destination has no address relation");
    const std::string suffix = vectorSuffix(operation.getValue());
    const unsigned width =
        riscv_internal::logicalBitWidth(operation.getValue().getType());
    auto elementType = scalarCType(element);
    if (!width || !elementType)
      return fail(operation, "vector commit element has no fixed C representation");
    std::string statement;
    if (memoryForm == "unit")
      statement = "__riscv_vse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  value.parts[part] + ", " +
                  partVL(operation.getValue(), part) + ");";
    else
      statement = "__riscv_vsse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  memory.memory.strides[laneDimension] +
                  " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                  value.parts[part] + ", " +
                  partVL(operation.getValue(), part) + ");";
    if (!registerAxes.empty())
      line("if (" + active + ") " + statement);
    else
      line(statement);
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult weft::emitSelectedRISCVIntrinsicC(mlir::ModuleOp module,
                                                      std::string &result) {
  std::string body;
  llvm::raw_string_ostream output(body);
  output << "#include <stddef.h>\n"
         << "#include <stdint.h>\n"
         << "#include <math.h>\n"
         << "#include <string.h>\n"
         << "#include <riscv_vector.h>\n\n"
         << "static inline __attribute__((unused)) int16_t weft_load_i16_le(const uint8_t *p) {\n"
         << "  return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint16_t weft_load_u16_le(const uint8_t *p) {\n"
         << "  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint32_t weft_load_u32_le(const uint8_t *p) {\n"
         << "  uint32_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) _Float16 weft_load_f16_le(const uint8_t *p) {\n"
         << "  _Float16 value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_load_f32_le(const uint8_t *p) {\n"
         << "  float value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_i16_le(uint8_t *p, int16_t value) {\n"
         << "  uint16_t bits = (uint16_t)value; p[0] = (uint8_t)bits; p[1] = (uint8_t)(bits >> 8);\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f16_le(uint8_t *p, _Float16 value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f32_le(uint8_t *p, float value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "\n";
  bool found = false;
  for (riscv::ArtifactPackOp artifact :
       module.getOps<riscv::ArtifactPackOp>()) {
    Emitter emitter(module, artifact, output);
    if (mlir::failed(emitter.emitArtifact()))
      return mlir::failure();
  }
  for (riscv::KernelOp kernel : module.getOps<riscv::KernelOp>()) {
    Emitter emitter(module, kernel, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
    found = true;
  }
  if (!found)
    return module.emitError("intrinsic-C emission requires a verified RISC-V physical kernel");
  output.flush();
  result = std::move(body);
  return mlir::success();
}
