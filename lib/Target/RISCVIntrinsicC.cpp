#include "RISCVIntrinsicC.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

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
  mlir::ArrayAttr layouts;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
};

struct EncodingInfo {
  std::string family;
  std::string kind;
  std::string layoutIdentity;
  int64_t storageBits = 0;
  int64_t alignment = 1;
  int64_t logicalElements = 0;
  llvm::SmallVector<EncodingField> fields;
};

llvm::StringRef layoutKind(mlir::ArrayAttr layouts) {
  if (!layouts || layouts.empty())
    return {};
  auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(layouts[0]);
  auto kind = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                         : mlir::StringAttr();
  return kind ? kind.getValue() : llvm::StringRef();
}

struct StorageFragment {
  std::string byte;
  std::string shift;
  unsigned width = 0;
};

std::optional<StorageFragment>
singleStorageFragment(const EncodingField &field, llvm::StringRef logicalIndex,
                      unsigned logicalWidth) {
  llvm::StringRef kind = layoutKind(field.layouts);
  if (kind == "natural") {
    std::string bit = "(" + std::to_string(field.bitOffset) + " + (" +
                      logicalIndex.str() + ") * " +
                      std::to_string(logicalWidth) + ")";
    return StorageFragment{"(" + bit + " / 8)", "(" + bit + " % 8)",
                           logicalWidth};
  }
  if (kind != "grouped" || field.layouts.size() != 2)
    return std::nullopt;
  auto grouped = mlir::cast<mlir::DictionaryAttr>(field.layouts[0]);
  auto layered = mlir::cast<mlir::DictionaryAttr>(field.layouts[1]);
  int64_t group = grouped.getAs<mlir::IntegerAttr>("size").getInt();
  int64_t layer = layered.getAs<mlir::IntegerAttr>("size").getInt();
  llvm::StringRef order = layered.getAs<mlir::StringAttr>("order").getValue();
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
  bool packed = false;
  std::string sourceEncodingFamily;
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
  mlir::DictionaryAttr memoryEdge;
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
    DeferredMac,
    DeferredProduct,
    DeferredContract,
    DeferredWiden,
    DeferredFold,
    Point,
    Domain,
  } kind = Kind::None;

  std::string scalar;
  llvm::SmallVector<std::string> parts;
  MemoryInfo memory;
  SliceInfo slice;
  FieldInfo field;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value input;
  int64_t group = 0;
  mlir::Operation *sourceOperation = nullptr;
  PointInfo point;
  mlir::Value parentDomain;
  kernel::DomainType domainType;
  std::string recordPointer;
  std::string encodingFamily;
  int64_t interleaveRows = 0;
  llvm::SmallVector<std::pair<int64_t, std::string>> recordByteStrides;
};

class Emitter {
public:
  Emitter(mlir::ModuleOp module, kernel::KernelOp kernel,
          riscv::AssignmentOp assignment, llvm::raw_ostream &output)
      : module(module), kernel(kernel), assignment(assignment), output(output) {
    indexAssignment();
    collectEncodings();
    collectDerivedFamilies();
    collectWritableArguments();
    for (auto [index, symbol] : llvm::enumerate(kernel.getShapeSymbols()))
      axisSymbolIds[mlir::cast<mlir::StringAttr>(symbol).getValue()] = index + 1;
    auto bindings = assignment.getCandidate().getAs<mlir::DictionaryAttr>(
        "auto_bindings");
    if (bindings)
      for (mlir::NamedAttribute binding : bindings)
        autoBindings[binding.getName()] =
            mlir::cast<mlir::IntegerAttr>(binding.getValue()).getInt();
  }

  mlir::LogicalResult emit() {
    if (!kernel.getBody().hasOneBlock())
      return fail(kernel, "intrinsic-C emission requires one kernel entry block");
    if (mlir::failed(validateAssignment()))
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

  void emitBuilders() {
    mlir::Block &entry = kernel.getBody().front();
    auto names = kernel.getArgNames();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
      auto view = mlir::dyn_cast<kernel::ViewType>(argument.getType());
      if (!view)
        continue;
      auto encoding = mlir::dyn_cast<kernel::EncodingType>(view.getEncoding());
      if (!encoding || encoding.getKind() != "derived_instance")
        continue;
      auto source = derivedSources.find(encoding.getFamily());
      auto info = source == derivedSources.end()
                      ? encodings.end()
                      : encodings.find(source->second);
      mlir::DictionaryAttr physical = assigned(argument);
      int64_t rows =
          riscv_internal::integer(physical, "interleave_rows").value_or(0);
      if (info == encodings.end() || rows <= 0)
        continue;
      const std::string prefix = identifier(kernel.getSymName()) + "_" +
                                 identifier(mlir::cast<mlir::StringAttr>(names[index])
                                                .getValue());
      const int64_t recordBytes = info->second.storageBits / 8;
      const int64_t elements = info->second.logicalElements;
      output << "size_t " << prefix
             << "_packed_size(size_t M, size_t K) {\n"
             << "  if (K == 0 || K % " << elements << " != 0) return 0;\n"
             << "  return ((M + " << rows << " - 1) / "
             << rows << ") * (K / " << elements << ") * "
             << recordBytes * rows << ";\n"
             << "}\n\n";
      output << "void " << prefix
             << "_pack(const uint8_t *restrict source, uint8_t *restrict target, "
                "size_t M, size_t K) {\n"
             << "  const size_t blocks = K / " << elements << ";\n"
             << "  const size_t row_groups = (M + " << rows
             << " - 1) / " << rows << ";\n"
             << "  for (size_t rg = 0; rg < row_groups; ++rg)\n"
             << "    for (size_t kb = 0; kb < blocks; ++kb)\n"
             << "      for (size_t byte = 0; byte < " << recordBytes
             << "; ++byte)\n"
             << "        for (size_t lane = 0; lane < " << rows
             << "; ++lane) {\n"
             << "          const size_t row = rg * " << rows
             << " + lane;\n"
             << "          const size_t dst = ((rg * blocks + kb) * "
             << recordBytes << " + byte) * " << rows
             << " + lane;\n"
             << "          target[dst] = row < M ? source[(row * blocks + kb) * "
             << recordBytes << " + byte] : 0;\n"
             << "        }\n"
             << "}\n\n";
    }
  }

private:
  mlir::LogicalResult validateAssignment() {
    for (mlir::Attribute attribute : assignment.getValues()) {
      auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
      llvm::StringRef kind =
          riscv_internal::string(value, "physical_kind").value_or("");
      if (!kind.starts_with("rvv"))
        continue;
      auto lmul = value.getAs<mlir::StringAttr>("lmul");
      auto lmulEighths = value.getAs<mlir::IntegerAttr>("lmul_eighths");
      auto sew = value.getAs<mlir::IntegerAttr>("physical_sew");
      if (!lmul || !lmulEighths || !sew || lmulEighths.getInt() <= 0 ||
          sew.getInt() <= 0)
        return fail(assignment,
                    "intrinsic-C emission requires complete RVV SEW/LMUL assignments");
    }
    for (mlir::Attribute attribute : assignment.getOperations()) {
      auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
      auto realization = operation.getAs<mlir::StringAttr>("realization");
      if (!realization || realization.getValue().empty())
        return fail(assignment,
                    "intrinsic-C emission requires one selected realization per operation");
    }
    return mlir::success();
  }

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

  mlir::DictionaryAttr assigned(mlir::Value value) const {
    auto found = valueAssignments.find(value);
    return found == valueAssignments.end() ? mlir::DictionaryAttr()
                                           : found->second;
  }

  mlir::DictionaryAttr assigned(mlir::Operation *operation) const {
    auto found = operationAssignments.find(operation);
    return found == operationAssignments.end() ? mlir::DictionaryAttr()
                                               : found->second;
  }

  void indexAssignmentBlock(mlir::Block &block, unsigned &valueIndex,
                            unsigned &operationIndex,
                            llvm::ArrayRef<mlir::DictionaryAttr> values) {
    auto recordValue = [&](mlir::Value value) {
      mlir::DictionaryAttr assignment = values[valueIndex++];
      valueAssignments[value] = assignment;
      if (auto id = assignment.getAs<mlir::StringAttr>("id"))
        canonicalValuesById[id.getValue()] = value;
    };
    for (mlir::BlockArgument argument : block.getArguments())
      recordValue(argument);
    for (mlir::Operation &operation : block) {
      for (mlir::Value result : operation.getResults())
        recordValue(result);
      std::string id = "op" + std::to_string(operationIndex++);
      operationAssignments[&operation] = assignmentOperationsById.lookup(id);
      canonicalOperationsById[id] = &operation;
      for (mlir::Region &region : operation.getRegions())
        for (mlir::Block &nested : region)
          indexAssignmentBlock(nested, valueIndex, operationIndex, values);
    }
  }

  void indexAssignment() {
    llvm::SmallVector<mlir::DictionaryAttr> values;
    for (mlir::Attribute attribute : assignment.getValues())
      values.push_back(mlir::cast<mlir::DictionaryAttr>(attribute));
    for (mlir::Attribute attribute : assignment.getOperations()) {
      auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
      assignmentOperationsById[
          *riscv_internal::string(operation, "id")] = operation;
    }
    unsigned valueIndex = 0;
    unsigned operationIndex = 0;
    indexAssignmentBlock(kernel.getBody().front(), valueIndex, operationIndex,
                         values);
  }

  void collectEncodings() {
    for (kernel::EncodingDeclOp declaration :
         module.getOps<kernel::EncodingDeclOp>()) {
      EncodingInfo info;
      info.family = declaration.getSymName().str();
      info.kind = declaration.getKind().str();
      info.layoutIdentity = declaration.getLayoutIdentity().str();
      info.storageBits = declaration.getStorageBits();
      info.alignment = declaration.getAlignment();
      info.logicalElements = declaration.getElements();
      auto names = declaration.getFieldNames();
      auto types = declaration.getFieldTypes();
      auto shapes = declaration.getFieldShapes();
      auto layouts = declaration.getFieldLayouts();
      auto offsets = declaration.getFieldBitOffsets();
      auto storage = declaration.getFieldStorageBits();
      for (size_t index = 0; index < names.size(); ++index) {
        EncodingField field;
        field.name = mlir::cast<mlir::StringAttr>(names[index]).getValue().str();
        field.type = mlir::cast<mlir::TypeAttr>(types[index]).getValue();
        auto shape = mlir::cast<mlir::DenseI64ArrayAttr>(shapes[index]);
        field.shape.assign(shape.asArrayRef().begin(), shape.asArrayRef().end());
        field.layouts = mlir::cast<mlir::ArrayAttr>(layouts[index]);
        field.bitOffset = offsets[index];
        field.storageBits = storage[index];
        info.fields.push_back(std::move(field));
      }
      encodings[info.family] = std::move(info);
    }
  }

  void collectDerivedFamilies() {
    for (kernel::DeriveOp derive : module.getOps<kernel::DeriveOp>())
      derivedSources[derive.getResultFamily()] = derive.getSourceFamily().str();
  }

  mlir::Value rootView(mlir::Value value) const {
    while (true) {
      if (auto field = value.getDefiningOp<kernel::FieldOp>()) {
        value = field.getOwner();
        continue;
      }
      if (auto slice = value.getDefiningOp<kernel::SliceOp>()) {
        value = slice.getBase();
        continue;
      }
      return value;
    }
  }

  void collectWritableArguments() {
    kernel.walk([&](kernel::CommitOp commit) {
      mlir::Value root = rootView(commit.getRegion());
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root))
        if (argument.getOwner() == &kernel.getBody().front())
          writableArguments.insert(argument.getArgNumber());
    });
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

  std::optional<std::string> argumentCType(mlir::BlockArgument argument) const {
    auto view = mlir::cast<kernel::ViewType>(argument.getType());
    auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
    bool writable = writableArguments.contains(argument.getArgNumber());
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
    return (writable ? "" : "const ") + element + " *restrict";
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
      auto view = mlir::cast<kernel::ViewType>(argument.getType());
      MemoryInfo memory;
      memory.name = identifier(
          mlir::cast<mlir::StringAttr>(names[index]).getValue());
      memory.encoding = view.getEncoding();
      if (auto elementType = denseElementType(view.getEncoding()))
        memory.elementType = *elementType;
      memory.axes.assign(view.getAxisIds().asArrayRef().begin(),
                         view.getAxisIds().asArrayRef().end());
      for (auto [dimension, extent] : llvm::enumerate(view.getShape().asArrayRef())) {
        auto physicalExtent = extentForShape(extent, memory.axes[dimension]);
        if (!physicalExtent)
          return fail(kernel,
                      "dynamic view extent has no corresponding kernel shape symbol");
        memory.extents.push_back(std::move(*physicalExtent));
      }
      memory.strides.resize(memory.extents.size(), "1");
      for (int64_t dimension = memory.extents.size() - 2; dimension >= 0;
           --dimension)
        memory.strides[dimension] = "(" + memory.strides[dimension + 1] + " * " +
                                    memory.extents[dimension + 1] + ")";
      memory.origins.assign(memory.axes.size(), "0");
      memory.isConst = !writableArguments.contains(index);
      Binding binding;
      binding.kind = Binding::Kind::Memory;
      binding.memory = std::move(memory);
      bindings[argument] = std::move(binding);
    }
    return mlir::success();
  }

  // The remaining operation emitters are intentionally below the ABI and
  // representation plumbing: they consume canonical op semantics plus the
  // selected per-value/per-op assignment and never inspect kernel names.

  mlir::LogicalResult compileBlock(mlir::Block &block,
                                   llvm::ArrayRef<Binding> arguments) {
    conversionCaches.emplace_back();
    if (!arguments.empty()) {
      if (arguments.size() != block.getNumArguments())
        return fail(block.getParentOp(), "internal intrinsic-C block binding mismatch");
      for (auto [argument, binding] : llvm::zip(block.getArguments(), arguments))
        bindings[argument] = binding;
    }
    for (mlir::Operation &operation : block) {
      if (mlir::isa<kernel::ReturnOp, kernel::BirthsYieldOp,
                    kernel::HandoffOp, kernel::YieldOp,
                    kernel::ConditionOp>(operation))
        continue;
      if (mlir::failed(compileOperation(operation))) {
        conversionCaches.pop_back();
        return mlir::failure();
      }
    }
    conversionCaches.pop_back();
    return mlir::success();
  }

  mlir::LogicalResult compileOperation(mlir::Operation &operation);
  mlir::LogicalResult compileLevel(kernel::LevelOp level);
  mlir::LogicalResult compileFor(kernel::ForOp operation);
  mlir::LogicalResult compilePipelinedFor(
      kernel::ForOp operation, mlir::DictionaryAttr cluster,
      const Binding &lower, const Binding &upper, const Binding &step,
      llvm::SmallVectorImpl<Binding> &carried);
  mlir::LogicalResult compileIf(kernel::IfOp operation);
  mlir::LogicalResult compileSlice(kernel::SliceOp slice);
  mlir::LogicalResult compileAdmit(kernel::AdmitOp admit);
  mlir::LogicalResult compileMaterialize(kernel::MaterializeOp materialize);
  mlir::LogicalResult compileIota(kernel::IotaOp operation);
  mlir::LogicalResult compileNew(kernel::NewOp operation);
  mlir::LogicalResult compileField(kernel::FieldOp operation);
  mlir::LogicalResult compileExtract(kernel::ExtractOp operation);
  mlir::LogicalResult compileUnary(kernel::UnaryOp operation);
  mlir::LogicalResult compileCompare(kernel::CompareOp operation);
  mlir::LogicalResult compileCast(kernel::CastOp operation);
  mlir::LogicalResult compileWiden(kernel::WidenOp operation);
  mlir::LogicalResult compileReduce(kernel::ReduceOp operation);
  mlir::LogicalResult compileFold2(kernel::Fold2Op operation);
  mlir::LogicalResult compileMac(mlir::Operation &operation, mlir::Value lhs,
                                 mlir::Value rhs, int64_t group);
  mlir::LogicalResult compileDot(kernel::DotOp operation);
  mlir::LogicalResult compileLookup(kernel::LookupOp operation);
  mlir::LogicalResult compileContract(mlir::Operation &operation,
                                      mlir::Value lhs, mlir::Value rhs,
                                      mlir::Value result, bool outer,
                                      std::optional<Binding> initial = std::nullopt,
                                      mlir::Value initialValue = {});
  mlir::LogicalResult compileBinary(kernel::BinaryOp operation);
  mlir::LogicalResult compileCommit(kernel::CommitOp operation);

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
  int64_t physicalLanes(mlir::Value value) const;
  int64_t laneAxisFor(mlir::Value value) const;
  std::optional<Binding::Kind> selectedBindingKind(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerAxesFor(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerExtentsFor(mlir::Value value) const;
  std::optional<llvm::SmallVector<int64_t, 4>>
  registerCoordinates(mlir::Value value, size_t registerPart) const;
  std::optional<size_t> projectPart(mlir::Value source, mlir::Value result,
                                    size_t resultPart) const;
  mlir::DictionaryAttr useConversion(mlir::Operation *operation,
                                     size_t operand) const;
  std::optional<size_t> mappedPart(mlir::Operation *operation, size_t operand,
                                   size_t resultPart) const;
  std::optional<size_t> projectRegisterPart(mlir::Value source,
                                            mlir::Value result,
                                            size_t resultRegisterPart) const;
  std::optional<std::string>
  extractLaneForRegisterBroadcast(mlir::Operation *operation, size_t operand,
                                  mlir::Value source, mlir::Value result,
                                  size_t resultPart, const Binding &binding);
  mlir::FailureOr<Binding> projectBinding(mlir::Value source,
                                          mlir::Value result,
                                          const Binding &binding);
  mlir::LogicalResult assignBinding(mlir::Operation *operation,
                                    mlir::Value targetValue, Binding &target,
                                    mlir::Value sourceValue,
                                    const Binding &source,
                                    llvm::StringRef mismatch);
  std::string partOffset(mlir::Value value, size_t part) const;
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
  mlir::FailureOr<Binding> emitGroupedMacReduction(mlir::Value result,
                                                    const Binding &mac);
  mlir::FailureOr<Binding> emitQuantDot(kernel::DotOp operation,
                                        const Binding &lhs,
                                        const Binding &rhs);
  mlir::FailureOr<Binding>
  emitEncodedOuterMac(mlir::Operation &operation, mlir::Value lhs,
                      mlir::Value rhs, mlir::Value result,
                      const Binding &lhsBinding, const Binding &rhsBinding);
  mlir::FailureOr<Binding>
  emitEncodedOuterFold(mlir::Operation &operation, mlir::Value lhs,
                       mlir::Value rhs, mlir::Value result,
                       const Binding &lhsBinding, const Binding &rhsBinding);
  std::string laneVL() const {
    return laneVLStack.empty() ? "1" : laneVLStack.back();
  }

  mlir::ModuleOp module;
  kernel::KernelOp kernel;
  riscv::AssignmentOp assignment;
  llvm::raw_ostream &output;
  unsigned indent = 0;
  unsigned nextName = 0;
  llvm::DenseMap<mlir::Value, mlir::DictionaryAttr> valueAssignments;
  llvm::DenseMap<mlir::Operation *, mlir::DictionaryAttr> operationAssignments;
  llvm::StringMap<mlir::DictionaryAttr> assignmentOperationsById;
  llvm::StringMap<mlir::Operation *> canonicalOperationsById;
  llvm::StringMap<mlir::Value> canonicalValuesById;
  llvm::DenseMap<mlir::Value, Binding> bindings;
  llvm::StringMap<EncodingInfo> encodings;
  llvm::StringMap<std::string> derivedSources;
  llvm::StringMap<int64_t> autoBindings;
  llvm::StringMap<int64_t> axisSymbolIds;
  llvm::SmallSet<unsigned, 8> writableArguments;
  llvm::DenseMap<int64_t, llvm::SmallVector<PointInfo>> axisScopes;
  llvm::SmallVector<std::string> laneVLStack;
  llvm::SmallVector<llvm::StringMap<std::string>> conversionCaches;
};

std::string Emitter::vectorSuffix(mlir::Value value) const {
  mlir::DictionaryAttr physical = assigned(value);
  auto lmul = physical.getAs<mlir::StringAttr>("lmul");
  int64_t sew =
      physical.getAs<mlir::IntegerAttr>("physical_sew").getInt();
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  char category = 'i';
  if (mlir::isa<mlir::FloatType>(element))
    category = 'f';
  else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    category = integer.isUnsigned() ? 'u' : 'i';
  return std::string(1, category) + std::to_string(sew) +
         lmul.getValue().str();
}

std::string Emitter::vectorType(mlir::Value value) const {
  std::string suffix = vectorSuffix(value);
  char category = suffix.front();
  std::string prefix = category == 'f' ? "vfloat" : category == 'u' ? "vuint"
                                                                       : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t Emitter::streamPartCount(mlir::Value value) const {
  if (auto count = assigned(value).getAs<mlir::IntegerAttr>("stream_parts"))
    return std::max<int64_t>(1, count.getInt());
  return 1;
}

std::optional<Binding::Kind>
Emitter::selectedBindingKind(mlir::Value value) const {
  llvm::StringRef kind =
      riscv_internal::string(assigned(value), "physical_kind").value_or("");
  if (kind.starts_with("rvv"))
    return Binding::Kind::Vector;
  if (kind == "register-scalar-tuple")
    return registerPartCount(value) == 1 ? Binding::Kind::Scalar
                                         : Binding::Kind::ScalarTuple;
  if (kind == "scalar" || kind == "sequential")
    return Binding::Kind::Scalar;
  return std::nullopt;
}

int64_t Emitter::registerPartCount(mlir::Value value) const {
  if (auto count = assigned(value).getAs<mlir::IntegerAttr>("register_parts"))
    return std::max<int64_t>(1, count.getInt());
  return 1;
}

llvm::SmallVector<int64_t, 4>
Emitter::registerAxesFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto axes =
          assigned(value).getAs<mlir::DenseI64ArrayAttr>("register_axes"))
    result.append(axes.asArrayRef().begin(), axes.asArrayRef().end());
  else if (auto axis = assigned(value).getAs<mlir::IntegerAttr>("register_axis");
           axis && axis.getInt() > 0)
    result.push_back(axis.getInt());
  return result;
}

llvm::SmallVector<int64_t, 4>
Emitter::registerExtentsFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto extents =
          assigned(value).getAs<mlir::DenseI64ArrayAttr>("register_extents"))
    result.append(extents.asArrayRef().begin(), extents.asArrayRef().end());
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
  const int64_t resultStream = resultPart % resultStreams;
  int64_t resultRegister = resultPart / resultStreams;
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
  int64_t sourceRegister = 0;
  for (auto [axis, extent] : llvm::zip(sourceAxes, sourceExtents)) {
    auto found = llvm::find(resultAxes, axis);
    if (found == resultAxes.end() || extent <= 0)
      return std::nullopt;
    const size_t position = found - resultAxes.begin();
    if (resultCoordinates[position] >= extent)
      return std::nullopt;
    sourceRegister = sourceRegister * extent + resultCoordinates[position];
  }

  const int64_t sourceLane = laneAxisFor(source);
  const int64_t resultLane = laneAxisFor(result);
  int64_t sourceStream = 0;
  if (sourceLane > 0) {
    if (sourceLane != resultLane || resultStream >= sourceStreams)
      return std::nullopt;
    sourceStream = sourceStreams == 1 ? 0 : resultStream;
  }
  const int64_t projected = sourceRegister * sourceStreams + sourceStream;
  if (projected < 0 || projected >= vectorPartCount(source))
    return std::nullopt;
  return static_cast<size_t>(projected);
}

mlir::DictionaryAttr Emitter::useConversion(mlir::Operation *operation,
                                            size_t operand) const {
  auto conversions =
      assigned(operation).getAs<mlir::ArrayAttr>("use_conversions");
  if (!conversions)
    return {};
  for (mlir::Attribute attribute : conversions) {
    auto conversion = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    if (conversion &&
        riscv_internal::integer(conversion, "operand").value_or(-1) ==
            static_cast<int64_t>(operand))
      return conversion;
  }
  return {};
}

std::optional<size_t> Emitter::mappedPart(mlir::Operation *operation,
                                          size_t operand,
                                          size_t resultPart) const {
  mlir::DictionaryAttr conversion = useConversion(operation, operand);
  auto parts =
      conversion ? conversion.getAs<mlir::DenseI64ArrayAttr>("source_parts")
                 : mlir::DenseI64ArrayAttr();
  if (!parts || resultPart >= parts.size() || parts[resultPart] < 0)
    return std::nullopt;
  return static_cast<size_t>(parts[resultPart]);
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

std::optional<std::string> Emitter::extractLaneForRegisterBroadcast(
    mlir::Operation *operation, size_t operand, mlir::Value source,
    mlir::Value result, size_t resultPart, const Binding &binding) {
  (void)result;
  if (binding.kind != Binding::Kind::Vector)
    return std::nullopt;
  mlir::DictionaryAttr conversion = useConversion(operation, operand);
  if (riscv_internal::string(conversion, "relation").value_or("") !=
      "lane-to-register")
    return std::nullopt;
  auto sourceParts = conversion.getAs<mlir::DenseI64ArrayAttr>("source_parts");
  auto laneOffsets = conversion.getAs<mlir::DenseI64ArrayAttr>("lane_offsets");
  auto sourceId = conversion.getAs<mlir::StringAttr>("source");
  if (!sourceParts || !laneOffsets || !sourceId ||
      resultPart >= sourceParts.size() || resultPart >= laneOffsets.size())
    return std::nullopt;
  const int64_t sourcePart = sourceParts[resultPart];
  const int64_t laneOffset = laneOffsets[resultPart];
  if (sourcePart < 0 || laneOffset < 0 ||
      sourcePart >= static_cast<int64_t>(binding.parts.size()))
    return std::nullopt;

  const std::string cacheKey =
      sourceId.getValue().str() + ":part" + std::to_string(sourcePart) +
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
  const unsigned sew = riscv_internal::logicalBitWidth(source.getType());
  std::string expression;
  if (mlir::isa<mlir::FloatType>(element))
    expression = "__riscv_vfmv_f_s_" + suffix + "_f" +
                 std::to_string(sew) + "(" + shifted + ")";
  else {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!integer)
      return std::nullopt;
    expression = "__riscv_vmv_x_s_" + suffix + "_" +
                 std::string(integer.isUnsigned() ? "u" : "i") +
                 std::to_string(sew) + "(" + shifted + ")";
  }
  auto type = scalarCType(element);
  if (!type || conversionCaches.empty())
    return std::nullopt;
  std::string materialized = fresh("layout_convert");
  line(*type + " " + materialized + " = " + expression + ";");
  conversionCaches.back()[cacheKey] = materialized;
  return materialized;
}

mlir::FailureOr<Binding>
Emitter::projectBinding(mlir::Value source, mlir::Value result,
                        const Binding &binding) {
  if (binding.kind == Binding::Kind::Scalar)
    return binding;
  if (binding.kind != Binding::Kind::ScalarTuple &&
      binding.kind != Binding::Kind::Vector)
    return binding;
  const int64_t parts = binding.kind == Binding::Kind::Vector
                            ? vectorPartCount(result)
                            : registerPartCount(result);
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
  if (target.kind == Binding::Kind::Scalar &&
      source.kind == Binding::Kind::Scalar) {
    line(target.scalar + " = " + source.scalar + ";");
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
                            : registerPartCount(targetValue);
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
  if (auto lanes = assigned(value).getAs<mlir::IntegerAttr>("physical_lanes"))
    return std::max<int64_t>(1, lanes.getInt());
  return 1;
}

int64_t Emitter::laneAxisFor(mlir::Value value) const {
  return riscv_internal::integer(assigned(value), "lane_axis").value_or(0);
}

std::string Emitter::partOffset(mlir::Value value, size_t part) const {
  const int64_t streams = streamPartCount(value);
  const int64_t stream = static_cast<int64_t>(part % streams);
  return std::to_string(stream * physicalLanes(value));
}

std::string Emitter::partVL(mlir::Value value, size_t part) const {
  const int64_t streams = streamPartCount(value);
  const int64_t laneAxis = laneAxisFor(value);
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
  if (auto count = assigned(value).getAs<mlir::IntegerAttr>("vector_parts"))
    return std::max<int64_t>(1, count.getInt());
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
  if (auto fresh = mlir::dyn_cast_or_null<kernel::NewOp>(transfer);
      fresh && fresh.getInitialized())
    transfer = fresh.getInitial().getDefiningOp();
  mlir::DictionaryAttr selected = assigned(transfer);
  auto edge = selected
                  ? selected.getAs<mlir::DictionaryAttr>("memory_edge")
                  : mlir::DictionaryAttr();
  llvm::StringRef memoryForm =
      edge ? riscv_internal::string(edge, "form").value_or("")
           : llvm::StringRef();
  if (memoryForm != "unit-stride" && memoryForm != "runtime-strided") {
    result.getDefiningOp()->emitError(
        "dense load has no pass-selected unit or runtime-strided memory form");
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
    if (memoryForm == "unit-stride")
      expression = "__riscv_vle" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + partVL(result, part) + ")";
    else
      expression = "__riscv_vlse" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + memory.strides[laneDimension] +
                   " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                   partVL(result, part) + ")";
    if (registerAxes.empty() && streams > 1) {
      loaded.parts.push_back(std::move(expression));
      continue;
    }
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
    llvm::StringRef kind =
        riscv_internal::string(assigned(value), "physical_kind").value_or("");
    if (kind == "sequential" || kind == "scalar") {
      auto valueType = mlir::cast<kernel::ValueType>(value.getType());
      int64_t elements = 1;
      for (int64_t extent : valueType.getShape().asArrayRef())
        elements *= extent;
      if (elements != 1) {
        value.getDefiningOp()->emitError(
            "sequential dense load requires one logical element");
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
    if (!kind.starts_with("rvv")) {
      value.getDefiningOp()->emitError(
          "dense slice has no selected scalar or vector load realization");
      return mlir::failure();
    }
    mlir::FailureOr<Binding> loaded = loadDenseBlock(value, binding);
    if (mlir::failed(loaded))
      return mlir::failure();
    if (riscv_internal::string(assigned(value), "materialization")
            .value_or("") == "shared-register")
      bindings[value] = *loaded;
    return loaded;
  }
  if (binding.kind == Binding::Kind::Field) {
    llvm::StringRef physicalKind =
        riscv_internal::string(assigned(value), "physical_kind").value_or("");
    if (physicalKind == "register-scalar-tuple") {
      Binding owner = bindings.lookup(binding.field.owner);
      if (owner.kind == Binding::Kind::Slice) {
        mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
        if (mlir::failed(record))
          return mlir::failure();
        owner = std::move(*record);
      }
      auto field = fieldFor(binding);
      llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(value);
      const int64_t registerParts =
          riscv_internal::integer(assigned(value), "register_parts").value_or(0);
      llvm::SmallVector<std::string, 4> registerStrides;
      for (int64_t registerAxis : registerAxes) {
        std::string stride;
        for (const auto &[axis, byteStride] : owner.recordByteStrides)
          if (axis == registerAxis) {
            stride = byteStride;
            break;
          }
        if (stride.empty())
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple axis has no record-byte stride"),
                 mlir::failure();
        registerStrides.push_back(std::move(stride));
      }
      std::string logicalIndex = "0";
      if (binding.field.index) {
        const Binding &index = bindings.lookup(*binding.field.index);
        auto encoding = encodings.find(owner.encodingFamily);
        if (index.kind == Binding::Kind::Scalar) {
          logicalIndex = index.scalar;
        } else if (index.kind == Binding::Kind::Point &&
                   encoding != encodings.end()) {
          const int64_t elements = encoding->second.logicalElements;
          logicalIndex =
              binding.field.selector == "group_index"
                  ? "((" + index.point.base + " % " +
                        std::to_string(elements) + ") / " +
                        std::to_string(index.point.physicalExtent) + ")"
                  : "(" + index.point.base + " % " +
                        std::to_string(elements) + ")";
        } else {
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field has no usable logical index"),
                 mlir::failure();
        }
      }
      unsigned logicalWidth =
          field ? riscv_internal::logicalBitWidth(field->type) : 0;
      auto fragment = field
                          ? singleStorageFragment(*field, logicalIndex, logicalWidth)
                            : std::optional<StorageFragment>();
      if (owner.kind != Binding::Kind::Record || !field || registerAxes.empty() ||
          registerParts <= 0 || registerAxes.size() != registerStrides.size() ||
          !fragment ||
          layoutKind(field->layouts) != "natural" || field->bitOffset % 8 ||
          logicalWidth % 8)
        return value.getDefiningOp()->emitError(
                   "selected scalar tuple has no byte-addressable record mapping"),
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
          record += " + " + std::to_string(coordinate) + " * " + stride;
        record += ")";
        if (field->type.isF32())
          tuple.parts.push_back("weft_load_f32_le(" + record + " + " +
                                fragment->byte + ")");
        else if (field->type.isF16())
          tuple.parts.push_back("weft_load_f16_le(" + record + " + " +
                                fragment->byte + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 16)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "weft_load_u16_le(" :
                                                 "weft_load_i16_le(") +
              record + " + " + fragment->byte + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 8)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "*(const uint8_t *)(" :
                                                 "*(const int8_t *)(") +
              record + " + " + fragment->byte + ")");
        else
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field type has no intrinsic-C load"),
                 mlir::failure();
      }
      return tuple;
    }
    Binding loaded = emitInterleavedField(value, binding, "0");
    if (loaded.kind == Binding::Kind::None) {
      value.getDefiningOp()->emitError(
          "encoded field has no selected numeric load realization");
      return mlir::failure();
    }
    return loaded;
  }
  return binding;
}

mlir::LogicalResult Emitter::compileOperation(mlir::Operation &operation) {
  if (auto symbol = mlir::dyn_cast<kernel::SymbolOp>(operation)) {
    bindings[symbol.getResult()] = scalar(identifier(symbol.getName()));
    return mlir::success();
  }
  if (auto root = mlir::dyn_cast<kernel::RootDomainOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.domainType = root.getResult().getType();
    bindings[root.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto domain = mlir::dyn_cast<kernel::DomainOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.parentDomain = domain.getParent();
    binding.domainType = domain.getResult().getType();
    bindings[domain.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto constant = mlir::dyn_cast<kernel::ConstantOp>(operation)) {
    std::string expression;
    llvm::raw_string_ostream stream(expression);
    constant.getValue().print(stream);
    stream.flush();
    size_t typeMarker = expression.find(" : ");
    if (typeMarker != std::string::npos)
      expression.resize(typeMarker);
    mlir::Type element =
        riscv_internal::logicalElement(constant.getResult().getType());
    if (element.isF32())
      expression += "f";
    else if (element.isF16())
      expression = "((_Float16)(" + expression + "f))";
    bindings[constant.getResult()] = scalar(expression);
    return mlir::success();
  }
  if (auto level = mlir::dyn_cast<kernel::LevelOp>(operation))
    return compileLevel(level);
  if (auto loop = mlir::dyn_cast<kernel::ForOp>(operation))
    return compileFor(loop);
  if (auto branch = mlir::dyn_cast<kernel::IfOp>(operation))
    return compileIf(branch);
  if (auto slice = mlir::dyn_cast<kernel::SliceOp>(operation))
    return compileSlice(slice);
  if (auto admit = mlir::dyn_cast<kernel::AdmitOp>(operation))
    return compileAdmit(admit);
  if (auto materialize = mlir::dyn_cast<kernel::MaterializeOp>(operation))
    return compileMaterialize(materialize);
  if (auto iota = mlir::dyn_cast<kernel::IotaOp>(operation))
    return compileIota(iota);
  if (auto freshValue = mlir::dyn_cast<kernel::NewOp>(operation))
    return compileNew(freshValue);
  if (auto field = mlir::dyn_cast<kernel::FieldOp>(operation))
    return compileField(field);
  if (auto extract = mlir::dyn_cast<kernel::ExtractOp>(operation))
    return compileExtract(extract);
  if (auto unary = mlir::dyn_cast<kernel::UnaryOp>(operation))
    return compileUnary(unary);
  if (auto compare = mlir::dyn_cast<kernel::CompareOp>(operation))
    return compileCompare(compare);
  if (auto cast = mlir::dyn_cast<kernel::CastOp>(operation))
    return compileCast(cast);
  if (auto mac = mlir::dyn_cast<kernel::MacPairsOp>(operation))
    return compileMac(operation, mac.getLhs(), mac.getRhs(), mac.getGroup());
  if (auto mac = mlir::dyn_cast<kernel::MacGroupsOp>(operation))
    return compileMac(operation, mac.getLhs(), mac.getRhs(), mac.getGroup());
  if (auto widen = mlir::dyn_cast<kernel::WidenOp>(operation))
    return compileWiden(widen);
  if (auto reduce = mlir::dyn_cast<kernel::ReduceOp>(operation))
    return compileReduce(reduce);
  if (auto fold = mlir::dyn_cast<kernel::Fold2Op>(operation))
    return compileFold2(fold);
  if (auto dot = mlir::dyn_cast<kernel::DotOp>(operation))
    return compileDot(dot);
  if (auto lookup = mlir::dyn_cast<kernel::LookupOp>(operation))
    return compileLookup(lookup);
  if (auto contract = mlir::dyn_cast<kernel::ContractOp>(operation))
    return compileContract(operation, contract.getLhs(), contract.getRhs(),
                           contract.getResult(), false);
  if (auto contract = mlir::dyn_cast<kernel::OuterContractOp>(operation))
    return compileContract(operation, contract.getLhs(), contract.getRhs(),
                           contract.getResult(), true);
  if (auto binary = mlir::dyn_cast<kernel::BinaryOp>(operation))
    return compileBinary(binary);
  if (auto commit = mlir::dyn_cast<kernel::CommitOp>(operation))
    return compileCommit(commit);
  return fail(&operation, "intrinsic-C emission has no rule for canonical operation");
}

mlir::LogicalResult Emitter::compileLevel(kernel::LevelOp level) {
  const Binding &domainBinding = bindings.lookup(level.getDomain());
  kernel::DomainType domain = domainBinding.domainType;
  const int64_t axis = domain.getAxisId();
  const std::string partition = resolve(domain.getPartition());
  int64_t physicalExtent = 0;
  if (llvm::StringRef(partition).getAsInteger(10, physicalExtent) ||
      physicalExtent <= 0)
    return fail(level, "selected Level partition is not a positive compile-time integer");
  std::string origin = "0";
  std::string total = resolve(domain.getExtent());
  auto parentScope = axisScopes.find(axis);
  if (parentScope != axisScopes.end() && !parentScope->second.empty()) {
    origin = parentScope->second.back().base;
    total = parentScope->second.back().active;
  }

  llvm::SmallVector<Binding, 0> carried;
  for (mlir::Value value : level.getCarried()) {
    Binding source = bindings.lookup(value);
    if (source.kind == Binding::Kind::Vector) {
      mlir::FailureOr<Binding> copy = makeVector(value, "carry", source, value);
      if (mlir::failed(copy))
        return mlir::failure();
      carried.push_back(std::move(*copy));
    } else if (source.kind == Binding::Kind::Scalar) {
      Binding copy = source;
      copy.scalar = fresh("carry");
      auto type = scalarCType(value.getType());
      if (!type)
        return fail(level, "intrinsic-C emission does not support a carried scalar type");
      line(*type + " " + copy.scalar + " = " + source.scalar + ";");
      carried.push_back(std::move(copy));
    } else if (source.kind == Binding::Kind::ScalarTuple) {
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return fail(level,
                    "intrinsic-C emission does not support a carried tuple type");
      Binding copy;
      copy.kind = Binding::Kind::ScalarTuple;
      for (llvm::StringRef expression : source.parts) {
        std::string name = fresh("carry");
        line(*type + " " + name + " = " + expression.str() + ";");
        copy.parts.push_back(std::move(name));
      }
      carried.push_back(std::move(copy));
    } else {
      carried.push_back(std::move(source));
    }
  }

  std::string iterator = fresh("point");
  std::string active = fresh("active");
  std::string base = fresh("base");
  line("for (size_t " + iterator + " = 0; " + iterator + " < (size_t)(" +
       total + "); " + iterator + " += " + partition + ") {");
  ++indent;
  line("const size_t " + active + " = (size_t)(" + total + ") - " + iterator +
       " < (size_t)(" + partition + ") ? (size_t)(" + total + ") - " +
       iterator + " : (size_t)(" + partition + ");");
  line("(void)" + active + ";");
  line("const size_t " + base +
       " __attribute__((unused)) = (size_t)(" + origin + ") + " + iterator +
       ";");
  PointInfo point{axis, base, active, physicalExtent};
  axisScopes[axis].push_back(point);

  bool laneLevel = false;
  int64_t levelPhysicalLanes = physicalExtent;
  if (mlir::DictionaryAttr selected = assigned(level))
    if (auto mapping = selected.getAs<mlir::DictionaryAttr>("level_mapping")) {
      auto iteration = riscv_internal::string(mapping, "physical_iteration");
      laneLevel = iteration && iteration->starts_with("rvv-");
      levelPhysicalLanes =
          riscv_internal::integer(mapping, "physical_lanes")
              .value_or(physicalExtent);
    }
  if (laneLevel)
    laneVLStack.push_back("(" + active + " < " +
                          std::to_string(levelPhysicalLanes) + " ? " + active +
                          " : " + std::to_string(levelPhysicalLanes) + ")");

  Binding pointBinding;
  pointBinding.kind = Binding::Kind::Point;
  pointBinding.point = point;

  auto compileBirths = [&](mlir::Region &region,
                           llvm::SmallVectorImpl<Binding> &yielded,
                           llvm::SmallVectorImpl<mlir::Value> &yieldedValues) {
    mlir::Block &block = region.front();
    if (mlir::failed(compileBlock(block, {pointBinding})))
      return mlir::failure();
    auto terminator = mlir::cast<kernel::BirthsYieldOp>(block.getTerminator());
    for (mlir::Value value : terminator.getValues()) {
      yielded.push_back(bindings.lookup(value));
      yieldedValues.push_back(value);
    }
    return mlir::success();
  };

  llvm::SmallVector<Binding, 0> stateBirths;
  llvm::SmallVector<Binding, 0> stagedBirths;
  llvm::SmallVector<mlir::Value, 0> stateBirthValues;
  llvm::SmallVector<mlir::Value, 0> stagedBirthValues;
  if (mlir::failed(compileBirths(level.getStateBirths(), stateBirths,
                                stateBirthValues)) ||
      mlir::failed(compileBirths(level.getStagedBirths(), stagedBirths,
                                stagedBirthValues)))
    return mlir::failure();

  llvm::SmallVector<Binding, 0> bodyArguments{pointBinding};
  mlir::Block &body = level.getBody().front();
  size_t argumentIndex = 1;
  for (auto [index, binding] : llvm::enumerate(carried)) {
    mlir::FailureOr<Binding> projected = projectBinding(
        level.getCarried()[index], body.getArgument(argumentIndex++), binding);
    if (mlir::failed(projected))
      return fail(level,
                  "level carried argument requires an explicit physical layout conversion");
    bodyArguments.push_back(std::move(*projected));
  }
  for (auto [value, binding] : llvm::zip(stateBirthValues, stateBirths)) {
    mlir::FailureOr<Binding> projected =
        projectBinding(value, body.getArgument(argumentIndex++), binding);
    if (mlir::failed(projected))
      return fail(level,
                  "level state birth requires an explicit physical layout conversion");
    bodyArguments.push_back(std::move(*projected));
  }
  for (auto [value, binding] : llvm::zip(stagedBirthValues, stagedBirths)) {
    mlir::FailureOr<Binding> projected =
        projectBinding(value, body.getArgument(argumentIndex++), binding);
    if (mlir::failed(projected))
      return fail(level,
                  "level staged birth requires an explicit physical layout conversion");
    bodyArguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(body, bodyArguments)))
    return mlir::failure();
  auto handoff = mlir::cast<kernel::HandoffOp>(body.getTerminator());
  if (handoff.getValues().size() != carried.size())
    return fail(level,
                "level handoff value count does not match its carried state");
  for (auto [index, value] : llvm::enumerate(handoff.getValues())) {
    Binding next = bindings.lookup(value);
    if (index >= carried.size())
      return fail(level, "handoff produced more carried values than the level owns");
    Binding &state = carried[index];
    if (mlir::failed(assignBinding(
            level, level.getCarried()[index], state, value, next,
            "level handoff requires an explicit physical layout conversion")))
      return mlir::failure();
  }

  if (laneLevel)
    laneVLStack.pop_back();
  axisScopes[axis].pop_back();
  --indent;
  line("}");
  for (auto [index, result] : llvm::enumerate(level.getResults())) {
    mlir::FailureOr<Binding> projected = projectBinding(
        level.getCarried()[index], result, carried[index]);
    if (mlir::failed(projected))
      return fail(level,
                  "level result requires an explicit physical layout conversion");
    bindings[result] = std::move(*projected);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFor(kernel::ForOp operation) {
  Binding lower = bindings.lookup(operation.getLower());
  Binding upper = bindings.lookup(operation.getUpper());
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
    } else {
      return fail(operation,
                  "ordered for carry has no selected scalar/vector/tuple handoff");
    }
  }
  if (auto cluster =
          assigned(operation.getOperation()).getAs<mlir::DictionaryAttr>(
              "local_cluster");
      cluster &&
      riscv_internal::integer(cluster, "pipeline_depth").value_or(1) > 1)
    return compilePipelinedFor(operation, cluster, lower, upper, step, carried);
  std::string iterator = fresh("for_index");
  line("for (size_t " + iterator + " = (size_t)(" + lower.scalar + "); " +
       iterator + " < (size_t)(" + upper.scalar + "); " + iterator + " += " +
       step.scalar + ") {");
  ++indent;
  llvm::SmallVector<Binding, 0> arguments{scalar(iterator)};
  mlir::Block &body = operation.getBody().front();
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
  auto yield = mlir::cast<kernel::YieldOp>(body.getTerminator());
  if (yield.getValues().size() != carried.size())
    return fail(operation,
                "ordered for yield count does not match its carried state");
  for (auto [index, nextValue] : llvm::enumerate(yield.getValues())) {
    Binding &state = carried[index];
    Binding next = bindings.lookup(nextValue);
    if (mlir::failed(assignBinding(
            operation, operation.getInitArgs()[index], state, nextValue, next,
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

mlir::LogicalResult Emitter::compilePipelinedFor(
    kernel::ForOp operation, mlir::DictionaryAttr cluster,
    const Binding &lower, const Binding &upper, const Binding &step,
    llvm::SmallVectorImpl<Binding> &carried) {
  auto producerIds = cluster.getAs<mlir::ArrayAttr>("producer_ops");
  auto consumerIds = cluster.getAs<mlir::ArrayAttr>("consumer_ops");
  auto frontierIds = cluster.getAs<mlir::ArrayAttr>("frontier_values");
  if (!producerIds || producerIds.empty() || !consumerIds ||
      consumerIds.empty() || !frontierIds || frontierIds.empty())
    return fail(operation,
                "selected local pipeline has no producer/consumer frontier");

  mlir::Block &body = operation.getBody().front();
  llvm::SmallPtrSet<mlir::Operation *, 32> producers;
  llvm::SmallPtrSet<mlir::Operation *, 32> consumers;
  auto collectOperations = [&](mlir::ArrayAttr ids,
                               llvm::SmallPtrSetImpl<mlir::Operation *> &target) {
    for (mlir::Attribute attribute : ids) {
      llvm::StringRef id =
          mlir::cast<mlir::StringAttr>(attribute).getValue();
      mlir::Operation *selected = canonicalOperationsById.lookup(id);
      if (!selected || selected->getBlock() != &body)
        return false;
      target.insert(selected);
    }
    return true;
  };
  if (!collectOperations(producerIds, producers) ||
      !collectOperations(consumerIds, consumers))
    return fail(operation,
                "selected local pipeline references an operation outside its loop body");
  for (mlir::Operation *producer : producers)
    if (consumers.contains(producer))
      return fail(operation,
                  "selected local pipeline assigns one operation to two stages");

  llvm::SmallVector<mlir::Value> frontierValues;
  for (mlir::Attribute attribute : frontierIds) {
    llvm::StringRef id = mlir::cast<mlir::StringAttr>(attribute).getValue();
    mlir::Value value = canonicalValuesById.lookup(id);
    if (!value)
      return fail(operation,
                  "selected local pipeline references an unknown frontier value");
    frontierValues.push_back(value);
  }

  auto bindBodyArguments = [&](llvm::StringRef iterator) -> mlir::LogicalResult {
    bindings[body.getArgument(0)] = scalar(iterator);
    for (auto [index, binding] : llvm::enumerate(carried)) {
      mlir::FailureOr<Binding> projected = projectBinding(
          operation.getInitArgs()[index], body.getArgument(index + 1), binding);
      if (mlir::failed(projected))
        return fail(operation,
                    "pipelined for argument requires an explicit physical layout conversion");
      bindings[body.getArgument(index + 1)] = std::move(*projected);
    }
    return mlir::success();
  };

  auto compileStage = [&](const llvm::SmallPtrSetImpl<mlir::Operation *> &stage,
                          llvm::StringRef iterator) -> mlir::LogicalResult {
    if (mlir::failed(bindBodyArguments(iterator)))
      return mlir::failure();
    conversionCaches.emplace_back();
    for (mlir::Operation &nested : body) {
      if (!stage.contains(&nested))
        continue;
      if (mlir::failed(compileOperation(nested))) {
        conversionCaches.pop_back();
        return mlir::failure();
      }
    }
    conversionCaches.pop_back();
    return mlir::success();
  };

  auto storeFrontier = [&](llvm::MutableArrayRef<Binding> storage)
      -> mlir::LogicalResult {
    for (auto [index, value] : llvm::enumerate(frontierValues)) {
      auto found = bindings.find(value);
      if (found == bindings.end())
        return fail(operation,
                    "pipeline producer did not materialize a frontier value");
      if (mlir::failed(assignBinding(
              operation, value, storage[index], value, found->second,
              "pipeline frontier requires an explicit physical layout conversion")))
        return mlir::failure();
    }
    return mlir::success();
  };

  auto bindFrontier = [&](llvm::ArrayRef<Binding> storage) {
    for (auto [value, binding] : llvm::zip(frontierValues, storage))
      bindings[value] = binding;
  };

  auto updateCarry = [&]() -> mlir::LogicalResult {
    auto yield = mlir::cast<kernel::YieldOp>(body.getTerminator());
    if (yield.getValues().size() != carried.size())
      return fail(operation,
                  "pipelined for yield count does not match its carried state");
    for (auto [index, nextValue] : llvm::enumerate(yield.getValues())) {
      auto found = bindings.find(nextValue);
      if (found == bindings.end())
        return fail(operation,
                    "pipeline consumer did not materialize its yielded value");
      if (mlir::failed(assignBinding(
              operation, operation.getInitArgs()[index], carried[index],
              nextValue, found->second,
              "pipelined for body requires an explicit physical layout conversion")))
        return mlir::failure();
    }
    return mlir::success();
  };

  std::string currentIndex = fresh("pipeline_current_index");
  line("size_t " + currentIndex + " = (size_t)(" + lower.scalar + ");");
  line("if (" + currentIndex + " < (size_t)(" + upper.scalar + ")) {");
  ++indent;

  llvm::SmallVector<Binding, 0> currentFrontier;
  llvm::SmallVector<Binding, 0> nextFrontier;
  for (mlir::Value value : frontierValues) {
    mlir::FailureOr<Binding> current =
        declareMutableBinding(value, "pipeline_current");
    mlir::FailureOr<Binding> next =
        declareMutableBinding(value, "pipeline_next");
    if (mlir::failed(current) || mlir::failed(next))
      return fail(operation,
                  "pipeline frontier has no mutable scalar/vector representation");
    currentFrontier.push_back(std::move(*current));
    nextFrontier.push_back(std::move(*next));
  }

  if (mlir::failed(compileStage(producers, currentIndex)) ||
      mlir::failed(storeFrontier(currentFrontier)))
    return mlir::failure();

  std::string nextIndex = fresh("pipeline_next_index");
  line("for (size_t " + nextIndex + " = " + currentIndex +
       " + (size_t)(" + step.scalar + "); " + nextIndex +
       " < (size_t)(" + upper.scalar + "); " + nextIndex +
       " += (size_t)(" + step.scalar + ")) {");
  ++indent;
  if (mlir::failed(compileStage(producers, nextIndex)) ||
      mlir::failed(storeFrontier(nextFrontier)))
    return mlir::failure();
  bindFrontier(currentFrontier);
  if (mlir::failed(compileStage(consumers, currentIndex)) ||
      mlir::failed(updateCarry()))
    return mlir::failure();
  for (auto [index, value] : llvm::enumerate(frontierValues))
    if (mlir::failed(assignBinding(
            operation, value, currentFrontier[index], value,
            nextFrontier[index],
            "pipeline buffer rotation requires matching frontier representations")))
      return mlir::failure();
  line(currentIndex + " = " + nextIndex + ";");
  --indent;
  line("}");

  bindFrontier(currentFrontier);
  if (mlir::failed(compileStage(consumers, currentIndex)) ||
      mlir::failed(updateCarry()))
    return mlir::failure();
  --indent;
  line("}");

  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInitArgs()[index], result, carried[index]);
    if (mlir::failed(projected))
      return fail(operation,
                  "pipelined for result requires an explicit physical layout conversion");
    bindings[result] = std::move(*projected);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIf(kernel::IfOp operation) {
  Binding condition = bindings.lookup(operation.getCondition());
  if (condition.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered if requires a selected scalar condition");

  llvm::SmallVector<Binding, 0> results;
  for (mlir::Value resultValue : operation.getResults()) {
    llvm::StringRef kind =
        riscv_internal::string(assigned(resultValue), "physical_kind").value_or("");
    if (kind.starts_with("rvv")) {
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
    if (kind == "register-scalar-tuple") {
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
    auto yield = mlir::cast<kernel::YieldOp>(block.getTerminator());
    if (yield.getValues().size() != results.size())
      return fail(operation,
                  "ordered if yield count does not match its results");
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
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

mlir::LogicalResult Emitter::compileSlice(kernel::SliceOp slice) {
  Binding base = bindings.lookup(slice.getBase());
  if (base.kind == Binding::Kind::Field) {
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : slice.getSelectors()) {
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
    }
    bindings[slice.getResult()] = std::move(base);
    return mlir::success();
  }
  Binding binding;
  binding.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Slice) {
    binding.slice = base.slice;
    auto baseType = mlir::cast<kernel::SliceType>(slice.getBase().getType());
    size_t cursor = 0;
    for (auto [position, selectorAttribute] : llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size() ||
          position >= baseType.getAxisIds().size())
        return fail(slice, "nested slice selector has no logical axis or index");
      binding.slice.localOffsets.emplace_back(
          baseType.getAxisIds()[position], slice.getIndices()[cursor++]);
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
  auto sliceType = mlir::dyn_cast<kernel::SliceType>(value.getType());
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
  std::string sourceFamily = base.memory.sourceEncodingFamily.empty()
                                 ? family
                                 : base.memory.sourceEncodingFamily;
  mlir::DictionaryAttr physical = assigned(value);
  if (auto selectedBase = physical.getAs<mlir::StringAttr>(
          "base_encoding_family"))
    sourceFamily = selectedBase.getValue().str();
  else if (auto derived = derivedSources.find(family);
           derived != derivedSources.end())
    sourceFamily = derived->second;
  auto info = encodings.find(sourceFamily);
  if (info == encodings.end())
    return value.getDefiningOp()->emitError(
               "encoded record references an undeclared encoding family"),
           mlir::failure();

  llvm::DenseMap<int64_t, PointInfo> points;
  llvm::SmallVector<std::string> coordinates(base.memory.axes.size(), "0");
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
  result.encodingFamily = sourceFamily;
  const int64_t recordBytes = info->second.storageBits / 8;
  const int64_t elements = info->second.logicalElements;
  if (encoding.getKind() == "derived_instance" ||
      base.memory.interleaveRows > 0) {
    int64_t rows = base.memory.interleaveRows > 0
                       ? base.memory.interleaveRows
                       : riscv_internal::integer(physical, "interleave_rows")
                             .value_or(0);
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
      std::string stride = "1";
      for (size_t following = dimension + 1;
           following < base.memory.axes.size(); ++following) {
        std::string extent = base.memory.extents[following];
        if (following == recordDimension)
          extent = "(" + extent + " / " + std::to_string(elements) + ")";
        stride = "(" + stride + " * " + extent + ")";
      }
      result.recordByteStrides.emplace_back(
          base.memory.axes[dimension],
          "(" + stride + " * " + std::to_string(recordBytes) + ")");
    }
    coordinates[recordDimension] = "(" + coordinates[recordDimension] + " / " +
                                   std::to_string(elements) + ")";
    std::string linear = "0";
    for (size_t dimension = 0; dimension < base.memory.axes.size(); ++dimension) {
      std::string stride = "1";
      for (size_t following = dimension + 1;
           following < base.memory.axes.size(); ++following) {
        std::string extent = base.memory.extents[following];
        if (following == recordDimension)
          extent = "(" + extent + " / " + std::to_string(elements) + ")";
        stride = "(" + stride + " * " + extent + ")";
      }
      linear = "(" + linear + " + (" + coordinates[dimension] + ") * " +
               stride + ")";
    }
    result.recordPointer = base.memory.name + " + (" + linear + ") * " +
                           std::to_string(recordBytes);
  }
  return result;
}

mlir::LogicalResult Emitter::compileAdmit(kernel::AdmitOp admit) {
  Binding region = bindings.lookup(admit.getRegion());
  if (region.kind == Binding::Kind::Memory) {
    auto view = mlir::dyn_cast<kernel::ViewType>(admit.getRegion().getType());
    auto encoding = view ? mlir::dyn_cast<kernel::EncodingType>(view.getEncoding())
                         : kernel::EncodingType();
    if (!encoding || encoding.getKind() != "dense")
      return fail(admit,
                  "whole-view admit is only defined for a dense read-only value");
    Binding slice;
    slice.kind = Binding::Kind::Slice;
    slice.slice.base = admit.getRegion();
    bindings[admit.getResult()] = std::move(slice);
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(admit, "admit currently requires a canonical slice region");
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(
      mlir::cast<kernel::SliceType>(admit.getRegion().getType()).getEncoding());
  if (!encoding)
    return fail(admit, "admit slice has no encoding");
  if (encoding.getKind() == "dense") {
    bindings[admit.getResult()] = std::move(region);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = recordForSlice(admit.getRegion());
  if (mlir::failed(result))
    return mlir::failure();
  bindings[admit.getResult()] = std::move(*result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIota(kernel::IotaOp operation) {
  llvm::StringRef realization =
      riscv_internal::string(assigned(operation.getOperation()), "realization")
          .value_or("");
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getResult().getType()));
  if (!integer || integer.isSigned())
    return fail(operation, "iota requires an unsigned integer result");
  if (realization == "register.iota") {
    auto type = scalarCType(integer);
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
  const bool rematerialize =
      riscv_internal::string(assigned(operation.getResult()), "materialization")
          .value_or("") == "rematerialize-per-register-part";
  const std::string type = vectorType(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string expression = "__riscv_vid_v_" + suffix + "(" + vl + ")";
    const std::string offset = "(" + std::to_string(operation.getStart()) +
                               " + " + partOffset(operation.getResult(), part) + ")";
    expression = "__riscv_vadd_vx_" + suffix + "(" + expression + ", " +
                 offset + ", " + vl + ")";
    if (rematerialize) {
      result.parts.push_back(std::move(expression));
    } else {
      std::string name = fresh("iota");
      line(type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileNew(kernel::NewOp operation) {
  Binding initial = operation.getInitialized()
                        ? bindings.lookup(operation.getInitial())
                        : scalar("0");
  mlir::DictionaryAttr physical = assigned(operation.getResult());
  llvm::StringRef kind =
      riscv_internal::string(physical, "physical_kind").value_or("");
  if (kind.starts_with("rvv")) {
    mlir::FailureOr<Binding> result =
        initial.kind == Binding::Kind::Slice
            ? loadDenseBlock(operation.getResult(), initial)
            : makeVector(operation.getResult(), "state", initial,
                         operation.getInitialized() ? operation.getInitial()
                                                    : mlir::Value());
    if (mlir::failed(result))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
    return mlir::success();
  }
  if (kind == "register-scalar-tuple") {
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
  if (kind != "scalar")
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

mlir::LogicalResult Emitter::compileField(kernel::FieldOp operation) {
  mlir::DictionaryAttr selected = assigned(operation.getOperation());
  auto memoryEdge = selected.getAs<mlir::DictionaryAttr>("memory_edge");
  llvm::StringRef realization =
      riscv_internal::string(selected, "realization").value_or("");
  if (!memoryEdge || !realization.starts_with("encoded-field."))
    return fail(operation,
                "field emission requires one selected memory-edge mapping");
  Binding binding;
  binding.kind = Binding::Kind::Field;
  binding.field.owner = operation.getOwner();
  binding.field.name = operation.getName().str();
  binding.field.memoryEdge = memoryEdge;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileExtract(kernel::ExtractOp operation) {
  Binding binding = bindings.lookup(operation.getInput());
  if (binding.kind == Binding::Kind::Field) {
    mlir::DictionaryAttr selected = assigned(operation.getOperation());
    auto memoryEdge = selected
                          ? selected.getAs<mlir::DictionaryAttr>("memory_edge")
                          : mlir::DictionaryAttr();
    if (!memoryEdge)
      return fail(operation,
                  "encoded extract has no pass-selected per-use memory edge");
    binding.field.memoryEdge = memoryEdge;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size())
        return fail(operation, "extract selector has no corresponding index");
      binding.field.index = operation.getIndices()[cursor++];
      binding.field.selector = selector.str();
    }
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }

  if (binding.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(materialized))
      return mlir::failure();
    auto inputType = mlir::cast<kernel::ValueType>(operation.getInput().getType());
    int64_t elements = 1;
    for (int64_t extent : inputType.getShape().asArrayRef())
      elements *= extent;
    if (materialized->kind == Binding::Kind::Scalar && elements == 1 &&
        !mlir::isa<kernel::ValueType>(operation.getResult().getType())) {
      bindings[operation.getResult()] = std::move(*materialized);
      return mlir::success();
    }
    binding = std::move(*materialized);
  }

  if (binding.kind != Binding::Kind::Vector)
    return fail(operation,
                "extract has no selected local vector or encoded-field input");
  if (!mlir::isa<mlir::IntegerType>(
          riscv_internal::logicalElement(operation.getInput().getType())) ||
      riscv_internal::logicalElement(operation.getInput().getType()) !=
          riscv_internal::logicalElement(operation.getResult().getType()))
    return fail(operation,
                "current local vector extract requires one unchanged integer element type");
  if (streamPartCount(operation.getResult()) != 1)
    return fail(operation,
                "current local vector extract result must fit one RVV value");

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

  auto inputType = mlir::cast<kernel::ValueType>(operation.getInput().getType());
  int64_t logicalExtent = 0;
  for (auto [extent, axis] : llvm::zip(inputType.getShape().asArrayRef(),
                                       inputType.getAxisIds().asArrayRef()))
    if (axis == laneAxis)
      logicalExtent = extent;
  const int64_t inputLanes = physicalLanes(operation.getInput());
  const int64_t resultLanes = physicalLanes(operation.getResult());
  if (logicalExtent <= 0 || inputLanes <= 0 || resultLanes <= 0 ||
      inputLanes % resultLanes ||
      point.point.physicalExtent > resultLanes ||
      static_cast<int64_t>(binding.parts.size()) !=
          streamPartCount(operation.getInput()))
    return fail(operation,
                "selected local vector subview has incompatible strip geometry");

  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const std::string resultName = fresh("extract");
  line(resultType + " " + resultName + " = __riscv_vmv_v_x_" + resultSuffix +
       "(0, " + laneVL() + ");");
  const std::string relative = "(" + point.point.base + " % " +
                               std::to_string(logicalExtent) + ")";
  for (auto [part, source] : llvm::enumerate(binding.parts)) {
    const std::string shifted =
        "__riscv_vslidedown_vx_" + inputSuffix + "(" + source + ", " +
        relative + " % " + std::to_string(inputLanes) + ", " +
        std::to_string(inputLanes) + ")";
    const std::string extracted =
        inputSuffix == resultSuffix
            ? shifted
            : "__riscv_vlmul_trunc_v_" + inputSuffix + "_" + resultSuffix +
                  "(" + shifted + ")";
    line("if ((" + relative + " / " + std::to_string(inputLanes) + ") == " +
         std::to_string(part) + ") " + resultName + " = " + extracted + ";");
  }
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.push_back(std::move(resultName));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileUnary(kernel::UnaryOp operation) {
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type element =
      riscv_internal::logicalElement(operation.getResult().getType());
  auto scalarExpression = [&](llvm::StringRef source)
      -> std::optional<std::string> {
    std::string expression;
    if (operation.getKind() == "neg")
      expression = "(-(" + source.str() + "))";
    else if (operation.getKind() == "abs" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "fabs(" + source.str() + ")"
                                   : "fabsf(" + source.str() + ")";
    else if (operation.getKind() == "abs")
      expression = "((" + source.str() + ") < 0 ? -(" + source.str() +
                   ") : (" + source.str() + "))";
    else if (operation.getKind() == "exp" && mlir::isa<mlir::FloatType>(element))
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
    mlir::DictionaryAttr conversion = useConversion(operation.getOperation(), 0);
    llvm::StringRef relation =
        riscv_internal::string(conversion, "relation").value_or("");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : registerPartCount(operation.getResult());
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
      } else if (input->kind == Binding::Kind::Vector &&
                 relation == "lane-to-register") {
        source = extractLaneForRegisterBroadcast(
            operation.getOperation(), 0, operation.getInput(),
            operation.getResult(), part, *input);
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
  if (operation.getKind() == "exp")
    return fail(operation,
                "wide exp has no selected local RVV implementation");
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
    if (operation.getKind() == "neg")
      expression = "__riscv_" + std::string(floating ? "vfneg_v_" : "vneg_v_") +
                   suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else if (operation.getKind() == "abs" && floating)
      expression = "__riscv_vfabs_v_" + suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else
      return fail(operation,
                  "selected integer vector absolute value is not implemented");
    if (streamPartCount(operation.getResult()) > 1 ||
        riscv_internal::string(assigned(operation.getResult()),
                               "materialization")
                .value_or("") == "rematerialize-per-register-part") {
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

mlir::LogicalResult Emitter::compileCompare(kernel::CompareOp operation) {
  mlir::FailureOr<Binding> lhs =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhs =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();
  if (lhs->kind != Binding::Kind::Scalar || rhs->kind != Binding::Kind::Scalar)
    return fail(operation,
                "current comparison emission requires selected scalar operands");
  llvm::StringRef spelling = operation.getPredicate() == "eq" ? "=="
                            : operation.getPredicate() == "ne" ? "!="
                            : operation.getPredicate() == "lt" ? "<"
                            : operation.getPredicate() == "le" ? "<="
                            : operation.getPredicate() == "gt" ? ">"
                            : operation.getPredicate() == "ge" ? ">="
                                                                 : "";
  if (spelling.empty())
    return fail(operation, "unknown selected comparison predicate");
  bindings[operation.getResult()] =
      scalar("(" + lhs->scalar + " " + spelling.str() + " " + rhs->scalar +
             ")");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCast(kernel::CastOp operation) {
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type source =
      riscv_internal::logicalElement(operation.getInput().getType());
  mlir::Type target =
      riscv_internal::logicalElement(operation.getResult().getType());
  auto targetCType = scalarCType(target);
  if (!targetCType)
    return fail(operation, "cast target has no intrinsic-C type");
  auto rounding = operation->getAttrOfType<mlir::StringAttr>("rounding");
  auto saturate = operation->getAttrOfType<mlir::BoolAttr>("saturate");
  auto scalarExpression = [&](llvm::StringRef inputExpression)
      -> mlir::FailureOr<std::string> {
    std::string expression = inputExpression.str();
    if (rounding) {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(target);
      if (!integer)
        return mlir::failure();
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
      llvm::StringRef roundFunction = rounding.getValue() == "rne" ? "nearbyint"
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
    mlir::DictionaryAttr conversion = useConversion(operation.getOperation(), 0);
    llvm::StringRef relation =
        riscv_internal::string(conversion, "relation").value_or("");
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
      } else if (input->kind == Binding::Kind::Vector &&
                 relation == "lane-to-register") {
        sourceExpression = extractLaneForRegisterBroadcast(
            operation.getOperation(), 0, operation.getInput(),
            operation.getResult(), part, *input);
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
  mlir::DictionaryAttr local =
      assigned(operation.getOperation()).getAs<mlir::DictionaryAttr>(
          "local_operation");
  llvm::StringRef conversion =
      riscv_internal::string(local, "conversion").value_or("");
  if (!rounding && conversion == "identity") {
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
  if (!rounding && conversion == "integer-reinterpret") {
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
  if (!rounding && ((source.isF32() && target.isF16()) ||
                    (source.isF16() && target.isF32()))) {
    int64_t sourceLMUL = assigned(operation.getInput())
                             .getAs<mlir::IntegerAttr>("lmul_eighths")
                             .getInt();
    int64_t targetLMUL = assigned(operation.getResult())
                             .getAs<mlir::IntegerAttr>("lmul_eighths")
                             .getInt();
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
  if (!rounding || !saturate || !saturate.getValue() || !source.isF32())
    return fail(operation,
                "current vector cast implements explicit saturated f32 narrowing only");
  auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
  if (!targetInteger || targetInteger.getWidth() != 8)
    return fail(operation,
                "current vector narrowing implements an eight-bit integer target");
  int64_t sourceLMUL =
      assigned(operation.getInput()).getAs<mlir::IntegerAttr>("lmul_eighths").getInt();
  int64_t targetLMUL =
      assigned(operation.getResult()).getAs<mlir::IntegerAttr>("lmul_eighths").getInt();
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  if (sourceLMUL < 4 || targetLMUL * 4 != sourceLMUL)
    return fail(operation,
                "selected f32-to-i8 LMUL relation cannot be narrowed mechanically");
  const std::string i32Suffix = "i32" + lmul(sourceLMUL);
  const std::string i32Type = "vint32" + lmul(sourceLMUL) + "_t";
  const std::string i16Suffix = "i16" + lmul(sourceLMUL / 2);
  const std::string i16Type = "vint16" + lmul(sourceLMUL / 2) + "_t";
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  const std::string targetType = vectorType(operation.getResult());
  llvm::StringRef frm = rounding.getValue() == "rne" ? "__RISCV_FRM_RNE"
                       : rounding.getValue() == "rtz" ? "__RISCV_FRM_RTZ"
                       : rounding.getValue() == "rdn" ? "__RISCV_FRM_RDN"
                                                       : "__RISCV_FRM_RUP";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, part);
    if (!sourcePart || *sourcePart >= input->parts.size())
      return fail(operation,
                  "rounded narrowing requires an explicit physical layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    std::string i32Value = fresh("rounded_i32");
    line(i32Type + " " + i32Value + " = __riscv_vfcvt_" +
         std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_v_" +
         i32Suffix + "_rm(" + input->parts[*sourcePart] + ", " + frm.str() +
         ", " + vl +
         ");");
    std::string i16Value = fresh("narrow_i16");
    line(i16Type + " " + i16Value + " = __riscv_" +
         std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
         "_wx_" + i16Suffix + "(" + i32Value +
         ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    std::string i8Value = fresh("narrow_i8");
    line(targetType + " " + i8Value + " = __riscv_" +
         std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
         "_wx_" + targetSuffix + "(" + i16Value +
         ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    result.parts.push_back(std::move(i8Value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileMac(mlir::Operation &operation,
                                        mlir::Value lhs, mlir::Value rhs,
                                        int64_t group) {
  llvm::StringRef realization =
      riscv_internal::string(assigned(&operation), "realization").value_or("");
  if (realization != "rvv.vwmaccsu.typed")
    return fail(&operation,
                "intrinsic-C emitter only implements the selected rvv.vwmaccsu MAC");
  Binding binding;
  binding.kind = Binding::Kind::DeferredMac;
  binding.lhs = lhs;
  binding.rhs = rhs;
  binding.group = group;
  binding.sourceOperation = &operation;
  bindings[operation.getResult(0)] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileWiden(kernel::WidenOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  llvm::StringRef realization =
      riscv_internal::string(assigned(operation.getOperation()), "realization")
          .value_or("");
  if (input.kind == Binding::Kind::DeferredMac ||
      realization == "rvv.widen-deferred-to-reduction") {
    Binding binding;
    binding.kind = Binding::Kind::DeferredWiden;
    binding.input = operation.getInput();
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getInput(), std::move(input));
  if (mlir::failed(materialized))
    return mlir::failure();
  input = std::move(*materialized);
  mlir::Type source =
      riscv_internal::logicalElement(operation.getInput().getType());
  mlir::Type target =
      riscv_internal::logicalElement(operation.getResult().getType());
  auto targetCType = scalarCType(target);
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "widen result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    if (!targetCType)
      return fail(operation, "scalar or tuple widening target has no C type");
    mlir::DictionaryAttr conversion = useConversion(operation.getOperation(), 0);
    llvm::StringRef relation =
        riscv_internal::string(conversion, "relation").value_or("");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : registerPartCount(operation.getResult());
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
      } else if (input.kind == Binding::Kind::Vector &&
                 relation == "lane-to-register") {
        sourceExpression = extractLaneForRegisterBroadcast(
            operation.getOperation(), 0, operation.getInput(),
            operation.getResult(), part, input);
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
    if (source.isF16() && target.isF32())
      expression = "__riscv_vfwcvt_f_f_v_" + targetSuffix + "(" +
                   sourceExpression + ", " + vl + ")";
    else if (source.isInteger(32) && target.isF32())
      expression = "__riscv_vfcvt_f_x_v_" + targetSuffix + "(" +
                   sourceExpression + ", " + vl + ")";
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
    if (riscv_internal::string(assigned(operation.getResult()),
                               "materialization")
            .value_or("") == "rematerialize-per-register-part") {
      result.parts.push_back(std::move(expression));
    } else {
      std::string name = fresh("widen");
      line(targetType + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReduce(kernel::ReduceOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  mlir::DictionaryAttr selected = assigned(operation.getOperation());
  llvm::StringRef selectedRealization =
      riscv_internal::string(selected, "realization").value_or("");
  if (selectedRealization == "rvv.co-reduce.max-min.follower") {
    if (!bindings.count(operation.getResult()))
      return fail(operation,
                  "co-reduction follower was emitted before its selected leader");
    return mlir::success();
  }
  if (selectedRealization == "rvv.co-reduce.max-min.leader") {
    auto partnerId = selected.getAs<mlir::StringAttr>("co_reduce_partner");
    auto partnerIt = partnerId
                         ? canonicalOperationsById.find(partnerId.getValue())
                         : canonicalOperationsById.end();
    auto partner = partnerIt != canonicalOperationsById.end()
                       ? mlir::dyn_cast<kernel::ReduceOp>(partnerIt->second)
                       : kernel::ReduceOp();
    if (!partner || partner.getInput() != operation.getInput() ||
        operation.getKind() == partner.getKind() ||
        (operation.getKind() != "max" && operation.getKind() != "min") ||
        (partner.getKind() != "max" && partner.getKind() != "min"))
      return fail(operation,
                  "selected co-reduction does not name one max/min input pair");
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    mlir::Type element =
        riscv_internal::logicalElement(operation.getInput().getType());
    if (materialized->kind != Binding::Kind::Vector || !element.isF32())
      return fail(operation,
                  "current selected co-reduction requires an f32 vector input");
    const std::string suffix = vectorSuffix(operation.getInput());
    const std::string type = vectorType(operation.getInput());
    const std::string vl = std::to_string(physicalLanes(operation.getInput()));
    std::string maximum = fresh("co_reduce_max");
    std::string minimum = fresh("co_reduce_min");
    line(type + " " + maximum + " = __riscv_vfmv_v_f_" + suffix +
         "(-INFINITY, " + vl + ");");
    line(type + " " + minimum + " = __riscv_vfmv_v_f_" + suffix +
         "(INFINITY, " + vl + ");");
    for (auto [index, expression] : llvm::enumerate(materialized->parts)) {
      std::string loaded = fresh("co_reduce_load");
      line(type + " " + loaded + " = " + expression + ";");
      const std::string active = partVL(operation.getInput(), index);
      line(maximum + " = __riscv_vfmax_vv_" + suffix + "(" + maximum + ", " +
           loaded + ", " + active + ");");
      line(minimum + " = __riscv_vfmin_vv_" + suffix + "(" + minimum + ", " +
           loaded + ", " + active + ");");
    }
    auto finish = [&](kernel::ReduceOp reduce, llvm::StringRef vector,
                      llvm::StringRef kind) {
      const std::string scalarValue = fresh("co_reduce_scalar");
      const std::string seed = fresh("co_reduce_seed");
      const std::string reduced = fresh("co_reduce_vector");
      line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
           std::string(kind == "max" ? "-INFINITY" : "INFINITY") + ", 1);");
      line("vfloat32m1_t " + reduced + " = __riscv_vfred" + kind.str() +
           "_vs_" + suffix + "_f32m1(" + vector.str() + ", " + seed + ", " +
           vl + ");");
      line("float " + scalarValue + " = __riscv_vfmv_f_s_f32m1_f32(" +
           reduced + ");");
      bindings[reduce.getResult()] = scalar(scalarValue);
    };
    kernel::ReduceOp maximumOp =
        operation.getKind() == "max" ? operation : partner;
    kernel::ReduceOp minimumOp =
        operation.getKind() == "min" ? operation : partner;
    finish(maximumOp, maximum, "max");
    finish(minimumOp, minimum, "min");
    return mlir::success();
  }
  if (selectedRealization == "rvv.reduce.register-axis") {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    if (materialized->kind != Binding::Kind::Vector)
      return fail(operation,
                  "register-axis reduction requires a selected vector input");
    mlir::DictionaryAttr local =
        selected.getAs<mlir::DictionaryAttr>("local_operation");
    auto eliminated =
        local ? local.getAs<mlir::IntegerAttr>("eliminated_register_axis")
              : mlir::IntegerAttr();
    if (!eliminated)
      return fail(operation,
                  "register-axis reduction has no pass-selected axis");

    llvm::SmallVector<int64_t, 4> inputAxes =
        registerAxesFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> inputExtents =
        registerExtentsFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> resultAxes =
        registerAxesFor(operation.getResult());
    auto eliminatedPosition = llvm::find(inputAxes, eliminated.getInt());
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
    llvm::StringRef kind = operation.getKind();
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
            if (axis != eliminated.getInt()) {
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
  if (input.kind == Binding::Kind::DeferredWiden) {
    Binding source = bindings.lookup(input.input);
    if (source.kind == Binding::Kind::DeferredMac) {
      mlir::FailureOr<Binding> result =
          emitGroupedMacReduction(operation.getResult(), source);
      if (mlir::failed(result))
        return mlir::failure();
      bindings[operation.getResult()] = std::move(*result);
      return mlir::success();
    }
    llvm::StringRef realization =
        riscv_internal::string(assigned(operation.getOperation()), "realization")
            .value_or("");
    if (!realization.starts_with("rvv.widen-reduce."))
      return fail(operation,
                  "deferred widening has no selected reduction realization");
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(input.input, std::move(source));
    if (mlir::failed(materialized))
      return mlir::failure();
    if (materialized->kind != Binding::Kind::Vector || operation.getKind() != "add")
      return fail(operation,
                  "selected widening reduction requires an integer vector add");
    auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(
        riscv_internal::logicalElement(input.input.getType()));
    auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(
        riscv_internal::logicalElement(operation.getResult().getType()));
    if (!sourceInteger || !targetInteger ||
        targetInteger.getWidth() != sourceInteger.getWidth() * 2 ||
        sourceInteger.isUnsigned() != targetInteger.isUnsigned())
      return fail(operation,
                  "selected widening reduction has incompatible integer types");
    auto scalarType = scalarCType(targetInteger);
    if (!scalarType)
      return fail(operation,
                  "selected widening reduction result has no scalar C type");
    const std::string sourceSuffix = vectorSuffix(input.input);
    const std::string targetSuffix =
        std::string(targetInteger.isUnsigned() ? "u" : "i") +
        std::to_string(targetInteger.getWidth()) + "m1";
    const std::string targetType =
        std::string(targetInteger.isUnsigned() ? "vuint" : "vint") +
        std::to_string(targetInteger.getWidth()) + "m1_t";
    const int64_t resultParts = registerPartCount(operation.getResult());
    const int64_t streams = streamPartCount(input.input);
    if (resultParts <= 0 || streams <= 0)
      return fail(operation,
                  "widening reduction input/result mappings do not form a slice");
    Binding result;
    result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                   : Binding::Kind::ScalarTuple;
    for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
      auto sourceRegister =
          projectRegisterPart(input.input, operation.getResult(), resultPart);
      if (!sourceRegister)
        return fail(operation,
                    "widening reduction free-axis mapping is not projectable");
      std::string accumulator = fresh("widen_reduce_scalar");
      line(*scalarType + " " + accumulator + " = 0;");
      for (int64_t stream = 0; stream < streams; ++stream) {
        const size_t index = *sourceRegister * streams + stream;
        if (index >= materialized->parts.size())
          return fail(operation,
                      "widening reduction source mapping is incomplete");
        std::string seed = fresh("widen_reduce_seed");
        line(targetType + " " + seed + " = __riscv_vmv_v_x_" + targetSuffix +
             "(" + accumulator + ", 1);");
        std::string reduced = fresh("widen_reduce_vector");
        line(targetType + " " + reduced + " = __riscv_" +
             std::string(targetInteger.isUnsigned() ? "vwredsumu" : "vwredsum") +
             "_vs_" + sourceSuffix + "_" + targetSuffix + "(" +
             materialized->parts[index] + ", " + seed + ", " +
             partVL(input.input, index) + ");");
        line(accumulator + " = __riscv_vmv_x_s_" + targetSuffix + "_" +
             std::string(targetInteger.isUnsigned() ? "u" : "i") +
             std::to_string(targetInteger.getWidth()) + "(" + reduced + ");");
      }
      if (resultParts == 1)
        result.scalar = accumulator;
      else
        result.parts.push_back(std::move(accumulator));
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
  llvm::StringRef kind = operation.getKind();
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
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    auto sourceRegister = projectRegisterPart(
        operation.getInput(), operation.getResult(), resultPart);
    if (!sourceRegister)
      return fail(operation,
                  "reduce free-axis mapping is not projectable");
    std::string accumulator = fresh("reduce_scalar");
    line(*scalarType + " " + accumulator + " = (" + *scalarType + ")(" +
         initial + ");");
    for (int64_t stream = 0; stream < streams; ++stream) {
      const size_t index = *sourceRegister * streams + stream;
      if (index >= materialized->parts.size())
        return fail(operation, "reduce source mapping is incomplete");
      const std::string seed = fresh("reduce_seed");
      line(baseType + " " + seed + " = __riscv_" +
           std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + baseSuffix +
           "(" + accumulator + ", 1);");
      std::string stem;
      if (floating)
        stem = kind == "add"   ? "vfredusum"
               : kind == "max" ? "vfredmax"
                                 : "vfredmin";
      else
        stem = "vredsum";
      const std::string reduced = fresh("reduce_vector");
      line(baseType + " " + reduced + " = __riscv_" + stem + "_vs_" +
           sourceSuffix + "_" + baseSuffix + "(" +
           materialized->parts[index] + ", " + seed + ", " +
           partVL(operation.getInput(), index) + ");");
      if (floating)
        line(accumulator + " = __riscv_vfmv_f_s_" + baseSuffix + "_f" +
             std::to_string(sew) + "(" + reduced + ");");
      else
        line(accumulator + " = __riscv_vmv_x_s_" + baseSuffix + "_" +
             std::string(unsignedInteger ? "u" : "i") +
             std::to_string(sew) + "(" + reduced + ");");
    }
    if (resultParts == 1)
      result.scalar = accumulator;
    else
      result.parts.push_back(std::move(accumulator));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFold2(kernel::Fold2Op operation) {
  Binding binding;
  binding.kind = Binding::Kind::DeferredFold;
  binding.input = operation.getInput();
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileDot(kernel::DotOp operation) {
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  llvm::StringRef realization =
      riscv_internal::string(assigned(operation.getOperation()), "realization")
          .value_or("");
  if (realization == "rvv.dot.encoded-field-folded-i16") {
    mlir::FailureOr<Binding> result = emitQuantDot(operation, lhs, rhs);
    if (mlir::failed(result))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
    return mlir::success();
  }
  return compileContract(*operation, operation.getLhs(), operation.getRhs(),
                         operation.getResult(), false);
}

mlir::LogicalResult Emitter::compileLookup(kernel::LookupOp operation) {
  llvm::StringRef realization =
      riscv_internal::string(assigned(operation.getOperation()), "realization")
          .value_or("");
  Binding table = bindings.lookup(operation.getTable());
  if (table.kind != Binding::Kind::Slice)
    return fail(operation,
                "selected lookup currently requires an admitted dense table");
  auto address = denseAddress(table.slice, {});
  if (!address)
    return fail(operation, "selected lookup table has no dense base address");

  mlir::Type tableElement =
      riscv_internal::logicalElement(operation.getTable().getType());
  auto tableCType = scalarCType(tableElement);
  const unsigned tableBits = riscv_internal::logicalBitWidth(tableElement);
  if (!tableCType || !tableBits || tableBits % 8)
    return fail(operation,
                "selected lookup requires a byte-addressable numeric table");
  const unsigned tableBytes = tableBits / 8;
  const std::string base = "((const " + *tableCType + " *)(" + *address + "))";

  Binding indices = bindings.lookup(operation.getIndices());
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getIndices(), indices);
  if (mlir::failed(materialized))
    return mlir::failure();
  indices = std::move(*materialized);

  if (realization == "scalar.lookup") {
    if (indices.kind != Binding::Kind::Scalar)
      return fail(operation,
                  "scalar lookup requires a scalar unsigned index");
    bindings[operation.getResult()] =
        scalar(base + "[(size_t)(" + indices.scalar + ")]");
    return mlir::success();
  }
  if (realization != "rvv.indexed-lookup" &&
      realization != "rvv.unit-stride-lookup-window")
    return fail(operation, "lookup has an unknown selected realization");
  if (indices.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV indexed lookup requires vector unsigned indices");

  const int64_t parts = vectorPartCount(operation.getResult());
  mlir::DictionaryAttr indexPhysical = assigned(operation.getIndices());
  auto indexSEW = indexPhysical.getAs<mlir::IntegerAttr>("physical_sew");
  auto indexLMUL = indexPhysical.getAs<mlir::StringAttr>("lmul");
  if (!indexSEW || !indexLMUL)
    return fail(operation,
                "RVV indexed lookup has no selected index SEW/LMUL");
  const std::string indexSuffix =
      "u" + std::to_string(indexSEW.getInt()) + indexLMUL.getValue().str();
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const bool share =
      riscv_internal::string(assigned(operation.getResult()), "materialization")
          .value_or("") == "shared-register";

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto indexPart = mappedPart(operation.getOperation(), 1, part);
    if (!indexPart || *indexPart >= indices.parts.size())
      return fail(operation,
                  "lookup index mapping requires an explicit layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    if (realization == "rvv.unit-stride-lookup-window") {
      std::string first = "__riscv_vmv_x_s_" + indexSuffix + "_u" +
                          std::to_string(indexSEW.getInt()) + "(" +
                          indices.parts[*indexPart] + ")";
      std::string expression =
          "__riscv_vle" + std::to_string(tableBits) + "_v_" + resultSuffix +
          "(" + base + " + (size_t)(" + first + "), " + vl + ")";
      if (!share) {
        result.parts.push_back(std::move(expression));
      } else {
        std::string name = fresh("lookup_window");
        line(resultType + " " + name + " = " + expression + ";");
        result.parts.push_back(std::move(name));
      }
      continue;
    }
    std::string byteOffsets = indices.parts[*indexPart];
    if (tableBytes != 1)
      byteOffsets = "__riscv_vmul_vx_" + indexSuffix + "(" + byteOffsets +
                    ", " + std::to_string(tableBytes) + ", " + vl + ")";
    std::string expression =
        "__riscv_vluxei" + std::to_string(indexSEW.getInt()) + "_v_" +
        resultSuffix + "(" + base + ", " + byteOffsets + ", " + vl + ")";
    if (!share) {
      result.parts.push_back(std::move(expression));
      continue;
    }
    std::string name = fresh("lookup");
    line(resultType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileBinary(kernel::BinaryOp operation) {
  llvm::StringRef realization =
      riscv_internal::string(assigned(operation.getOperation()), "realization")
          .value_or("");
  if (realization == "rvv.outer.accumulate") {
    Binding lhs = bindings.lookup(operation.getLhs());
    Binding rhs = bindings.lookup(operation.getRhs());
    const Binding *contract = lhs.kind == Binding::Kind::DeferredContract
                                  ? &lhs
                              : rhs.kind == Binding::Kind::DeferredContract
                                  ? &rhs
                                  : nullptr;
    mlir::Value accumulatorValue = lhs.kind == Binding::Kind::DeferredContract
                                       ? operation.getRhs()
                                       : operation.getLhs();
    if (!contract || !contract->sourceOperation)
      return fail(operation,
                  "selected outer accumulation has no deferred contract");
    mlir::FailureOr<Binding> accumulator = materializeNumeric(
        accumulatorValue, bindings.lookup(accumulatorValue));
    if (mlir::failed(accumulator) ||
        accumulator->kind != Binding::Kind::Vector)
      return fail(operation,
                  "selected outer accumulation requires a vector accumulator");
    return compileContract(*contract->sourceOperation, contract->lhs,
                           contract->rhs, operation.getResult(), true,
                           std::move(*accumulator), accumulatorValue);
  }
  if (realization == "rvv.fma.product" ||
      realization == "rvv.fmsac.product") {
    Binding product;
    product.kind = Binding::Kind::DeferredProduct;
    product.lhs = operation.getLhs();
    product.rhs = operation.getRhs();
    product.sourceOperation = operation.getOperation();
    bindings[operation.getResult()] = std::move(product);
    return mlir::success();
  }
  if (realization == "rvv.fma.accumulate" ||
      realization == "rvv.fmsac.accumulate") {
    mlir::DictionaryAttr selected = assigned(operation.getOperation());
    auto local = selected
                     ? selected.getAs<mlir::DictionaryAttr>("local_operation")
                     : mlir::DictionaryAttr();
    llvm::StringRef instruction =
        riscv_internal::string(local, "instruction").value_or("");
    llvm::StringRef operandForm =
        riscv_internal::string(local, "operand_form").value_or("");
    if ((instruction != "rvv.vfmacc" && instruction != "rvv.vfmsac") ||
        (operandForm != "vv" && operandForm != "vf-lhs-scalar" &&
         operandForm != "vf-rhs-scalar"))
      return fail(operation,
                  "selected fused accumulation has no complete local operation");
    Binding lhs = bindings.lookup(operation.getLhs());
    Binding rhs = bindings.lookup(operation.getRhs());
    const Binding *product = lhs.kind == Binding::Kind::DeferredProduct ? &lhs
                              : rhs.kind == Binding::Kind::DeferredProduct
                                  ? &rhs
                                  : nullptr;
    mlir::Value accumulatorValue = lhs.kind == Binding::Kind::DeferredProduct
                                       ? operation.getRhs()
                                       : operation.getLhs();
    if (!product)
      return fail(operation,
                  "selected fused accumulation has no deferred product");
    mlir::FailureOr<Binding> accumulator = materializeNumeric(
        accumulatorValue, bindings.lookup(accumulatorValue));
    mlir::FailureOr<Binding> productLhs = materializeNumeric(
        product->lhs, bindings.lookup(product->lhs));
    mlir::FailureOr<Binding> productRhs = materializeNumeric(
        product->rhs, bindings.lookup(product->rhs));
    if (mlir::failed(accumulator) || mlir::failed(productLhs) ||
        mlir::failed(productRhs))
      return mlir::failure();
    const int64_t resultParts = vectorPartCount(operation.getResult());
    auto vectorCompatible = [&](mlir::Value value, const Binding &binding) {
      if (binding.kind != Binding::Kind::Vector)
        return false;
      for (int64_t part = 0; part < resultParts; ++part) {
        auto projected = projectPart(value, operation.getResult(), part);
        if (!projected || *projected >= binding.parts.size())
          return false;
      }
      return true;
    };
    const bool lhsScalar = operandForm == "vf-lhs-scalar";
    const bool rhsScalar = operandForm == "vf-rhs-scalar";
    const bool productCompatible =
        operandForm == "vv"
            ? vectorCompatible(product->lhs, *productLhs) &&
                  vectorCompatible(product->rhs, *productRhs)
            : lhsScalar ? productLhs->kind == Binding::Kind::Scalar &&
                              vectorCompatible(product->rhs, *productRhs)
                        : productRhs->kind == Binding::Kind::Scalar &&
                              vectorCompatible(product->lhs, *productLhs);
    if (resultParts <= 0 ||
        !vectorCompatible(accumulatorValue, *accumulator) ||
        !productCompatible)
      return fail(operation,
                  "selected fused accumulation mappings do not agree");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string type = vectorType(operation.getResult());
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string intrinsic =
        instruction == "rvv.vfmacc" ? "vfmacc" : "vfmsac";
    for (int64_t part = 0; part < resultParts; ++part) {
      const size_t accPart = *projectPart(accumulatorValue,
                                          operation.getResult(), part);
      const std::string &acc = accumulator->parts[accPart];
      std::string value = fresh("fma");
      if (operandForm == "vv") {
        const size_t lhsPart =
            *projectPart(product->lhs, operation.getResult(), part);
        const size_t rhsPart =
            *projectPart(product->rhs, operation.getResult(), part);
        line(type + " " + value + " = __riscv_" + intrinsic + "_vv_" + suffix + "(" +
             acc + ", " + productLhs->parts[lhsPart] + ", " +
             productRhs->parts[rhsPart] + ", " +
             partVL(operation.getResult(), part) + ");");
      } else {
        const Binding &scalarOperand = lhsScalar ? *productLhs : *productRhs;
        const Binding &vectorOperand = lhsScalar ? *productRhs : *productLhs;
        mlir::Value vectorValue = lhsScalar ? product->rhs : product->lhs;
        const size_t vectorPart =
            *projectPart(vectorValue, operation.getResult(), part);
        line(type + " " + value + " = __riscv_" + intrinsic + "_vf_" + suffix + "(" +
             acc + ", " + scalarOperand.scalar + ", " +
             vectorOperand.parts[vectorPart] + ", " +
             partVL(operation.getResult(), part) + ");");
      }
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
  mlir::DictionaryAttr lhsUse = useConversion(operation.getOperation(), 0);
  mlir::DictionaryAttr rhsUse = useConversion(operation.getOperation(), 1);
  if (!lhsUse || !rhsUse)
    return fail(operation,
                "pointwise operation has no typed value-use conversions");
  llvm::StringRef lhsLayoutConversion =
      riscv_internal::string(lhsUse, "relation").value_or("");
  llvm::StringRef rhsLayoutConversion =
      riscv_internal::string(rhsUse, "relation").value_or("");
  const bool lhsLaneToRegister =
      lhsLayoutConversion == "lane-to-register";
  const bool rhsLaneToRegister =
      rhsLayoutConversion == "lane-to-register";
  if (lhs.kind == Binding::Kind::Scalar && rhs.kind == Binding::Kind::Scalar) {
    llvm::StringRef kind = operation.getKind();
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
      bindings[operation.getResult()] = scalar(
          "(" + lhs.scalar + " " + spelling.str() + " " + rhs.scalar + ")");
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
    llvm::StringRef kind = operation.getKind();
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
          return extractLaneForRegisterBroadcast(
              operation.getOperation(), operand, value, operation.getResult(),
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
      else
        result.parts.push_back(std::move(expression));
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
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getResult().getType()));
  bool unsignedInteger = integer && integer.isUnsigned();
  llvm::StringRef kind = operation.getKind();
  std::string stem =
      kind == "add" ? (floating ? "vfadd" : "vadd")
      : kind == "sub" ? (floating ? "vfsub" : "vsub")
      : kind == "mul" ? (floating ? "vfmul" : "vmul")
      : kind == "div" ? (floating ? "vfdiv" : unsignedInteger ? "vdivu" : "vdiv")
      : kind == "mod" ? (unsignedInteger ? "vremu" : "vrem")
      : kind == "and" ? "vand"
      : kind == "or" ? "vor"
      : kind == "xor" ? "vxor"
      : kind == "shl" ? "vsll"
      : kind == "shr" ? (unsignedInteger ? "vsrl" : "vsra")
      : kind == "max" ? (floating ? "vfmax" : unsignedInteger ? "vmaxu" : "vmax")
      : kind == "min" ? (floating ? "vfmin" : unsignedInteger ? "vminu" : "vmin")
                        : "unsupported";
  if (stem == "unsupported")
    return fail(operation, "unsupported selected pointwise binary kind");
  const int64_t resultParts = vectorPartCount(operation.getResult());
  auto compatibleParts = [&](size_t operand, mlir::Value value,
                             const Binding &binding, bool laneToRegister) {
    if (binding.kind == Binding::Kind::Scalar)
      return true;
    if (laneToRegister) {
      if (binding.kind != Binding::Kind::Vector)
        return false;
      for (int64_t part = 0; part < resultParts; ++part)
        if (!extractLaneForRegisterBroadcast(
                operation.getOperation(), operand, value,
                operation.getResult(), part, binding))
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
              ? extractLaneForRegisterBroadcast(
                    operation.getOperation(), otherOperand, otherValue,
                    operation.getResult(), index, other)
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
      bool commutative = kind == "add" || kind == "mul" || kind == "and" ||
                         kind == "or" || kind == "xor" || kind == "max" ||
                         kind == "min";
      if (scalarOnLeft && !commutative) {
        std::string splat = "__riscv_" +
                            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
                            suffix + "(" + scalarPart + ", " +
                            partVL(operation.getResult(), index) + ")";
        expression = "__riscv_" + stem + "_vv_" + suffix + "(" + splat +
                     ", " + vectorPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      } else {
        expression = "__riscv_" + stem + (floating ? "_vf_" : "_vx_") +
                     suffix + "(" + vectorPart + ", " + scalarPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      }
    } else {
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
    if (streamPartCount(operation.getResult()) > 1 ||
        riscv_internal::string(assigned(operation.getResult()),
                               "materialization")
                .value_or("") == "rematerialize-per-register-part") {
      result.parts.push_back(std::move(expression));
    } else {
      std::string name = fresh("pointwise");
      line(type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

std::optional<EncodingField>
Emitter::fieldFor(const Binding &fieldBinding) const {
  if (fieldBinding.kind != Binding::Kind::Field)
    return std::nullopt;
  mlir::DictionaryAttr mapping = fieldBinding.field.memoryEdge;
  if (!mapping)
    return std::nullopt;
  auto name = mapping.getAs<mlir::StringAttr>("field");
  auto type = mapping.getAs<mlir::TypeAttr>("field_type");
  auto shape = mapping.getAs<mlir::DenseI64ArrayAttr>("field_shape");
  auto layouts = mapping.getAs<mlir::ArrayAttr>("layout");
  auto offset = mapping.getAs<mlir::IntegerAttr>("bit_offset");
  auto storage = mapping.getAs<mlir::IntegerAttr>("storage_bits");
  if (!name || !type || !shape || !layouts || !offset || !storage)
    return std::nullopt;
  EncodingField result;
  result.name = name.getValue().str();
  result.type = type.getValue();
  result.shape.assign(shape.asArrayRef().begin(), shape.asArrayRef().end());
  result.layouts = layouts;
  result.bitOffset = offset.getInt();
  result.storageBits = storage.getInt();
  return result;
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
  if (fieldBinding.field.index) {
    const Binding &index = bindings.lookup(*fieldBinding.field.index);
    auto encoding = encodings.find(owner.encodingFamily);
    if (encoding == encodings.end())
      return {};
    if (index.kind == Binding::Kind::Vector) {
      llvm::StringRef form =
          riscv_internal::string(fieldBinding.field.memoryEdge, "form")
              .value_or("");
      if ((form != "indexed-gather" &&
           form != "unit-stride-index-window") ||
          owner.interleaveRows != 0 ||
          field->bitOffset % 8 || layoutKind(field->layouts) != "natural" ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32))
        return {};
      mlir::Value indexValue = *fieldBinding.field.index;
      auto indexSEW =
          assigned(indexValue).getAs<mlir::IntegerAttr>("physical_sew");
      auto indexLMUL = assigned(indexValue).getAs<mlir::StringAttr>("lmul");
      auto elementType = scalarCType(field->type);
      if (!indexSEW || !indexLMUL || !elementType)
        return {};
      const std::string indexSuffix =
          "u" + std::to_string(indexSEW.getInt()) +
          indexLMUL.getValue().str();
      const std::string resultSuffix = vectorSuffix(result);
      const std::string resultType = vectorType(result);
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
        if (form == "unit-stride-index-window") {
          std::string first =
              "__riscv_vmv_x_s_" + indexSuffix + "_u" +
              std::to_string(indexSEW.getInt()) + "(" +
              index.parts[*indexPart] + ")";
          std::string expression =
              "__riscv_vle" + std::to_string(logicalWidth) + "_v_" +
              resultSuffix + "((const " + *elementType + " *)(" + record +
              ") + (size_t)(" + first + "), " + partVL(result, part) + ")";
          if (riscv_internal::string(assigned(result), "materialization")
                  .value_or("") == "rematerialize-per-register-part") {
            gathered.parts.push_back(std::move(expression));
          } else {
            std::string loaded = fresh("window");
            line(resultType + " " + loaded + " = " + expression + ";");
            gathered.parts.push_back(std::move(loaded));
          }
          continue;
        }
        std::string offsets = index.parts[*indexPart];
        if (logicalWidth != 8)
          offsets = "__riscv_vmul_vx_" + indexSuffix + "(" + offsets +
                    ", " + std::to_string(logicalWidth / 8) + ", " +
                    partVL(result, part) + ")";
        std::string expression =
            "__riscv_vluxei" + std::to_string(indexSEW.getInt()) + "_v_" +
            resultSuffix + "((const " + *elementType + " *)(" + record +
            "), " + offsets + ", " + partVL(result, part) + ")";
        if (riscv_internal::string(assigned(result), "materialization")
                .value_or("") == "rematerialize-per-register-part") {
          gathered.parts.push_back(std::move(expression));
        } else {
          std::string loaded = fresh("gather");
          line(resultType + " " + loaded + " = " + expression + ";");
          gathered.parts.push_back(std::move(loaded));
        }
      }
      return gathered;
    } else if (index.kind == Binding::Kind::Scalar) {
      logicalIndex = index.scalar;
    } else if (index.kind == Binding::Kind::Point) {
      const int64_t elements = encoding->second.logicalElements;
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
  if (owner.interleaveRows == 0) {
    llvm::StringRef physicalKind =
        riscv_internal::string(assigned(result), "physical_kind").value_or("");
    if (physicalKind.starts_with("rvv") && !field->shape.empty()) {
      if (field->bitOffset % 8 || layoutKind(field->layouts) != "natural" ||
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
        const std::string address =
            owner.recordPointer + " + " + std::to_string(field->bitOffset / 8) +
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
    llvm::StringRef kind = layoutKind(field->layouts);
    if (kind == "joined") {
      auto joined = mlir::cast<mlir::DictionaryAttr>(field->layouts[0]);
      int64_t group = joined.getAs<mlir::IntegerAttr>("size").getInt();
      int64_t fields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
      int64_t lowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
      int64_t role = joined.getAs<mlir::IntegerAttr>("role").getInt();
      llvm::StringRef order =
          joined.getAs<mlir::StringAttr>("order").getValue();
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
      scalarResult.scalar = "weft_load_f32_le(" + owner.recordPointer + " + " +
                            fragment->byte + ")";
    else if (field->type.isF16() && logicalWidth == 16 && naturallyByteAligned)
      scalarResult.scalar = "weft_load_f16_le(" + owner.recordPointer + " + " +
                            fragment->byte + ")";
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

  mlir::DictionaryAttr edge = fieldBinding.field.memoryEdge;
  int64_t rawLMUL =
      riscv_internal::integer(edge, "raw_lmul_eighths").value_or(0);
  if (rawLMUL <= 0)
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
  llvm::StringRef kind = layoutKind(field->layouts);
  if (kind == "joined") {
    auto joined = mlir::cast<mlir::DictionaryAttr>(field->layouts[0]);
    int64_t group = joined.getAs<mlir::IntegerAttr>("size").getInt();
    int64_t fields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
    int64_t lowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
    int64_t role = joined.getAs<mlir::IntegerAttr>("role").getInt();
    llvm::StringRef order =
        joined.getAs<mlir::StringAttr>("order").getValue();
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

mlir::FailureOr<Binding>
Emitter::emitGroupedMacReduction(mlir::Value result, const Binding &mac) {
  if (!mac.sourceOperation) {
    kernel.emitError("intrinsic-C MAC binding lost its source operation");
    return mlir::failure();
  }
  const Binding &lhs = bindings.lookup(mac.lhs);
  const Binding &rhs = bindings.lookup(mac.rhs);
  if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field) {
    mac.sourceOperation->emitError("selected grouped MAC requires two encoded fields");
    return mlir::failure();
  }
  const Binding &lhsOwner = bindings.lookup(lhs.field.owner);
  const Binding &rhsOwner = bindings.lookup(rhs.field.owner);
  auto lhsField = fieldFor(lhs);
  auto rhsField = fieldFor(rhs);
  if (!lhsField || !rhsField || !lhs.field.index || !rhs.field.index)
    return mac.sourceOperation->emitError(
               "selected grouped MAC requires indexed encoded fields"),
           mlir::failure();
  if (mac.group <= 0)
    return mac.sourceOperation->emitError("selected grouped MAC has an invalid group"),
           mlir::failure();
  mlir::DictionaryAttr selected = assigned(mac.sourceOperation);
  auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
  auto instruction = local ? local.getAs<mlir::StringAttr>("instruction")
                           : mlir::StringAttr();
  auto lhsLMULAttr =
      local ? local.getAs<mlir::IntegerAttr>("lhs_lmul_eighths")
            : mlir::IntegerAttr();
  auto partialLMULAttr =
      local ? local.getAs<mlir::IntegerAttr>("partial_lmul_eighths")
            : mlir::IntegerAttr();
  auto partialSEWAttr =
      local ? local.getAs<mlir::IntegerAttr>("partial_sew")
            : mlir::IntegerAttr();
  auto lhsAccess =
      local ? local.getAs<mlir::StringAttr>("lhs_access") : mlir::StringAttr();
  auto lhsGroupAttr =
      local ? local.getAs<mlir::IntegerAttr>("lhs_group_size")
            : mlir::IntegerAttr();
  auto lhsLayerAttr =
      local ? local.getAs<mlir::IntegerAttr>("lhs_layer_size")
            : mlir::IntegerAttr();
  auto lhsLayerOrder =
      local ? local.getAs<mlir::StringAttr>("lhs_layer_order")
            : mlir::StringAttr();
  auto lhsBitOffset =
      local ? local.getAs<mlir::IntegerAttr>("lhs_bit_offset")
            : mlir::IntegerAttr();
  auto rhsAccess =
      local ? local.getAs<mlir::StringAttr>("rhs_access") : mlir::StringAttr();
  auto rhsBitOffset =
      local ? local.getAs<mlir::IntegerAttr>("rhs_bit_offset")
            : mlir::IntegerAttr();
  auto schedule = selected.getAs<mlir::DictionaryAttr>("schedule");
  int64_t unroll =
      schedule ? riscv_internal::integer(schedule, "unroll").value_or(0) : 0;
  int64_t pipelineDepth =
      schedule
          ? riscv_internal::integer(schedule, "pipeline_depth").value_or(0)
          : 0;
  auto loopStructure =
      schedule ? schedule.getAs<mlir::StringAttr>("loop_structure")
               : mlir::StringAttr();
  if (!instruction || instruction.getValue() != "rvv.vwmaccsu" ||
      !lhsLMULAttr || !partialLMULAttr || !partialSEWAttr ||
      partialSEWAttr.getInt() != 16 || unroll <= 0 || pipelineDepth <= 0 ||
      !lhsAccess ||
      lhsAccess.getValue() != "grouped-layered-constant-stride-window" ||
      !lhsGroupAttr || !lhsLayerAttr || !lhsLayerOrder || !lhsBitOffset ||
      !rhsAccess || rhsAccess.getValue() != "natural-unit-stride-window" ||
      !rhsBitOffset || !loopStructure)
    return mac.sourceOperation->emitError(
               "grouped MAC has no complete selected local operation"),
           mlir::failure();
  const Binding &point = bindings.lookup(*lhs.field.index);
  const Binding &rhsPoint = bindings.lookup(*rhs.field.index);
  if (point.kind != Binding::Kind::Point || rhsPoint.kind != Binding::Kind::Point ||
      point.point.axis != rhsPoint.point.axis || point.point.base != rhsPoint.point.base ||
      point.point.active != rhsPoint.point.active)
    return mac.sourceOperation->emitError(
               "grouped MAC operands do not share one logical reduction point"),
           mlir::failure();
  auto encoding = encodings.find(lhsOwner.encodingFamily);
  if (encoding == encodings.end())
    return mac.sourceOperation->emitError("grouped MAC encoding family is unavailable"),
           mlir::failure();
  auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(lhsField->type);
  if (!lhsInteger || lhsInteger.getWidth() == 0 || lhsInteger.getWidth() >= 8 ||
      !lhsInteger.isUnsigned())
    return mac.sourceOperation->emitError(
               "rvv.vwmaccsu grouped MAC requires an unsigned sub-byte lhs"),
           mlir::failure();
  const int64_t storageGroup = lhsGroupAttr.getInt();
  const int64_t storageLayer = lhsLayerAttr.getInt();
  if (storageGroup <= 0 || storageLayer <= 0 ||
      storageGroup % storageLayer != 0)
    return mac.sourceOperation->emitError(
               "selected grouped MAC has an invalid storage grouping"),
           mlir::failure();
  const int64_t storageLayers = storageGroup / storageLayer;
  const bool lowFirst = lhsLayerOrder.getValue() == "lo_first";
  if ((lhsLayerOrder.getValue() != "lo_first" &&
       lhsLayerOrder.getValue() != "hi_first") ||
      storageLayers * lhsInteger.getWidth() > 8 ||
      lhsBitOffset.getInt() != lhsField->bitOffset ||
      rhsBitOffset.getInt() != rhsField->bitOffset ||
      lhsOwner.interleaveRows <= 0 || point.point.physicalExtent <= 0 ||
      point.point.physicalExtent > storageLayer ||
      storageLayer % point.point.physicalExtent != 0)
    return mac.sourceOperation->emitError(
               "selected grouped MAC window is incompatible with its field mapping"),
           mlir::failure();
  const bool doubleBuffered =
      loopStructure.getValue() == "cross-iteration-double-buffer";
  if ((pipelineDepth == 1 &&
       loopStructure.getValue() != "sequential-stream") ||
      (pipelineDepth == 2 && !doubleBuffered) || pipelineDepth > 2)
    return mac.sourceOperation->emitError(
               "selected grouped MAC schedule has no generated loop structure"),
           mlir::failure();

  const int64_t elements = encoding->second.logicalElements;
  const std::string outSuffix = vectorSuffix(result);
  const std::string outType = vectorType(result);
  const std::string outName = fresh("mac_reduce");
  line(outType + " " + outName + " = __riscv_vmv_v_x_" + outSuffix +
       "(0, " + laneVL() + ");");
  const int64_t partialLMUL = partialLMULAttr.getInt();
  const int64_t rawLMUL = lhsLMULAttr.getInt();
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  const std::string partialSuffix = "i16" + lmul(partialLMUL);
  const std::string partialType = "vint16" + lmul(partialLMUL) + "_t";
  const std::string rawSuffix = "u8" + lmul(rawLMUL);
  const std::string rawType = "vuint8" + lmul(rawLMUL) + "_t";
  const std::string logicalBase = fresh("mac_logical_base");
  const std::string active = fresh("mac_active");
  const std::string fullGroups = fresh("mac_full_groups");
  const std::string tail = fresh("mac_tail");
  const std::string layerWithin = fresh("mac_layer_within");
  const std::string physicalLayer = fresh("mac_physical_layer");
  const std::string storageByte = fresh("mac_storage_byte");
  const std::string shift = fresh("mac_shift");
  const std::string qWindow = fresh("mac_q_window");
  const std::string xWindow = fresh("mac_x_window");
  line("const size_t " + logicalBase + " = " + point.point.base + " % " +
       std::to_string(elements) + ";");
  line("const size_t " + active + " = " + point.point.active + ";");
  line("const size_t " + fullGroups + " = " + active + " / " +
       std::to_string(mac.group) + ";");
  line("const size_t " + tail + " = " + active + " % " +
       std::to_string(mac.group) + ";");
  line("const size_t " + layerWithin + " = " + logicalBase + " % " +
       std::to_string(storageGroup) + ";");
  if (lowFirst)
    line("const size_t " + physicalLayer + " = " + layerWithin + " / " +
         std::to_string(storageLayer) + ";");
  else
    line("const size_t " + physicalLayer + " = " +
         std::to_string(storageLayers - 1) + " - " + layerWithin + " / " +
         std::to_string(storageLayer) + ";");
  line("const size_t " + storageByte + " = " +
       std::to_string(lhsBitOffset.getInt() / 8) + " + (" + logicalBase + " / " +
       std::to_string(storageGroup) + ") * " + std::to_string(storageLayer) +
       " + " + layerWithin + " % " + std::to_string(storageLayer) + ";");
  line("const size_t " + shift + " = " + physicalLayer + " * " +
       std::to_string(lhsInteger.getWidth()) + ";");
  line("const uint8_t *" + qWindow + " = " + lhsOwner.recordPointer + " + " +
       storageByte + " * " + std::to_string(lhsOwner.interleaveRows) + ";");
  line("const int8_t *" + xWindow + " = (const int8_t *)(" +
       rhsOwner.recordPointer + " + " +
       std::to_string(rhsBitOffset.getInt() / 8) + " + " + logicalBase + ");");

  using Bank = llvm::SmallVector<llvm::SmallVector<std::string>>;
  auto declareBank = [&](Bank &q, Bank &x, llvm::StringRef prefix) {
    q.resize(unroll);
    x.resize(unroll);
    for (int64_t slot = 0; slot < unroll; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        std::string qName = fresh((prefix + "_q").str());
        std::string xName = fresh((prefix + "_x").str());
        line(rawType + " " + qName + ";");
        line("int8_t " + xName + ";");
        q[slot].push_back(std::move(qName));
        x[slot].push_back(std::move(xName));
      }
  };
  auto decodedLoad = [&](llvm::StringRef qPointer, int64_t linearTerm) {
    return "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" + rawSuffix +
           "(__riscv_vle8_v_" + rawSuffix + "(" + qPointer.str() + " + " +
           std::to_string(linearTerm * lhsOwner.interleaveRows) + ", " + laneVL() +
           "), " + shift + ", " + laneVL() + "), " +
           std::to_string((1u << lhsInteger.getWidth()) - 1) + ", " + laneVL() +
           ")";
  };
  auto loadBank = [&](Bank &q, Bank &x, llvm::StringRef qPointer,
                      llvm::StringRef xPointer) {
    for (int64_t slot = 0; slot < unroll; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        int64_t linearTerm = slot * mac.group + term;
        line(q[slot][term] + " = " + decodedLoad(qPointer, linearTerm) + ";");
        line(x[slot][term] + " = *(" + xPointer.str() + " + " +
             std::to_string(linearTerm) + ");");
      }
  };
  auto beginPartials = [&](int64_t count) {
    llvm::SmallVector<std::string> partials;
    for (int64_t slot = 0; slot < count; ++slot) {
      std::string partial = fresh("group_partial");
      line(partialType + " " + partial + " = __riscv_vmv_v_x_" +
           partialSuffix + "(0, " + laneVL() + ");");
      partials.push_back(std::move(partial));
    }
    return partials;
  };
  auto accumulate = [&](llvm::ArrayRef<std::string> partials, const Bank &q,
                        const Bank &x, int64_t slots) {
    for (int64_t slot = 0; slot < slots; ++slot)
      for (int64_t term = 0; term < mac.group; ++term)
        line(partials[slot] + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
             partials[slot] + ", " + x[slot][term] + ", " + q[slot][term] +
             ", " + laneVL() + ");");
  };
  auto mergePartials = [&](llvm::ArrayRef<std::string> partials) {
    for (llvm::StringRef partial : partials) {
      std::string widened = fresh("partial_wide");
      line(outType + " " + widened + " = __riscv_vwcvt_x_x_v_" + outSuffix +
           "(" + partial.str() + ", " + laneVL() + ");");
      line(outName + " = __riscv_vadd_vv_" + outSuffix + "(" + outName + ", " +
           widened + ", " + laneVL() + ");");
    }
  };
  auto emitImmediate = [&](llvm::StringRef qPointer, llvm::StringRef xPointer,
                           int64_t slots) {
    llvm::SmallVector<std::string> partials = beginPartials(slots);
    for (int64_t slot = 0; slot < slots; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        int64_t linearTerm = slot * mac.group + term;
        std::string q = fresh("q_value");
        line(rawType + " " + q + " = " + decodedLoad(qPointer, linearTerm) +
             ";");
        const std::string x = "*(" + xPointer.str() + " + " +
                              std::to_string(linearTerm) + ")";
        line(partials[slot] + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
             partials[slot] + ", " + x + ", " + q + ", " + laneVL() + ");");
      }
    mergePartials(partials);
  };

  const int64_t chunkTerms = unroll * mac.group;
  const int64_t chunkQBytes = chunkTerms * lhsOwner.interleaveRows;
  const std::string fullChunks = fresh("mac_full_chunks");
  line("const size_t " + fullChunks + " = " + fullGroups + " / " +
       std::to_string(unroll) + ";");

  if (doubleBuffered) {
    line("if (" + fullChunks + " != 0) {");
    ++indent;
    Bank currentQ;
    Bank currentX;
    Bank nextQ;
    Bank nextX;
    declareBank(currentQ, currentX, "current");
    declareBank(nextQ, nextX, "next");
    loadBank(currentQ, currentX, qWindow, xWindow);
    const std::string nextQPointer = fresh("mac_next_q");
    const std::string nextXPointer = fresh("mac_next_x");
    line("const uint8_t *" + nextQPointer + " = " + qWindow + " + " +
         std::to_string(chunkQBytes) + ";");
    line("const int8_t *" + nextXPointer + " = " + xWindow + " + " +
         std::to_string(chunkTerms) + ";");
    const std::string chunk = fresh("mac_chunk");
    line("#pragma GCC unroll 1");
    line("for (size_t " + chunk + " = 1; " + chunk + " < " + fullChunks +
         "; ++" + chunk + ") {");
    ++indent;
    llvm::SmallVector<std::string> partials = beginPartials(unroll);
    for (int64_t slot = 0; slot < unroll; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        int64_t linearTerm = slot * mac.group + term;
        line(nextQ[slot][term] + " = " +
             decodedLoad(nextQPointer, linearTerm) + ";");
        line(nextX[slot][term] + " = *(" + nextXPointer + " + " +
             std::to_string(linearTerm) + ");");
        line(partials[slot] + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
             partials[slot] + ", " + currentX[slot][term] + ", " +
             currentQ[slot][term] + ", " + laneVL() + ");");
      }
    mergePartials(partials);
    for (int64_t slot = 0; slot < unroll; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        line(currentQ[slot][term] + " = " + nextQ[slot][term] + ";");
        line(currentX[slot][term] + " = " + nextX[slot][term] + ";");
      }
    line(nextQPointer + " += " + std::to_string(chunkQBytes) + ";");
    line(nextXPointer + " += " + std::to_string(chunkTerms) + ";");
    --indent;
    line("}");
    llvm::SmallVector<std::string> epilogue = beginPartials(unroll);
    accumulate(epilogue, currentQ, currentX, unroll);
    mergePartials(epilogue);
    --indent;
    line("}");
  } else {
    const std::string chunkQPointer = fresh("mac_chunk_q");
    const std::string chunkXPointer = fresh("mac_chunk_x");
    line("const uint8_t *" + chunkQPointer + " = " + qWindow + ";");
    line("const int8_t *" + chunkXPointer + " = " + xWindow + ";");
    const std::string chunk = fresh("mac_chunk");
    line("#pragma GCC unroll 1");
    line("for (size_t " + chunk + " = 0; " + chunk + " < " + fullChunks +
         "; ++" + chunk + ") {");
    ++indent;
    emitImmediate(chunkQPointer, chunkXPointer, unroll);
    line(chunkQPointer + " += " + std::to_string(chunkQBytes) + ";");
    line(chunkXPointer + " += " + std::to_string(chunkTerms) + ";");
    --indent;
    line("}");
  }

  const std::string processedGroups = fresh("mac_processed_groups");
  const std::string remainingGroups = fresh("mac_remaining_groups");
  const std::string remainderQ = fresh("mac_remainder_q");
  const std::string remainderX = fresh("mac_remainder_x");
  line("const size_t " + processedGroups + " = " + fullChunks + " * " +
       std::to_string(unroll) + ";");
  line("size_t " + remainingGroups + " = " + fullGroups + " - " +
       processedGroups + ";");
  line("const uint8_t *" + remainderQ + " = " + qWindow + " + " +
       processedGroups + " * " +
       std::to_string(mac.group * lhsOwner.interleaveRows) + ";");
  line("const int8_t *" + remainderX + " = " + xWindow + " + " +
       processedGroups + " * " + std::to_string(mac.group) + ";");
  line("while (" + remainingGroups + " != 0) {");
  ++indent;
  emitImmediate(remainderQ, remainderX, 1);
  line(remainderQ + " += " +
       std::to_string(mac.group * lhsOwner.interleaveRows) + ";");
  line(remainderX + " += " + std::to_string(mac.group) + ";");
  line("--" + remainingGroups + ";");
  --indent;
  line("}");

  line("if (" + tail + " != 0) {");
  ++indent;
  llvm::SmallVector<std::string> tailPartial = beginPartials(1);
  const std::string tailQ = fresh("mac_tail_q");
  const std::string tailX = fresh("mac_tail_x");
  line("const uint8_t *" + tailQ + " = " + qWindow + " + " + fullGroups +
       " * " + std::to_string(mac.group * lhsOwner.interleaveRows) + ";");
  line("const int8_t *" + tailX + " = " + xWindow + " + " + fullGroups +
       " * " + std::to_string(mac.group) + ";");
  for (int64_t term = 0; term < mac.group; ++term) {
    line("if (" + std::to_string(term) + " < " + tail + ") {");
    ++indent;
    std::string q = fresh("q_tail_value");
    line(rawType + " " + q + " = " + decodedLoad(tailQ, term) + ";");
    line(tailPartial.front() + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
         tailPartial.front() + ", *(" + tailX + " + " + std::to_string(term) +
         "), " + q + ", " + laneVL() + ");");
    --indent;
    line("}");
  }
  mergePartials(tailPartial);
  --indent;
  line("}");
  Binding resultBinding;
  resultBinding.kind = Binding::Kind::Vector;
  resultBinding.parts.push_back(outName);
  return resultBinding;
}

mlir::FailureOr<Binding>
Emitter::emitQuantDot(kernel::DotOp operation, const Binding &lhs,
                      const Binding &rhs) {
  Binding resultBinding;
  if (lhs.kind != Binding::Kind::Field ||
      rhs.kind != Binding::Kind::DeferredFold)
    return operation.emitError(
               "selected encoded dot requires one field and one ordered pair fold"),
           mlir::failure();
  mlir::DictionaryAttr selected = assigned(operation.getOperation());
  auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
  auto instruction = local ? local.getAs<mlir::StringAttr>("instruction")
                           : mlir::StringAttr();
  auto accumulatorSEW =
      local ? local.getAs<mlir::IntegerAttr>("accumulator_sew")
            : mlir::IntegerAttr();
  auto accumulatorLMUL =
      local ? local.getAs<mlir::IntegerAttr>("accumulator_lmul_eighths")
            : mlir::IntegerAttr();
  auto groupCount = local ? local.getAs<mlir::IntegerAttr>("group_count")
                          : mlir::IntegerAttr();
  auto rhsPairs =
      local ? local.getAs<mlir::IntegerAttr>("rhs_pairs_per_group")
            : mlir::IntegerAttr();
  if (!instruction || instruction.getValue() != "rvv.vmacc" ||
      !accumulatorSEW || accumulatorSEW.getInt() != 32 || !accumulatorLMUL ||
      !groupCount || !rhsPairs)
    return operation.emitError(
               "selected encoded dot has no complete local operation"),
           mlir::failure();
  const Binding &lhsOwner = bindings.lookup(lhs.field.owner);
  const Binding &foldInput = bindings.lookup(rhs.input);
  if (foldInput.kind != Binding::Kind::Field)
    return operation.emitError("selected ordered pair fold lost its encoded field"),
           mlir::failure();
  const Binding &rhsOwner = bindings.lookup(foldInput.field.owner);
  auto lhsField = fieldFor(lhs);
  auto rhsField = fieldFor(foldInput);
  if (!lhsField || !rhsField)
    return operation.emitError("selected encoded dot has incomplete memory edges"),
           mlir::failure();
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  const std::string i32Suffix = "i32" + lmul(accumulatorLMUL.getInt());
  const std::string i32Type =
      "vint32" + lmul(accumulatorLMUL.getInt()) + "_t";
  std::string acc = fresh("min_acc");
  line(i32Type + " " + acc + " = __riscv_vmv_v_x_" + i32Suffix + "(0, " +
       laneVL() + ");");
  std::string index = fresh("min_group");
  line("for (size_t " + index + " = 0; " + index + " < " +
       std::to_string(groupCount.getInt()) + "; ++" + index + ") {");
  ++indent;
  Binding indexed = lhs;
  indexed.field.index.reset();
  Binding m8 = emitInterleavedField(operation.getLhs(), indexed, index);
  if (m8.kind != Binding::Kind::Vector) {
    --indent;
    line("}");
    return operation.emitError(
               "selected encoded dot could not materialize its lhs field mapping"),
           mlir::failure();
  }
  std::string m32 = fresh("min_scale");
  std::string u32Suffix = i32Suffix;
  u32Suffix[0] = 'u';
  line(i32Type + " " + m32 + " = __riscv_vreinterpret_v_" + u32Suffix + "_" +
       i32Suffix + "(__riscv_vzext_vf4_" + u32Suffix + "(" + m8.parts.front() +
       ", " + laneVL() + "));" );
  const std::string pairCount = std::to_string(rhsPairs.getInt());
  const std::string bsum =
      "((int32_t)weft_load_i16_le(" + rhsOwner.recordPointer + " + " +
      std::to_string(rhsField->bitOffset / 8) + " + 2 * (" + pairCount + " * " +
      index + ")) + (int32_t)weft_load_i16_le(" + rhsOwner.recordPointer + " + " +
      std::to_string(rhsField->bitOffset / 8) + " + 2 * (" + pairCount + " * " +
      index + " + 1)))";
  line(acc + " = __riscv_vmacc_vx_" + i32Suffix + "(" + acc + ", " + bsum +
       ", " + m32 + ", " + laneVL() + ");");
  --indent;
  line("}");
  std::string converted = fresh("min_term");
  line(vectorType(operation.getResult()) + " " + converted +
       " = __riscv_vfcvt_f_x_v_" + vectorSuffix(operation.getResult()) + "(" +
       acc + ", " + laneVL() + ");");
  resultBinding.kind = Binding::Kind::Vector;
  resultBinding.parts.push_back(std::move(converted));
  return resultBinding;
}

mlir::FailureOr<Binding> Emitter::emitEncodedOuterMac(
    mlir::Operation &operation, mlir::Value lhsValue, mlir::Value rhsValue,
    mlir::Value resultValue, const Binding &lhsBinding,
    const Binding &rhsBinding) {
  mlir::DictionaryAttr selected = assigned(&operation);
  auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
  llvm::StringRef instruction =
      riscv_internal::string(local, "instruction").value_or("");
  llvm::StringRef laneOperand =
      riscv_internal::string(local, "lane_operand").value_or("");
  int64_t partialLMUL =
      riscv_internal::integer(local, "partial_lmul_eighths").value_or(0);
  int64_t partialTerms =
      riscv_internal::integer(local, "partial_terms").value_or(0);
  int64_t accumulatorParts =
      riscv_internal::integer(local, "accumulator_parts").value_or(0);
  int64_t accumulatorAxis =
      riscv_internal::integer(local, "accumulator_axis").value_or(0);
  int64_t reductionAxis =
      riscv_internal::integer(local, "reduction_axis").value_or(0);
  llvm::StringRef laneReductionAccess =
      riscv_internal::string(local, "lane_reduction_access").value_or("");
  llvm::StringRef repeatedReductionAccess =
      riscv_internal::string(local, "repeated_reduction_access").value_or("");
  auto schedule = selected.getAs<mlir::DictionaryAttr>("schedule");
  int64_t pipelineDepth =
      schedule ? riscv_internal::integer(schedule, "pipeline_depth").value_or(0)
               : 0;
  llvm::StringRef loopStructure =
      schedule ? riscv_internal::string(schedule, "loop_structure").value_or("")
               : llvm::StringRef();
  int64_t operandBufferCount =
      riscv_internal::integer(local, "operand_buffer_count").value_or(0);
  llvm::StringRef laneAccess =
      riscv_internal::string(local, "lane_access").value_or("");
  if (instruction != "rvv.vwmaccsu" ||
      (laneOperand != "lhs" && laneOperand != "rhs") || partialLMUL <= 0 ||
      partialTerms <= 0 || accumulatorParts <= 0 || accumulatorAxis <= 0 ||
      reductionAxis <= 0 ||
      (laneReductionAccess != "indexed-point" &&
       laneReductionAccess != "enclosing-level-window") ||
      (repeatedReductionAccess != "indexed-point" &&
       repeatedReductionAccess != "enclosing-level-window") ||
      pipelineDepth <= 0 || operandBufferCount <= 0 ||
      (pipelineDepth == 1 && loopStructure != "sequential-stream") ||
      (pipelineDepth == 2 &&
       loopStructure != "cross-iteration-double-buffer") ||
      pipelineDepth > 2 || operandBufferCount != pipelineDepth)
    return operation.emitError(
               "selected encoded outer MAC has no complete local operation"),
           mlir::failure();

  const bool laneIsLhs = laneOperand == "lhs";
  mlir::Value laneValue = laneIsLhs ? lhsValue : rhsValue;
  const Binding &lane = laneIsLhs ? lhsBinding : rhsBinding;
  const Binding &repeated = laneIsLhs ? rhsBinding : lhsBinding;
  if (lane.kind != Binding::Kind::Field ||
      repeated.kind != Binding::Kind::Field)
    return operation.emitError(
               "selected encoded outer MAC requires two encoded fields"),
           mlir::failure();
  const Binding &laneOwner = bindings.lookup(lane.field.owner);
  const Binding &repeatedOwner = bindings.lookup(repeated.field.owner);
  auto laneField = fieldFor(lane);
  auto repeatedField = fieldFor(repeated);
  if (laneOwner.kind != Binding::Kind::Record ||
      repeatedOwner.kind != Binding::Kind::Record || !laneField ||
      !repeatedField)
    return operation.emitError(
               "selected encoded outer MAC has incomplete record mappings"),
           mlir::failure();
  auto reductionPointFor = [&](const Binding &field,
                               llvm::StringRef access)
      -> std::optional<PointInfo> {
    if (access == "indexed-point") {
      if (!field.field.index)
        return std::nullopt;
      const Binding &point = bindings.lookup(*field.field.index);
      if (point.kind != Binding::Kind::Point ||
          point.point.axis != reductionAxis)
        return std::nullopt;
      return point.point;
    }
    auto scope = axisScopes.find(reductionAxis);
    if (access != "enclosing-level-window" || scope == axisScopes.end() ||
        scope->second.empty())
      return std::nullopt;
    return scope->second.back();
  };
  std::optional<PointInfo> lanePoint =
      reductionPointFor(lane, laneReductionAccess);
  std::optional<PointInfo> repeatedPoint =
      reductionPointFor(repeated, repeatedReductionAccess);
  if (!lanePoint || !repeatedPoint || lanePoint->axis != repeatedPoint->axis ||
      lanePoint->base != repeatedPoint->base ||
      lanePoint->active != repeatedPoint->active)
    return operation.emitError(
               "selected encoded outer MAC operands do not share one reduction point"),
           mlir::failure();
  const PointInfo &reductionPoint = *lanePoint;
  auto encoding = encodings.find(laneOwner.encodingFamily);
  if (encoding == encodings.end())
    return operation.emitError(
               "selected encoded outer MAC has no lane encoding"),
           mlir::failure();
  auto repeatedInteger = mlir::dyn_cast<mlir::IntegerType>(repeatedField->type);
  auto laneInteger = mlir::dyn_cast<mlir::IntegerType>(laneField->type);
  if (!laneInteger || !laneInteger.isUnsigned() || laneInteger.getWidth() == 0 ||
      laneInteger.getWidth() >= 8 || !repeatedInteger ||
      !repeatedInteger.isSigned() ||
      repeatedInteger.getWidth() != 8 ||
      layoutKind(repeatedField->layouts) != "natural" ||
      repeatedField->bitOffset % 8)
    return operation.emitError(
               "selected encoded outer MAC scalar field is not natural signed i8"),
           mlir::failure();

  mlir::FailureOr<Binding> resultOr =
      makeVector(resultValue, "outer_acc", scalar("0"));
  if (mlir::failed(resultOr))
    return mlir::failure();
  Binding result = std::move(*resultOr);
  if (static_cast<int64_t>(result.parts.size()) != accumulatorParts)
    return operation.emitError(
               "selected encoded outer MAC accumulator part count disagrees with its value representation"),
           mlir::failure();
  llvm::SmallVector<int64_t, 4> accumulatorAxes =
      registerAxesFor(resultValue);
  if ((accumulatorParts == 1 && !accumulatorAxes.empty() &&
       (accumulatorAxes.size() != 1 ||
        accumulatorAxes.front() != accumulatorAxis)) ||
      (accumulatorParts > 1 &&
       (accumulatorAxes.size() != 1 ||
        accumulatorAxes.front() != accumulatorAxis)))
    return operation.emitError(
               "selected encoded outer MAC requires one explicit accumulator axis"),
           mlir::failure();

  std::string repeatedStride;
  for (const auto &[axis, stride] : repeatedOwner.recordByteStrides)
    if (axis == accumulatorAxis)
      repeatedStride = stride;
  if (accumulatorParts > 1 && repeatedStride.empty())
    return operation.emitError(
               "selected encoded outer MAC has no repeated-record byte stride"),
           mlir::failure();
  if (repeatedStride.empty())
    repeatedStride = "0";

  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  const std::string partialSuffix = "i16" + lmul(partialLMUL);
  const std::string partialType = "vint16" + lmul(partialLMUL) + "_t";
  const std::string resultSuffix = vectorSuffix(resultValue);
  const std::string resultType = vectorType(resultValue);
  const std::string operationVL = partVL(resultValue, 0);
  auto accumulatorScope = axisScopes.find(accumulatorAxis);
  const std::string accumulatorActive =
      accumulatorScope == axisScopes.end() || accumulatorScope->second.empty()
          ? std::to_string(accumulatorParts)
          : accumulatorScope->second.back().active;
  Binding laneFieldBinding = lane;
  laneFieldBinding.field.index.reset();
  const bool windowed =
      laneAccess == "grouped-layered-constant-stride-window";
  std::string qWindow;
  std::string qShift;
  llvm::SmallVector<std::string> repeatedWindows;
  if (windowed) {
    int64_t groupSize =
        riscv_internal::integer(local, "lane_group_size").value_or(0);
    int64_t layerSize =
        riscv_internal::integer(local, "lane_layer_size").value_or(0);
    llvm::StringRef layerOrder =
        riscv_internal::string(local, "lane_layer_order").value_or("");
    int64_t laneBitOffset =
        riscv_internal::integer(local, "lane_bit_offset").value_or(-1);
    int64_t repeatedBitOffset =
        riscv_internal::integer(local, "repeated_bit_offset").value_or(-1);
    if (groupSize <= 0 || layerSize <= 0 || groupSize % layerSize ||
        (layerOrder != "lo_first" && layerOrder != "hi_first") ||
        laneBitOffset < 0 || laneBitOffset % 8 || repeatedBitOffset < 0 ||
        repeatedBitOffset % 8 || laneOwner.interleaveRows <= 0 ||
        reductionPoint.physicalExtent <= 0 ||
        reductionPoint.physicalExtent > layerSize ||
        layerSize % reductionPoint.physicalExtent)
      return operation.emitError(
                 "selected encoded outer MAC window is incompatible with its field mapping"),
             mlir::failure();
    const int64_t layers = groupSize / layerSize;
    const std::string logicalBase = fresh("outer_logical_base");
    const std::string layerWithin = fresh("outer_layer_within");
    const std::string physicalLayer = fresh("outer_physical_layer");
    const std::string storageByte = fresh("outer_storage_byte");
    qShift = fresh("outer_shift");
    qWindow = fresh("outer_q_window");
    line("const size_t " + logicalBase + " = " + reductionPoint.base + " % " +
         std::to_string(encoding->second.logicalElements) + ";");
    line("const size_t " + layerWithin + " = " + logicalBase + " % " +
         std::to_string(groupSize) + ";");
    if (layerOrder == "lo_first")
      line("const size_t " + physicalLayer + " = " + layerWithin + " / " +
           std::to_string(layerSize) + ";");
    else
      line("const size_t " + physicalLayer + " = " +
           std::to_string(layers - 1) + " - " + layerWithin + " / " +
           std::to_string(layerSize) + ";");
    line("const size_t " + storageByte + " = " +
         std::to_string(laneBitOffset / 8) + " + (" + logicalBase + " / " +
         std::to_string(groupSize) + ") * " + std::to_string(layerSize) +
         " + " + layerWithin + " % " + std::to_string(layerSize) + ";");
    line("const size_t " + qShift + " = " + physicalLayer + " * " +
         std::to_string(laneInteger.getWidth()) + ";");
    line("const uint8_t *" + qWindow + " = " + laneOwner.recordPointer + " + " +
         storageByte + " * " + std::to_string(laneOwner.interleaveRows) + ";");
    for (int64_t part = 0; part < accumulatorParts; ++part) {
      std::string window = fresh("outer_x_window");
      line("const int8_t *" + window + " = (const int8_t *)((" +
           repeatedOwner.recordPointer + " + " + std::to_string(part) + " * " +
           repeatedStride + ") + " + std::to_string(repeatedBitOffset / 8) +
           " + " + logicalBase + ");");
      repeatedWindows.push_back(std::move(window));
    }
  } else if (laneAccess != "mapped-field-per-term") {
    return operation.emitError(
               "selected encoded outer MAC has an unknown lane access"),
           mlir::failure();
  }
  auto logicalIndex = [&](llvm::StringRef group, int64_t term) {
    return "((" + reductionPoint.base + " % " +
           std::to_string(encoding->second.logicalElements) + ") + " +
           group.str() + " + " + std::to_string(term) + ")";
  };
  auto laneVector = [&](llvm::StringRef group, int64_t term)
      -> mlir::FailureOr<std::string> {
    if (windowed) {
      const std::string suffix = vectorSuffix(laneValue);
      std::string loaded = fresh("outer_q");
      const std::string offset = "(" + group.str() + " + " +
                                 std::to_string(term) + ") * " +
                                 std::to_string(laneOwner.interleaveRows);
      line(vectorType(laneValue) + " " + loaded + " = __riscv_vand_vx_" +
           suffix + "(__riscv_vsrl_vx_" + suffix +
           "(__riscv_vle8_v_" + suffix + "(" + qWindow + " + " + offset +
           ", " + operationVL + "), " + qShift + ", " + operationVL + "), " +
           std::to_string((1u << laneInteger.getWidth()) - 1) + ", " +
           operationVL + ");");
      return loaded;
    }
    Binding loaded = emitInterleavedField(
        laneValue, laneFieldBinding, logicalIndex(group, term));
    if (loaded.kind != Binding::Kind::Vector || loaded.parts.size() != 1)
      return operation.emitError(
                 "selected encoded outer MAC could not materialize its lane field"),
             mlir::failure();
    return loaded.parts.front();
  };
  auto repeatedScalar = [&](int64_t part, llvm::StringRef group,
                            int64_t term) -> std::optional<std::string> {
    if (windowed)
      return "*(" + repeatedWindows[part] + " + " + group.str() + " + " +
             std::to_string(term) + ")";
    auto fragment = singleStorageFragment(
        *repeatedField, logicalIndex(group, term),
        static_cast<unsigned>(repeatedInteger.getWidth()));
    if (!fragment)
      return std::nullopt;
    const std::string record =
        "(" + repeatedOwner.recordPointer + " + " + std::to_string(part) +
        " * " + repeatedStride + ")";
    return "*(const int8_t *)(" + record + " + " + fragment->byte + ")";
  };
  auto beginPartials = [&]() {
    llvm::SmallVector<std::string> partials;
    for (int64_t part = 0; part < accumulatorParts; ++part) {
      std::string partial = fresh("outer_partial");
      line(partialType + " " + partial + " = __riscv_vmv_v_x_" +
           partialSuffix + "(0, " + operationVL + ");");
      partials.push_back(std::move(partial));
    }
    return partials;
  };
  auto mergePartials = [&](llvm::ArrayRef<std::string> partials) {
    for (int64_t part = 0; part < accumulatorParts; ++part) {
      std::string widened = fresh("outer_wide");
      line(resultType + " " + widened + " = __riscv_vwcvt_x_x_v_" +
           resultSuffix + "(" + partials[part] + ", " + operationVL + ");");
      line(result.parts[part] + " = __riscv_vadd_vv_" + resultSuffix + "(" +
           result.parts[part] + ", " + widened + ", " + operationVL + ");");
    }
  };
  auto emitImmediate = [&](llvm::StringRef group, bool guarded) {
    llvm::SmallVector<std::string> partials = beginPartials();
    for (int64_t term = 0; term < partialTerms; ++term) {
      if (guarded) {
        line("if (" + group.str() + " + " + std::to_string(term) + " < " +
             reductionPoint.active + ") {");
        ++indent;
      }
      mlir::FailureOr<std::string> q = laneVector(group, term);
      if (mlir::failed(q))
        return mlir::failure();
      for (int64_t part = 0; part < accumulatorParts; ++part) {
        auto x = repeatedScalar(part, group, term);
        if (!x)
          return operation.emitError(
                     "selected encoded outer MAC scalar field has no byte-aligned mapping"),
                 mlir::failure();
        const std::string statement =
            partials[part] + " = __riscv_vwmaccsu_vx_" + partialSuffix +
            "(" + partials[part] + ", " + *x + ", " + *q + ", " + operationVL +
            ");";
        line("if (" + std::to_string(part) + " < " + accumulatorActive + ") " +
             statement);
      }
      if (guarded) {
        --indent;
        line("}");
      }
    }
    mergePartials(partials);
    return mlir::success();
  };

  const std::string group = fresh("outer_k_group");
  line("size_t " + group + " = 0;");
  if (pipelineDepth == 2) {
    using ScalarBank = llvm::SmallVector<llvm::SmallVector<std::string>>;
    const std::string laneType = vectorType(laneValue);
    llvm::SmallVector<std::string> currentQ;
    llvm::SmallVector<std::string> nextQ;
    ScalarBank currentX(accumulatorParts);
    ScalarBank nextX(accumulatorParts);
    for (int64_t term = 0; term < partialTerms; ++term) {
      currentQ.push_back(fresh("outer_current_q"));
      nextQ.push_back(fresh("outer_next_q"));
      line(laneType + " " + currentQ.back() + ";");
      line(laneType + " " + nextQ.back() + ";");
      for (int64_t part = 0; part < accumulatorParts; ++part) {
        currentX[part].push_back(fresh("outer_current_x"));
        nextX[part].push_back(fresh("outer_next_x"));
        line("int8_t " + currentX[part].back() + ";");
        line("int8_t " + nextX[part].back() + ";");
      }
    }
    line("if (" + reductionPoint.active + " >= " +
         std::to_string(partialTerms) + ") {");
    ++indent;
    for (int64_t term = 0; term < partialTerms; ++term) {
      mlir::FailureOr<std::string> q = laneVector("0", term);
      if (mlir::failed(q))
        return mlir::failure();
      line(currentQ[term] + " = " + *q + ";");
      for (int64_t part = 0; part < accumulatorParts; ++part) {
        auto x = repeatedScalar(part, "0", term);
        if (!x)
          return mlir::failure();
        line(currentX[part][term] + " = " + *x + ";");
      }
    }
    line(group + " = " + std::to_string(partialTerms) + ";");
    line("#pragma GCC unroll 1");
    line("for (; " + group + " + " + std::to_string(partialTerms) + " <= " +
         reductionPoint.active + "; " + group + " += " +
         std::to_string(partialTerms) + ") {");
    ++indent;
    llvm::SmallVector<std::string> partials = beginPartials();
    for (int64_t term = 0; term < partialTerms; ++term) {
      mlir::FailureOr<std::string> q = laneVector(group, term);
      if (mlir::failed(q))
        return mlir::failure();
      line(nextQ[term] + " = " + *q + ";");
      for (int64_t part = 0; part < accumulatorParts; ++part) {
        auto x = repeatedScalar(part, group, term);
        if (!x)
          return mlir::failure();
        line(nextX[part][term] + " = " + *x + ";");
        const std::string statement =
            partials[part] + " = __riscv_vwmaccsu_vx_" + partialSuffix +
            "(" + partials[part] + ", " + currentX[part][term] + ", " +
            currentQ[term] + ", " + operationVL + ");";
        line("if (" + std::to_string(part) + " < " + accumulatorActive + ") " +
             statement);
      }
    }
    mergePartials(partials);
    for (int64_t term = 0; term < partialTerms; ++term) {
      line(currentQ[term] + " = " + nextQ[term] + ";");
      for (int64_t part = 0; part < accumulatorParts; ++part)
        line(currentX[part][term] + " = " + nextX[part][term] + ";");
    }
    --indent;
    line("}");
    llvm::SmallVector<std::string> epilogue = beginPartials();
    for (int64_t term = 0; term < partialTerms; ++term)
      for (int64_t part = 0; part < accumulatorParts; ++part) {
        const std::string statement =
            epilogue[part] + " = __riscv_vwmaccsu_vx_" + partialSuffix +
            "(" + epilogue[part] + ", " + currentX[part][term] + ", " +
            currentQ[term] + ", " + operationVL + ");";
        line("if (" + std::to_string(part) + " < " + accumulatorActive + ") " +
             statement);
      }
    mergePartials(epilogue);
    --indent;
    line("}");
  }
  line("#pragma GCC unroll 1");
  line("for (; " + group + " + " + std::to_string(partialTerms) + " <= " +
       reductionPoint.active + "; " + group + " += " +
       std::to_string(partialTerms) + ") {");
  ++indent;
  if (mlir::failed(emitImmediate(group, false)))
    return mlir::failure();
  --indent;
  line("}");
  if (partialTerms > 1) {
    line("for (; " + group + " < " + reductionPoint.active + "; " + group +
         " += " + std::to_string(partialTerms) + ") {");
    ++indent;
    if (mlir::failed(emitImmediate(group, true)))
      return mlir::failure();
    --indent;
    line("}");
  }
  return result;
}

mlir::FailureOr<Binding> Emitter::emitEncodedOuterFold(
    mlir::Operation &operation, mlir::Value lhsValue, mlir::Value rhsValue,
    mlir::Value resultValue, const Binding &lhsBinding,
    const Binding &rhsBinding) {
  mlir::DictionaryAttr selected = assigned(&operation);
  auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
  llvm::StringRef instruction =
      riscv_internal::string(local, "instruction").value_or("");
  llvm::StringRef laneOperand =
      riscv_internal::string(local, "lane_operand").value_or("");
  int64_t groupCount =
      riscv_internal::integer(local, "group_count").value_or(0);
  int64_t pairsPerGroup =
      riscv_internal::integer(local, "rhs_pairs_per_group").value_or(0);
  int64_t accumulatorParts =
      riscv_internal::integer(local, "accumulator_parts").value_or(0);
  int64_t accumulatorAxis =
      riscv_internal::integer(local, "accumulator_axis").value_or(0);
  if (instruction != "rvv.vmacc" ||
      (laneOperand != "lhs" && laneOperand != "rhs") || groupCount <= 0 ||
      pairsPerGroup <= 0 || accumulatorParts <= 0 || accumulatorAxis <= 0)
    return operation.emitError(
               "selected encoded outer fold product has no complete local operation"),
           mlir::failure();

  const bool laneIsLhs = laneOperand == "lhs";
  mlir::Value laneValue = laneIsLhs ? lhsValue : rhsValue;
  const Binding &lane = laneIsLhs ? lhsBinding : rhsBinding;
  const Binding &fold = laneIsLhs ? rhsBinding : lhsBinding;
  if (lane.kind != Binding::Kind::Field ||
      fold.kind != Binding::Kind::DeferredFold)
    return operation.emitError(
               "selected encoded outer fold product requires one field and one ordered pair fold"),
           mlir::failure();
  const Binding &laneOwner = bindings.lookup(lane.field.owner);
  const Binding &foldInput = bindings.lookup(fold.input);
  if (laneOwner.kind != Binding::Kind::Record ||
      foldInput.kind != Binding::Kind::Field)
    return operation.emitError(
               "selected encoded outer fold product lost one encoded record"),
           mlir::failure();
  const Binding &foldOwner = bindings.lookup(foldInput.field.owner);
  auto laneField = fieldFor(lane);
  auto foldField = fieldFor(foldInput);
  if (foldOwner.kind != Binding::Kind::Record || !laneField || !foldField ||
      foldField->bitOffset % 8 || layoutKind(foldField->layouts) != "natural")
    return operation.emitError(
               "selected encoded outer fold product has incomplete field mappings"),
           mlir::failure();

  mlir::FailureOr<Binding> resultOr =
      makeVector(resultValue, "outer_min_acc", scalar("0"));
  if (mlir::failed(resultOr))
    return mlir::failure();
  Binding result = std::move(*resultOr);
  if (static_cast<int64_t>(result.parts.size()) != accumulatorParts)
    return operation.emitError(
               "selected encoded outer fold product accumulator part count disagrees with its value representation"),
           mlir::failure();
  llvm::SmallVector<int64_t, 4> accumulatorAxes =
      registerAxesFor(resultValue);
  if (accumulatorAxes.size() != 1 ||
      accumulatorAxes.front() != accumulatorAxis)
    return operation.emitError(
               "selected encoded outer fold product requires one explicit accumulator axis"),
           mlir::failure();
  std::string foldStride;
  for (const auto &[axis, stride] : foldOwner.recordByteStrides)
    if (axis == accumulatorAxis)
      foldStride = stride;
  if (accumulatorParts > 1 && foldStride.empty())
    return operation.emitError(
               "selected encoded outer fold product has no repeated-record byte stride"),
           mlir::failure();
  if (foldStride.empty())
    foldStride = "0";

  auto accumulatorScope = axisScopes.find(accumulatorAxis);
  const std::string accumulatorActive =
      accumulatorScope == axisScopes.end() || accumulatorScope->second.empty()
          ? std::to_string(accumulatorParts)
          : accumulatorScope->second.back().active;
  const std::string resultSuffix = vectorSuffix(resultValue);
  std::string unsignedSuffix = resultSuffix;
  unsignedSuffix.front() = 'u';
  Binding laneFieldBinding = lane;
  laneFieldBinding.field.index.reset();
  const std::string group = fresh("outer_fold_group");
  line("#pragma GCC unroll 1");
  line("for (size_t " + group + " = 0; " + group + " < " +
       std::to_string(groupCount) + "; ++" + group + ") {");
  ++indent;
  Binding laneVector =
      emitInterleavedField(laneValue, laneFieldBinding, group);
  if (laneVector.kind != Binding::Kind::Vector ||
      laneVector.parts.size() != 1)
    return operation.emitError(
               "selected encoded outer fold product could not materialize its lane field"),
           mlir::failure();
  std::string laneI32 = fresh("outer_min_scale");
  line(vectorType(resultValue) + " " + laneI32 +
       " = __riscv_vreinterpret_v_" + unsignedSuffix + "_" +
       resultSuffix + "(__riscv_vzext_vf4_" + unsignedSuffix + "(" +
       laneVector.parts.front() + ", " + laneVL() + "));" );
  for (int64_t part = 0; part < accumulatorParts; ++part) {
    const std::string record =
        "(" + foldOwner.recordPointer + " + " + std::to_string(part) +
        " * " + foldStride + ")";
    const std::string pair =
        "((int32_t)weft_load_i16_le(" + record + " + " +
        std::to_string(foldField->bitOffset / 8) + " + 2 * (" +
        std::to_string(pairsPerGroup) + " * " + group +
        ")) + (int32_t)weft_load_i16_le(" + record + " + " +
        std::to_string(foldField->bitOffset / 8) + " + 2 * (" +
        std::to_string(pairsPerGroup) + " * " + group + " + 1)))";
    const std::string statement =
        result.parts[part] + " = __riscv_vmacc_vx_" + resultSuffix + "(" +
        result.parts[part] + ", " + pair + ", " + laneI32 + ", " +
        laneVL() + ");";
    line("if (" + std::to_string(part) + " < " + accumulatorActive + ") " +
         statement);
  }
  --indent;
  line("}");
  return result;
}

mlir::LogicalResult
Emitter::compileMaterialize(kernel::MaterializeOp materialize) {
  Binding input = bindings.lookup(materialize.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(materialize, "materialize input has no emitted value");
  bindings[materialize.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileContract(mlir::Operation &operation,
                                              mlir::Value lhsValue,
                                              mlir::Value rhsValue,
                                              mlir::Value resultValue,
                                              bool outer,
                                              std::optional<Binding> initial,
                                              mlir::Value initialValue) {
  Binding lhs = bindings.lookup(lhsValue);
  Binding rhs = bindings.lookup(rhsValue);
  mlir::DictionaryAttr selected = assigned(&operation);
  if (!selected)
    return fail(&operation,
                "intrinsic-C emission has no selected operation assignment");
  llvm::StringRef realization =
      riscv_internal::string(selected, "realization").value_or("");
  if (realization == "rvv.outer.deferred-reduction-product" && !initial) {
    Binding deferred;
    deferred.kind = Binding::Kind::DeferredContract;
    deferred.lhs = lhsValue;
    deferred.rhs = rhsValue;
    deferred.sourceOperation = &operation;
    bindings[resultValue] = std::move(deferred);
    return mlir::success();
  }
  if (realization == "rvv.outer.encoded-widening-mac") {
    mlir::FailureOr<Binding> result =
        emitEncodedOuterMac(operation, lhsValue, rhsValue, resultValue, lhs, rhs);
    if (mlir::failed(result))
      return mlir::failure();
    bindings[resultValue] = std::move(*result);
    return mlir::success();
  }
  if (realization == "rvv.outer.encoded-fold-product") {
    mlir::FailureOr<Binding> result = emitEncodedOuterFold(
        operation, lhsValue, rhsValue, resultValue, lhs, rhs);
    if (mlir::failed(result))
      return mlir::failure();
    bindings[resultValue] = std::move(*result);
    return mlir::success();
  }
  if (realization.starts_with("matrix.")) {
    auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
    llvm::StringRef instruction =
        riscv_internal::string(local, "instruction").value_or("");
    int64_t mFactor = riscv_internal::integer(local, "m_factor").value_or(0);
    int64_t nFactor = riscv_internal::integer(local, "n_factor").value_or(0);
    int64_t kFactor = riscv_internal::integer(local, "k_factor").value_or(0);
    int64_t interleaveRows =
        riscv_internal::integer(local, "interleave_rows").value_or(0);
    int64_t lhsGroupSize =
        riscv_internal::integer(local, "lhs_group_size").value_or(0);
    int64_t lhsLayerSize =
        riscv_internal::integer(local, "lhs_layer_size").value_or(0);
    llvm::StringRef lhsLayerOrder =
        riscv_internal::string(local, "lhs_layer_order").value_or("");
    llvm::StringRef resultSuffix =
        riscv_internal::string(local, "result_vector_suffix").value_or("");
    if (instruction != "spacemit-ime1-i4i8-mma" || mFactor != 1 ||
        nFactor != 16 || kFactor != 32)
      return fail(&operation,
                  "intrinsic-C emitter has no leaf for the selected matrix realization");
    if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field ||
        !lhs.field.index || !rhs.field.index)
      return fail(&operation,
                  "matrix local operation requires indexed encoded fields");
    const Binding &lhsOwner = bindings.lookup(lhs.field.owner);
    const Binding &rhsOwner = bindings.lookup(rhs.field.owner);
    auto lhsField = fieldFor(lhs);
    auto rhsField = fieldFor(rhs);
    if (!lhsField || !rhsField || lhsOwner.kind != Binding::Kind::Record ||
        rhsOwner.kind != Binding::Kind::Record ||
        lhsOwner.interleaveRows != interleaveRows)
      return fail(&operation,
                  "selected IME fragment requires a rows=16 derived encoding");
    if (lhsField->bitOffset % 8 || lhsGroupSize <= 0 || lhsLayerSize <= 0 ||
        lhsGroupSize != 2 * lhsLayerSize ||
        (lhsLayerOrder != "lo_first" && lhsLayerOrder != "hi_first"))
      return fail(&operation,
                  "selected IME fragment has an incomplete packed-field mapping");
    const Binding &point = bindings.lookup(*lhs.field.index);
    const Binding &rhsPoint = bindings.lookup(*rhs.field.index);
    if (point.kind != Binding::Kind::Point || rhsPoint.kind != Binding::Kind::Point ||
        point.point.axis != rhsPoint.point.axis || point.point.base != rhsPoint.point.base ||
        point.point.active != rhsPoint.point.active)
      return fail(&operation,
                  "selected IME fragment operands do not share one reduction point");
    if (vectorSuffix(resultValue) != resultSuffix)
      return fail(&operation,
                  "selected IME leaf result does not match its assigned value representation");
    auto encoding = encodings.find(lhsOwner.encodingFamily);
    if (encoding == encodings.end())
      return fail(&operation, "selected IME fragment encoding family is unavailable");
    const std::string logicalBase = "(" + point.point.base + " % " +
                                    std::to_string(encoding->second.logicalElements) +
                                    ")";
    Binding result;
    result.kind = Binding::Kind::Vector;
    std::string name = fresh("ime_fragment");
    line(vectorType(resultValue) + " " + name +
         " = weft_ime_i4_i8_m1n16k32(" + lhsOwner.recordPointer + ", " +
         std::to_string(lhsField->bitOffset) + ", " +
         std::to_string(lhsGroupSize) + ", " +
         std::to_string(lhsLayerSize) + ", " +
         (lhsLayerOrder == "lo_first" ? "1" : "0") + ", (const int8_t *)(" +
         rhsOwner.recordPointer + " + " +
         std::to_string(rhsField->bitOffset / 8) + " + " + logicalBase + "), " +
         logicalBase + ", " + point.point.active + ", " + laneVL() + ");");
    result.parts.push_back(std::move(name));
    bindings[resultValue] = std::move(result);
    return mlir::success();
  }
  if (!realization.starts_with("rvv."))
    return fail(&operation,
                "intrinsic-C emitter has no leaf for the selected contraction realization");
  if (lhs.kind != Binding::Kind::Slice || rhs.kind != Binding::Kind::Slice)
    return fail(&operation, "dense contraction requires admitted memory slices");
  auto local = selected.getAs<mlir::DictionaryAttr>("local_operation");
  if (!local)
    return fail(&operation,
                "selected dense contraction has no local operation decision");
  llvm::StringRef laneOperand =
      riscv_internal::string(local, "lane_operand").value_or("");
  if (outer && laneOperand != "lhs" && laneOperand != "rhs")
    return fail(&operation,
                "selected dense outer contraction has no lane operand");
  const Binding &laneSlice = laneOperand == "rhs" ? rhs : lhs;
  const Binding &repeatedSlice = laneOperand == "rhs" ? lhs : rhs;
  const int64_t laneAxis = laneAxisFor(resultValue);
  int64_t reductionAxis = -1;
  llvm::ArrayRef<int64_t> over;
  if (auto contract = mlir::dyn_cast<kernel::ContractOp>(operation))
    over = contract.getOver();
  else if (auto contract = mlir::dyn_cast<kernel::OuterContractOp>(operation))
    over = contract.getOver();
  else if (auto dot = mlir::dyn_cast<kernel::DotOp>(operation))
    over = dot.getOver();
  if (!over.empty())
    reductionAxis = over.front();
  auto reductionScope = axisScopes.find(reductionAxis);
  if (reductionScope == axisScopes.end() || reductionScope->second.empty())
    return fail(&operation, "contraction reduction axis has no enclosing level");
  const PointInfo &reduction = reductionScope->second.back();
  if (!riscv_internal::logicalElement(resultValue.getType()).isF32())
    return fail(&operation, "current dense contraction emission supports only f32 results");
  mlir::FailureOr<Binding> resultOr = makeVector(
      resultValue, "contract", initial ? *initial : scalar("0"), initialValue);
  if (mlir::failed(resultOr))
    return mlir::failure();
  Binding result = std::move(*resultOr);
  llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(resultValue);
  const int64_t streams = streamPartCount(resultValue);
  llvm::StringRef laneMemoryForm =
      riscv_internal::string(local, "lane_memory_form").value_or("");
  if (laneMemoryForm != "unit-stride" &&
      laneMemoryForm != "runtime-strided")
    return fail(&operation,
                "dense contraction has no pass-selected lane memory form");
  std::string k = fresh("reduce_k");
  line("for (size_t " + k + " = 0; " + k + " < " + reduction.active + "; ++" +
       k + ") {");
  ++indent;
  const Binding &laneMemory = bindings.lookup(laneSlice.slice.base);
  const Binding &repeatedMemory = bindings.lookup(repeatedSlice.slice.base);
  if (laneMemory.kind != Binding::Kind::Memory ||
      repeatedMemory.kind != Binding::Kind::Memory ||
      !laneMemory.memory.elementType || !repeatedMemory.memory.elementType ||
      !laneMemory.memory.elementType.isF32() ||
      !repeatedMemory.memory.elementType.isF32())
    return fail(&operation,
                "current dense contraction emission requires f32 memory operands");
  size_t laneDimension = llvm::find(laneMemory.memory.axes, laneAxis) -
                         laneMemory.memory.axes.begin();
  if (laneDimension >= laneMemory.memory.axes.size())
    return fail(&operation,
                "contraction lane operand has no selected lane axis");
  const std::string suffix = vectorSuffix(resultValue);
  llvm::StringMap<std::string> loadedLaneValues;
  for (size_t part = 0; part < result.parts.size(); ++part) {
    auto coordinates = registerCoordinates(resultValue, part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(&operation,
                  "dense contraction result has no register-coordinate mapping");
    llvm::SmallVector<std::pair<int64_t, std::string>> laneOffsets{
        {reductionAxis, k}};
    llvm::SmallVector<std::pair<int64_t, std::string>> repeatedOffsets{
        {reductionAxis, k}};
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
      if (llvm::is_contained(repeatedMemory.memory.axes, axis))
        repeatedOffsets.push_back({axis, std::to_string(coordinate)});
    }
    std::string loaded;
    auto existingLoad = loadedLaneValues.find(laneKey);
    if (existingLoad != loadedLaneValues.end()) {
      loaded = existingLoad->second;
    } else {
      auto laneAddress = denseAddress(laneSlice.slice, laneOffsets);
      if (!laneAddress)
        return fail(&operation,
                    "contraction lane operand has no address relation");
      loaded = fresh("operand");
      if (laneMemoryForm == "unit-stride")
        line(vectorType(resultValue) + " " + loaded + " = __riscv_vle32_v_" +
             suffix + "(" + *laneAddress + ", " +
             partVL(resultValue, part) + ");");
      else
        line(vectorType(resultValue) + " " + loaded + " = __riscv_vlse32_v_" +
             suffix + "(" + *laneAddress + ", " +
             laneMemory.memory.strides[laneDimension] +
             " * (ptrdiff_t)sizeof(float), " + partVL(resultValue, part) +
             ");");
      loadedLaneValues[laneKey] = loaded;
    }
    auto repeatedAddress = denseAddress(repeatedSlice.slice, repeatedOffsets);
    if (!repeatedAddress)
      return fail(&operation,
                  "contraction repeated operand has no address relation");
    std::string statement = result.parts[part] + " = __riscv_vfmacc_vf_" +
                            suffix + "(" + result.parts[part] + ", *(" +
                            *repeatedAddress + "), " + loaded + ", " +
                            partVL(resultValue, part) + ");";
    if (!registerAxes.empty())
      line("if (" + active + ") " + statement);
    else
      line(statement);
  }
  --indent;
  line("}");
  bindings[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCommit(kernel::CommitOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding region = bindings.lookup(operation.getRegion());
  mlir::DictionaryAttr selected = assigned(operation.getOperation());
  auto selectedEdge = selected
                          ? selected.getAs<mlir::DictionaryAttr>("memory_edge")
                          : mlir::DictionaryAttr();
  if (region.kind == Binding::Kind::Field) {
    auto field = fieldFor(region);
    if (!field || field->bitOffset % 8 ||
        layoutKind(field->layouts) != "natural")
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
      auto encoding = encodings.find(record.encodingFamily);
      if (point.kind != Binding::Kind::Point || encoding == encodings.end())
        return fail(operation,
                    "encoded field commit index has no record-domain relation");
      logicalIndex = "(" + point.point.base + " % " +
                     std::to_string(encoding->second.logicalElements) + ")";
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
        line("weft_store_f16_le(" + address + ", " + value.scalar + ");");
      else if (element.isF32())
        line("weft_store_f32_le(" + address + ", " + value.scalar + ");");
      else if (element.isInteger(16))
        line("weft_store_i16_le(" + address + ", " + value.scalar + ");");
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
      selectedEdge ? riscv_internal::string(selectedEdge, "form").value_or("")
                   : llvm::StringRef();
  if (memoryForm != "unit-stride" && memoryForm != "runtime-strided")
    return fail(operation,
                "dense commit has no pass-selected unit or runtime-strided form");
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
    if (memoryForm == "unit-stride")
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
         << "}\n\n";
  bool needsIME = false;
  for (riscv::AssignmentOp assignment : module.getOps<riscv::AssignmentOp>())
    for (mlir::Attribute attribute : assignment.getOperations()) {
      auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
      llvm::StringRef realization =
          riscv_internal::string(operation, "realization").value_or("");
      needsIME |= realization.starts_with("matrix.spacemit-ime1-i4i8-mma.");
    }
  if (needsIME)
    output << "static inline vint32m2_t weft_ime_i4_i8_m1n16k32(\n"
           << "    const uint8_t *panel, size_t q_bit_offset, size_t q_group,\n"
           << "    size_t q_layer, int q_low_first, const int8_t *x,\n"
           << "    size_t logical_base, size_t extent, size_t vl) {\n"
           << "  _Alignas(32) int32_t output[16];\n"
           << "  _Alignas(16) int8_t activation[8];\n"
           << "  _Alignas(32) uint8_t weights[32];\n"
           << "  for (size_t group = 0; group < 4; ++group) {\n"
           << "    int32_t *acc = output + group * 4;\n"
           << "    memset(acc, 0, 4 * sizeof(int32_t));\n"
           << "    for (size_t chunk = 0; chunk < 4; ++chunk) {\n"
           << "      for (size_t row = 0; row < 4; ++row)\n"
           << "        for (size_t k = 0; k < 8; ++k) {\n"
           << "          if (chunk * 8 + k < extent) {\n"
           << "            const size_t logical = logical_base + chunk * 8 + k;\n"
           << "            const size_t within = logical % q_group;\n"
           << "            const size_t layer_index = within / q_layer;\n"
           << "            const size_t storage_byte = q_bit_offset / 8 +\n"
           << "                (logical / q_group) * q_layer + within % q_layer;\n"
           << "            const size_t bit_layer = q_low_first ? layer_index :\n"
           << "                (q_group / q_layer - 1 - layer_index);\n"
           << "            const uint8_t packed =\n"
           << "                panel[storage_byte * 16 + group * 4 + row];\n"
           << "            weights[row * 8 + k] =\n"
           << "                (packed >> (4 * bit_layer)) & 15;\n"
           << "          } else {\n"
           << "            weights[row * 8 + k] = 0;\n"
           << "          }\n"
           << "        }\n"
           << "      const size_t active = extent > chunk * 8 ?\n"
           << "          (extent - chunk * 8 < 8 ? extent - chunk * 8 : 8) : 0;\n"
           << "      for (size_t k = 0; k < 8; ++k)\n"
           << "        activation[k] = k < active ? x[chunk * 8 + k] : 0;\n"
           << "      __asm__ volatile(\n"
           << "          \"vsetivli t0, 4, e32, mf2\\n\\t\"\n"
           << "          \"vle32.v v16, (%[acc])\\n\\t\"\n"
           << "          \"vsetivli t0, 8, e8, mf4\\n\\t\"\n"
           << "          \"vle8.v v14, (%[activation])\\n\\t\"\n"
           << "          \"li t0, 32\\n\\t\"\n"
           << "          \"vsetvli t0, t0, e8, m1\\n\\t\"\n"
           << "          \"vle8.v v0, (%[weights])\\n\\t\"\n"
           << "          \"vmadot v16, v14, v0\\n\\t\"\n"
           << "          \"vsetivli t0, 4, e32, mf2\\n\\t\"\n"
           << "          \"vse32.v v16, (%[acc])\"\n"
           << "          :\n"
           << "          : [acc] \"r\"(acc), [activation] \"r\"(activation),\n"
           << "            [weights] \"r\"(weights)\n"
           << "          : \"memory\", \"t0\", \"v0\", \"v14\", \"v16\");\n"
           << "    }\n"
           << "  }\n"
           << "  return __riscv_vle32_v_i32m2(output, vl);\n"
           << "}\n\n";
  bool found = false;
  for (riscv::AssignmentOp assignment : module.getOps<riscv::AssignmentOp>()) {
    kernel::KernelOp kernel =
        riscv_internal::findKernel(module, assignment.getKernelAttr());
    if (!kernel)
      return assignment.emitError("selected assignment references no canonical kernel");
    Emitter emitter(module, kernel, assignment, output);
    emitter.emitBuilders();
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
    found = true;
  }
  if (!found)
    return module.emitError("intrinsic-C emission requires a solved RISC-V assignment");
  output.flush();
  result = std::move(body);
  return mlir::success();
}
