#include "RISCVIntrinsicC.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
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
  int64_t logicalElements = 1;
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
};

struct SliceInfo {
  mlir::Value base;
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<std::string> selectors;
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
    Vector,
    Memory,
    Slice,
    Record,
    Field,
    DeferredMac,
    DeferredWiden,
    DeferredFold,
    PackRequest,
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
      if (!encoding || encoding.getKind() != "derived_family")
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
    for (mlir::BlockArgument argument : block.getArguments())
      valueAssignments[argument] = values[valueIndex++];
    for (mlir::Operation &operation : block) {
      for (mlir::Value result : operation.getResults())
        valueAssignments[result] = values[valueIndex++];
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
        if (!field.shape.empty())
          info.logicalElements = std::max(info.logicalElements, field.shape.back());
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
      if (mlir::failed(compileOperation(operation)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult compileOperation(mlir::Operation &operation);
  mlir::LogicalResult compileLevel(kernel::LevelOp level);
  mlir::LogicalResult compileFor(kernel::ForOp operation);
  mlir::LogicalResult compileIf(kernel::IfOp operation);
  mlir::LogicalResult compileSlice(kernel::SliceOp slice);
  mlir::LogicalResult compileAdmit(kernel::AdmitOp admit);
  mlir::LogicalResult compileMaterialize(kernel::MaterializeOp materialize);
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
                                      mlir::Value result, bool outer);
  mlir::LogicalResult compileBinary(kernel::BinaryOp operation);
  mlir::LogicalResult compileCommit(kernel::CommitOp operation);
  mlir::LogicalResult compilePack(kernel::PackOp operation);

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
  int64_t physicalLanes(mlir::Value value) const;
  int64_t laneAxisFor(mlir::Value value) const;
  std::string partOffset(mlir::Value value, size_t part) const;
  std::string partVL(mlir::Value value, size_t part) const;
  mlir::FailureOr<Binding>
  makeVector(mlir::Value value, llvm::StringRef prefix,
             std::optional<Binding> initial = std::nullopt);
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
  llvm::DenseMap<mlir::Value, Binding> bindings;
  llvm::StringMap<EncodingInfo> encodings;
  llvm::StringMap<std::string> derivedSources;
  llvm::StringMap<int64_t> autoBindings;
  llvm::StringMap<int64_t> axisSymbolIds;
  llvm::SmallSet<unsigned, 8> writableArguments;
  llvm::DenseMap<int64_t, llvm::SmallVector<PointInfo>> axisScopes;
  llvm::SmallVector<std::string> laneVLStack;
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
    return laneVL();
  const std::string active = scope->second.back().active;
  const std::string offset = partOffset(value, part);
  const std::string lanes = std::to_string(physicalLanes(value));
  return "((" + active + " > " + offset + ") ? ((" + active + " - " +
         offset + " < " + lanes + ") ? " + active + " - " + offset + " : " +
         lanes + ") : 0)";
}

int64_t Emitter::vectorPartCount(mlir::Value value) const {
  auto type = mlir::dyn_cast<kernel::ValueType>(value.getType());
  if (!type)
    return 1;
  int64_t count = 1;
  const int64_t laneAxis = laneAxisFor(value);
  for (auto [extent, axis] :
       llvm::zip(type.getShape().asArrayRef(), type.getAxisIds().asArrayRef())) {
    if (axis == laneAxis)
      continue;
    auto scope = axisScopes.find(axis);
    if (scope != axisScopes.end() && !scope->second.empty())
      count *= scope->second.back().physicalExtent;
    else if (extent > 0)
      count *= extent;
  }
  return std::max<int64_t>(1, count) * streamPartCount(value);
}

mlir::FailureOr<Binding>
Emitter::makeVector(mlir::Value value, llvm::StringRef prefix,
                    std::optional<Binding> initial) {
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
      } else if (initial->kind == Binding::Kind::Vector &&
                 index < static_cast<int64_t>(initial->parts.size())) {
        expression = initial->parts[index];
      }
    }
    if (expression.empty())
      expression = "(" + type + "){0}";
    line(type + " " + name + " = " + expression + ";");
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
  auto resultType = mlir::cast<kernel::ValueType>(result.getType());
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
  int64_t nonLaneAxis = -1;
  int64_t count = 1;
  const int64_t laneAxis = laneAxisFor(result);
  for (auto [extent, axis] : llvm::zip(resultType.getShape().asArrayRef(),
                                       resultType.getAxisIds().asArrayRef()))
    if (axis != laneAxis) {
      nonLaneAxis = axis;
      auto scope = axisScopes.find(axis);
      count = scope != axisScopes.end() && !scope->second.empty()
                  ? scope->second.back().physicalExtent
                  : std::max<int64_t>(1, extent);
    }
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
  const int64_t totalParts = count * streams;
  for (int64_t part = 0; part < totalParts; ++part) {
    const int64_t nonLanePart = part / streams;
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    if (nonLaneAxis >= 0)
      offsets.push_back({nonLaneAxis, std::to_string(nonLanePart)});
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
    if (streams > 1 && nonLaneAxis < 0) {
      loaded.parts.push_back(std::move(expression));
      continue;
    }
    std::string name = fresh("load");
    if (nonLaneAxis >= 0) {
      const std::string active = axisScopes.lookup(nonLaneAxis).back().active;
      line(type + " " + name + " = __riscv_vfmv_v_f_" + suffix + "(0.0f, " +
           partVL(result, part) + ");");
      line("if (" + std::to_string(nonLanePart) + " < " + active + ") " + name +
           " = " + expression + ";");
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
  if (auto pack = mlir::dyn_cast<kernel::PackOp>(operation))
    return compilePack(pack);
  if (auto materialize = mlir::dyn_cast<kernel::MaterializeOp>(operation))
    return compileMaterialize(materialize);
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
  if (mlir::isa<kernel::StageHandoffOp>(operation)) {
    auto handoff = mlir::cast<kernel::StageHandoffOp>(operation);
    bindings[handoff.getResult()] = bindings.lookup(handoff.getInput());
    return mlir::success();
  }
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
      mlir::FailureOr<Binding> copy = makeVector(value, "carry", source);
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
  line("const size_t " + base + " = (size_t)(" + origin + ") + " + iterator +
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
                           llvm::SmallVectorImpl<Binding> &yielded) {
    mlir::Block &block = region.front();
    if (mlir::failed(compileBlock(block, {pointBinding})))
      return mlir::failure();
    auto terminator = mlir::cast<kernel::BirthsYieldOp>(block.getTerminator());
    for (mlir::Value value : terminator.getValues())
      yielded.push_back(bindings.lookup(value));
    return mlir::success();
  };

  llvm::SmallVector<Binding, 0> stateBirths;
  llvm::SmallVector<Binding, 0> stagedBirths;
  if (mlir::failed(compileBirths(level.getStateBirths(), stateBirths)) ||
      mlir::failed(compileBirths(level.getStagedBirths(), stagedBirths)))
    return mlir::failure();

  llvm::SmallVector<Binding, 0> bodyArguments{pointBinding};
  bodyArguments.append(carried.begin(), carried.end());
  bodyArguments.append(stateBirths.begin(), stateBirths.end());
  bodyArguments.append(stagedBirths.begin(), stagedBirths.end());
  if (mlir::failed(compileBlock(level.getBody().front(), bodyArguments)))
    return mlir::failure();
  auto handoff = mlir::cast<kernel::HandoffOp>(level.getBody().front().getTerminator());
  for (auto [index, value] : llvm::enumerate(handoff.getValues())) {
    Binding next = bindings.lookup(value);
    if (index >= carried.size())
      return fail(level, "handoff produced more carried values than the level owns");
    Binding &state = carried[index];
    if (state.kind == Binding::Kind::Vector && next.kind == Binding::Kind::Vector) {
      if (state.parts.size() != next.parts.size())
        return fail(level, "vector handoff shape changed inside a frozen level");
      for (auto [target, source] : llvm::zip(state.parts, next.parts))
        line(target + " = " + source + ";");
    } else if (state.kind == Binding::Kind::Scalar &&
               next.kind == Binding::Kind::Scalar) {
      line(state.scalar + " = " + next.scalar + ";");
    } else {
      state = std::move(next);
    }
  }

  if (laneLevel)
    laneVLStack.pop_back();
  axisScopes[axis].pop_back();
  --indent;
  line("}");
  for (auto [result, binding] : llvm::zip(level.getResults(), carried))
    bindings[result] = binding;
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
      mlir::FailureOr<Binding> copy = makeVector(value, "for_carry", source);
      if (mlir::failed(copy))
        return mlir::failure();
      carried.push_back(std::move(*copy));
    } else {
      return fail(operation,
                  "ordered for carry has no selected scalar/vector handoff");
    }
  }
  std::string iterator = fresh("for_index");
  line("for (size_t " + iterator + " = (size_t)(" + lower.scalar + "); " +
       iterator + " < (size_t)(" + upper.scalar + "); " + iterator + " += " +
       step.scalar + ") {");
  ++indent;
  llvm::SmallVector<Binding, 0> arguments{scalar(iterator)};
  arguments.append(carried.begin(), carried.end());
  mlir::Block &body = operation.getBody().front();
  if (mlir::failed(compileBlock(body, arguments)))
    return mlir::failure();
  auto yield = mlir::cast<kernel::YieldOp>(body.getTerminator());
  for (auto [state, nextValue] : llvm::zip(carried, yield.getValues())) {
    Binding next = bindings.lookup(nextValue);
    if (state.kind == Binding::Kind::Scalar && next.kind == Binding::Kind::Scalar)
      line(state.scalar + " = " + next.scalar + ";");
    else if (state.kind == Binding::Kind::Vector &&
             next.kind == Binding::Kind::Vector &&
             state.parts.size() == next.parts.size())
      for (auto [destination, expression] : llvm::zip(state.parts, next.parts))
        line(destination + " = " + expression + ";");
    else
      return fail(operation,
                  "ordered for body changed a carried physical handoff");
  }
  --indent;
  line("}");
  for (auto [result, binding] : llvm::zip(operation.getResults(), carried))
    bindings[result] = std::move(binding);
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
    result.kind = Binding::Kind::Scalar;
    result.scalar = fresh("if_result");
    line(*type + " " + result.scalar + ";");
    results.push_back(std::move(result));
  }

  auto compileBranch = [&](mlir::Region &region) -> mlir::LogicalResult {
    mlir::Block &block = region.front();
    if (mlir::failed(compileBlock(block, {})))
      return mlir::failure();
    auto yield = mlir::cast<kernel::YieldOp>(block.getTerminator());
    for (auto [target, value] : llvm::zip(results, yield.getValues())) {
      Binding source = bindings.lookup(value);
      if (target.kind == Binding::Kind::Scalar &&
          source.kind == Binding::Kind::Scalar) {
        line(target.scalar + " = " + source.scalar + ";");
      } else if (target.kind == Binding::Kind::Vector &&
                 source.kind == Binding::Kind::Vector &&
                 target.parts.size() == source.parts.size()) {
        for (auto [destination, expression] :
             llvm::zip(target.parts, source.parts))
          line(destination + " = " + expression + ";");
      } else {
        return fail(operation,
                    "ordered if branches disagree on selected physical handoff");
      }
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
  binding.slice.base = slice.getBase();
  binding.slice.indices.assign(slice.getIndices().begin(), slice.getIndices().end());
  for (mlir::Attribute selector : slice.getSelectors())
    binding.slice.selectors.push_back(
        mlir::cast<mlir::StringAttr>(selector).getValue().str());
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
  if (!encoding || encoding.getKind() == "dense" ||
      encoding.getKind() == "ephemeral") {
    value.getDefiningOp()->emitError("record slice must carry a concrete encoding");
    return mlir::failure();
  }
  if (base.kind != Binding::Kind::Memory) {
    value.getDefiningOp()->emitError("encoded record requires a memory-backed view");
    return mlir::failure();
  }
  std::string family = encoding.getFamily().str();
  std::string sourceFamily = family;
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
  Binding result;
  result.kind = Binding::Kind::Record;
  result.encodingFamily = sourceFamily;
  const int64_t recordBytes = info->second.storageBits / 8;
  const int64_t elements = info->second.logicalElements;
  if (encoding.getKind() == "derived_family") {
    int64_t rows =
        riscv_internal::integer(physical, "interleave_rows").value_or(0);
    if (rows <= 0 || base.memory.axes.size() != 2)
      return value.getDefiningOp()->emitError(
                 "derived interleave requires a two-axis view and selected rows"),
             mlir::failure();
    const PointInfo row = points.lookup(base.memory.axes[0]);
    const PointInfo block = points.lookup(base.memory.axes[1]);
    const std::string blocks = "(" + base.memory.extents[1] + " / " +
                               std::to_string(elements) + ")";
    result.interleaveRows = rows;
    result.recordPointer = base.memory.name + " + (((" + row.base + " / " +
                           std::to_string(rows) + ") * " + blocks +
                           " + (" + block.base + " / " +
                           std::to_string(elements) + ")) * " +
                           std::to_string(recordBytes * rows) + ")";
  } else {
    if (base.memory.axes.empty())
      return value.getDefiningOp()->emitError(
                 "base encoded view has no logical axis"),
             mlir::failure();
    const size_t recordDimension = base.memory.axes.size() - 1;
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
  if (region.kind != Binding::Kind::Slice)
    return fail(admit, "admit currently requires a canonical slice region");
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(
      mlir::cast<kernel::SliceType>(admit.getRegion().getType()).getEncoding());
  if (!encoding)
    return fail(admit, "admit slice has no encoding");
  if (encoding.getKind() == "dense" || encoding.getKind() == "ephemeral") {
    bindings[admit.getResult()] = std::move(region);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = recordForSlice(admit.getRegion());
  if (mlir::failed(result))
    return mlir::failure();
  bindings[admit.getResult()] = std::move(*result);
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
            : makeVector(operation.getResult(), "state", initial);
    if (mlir::failed(result))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
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
  if (input->kind == Binding::Kind::Scalar) {
    std::string expression;
    if (operation.getKind() == "neg")
      expression = "(-(" + input->scalar + "))";
    else if (operation.getKind() == "abs" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "fabs(" + input->scalar + ")"
                                   : "fabsf(" + input->scalar + ")";
    else if (operation.getKind() == "abs")
      expression = "((" + input->scalar + ") < 0 ? -(" + input->scalar +
                   ") : (" + input->scalar + "))";
    else if (operation.getKind() == "exp" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "exp(" + input->scalar + ")"
                                   : "expf(" + input->scalar + ")";
    else
      return fail(operation, "selected scalar unary relation has no C spelling");
    bindings[operation.getResult()] = scalar(expression);
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
  for (auto [index, part] : llvm::enumerate(input->parts)) {
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
  if (input->kind == Binding::Kind::Scalar) {
    std::string expression = input->scalar;
    if (rounding) {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(target);
      if (!integer)
        return fail(operation, "rounded scalar narrowing requires an integer target");
      const unsigned width = integer.getWidth();
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
    bindings[operation.getResult()] =
        scalar("((" + *targetCType + ")(" + expression + "))");
    return mlir::success();
  }
  if (input->kind != Binding::Kind::Vector)
    return fail(operation, "cast has no selected numeric handoff");
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
  for (auto [index, part] : llvm::enumerate(input->parts)) {
    const std::string vl = partVL(operation.getResult(), index);
    std::string i32Value = fresh("rounded_i32");
    line(i32Type + " " + i32Value + " = __riscv_vfcvt_" +
         std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_v_" +
         i32Suffix + "_rm(" + part + ", " + frm.str() + ", " + vl +
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
  if (input.kind != Binding::Kind::Vector)
    return fail(operation, "widen requires a selected vector input");
  Binding result;
  result.kind = Binding::Kind::Vector;
  mlir::Type source = riscv_internal::logicalElement(operation.getInput().getType());
  mlir::Type target = riscv_internal::logicalElement(operation.getResult().getType());
  const std::string targetType = vectorType(operation.getResult());
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  for (auto [index, part] : llvm::enumerate(input.parts)) {
    const std::string vl = partVL(operation.getResult(), index);
    std::string expression;
    if (source.isF16() && target.isF32())
      expression = "__riscv_vfwcvt_f_f_v_" + targetSuffix + "(" + part +
                   ", " + vl + ")";
    else if (source.isInteger(32) && target.isF32())
      expression = "__riscv_vfcvt_f_x_v_" + targetSuffix + "(" + part +
                   ", " + vl + ")";
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(source)) {
      unsigned factor = mlir::cast<mlir::IntegerType>(target).getWidth() /
                        std::max(8u, integer.getWidth());
      std::string unsignedSuffix = targetSuffix;
      unsignedSuffix[0] = 'u';
      if (integer.isSigned()) {
        expression = "__riscv_vsext_vf" + std::to_string(factor) + "_" +
                     targetSuffix + "(" + part + ", " + vl + ")";
      } else {
        std::string widened =
            "__riscv_vzext_vf" + std::to_string(factor) + "_" + unsignedSuffix +
            "(" + part + ", " + vl + ")";
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
    std::string accumulator = fresh("widen_reduce_scalar");
    line(*scalarType + " " + accumulator + " = 0;");
    for (auto [index, part] : llvm::enumerate(materialized->parts)) {
      std::string seed = fresh("widen_reduce_seed");
      line(targetType + " " + seed + " = __riscv_vmv_v_x_" + targetSuffix +
           "(" + accumulator + ", 1);");
      std::string reduced = fresh("widen_reduce_vector");
      line(targetType + " " + reduced + " = __riscv_" +
           std::string(targetInteger.isUnsigned() ? "vwredsumu" : "vwredsum") +
           "_vs_" + sourceSuffix + "_" + targetSuffix + "(" + part + ", " +
           seed + ", " + partVL(input.input, index) + ");");
      line(accumulator + " = __riscv_vmv_x_s_" + targetSuffix + "_" +
           std::string(targetInteger.isUnsigned() ? "u" : "i") +
           std::to_string(targetInteger.getWidth()) + "(" + reduced + ");");
    }
    bindings[operation.getResult()] = scalar(accumulator);
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
  std::string accumulator = fresh("reduce_scalar");
  line(*scalarType + " " + accumulator + " = (" + *scalarType + ")(" + initial +
       ");");
  for (auto [index, part] : llvm::enumerate(materialized->parts)) {
    const std::string seed = fresh("reduce_seed");
    line(baseType + " " + seed + " = __riscv_" +
         std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + baseSuffix + "(" +
         accumulator + ", 1);");
    std::string stem;
    if (floating)
      stem = kind == "add" ? "vfredusum" : kind == "max" ? "vfredmax" : "vfredmin";
    else
      stem = "vredsum";
    const std::string reduced = fresh("reduce_vector");
    line(baseType + " " + reduced + " = __riscv_" + stem + "_vs_" +
         sourceSuffix + "_" + baseSuffix + "(" + part + ", " + seed +
         ", " + partVL(operation.getInput(), index) + ");");
    if (floating)
      line(accumulator + " = __riscv_vfmv_f_s_" + baseSuffix + "_f" +
           std::to_string(sew) + "(" + reduced + ");");
    else
      line(accumulator + " = __riscv_vmv_x_s_" + baseSuffix + "_" +
           std::string(unsignedInteger ? "u" : "i") + std::to_string(sew) +
           "(" + reduced + ");");
  }
  bindings[operation.getResult()] = scalar(accumulator);
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
  if (realization != "rvv.indexed-lookup")
    return fail(operation, "lookup has an unknown selected realization");
  if (indices.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV indexed lookup requires vector unsigned indices");

  const int64_t parts = vectorPartCount(operation.getResult());
  if (parts != static_cast<int64_t>(indices.parts.size()))
    return fail(operation,
                "lookup index and result physical mappings do not agree");
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
    const std::string vl = partVL(operation.getResult(), part);
    std::string byteOffsets = indices.parts[part];
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
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
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
  const Binding &vector = lhs.kind == Binding::Kind::Vector ? lhs : rhs;
  const Binding &other = lhs.kind == Binding::Kind::Vector ? rhs : lhs;
  if (vector.kind != Binding::Kind::Vector ||
      (other.kind != Binding::Kind::Vector && other.kind != Binding::Kind::Scalar))
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
  for (size_t index = 0; index < vector.parts.size(); ++index) {
    std::string expression;
    if (other.kind == Binding::Kind::Scalar) {
      std::string vectorPart = vector.parts[index];
      bool scalarOnLeft = lhs.kind == Binding::Kind::Scalar;
      bool commutative = kind == "add" || kind == "mul" || kind == "and" ||
                         kind == "or" || kind == "xor" || kind == "max" ||
                         kind == "min";
      if (scalarOnLeft && !commutative) {
        std::string splat = "__riscv_" +
                            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
                            suffix + "(" + lhs.scalar + ", " +
                            partVL(operation.getResult(), index) + ")";
        expression = "__riscv_" + stem + "_vv_" + suffix + "(" + splat +
                     ", " + vectorPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      } else {
        expression = "__riscv_" + stem + (floating ? "_vf_" : "_vx_") +
                     suffix + "(" + vectorPart + ", " + other.scalar + ", " +
                     partVL(operation.getResult(), index) + ")";
      }
    } else {
      const std::string &lhsPart = lhs.parts[std::min(index, lhs.parts.size() - 1)];
      const std::string &rhsPart = rhs.parts[std::min(index, rhs.parts.size() - 1)];
      expression = "__riscv_" + stem + "_vv_" + suffix + "(" + lhsPart +
                   ", " + rhsPart + ", " +
                   partVL(operation.getResult(), index) + ")";
    }
    if (streamPartCount(operation.getResult()) > 1) {
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
  std::string logicalIndex = requestedIndex.str();
  if (fieldBinding.field.index) {
    const Binding &point = bindings.lookup(*fieldBinding.field.index);
    auto encoding = encodings.find(owner.encodingFamily);
    if (encoding == encodings.end() || point.kind != Binding::Kind::Point)
      return {};
    const int64_t elements = encoding->second.logicalElements;
    if (fieldBinding.field.selector == "group_index")
      logicalIndex = "((" + point.point.base + " % " +
                     std::to_string(elements) + ") / " +
                     std::to_string(point.point.physicalExtent) + ")";
    else
      logicalIndex = "(" + point.point.base + " % " +
                     std::to_string(elements) + ")";
  }
  unsigned logicalWidth = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type))
    logicalWidth = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(field->type))
    logicalWidth = floating.getWidth();
  if (!logicalWidth)
    return {};

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
    const std::string byte = std::to_string(field->bitOffset / 8);
    if (field->type.isF32())
      scalarResult.scalar = "weft_load_f32_le(" + owner.recordPointer + " + " +
                            byte + ")";
    else if (field->type.isF16())
      scalarResult.scalar = "weft_load_f16_le(" + owner.recordPointer + " + " +
                            byte + ")";
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
             integer && integer.getWidth() == 8) {
      auto type = scalarCType(field->type);
      if (!type)
        return {};
      scalarResult.scalar = "*(const " + *type + " *)(" + owner.recordPointer +
                            " + " + byte + ")";
    } else if (field->type.isSignedInteger(16)) {
      scalarResult.scalar = "weft_load_i16_le(" + owner.recordPointer + " + " +
                            byte + ")";
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
  auto rhsAccess =
      local ? local.getAs<mlir::StringAttr>("rhs_access") : mlir::StringAttr();
  auto schedule = selected.getAs<mlir::DictionaryAttr>("schedule");
  int64_t unroll =
      schedule ? riscv_internal::integer(schedule, "unroll").value_or(0) : 0;
  int64_t pipelineDepth =
      schedule
          ? riscv_internal::integer(schedule, "pipeline_depth").value_or(0)
          : 0;
  int64_t prefetchDistance =
      schedule
          ? riscv_internal::integer(schedule, "prefetch_distance").value_or(-1)
          : -1;
  if (!instruction || instruction.getValue() != "rvv.vwmaccsu" ||
      !lhsLMULAttr || !partialLMULAttr || !partialSEWAttr ||
      partialSEWAttr.getInt() != 16 || unroll <= 0 || pipelineDepth <= 0 ||
      prefetchDistance < 0 || !rhsAccess ||
      rhsAccess.getValue() != "natural-scalar-field")
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
  const int64_t elements = encoding->second.logicalElements;
  const std::string base = "(" + point.point.base + " % " +
                           std::to_string(elements) + ")";
  const std::string groups = "((" + point.point.active + " + " +
                             std::to_string(mac.group - 1) + ") / " +
                             std::to_string(mac.group) + ")";
  const std::string outSuffix = vectorSuffix(result);
  const std::string outType = vectorType(result);
  const std::string outName = fresh("mac_reduce");
  line(outType + " " + outName + " = __riscv_vmv_v_x_" + outSuffix +
       "(0, " + laneVL() + ");");
  const std::string groupIndex = fresh("mac_group");
  line("for (size_t " + groupIndex + " = 0; " + groupIndex + " < " + groups +
       "; " + groupIndex + " += " + std::to_string(unroll) + ") {");
  ++indent;
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
  if (prefetchDistance > 0) {
    const std::string futureGroup =
        "(" + groupIndex + " + " +
        std::to_string(unroll * prefetchDistance) + ")";
    line("if (" + futureGroup + " < " + groups + ") {");
    ++indent;
    for (int64_t term = 0; term < mac.group; ++term) {
      const std::string logical =
          "(" + base + " + " + futureGroup + " * " +
          std::to_string(mac.group) + " + " + std::to_string(term) + ")";
      auto fragment =
          singleStorageFragment(*lhsField, logical, lhsInteger.getWidth());
      if (!fragment)
        return mac.sourceOperation->emitError(
                   "selected grouped MAC prefetch has no field mapping"),
               mlir::failure();
      line("__builtin_prefetch(" + lhsOwner.recordPointer + " + (" +
           fragment->byte + ") * " + std::to_string(lhsOwner.interleaveRows) +
           ", 0, 1);");
      line("__builtin_prefetch(" + rhsOwner.recordPointer + " + " +
           std::to_string(rhsField->bitOffset / 8) + " + " + logical +
           ", 0, 1);");
    }
    --indent;
    line("}");
  }

  llvm::SmallVector<std::string> partials;
  llvm::SmallVector<llvm::SmallVector<std::string>> bufferedQ(unroll);
  llvm::SmallVector<llvm::SmallVector<std::string>> bufferedX(unroll);
  llvm::SmallVector<llvm::SmallVector<std::string>> valid(unroll);
  for (int64_t slot = 0; slot < unroll; ++slot) {
    std::string partial = fresh("pair_partial");
    line(partialType + " " + partial + " = __riscv_vmv_v_x_" + partialSuffix +
         "(0, " + laneVL() + ");");
    partials.push_back(std::move(partial));
  }

  auto logicalFor = [&](int64_t slot, int64_t term) {
    return "(" + base + " + (" + groupIndex + " + " +
           std::to_string(slot) + ") * " + std::to_string(mac.group) + " + " +
           std::to_string(term) + ")";
  };
  auto conditionFor = [&](int64_t slot, int64_t term) {
    return "(" + groupIndex + " + " + std::to_string(slot) + " < " + groups +
           " && (" + groupIndex + " + " + std::to_string(slot) + ") * " +
           std::to_string(mac.group) + " + " + std::to_string(term) + " < " +
           point.point.active + ")";
  };
  auto fragmentFor = [&](int64_t slot, int64_t term)
      -> std::optional<StorageFragment> {
    return singleStorageFragment(*lhsField, logicalFor(slot, term),
                                 lhsInteger.getWidth());
  };

  if (pipelineDepth > 1) {
    for (int64_t slot = 0; slot < unroll; ++slot)
      for (int64_t term = 0; term < mac.group; ++term) {
        auto fragment = fragmentFor(slot, term);
        if (!fragment)
          return mac.sourceOperation->emitError(
                     "selected buffered grouped MAC has no field mapping"),
                 mlir::failure();
        std::string q = fresh("q_buffer");
        std::string x = fresh("x_buffer");
        std::string isValid = fresh("buffer_valid");
        line(rawType + " " + q + " = __riscv_vmv_v_x_" + rawSuffix +
             "(0, " + laneVL() + ");");
        line("int8_t " + x + " = 0;");
        line("const int " + isValid + " = " + conditionFor(slot, term) + ";");
        line("if (" + isValid + ") {");
        ++indent;
        std::string packed = fresh("q_packed");
        line(rawType + " " + packed + " = __riscv_vle8_v_" + rawSuffix + "(" +
             lhsOwner.recordPointer + " + (" + fragment->byte + ") * " +
             std::to_string(lhsOwner.interleaveRows) + ", " + laneVL() + ");");
        line(q + " = __riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
             rawSuffix + "(" + packed + ", " + fragment->shift + ", " +
             laneVL() + "), " +
             std::to_string((1u << lhsInteger.getWidth()) - 1) + ", " +
             laneVL() + ");");
        line(x + " = *(const int8_t *)(" + rhsOwner.recordPointer + " + " +
             std::to_string(rhsField->bitOffset / 8) + " + " +
             logicalFor(slot, term) + ");");
        --indent;
        line("}");
        bufferedQ[slot].push_back(std::move(q));
        bufferedX[slot].push_back(std::move(x));
        valid[slot].push_back(std::move(isValid));
      }
  }

  for (int64_t slot = 0; slot < unroll; ++slot) {
    for (int64_t term = 0; term < mac.group; ++term) {
      if (pipelineDepth > 1) {
        line("if (" + valid[slot][term] + ") " + partials[slot] +
             " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" + partials[slot] +
             ", " + bufferedX[slot][term] + ", " + bufferedQ[slot][term] +
             ", " + laneVL() + ");");
        continue;
      }
      auto fragment = fragmentFor(slot, term);
      if (!fragment)
        return mac.sourceOperation->emitError(
                   "selected grouped MAC field layout has no single-byte mapping"),
               mlir::failure();
      line("if (" + conditionFor(slot, term) + ") {");
      ++indent;
      std::string packed = fresh("q_packed");
      line(rawType + " " + packed + " = __riscv_vle8_v_" + rawSuffix + "(" +
           lhsOwner.recordPointer + " + (" + fragment->byte + ") * " +
           std::to_string(lhsOwner.interleaveRows) + ", " + laneVL() + ");");
      std::string q = fresh("q_value");
      line(rawType + " " + q + " = __riscv_vand_vx_" + rawSuffix + "(" +
           "__riscv_vsrl_vx_" + rawSuffix + "(" + packed + ", " +
           fragment->shift + ", " + laneVL() + "), " +
           std::to_string((1u << lhsInteger.getWidth()) - 1) + ", " + laneVL() +
           ");");
      const std::string x =
          "*(const int8_t *)(" + rhsOwner.recordPointer + " + " +
          std::to_string(rhsField->bitOffset / 8) + " + " +
          logicalFor(slot, term) + ")";
      line(partials[slot] + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
           partials[slot] + ", " + x + ", " + q + ", " + laneVL() + ");");
      --indent;
      line("}");
    }
    line("if (" + groupIndex + " + " + std::to_string(slot) + " < " + groups +
         ") {");
    ++indent;
    std::string widened = fresh("partial_wide");
    line(outType + " " + widened + " = __riscv_vwcvt_x_x_v_" + outSuffix +
         "(" + partials[slot] + ", " + laneVL() + ");");
    line(outName + " = __riscv_vadd_vv_" + outSuffix + "(" + outName + ", " +
         widened + ", " + laneVL() + ");");
    --indent;
    line("}");
  }
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

mlir::LogicalResult Emitter::compilePack(kernel::PackOp operation) {
  Binding input = bindings.lookup(operation.getView());
  if (input.kind != Binding::Kind::Slice)
    return fail(operation, "pack requires an explicit source slice");
  Binding result;
  result.kind = Binding::Kind::PackRequest;
  result.input = operation.getView();
  result.scalar = operation.getAlong().str();
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileMaterialize(kernel::MaterializeOp materialize) {
  Binding request = bindings.lookup(materialize.getInput());
  if (request.kind != Binding::Kind::PackRequest)
    return fail(materialize, "materialize currently requires a pack value");
  Binding sourceBinding = bindings.lookup(request.input);
  if (sourceBinding.kind != Binding::Kind::Slice)
    return fail(materialize, "packed materialize lost its source slice");
  const Binding &sourceMemory = bindings.lookup(sourceBinding.slice.base);
  if (sourceMemory.kind != Binding::Kind::Memory)
    return fail(materialize, "pack source is not memory-backed");
  auto resultType = mlir::cast<kernel::ViewType>(materialize.getResult().getType());
  if (!sourceMemory.memory.elementType ||
      !sourceMemory.memory.elementType.isF32())
    return fail(materialize,
                "current local pack emission supports only f32 source views");
  llvm::SmallVector<int64_t> axes(resultType.getAxisIds().asArrayRef().begin(),
                                  resultType.getAxisIds().asArrayRef().end());
  if (axes.size() != 2)
    return fail(materialize, "the current local pack emitter requires rank two");
  llvm::SmallVector<PointInfo> scopes;
  for (int64_t axis : axes) {
    auto found = axisScopes.find(axis);
    if (found == axisScopes.end() || found->second.empty())
      return fail(materialize, "pack axis has no enclosing level ownership");
    scopes.push_back(found->second.back());
  }
  const int64_t extent0 = scopes[0].physicalExtent;
  const int64_t extent1 = scopes[1].physicalExtent;
  std::string buffer = fresh("packed");
  line("_Alignas(64) float " + buffer + "[" + std::to_string(extent0 * extent1) +
       "];" );
  std::string i = fresh("pack_i");
  std::string j = fresh("pack_j");
  line("for (size_t " + i + " = 0; " + i + " < " + scopes[0].active + "; ++" +
       i + ")");
  ++indent;
  line("for (size_t " + j + " = 0; " + j + " < " + scopes[1].active + "; ++" +
       j + ") {");
  ++indent;
  auto source =
      denseAddress(sourceBinding.slice, {{axes[0], i}, {axes[1], j}});
  if (!source)
    return fail(materialize, "local pack source has no address relation");
  mlir::DictionaryAttr local =
      assigned(materialize.getOperation()).getAs<mlir::DictionaryAttr>(
          "local_operation");
  int64_t laneAxis =
      riscv_internal::integer(local, "lane_axis").value_or(0);
  if (laneAxis <= 0)
    return fail(materialize,
                "selected local pack has no operation-owned lane axis");
  const bool laneFirst = axes[0] == laneAxis;
  const bool laneSecond = axes[1] == laneAxis;
  std::string destination;
  if (laneFirst)
    destination = j + " * " + std::to_string(extent0) + " + " + i;
  else if (laneSecond)
    destination = i + " * " + std::to_string(extent1) + " + " + j;
  else
    destination = i + " * " + std::to_string(extent1) + " + " + j;
  line(buffer + "[" + destination + "] = *(" + *source + ");");
  --indent;
  line("}");
  --indent;

  Binding result;
  result.kind = Binding::Kind::Memory;
  result.memory.name = buffer;
  result.memory.encoding = resultType.getEncoding();
  result.memory.elementType = sourceMemory.memory.elementType;
  result.memory.axes = axes;
  result.memory.extents = {std::to_string(extent0), std::to_string(extent1)};
  result.memory.origins = {scopes[0].base, scopes[1].base};
  result.memory.isConst = true;
  result.memory.packed = true;
  if (laneFirst)
    result.memory.strides = {"1", std::to_string(extent0)};
  else
    result.memory.strides = {std::to_string(extent1), "1"};
  bindings[materialize.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileContract(mlir::Operation &operation,
                                              mlir::Value lhsValue,
                                              mlir::Value rhsValue,
                                              mlir::Value resultValue,
                                              bool outer) {
  Binding lhs = bindings.lookup(lhsValue);
  Binding rhs = bindings.lookup(rhsValue);
  mlir::DictionaryAttr selected = assigned(&operation);
  llvm::StringRef realization =
      riscv_internal::string(selected, "realization").value_or("");
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
  auto resultType = mlir::cast<kernel::ValueType>(resultValue.getType());
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
  mlir::FailureOr<Binding> resultOr =
      makeVector(resultValue, "contract", scalar("0"));
  if (mlir::failed(resultOr))
    return mlir::failure();
  Binding result = std::move(*resultOr);
  int64_t outputAxis = -1;
  for (int64_t axis : resultType.getAxisIds().asArrayRef())
    if (axis != laneAxis)
      outputAxis = axis;
  std::string k = fresh("reduce_k");
  line("for (size_t " + k + " = 0; " + k + " < " + reduction.active + "; ++" +
       k + ") {");
  ++indent;
  const Binding &lhsMemory = bindings.lookup(lhs.slice.base);
  const Binding &rhsMemory = bindings.lookup(rhs.slice.base);
  if (lhsMemory.kind != Binding::Kind::Memory ||
      rhsMemory.kind != Binding::Kind::Memory ||
      !lhsMemory.memory.elementType || !rhsMemory.memory.elementType ||
      !lhsMemory.memory.elementType.isF32() ||
      !rhsMemory.memory.elementType.isF32())
    return fail(&operation,
                "current dense contraction emission requires f32 memory operands");
  size_t lhsLaneDimension =
      llvm::find(lhsMemory.memory.axes, laneAxis) - lhsMemory.memory.axes.begin();
  if (lhsLaneDimension >= lhsMemory.memory.axes.size())
    return fail(&operation, "contraction lhs has no selected lane axis");
  llvm::SmallVector<std::pair<int64_t, std::string>> lhsOffsets{{reductionAxis, k}};
  auto lhsAddress = denseAddress(lhs.slice, lhsOffsets);
  if (!lhsAddress)
    return fail(&operation, "contraction lhs has no address relation");
  const std::string suffix = vectorSuffix(resultValue);
  std::string loaded = fresh("operand");
  if (lhsMemory.memory.strides[lhsLaneDimension] == "1")
    line(vectorType(resultValue) + " " + loaded + " = __riscv_vle32_v_" +
         suffix + "(" + *lhsAddress + ", " + laneVL() + ");");
  else
    line(vectorType(resultValue) + " " + loaded + " = __riscv_vlse32_v_" +
         suffix + "(" + *lhsAddress + ", " +
         lhsMemory.memory.strides[lhsLaneDimension] +
         " * (ptrdiff_t)sizeof(float), " + laneVL() + ");");
  std::string outputActive;
  if (outputAxis >= 0)
    outputActive = axisScopes.lookup(outputAxis).back().active;
  for (size_t part = 0; part < result.parts.size(); ++part) {
    llvm::SmallVector<std::pair<int64_t, std::string>> rhsOffsets{{reductionAxis, k}};
    if (outer && outputAxis >= 0)
      rhsOffsets.push_back({outputAxis, std::to_string(part)});
    auto rhsAddress = denseAddress(rhs.slice, rhsOffsets);
    if (!rhsAddress)
      return fail(&operation, "contraction rhs has no address relation");
    std::string statement = result.parts[part] + " = __riscv_vfmacc_vf_" +
                            suffix + "(" + result.parts[part] + ", *(" +
                            *rhsAddress + "), " + loaded + ", " + laneVL() +
                            ");";
    if (!outputActive.empty())
      line("if (" + std::to_string(part) + " < " + outputActive + ") " +
           statement);
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
  if (value.kind != Binding::Kind::Vector)
    return fail(operation, "dense commit has no selected numeric handoff");
  size_t laneDimension =
      llvm::find(memory.memory.axes, laneAxisFor(operation.getValue())) -
      memory.memory.axes.begin();
  if (laneDimension >= memory.memory.axes.size())
    return fail(operation, "commit destination has no selected lane axis");
  int64_t nonLaneAxis = -1;
  const int64_t laneAxis = laneAxisFor(operation.getValue());
  for (int64_t axis : memory.memory.axes)
    if (axis != laneAxis)
      nonLaneAxis = axis;
  const int64_t streams = streamPartCount(operation.getValue());
  llvm::StringRef memoryForm =
      selectedEdge ? riscv_internal::string(selectedEdge, "form").value_or("")
                   : llvm::StringRef();
  if (memoryForm != "unit-stride" && memoryForm != "runtime-strided")
    return fail(operation,
                "dense commit has no pass-selected unit or runtime-strided form");
  for (size_t part = 0; part < value.parts.size(); ++part) {
    const int64_t nonLanePart = static_cast<int64_t>(part) / streams;
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    if (nonLaneAxis >= 0)
      offsets.push_back({nonLaneAxis, std::to_string(nonLanePart)});
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
    if (nonLaneAxis >= 0)
      line("if (" + std::to_string(nonLanePart) + " < " +
           axisScopes.lookup(nonLaneAxis).back().active + ") " + statement);
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
         << "  uint16_t bits; memcpy(&bits, p, sizeof(bits)); return (int16_t)bits;\n"
         << "}\n"
         << "static inline __attribute__((unused)) _Float16 weft_load_f16_le(const uint8_t *p) {\n"
         << "  _Float16 value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_load_f32_le(const uint8_t *p) {\n"
         << "  float value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_i16_le(uint8_t *p, int16_t value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
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
