#include "Weft/Target/RISCVLowering.h"

#include "RISCVCABI.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace weft::kernel;

namespace {

std::string macroName(llvm::StringRef input) {
  std::string result = weft::riscv_internal::cABIIdentifier(input);
  llvm::transform(result, result.begin(), [](char character) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
  });
  return result;
}

bool isCOrCXXKeyword(llvm::StringRef input) {
  return llvm::StringSwitch<bool>(input)
      .Cases("alignas", "alignof", "and", "and_eq", "asm", true)
      .Cases("auto", "bitand", "bitor", "bool", "break", true)
      .Cases("case", "catch", "char", "class", "compl", true)
      .Cases("const", "constexpr", "continue", "decltype", "default", true)
      .Cases("delete", "do", "double", "dynamic_cast", "else", true)
      .Cases("enum", "explicit", "export", "extern", "false", true)
      .Cases("float", "for", "friend", "goto", "if", true)
      .Cases("inline", "int", "long", "mutable", "namespace", true)
      .Cases("new", "noexcept", "not", "not_eq", "nullptr", true)
      .Cases("operator", "or", "or_eq", "private", "protected", true)
      .Cases("public", "register", "reinterpret_cast", "restrict", true)
      .Cases("return", "short", "signed", "sizeof", "static", true)
      .Cases("static_assert", "static_cast", "struct", "switch", true)
      .Cases("template", "this", "thread_local", "throw", "true", true)
      .Cases("try", "typedef", "typeid", "typename", "union", true)
      .Cases("unsigned", "using", "virtual", "void", "volatile", true)
      .Cases("wchar_t", "while", "xor", "xor_eq", "_Alignas", true)
      .Cases("_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", true)
      .Cases("_Imaginary", "_Noreturn", "_Static_assert", true)
      .Case("_Thread_local", true)
      .Default(false);
}

bool isPortableABIIdentifier(llvm::StringRef input) {
  if (input.empty() ||
      !(std::isalpha(static_cast<unsigned char>(input.front())) ||
        input.front() == '_') ||
      isCOrCXXKeyword(input))
    return false;
  return llvm::all_of(input.drop_front(), [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_';
  });
}

llvm::StringRef argumentName(KernelOp kernel, unsigned index) {
  return mlir::cast<mlir::StringAttr>(kernel.getArgNames()[index]).getValue();
}

std::optional<unsigned> entryArgumentIndex(KernelOp kernel, mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  return argument.getArgNumber();
}

mlir::FailureOr<std::string> kernelDeclaration(KernelOp kernel) {
  llvm::SmallVector<std::string> parameters;
  mlir::Block &entry = kernel.getBody().front();
  for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
    llvm::StringRef kind =
        mlir::cast<mlir::StringAttr>(kernel.getArgKinds()[index]).getValue();
    if (kind == "constexpr")
      continue;
    std::string type;
    if (auto pointer = mlir::dyn_cast<PtrType>(argument.getType()))
      type = weft::riscv_internal::cABIPointerType(
          pointer,
          weft::riscv_internal::CPointerSpelling::PublicDeclaration);
    else
      type = weft::riscv_internal::cABIScalarType(argument.getType());
    if (type.empty()) {
      kernel.emitError("generated header has an unsupported ABI argument type");
      return mlir::failure();
    }
    parameters.push_back(
        type + " " +
        weft::riscv_internal::cABIIdentifier(argumentName(kernel, index)));
  }
  std::string returnType = mlir::isa<mlir::NoneType>(kernel.getReturnType())
                               ? "void"
                               : weft::riscv_internal::cABIScalarType(
                                     kernel.getReturnType());
  if (returnType.empty()) {
    kernel.emitError("generated header has an unsupported return type");
    return mlir::failure();
  }
  return returnType + " " +
         weft::riscv_internal::cABIIdentifier(kernel.getSymName()) + "(" +
         llvm::join(parameters, ", ") + ");";
}

struct HeaderExpression {
  std::string spelling;
  std::set<unsigned> runtimeArguments;
  std::optional<int64_t> staticValue;
};

mlir::FailureOr<HeaderExpression>
storageExpression(KernelOp kernel, mlir::Value value,
                  const weft::RISCVLoweringOptions &options,
                  llvm::DenseSet<mlir::Value> &visiting) {
  if (!visiting.insert(value).second) {
    kernel.emitError("storage extent expression contains a cycle");
    return mlir::failure();
  }
  auto finish = [&](mlir::FailureOr<HeaderExpression> result) {
    visiting.erase(value);
    return result;
  };
  if (std::optional<unsigned> index = entryArgumentIndex(kernel, value)) {
    if (!value.getType().isIndex()) {
      kernel.emitError("storage extent depends on a non-index ABI argument");
      return finish(mlir::failure());
    }
    HeaderExpression result{
        weft::riscv_internal::cABIIdentifier(argumentName(kernel, *index)),
        {*index}, std::nullopt};
    return finish(std::move(result));
  }
  if (auto constant = value.getDefiningOp<ConstantOp>()) {
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (!integer) {
      constant.emitError("storage extent constant must be an index integer");
      return finish(mlir::failure());
    }
    if (integer.getInt() <= 0) {
      constant.emitError("storage extent constant must be positive");
      return finish(mlir::failure());
    }
    return finish(HeaderExpression{std::to_string(integer.getInt()), {},
                                   integer.getInt()});
  }
  if (auto meta = value.getDefiningOp<MetaValueOp>()) {
    std::optional<unsigned> index = entryArgumentIndex(kernel, meta.getInput());
    if (!index) {
      meta.emitError("storage extent meta value is not an entry parameter");
      return finish(mlir::failure());
    }
    llvm::StringRef name = argumentName(kernel, *index);
    auto binding = options.metaBindings.find(name);
    if (binding == options.metaBindings.end()) {
      meta.emitError("storage extent requires a bound meta parameter");
      return finish(mlir::failure());
    }
    return finish(HeaderExpression{std::to_string(binding->second), {},
                                   binding->second});
  }
  if (auto binary = value.getDefiningOp<BinaryOp>()) {
    mlir::FailureOr<HeaderExpression> lhs =
        storageExpression(kernel, binary.getLhs(), options, visiting);
    mlir::FailureOr<HeaderExpression> rhs =
        storageExpression(kernel, binary.getRhs(), options, visiting);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return finish(mlir::failure());
    llvm::StringRef symbol =
        llvm::StringSwitch<llvm::StringRef>(binary.getKind())
            .Case("add", "+")
            .Case("sub", "-")
            .Case("mul", "*")
            .Case("div", "/")
            .Case("mod", "%")
            .Default("");
    if (symbol.empty()) {
      binary.emitError("storage extent uses an unsupported scalar operation");
      return finish(mlir::failure());
    }
    std::optional<int64_t> staticValue;
    if (lhs->staticValue && rhs->staticValue) {
      int64_t computed = 0;
      bool invalid = false;
      if (symbol == "+")
        invalid = __builtin_add_overflow(*lhs->staticValue, *rhs->staticValue,
                                         &computed);
      else if (symbol == "-")
        invalid = __builtin_sub_overflow(*lhs->staticValue, *rhs->staticValue,
                                         &computed);
      else if (symbol == "*")
        invalid = __builtin_mul_overflow(*lhs->staticValue, *rhs->staticValue,
                                         &computed);
      else if (*rhs->staticValue == 0 ||
               (*lhs->staticValue == std::numeric_limits<int64_t>::min() &&
                *rhs->staticValue == -1))
        invalid = true;
      else if (symbol == "/")
        computed = *lhs->staticValue / *rhs->staticValue;
      else
        computed = *lhs->staticValue % *rhs->staticValue;
      if (invalid) {
        binary.emitError(
            "storage extent static arithmetic is undefined or overflows");
        return finish(mlir::failure());
      }
      staticValue = computed;
    }
    HeaderExpression result{"(" + lhs->spelling + " " + symbol.str() + " " +
                                rhs->spelling + ")",
                            lhs->runtimeArguments, staticValue};
    result.runtimeArguments.insert(rhs->runtimeArguments.begin(),
                                   rhs->runtimeArguments.end());
    return finish(std::move(result));
  }
  if (auto cast = value.getDefiningOp<CastOp>())
    return finish(storageExpression(kernel, cast.getInput(), options, visiting));
  if (mlir::Operation *definition = value.getDefiningOp())
    definition->emitError(
        "storage extent must be computable from ABI index/meta parameters");
  else
    kernel.emitError(
        "storage extent must be computable from ABI index/meta parameters");
  return finish(mlir::failure());
}

std::string queryParameters(KernelOp kernel,
                            const std::set<unsigned> &dependencies) {
  llvm::SmallVector<std::string> parameters;
  mlir::Block &entry = kernel.getBody().front();
  for (unsigned index : dependencies) {
    std::string type = weft::riscv_internal::cABIScalarType(entry.getArgument(index).getType());
    parameters.push_back(type + " " + weft::riscv_internal::cABIIdentifier(argumentName(kernel, index)));
  }
  return parameters.empty() ? "void" : llvm::join(parameters, ", ");
}

mlir::LogicalResult emitStorageMetadata(
    KernelOp kernel, StorageOp storage,
    const weft::RISCVLoweringOptions &options, llvm::raw_ostream &output) {
  std::optional<unsigned> pointerIndex =
      entryArgumentIndex(kernel, storage.getPointer());
  if (!pointerIndex)
    return storage.emitError(
        "generated storage metadata must bind an entry pointer");
  PtrType pointer = mlir::cast<PtrType>(storage.getPointer().getType());
  std::string symbol = weft::riscv_internal::cABIIdentifier(kernel.getSymName());
  std::string argument = weft::riscv_internal::cABIIdentifier(argumentName(kernel, *pointerIndex));
  std::string prefix = macroName(symbol + "_" + argument);
  output << "#define WEFT_" << prefix << "_STORAGE_CLASS WEFT_STORAGE_"
         << macroName(pointer.getStorageClass()) << "\n";
  output << "#define WEFT_" << prefix << "_FORMAT \""
         << pointer.getStorageFormat()
         << "\"\n";
  output << "#define WEFT_" << prefix << "_ALIGNMENT "
         << pointer.getAlignment() << "\n";
  output << "#define WEFT_" << prefix << "_NOALIAS "
         << (pointer.getNoAlias() ? 1 : 0) << "\n";
  output << "#define WEFT_" << prefix << "_RESTRICT "
         << (pointer.getNoAlias() || pointer.getRestrictLike() ? 1 : 0)
         << "\n";
  output << "#define WEFT_" << prefix << "_RANK " << storage.getShape().size()
         << "\n";

  llvm::SmallVector<HeaderExpression> extents;
  std::set<unsigned> dependencies;
  for (mlir::Value extent : storage.getExtents()) {
    llvm::DenseSet<mlir::Value> visiting;
    mlir::FailureOr<HeaderExpression> expression =
        storageExpression(kernel, extent, options, visiting);
    if (mlir::failed(expression))
      return mlir::failure();
    if (expression->staticValue && *expression->staticValue <= 0)
      return storage.emitError(
          "storage extent must evaluate to a positive value");
    dependencies.insert(expression->runtimeArguments.begin(),
                        expression->runtimeArguments.end());
    extents.push_back(std::move(*expression));
  }
  for (auto [index, extent] : llvm::enumerate(extents)) {
    std::string parameters =
        queryParameters(kernel, extent.runtimeArguments);
    output << "static inline size_t " << symbol << "__" << argument
           << "_extent_" << index << "(" << parameters << ") { return "
           << extent.spelling << "; }\n";
  }
  output << "static inline size_t " << symbol << "__" << argument
         << "_elements(" << queryParameters(kernel, dependencies)
         << ") { return ";
  for (unsigned index = 0; index < extents.size(); ++index) {
    if (index != 0)
      output << " * ";
    output << "(" << extents[index].spelling << ")";
  }
  output << "; }\n\n";
  return mlir::success();
}

void emitPointerMetadata(KernelOp kernel, unsigned index, PtrType pointer,
                         llvm::raw_ostream &output) {
  std::string prefix = macroName(weft::riscv_internal::cABIIdentifier(kernel.getSymName()) + "_" +
                                 weft::riscv_internal::cABIIdentifier(argumentName(kernel, index)));
  output << "#define WEFT_" << prefix << "_STORAGE_CLASS WEFT_STORAGE_"
         << macroName(pointer.getStorageClass()) << "\n";
  output << "#define WEFT_" << prefix << "_FORMAT \""
         << pointer.getStorageFormat()
         << "\"\n";
  output << "#define WEFT_" << prefix << "_ALIGNMENT "
         << pointer.getAlignment() << "\n";
  output << "#define WEFT_" << prefix << "_NOALIAS "
         << (pointer.getNoAlias() ? 1 : 0) << "\n";
  output << "#define WEFT_" << prefix << "_RESTRICT "
         << (pointer.getNoAlias() || pointer.getRestrictLike() ? 1 : 0)
         << "\n\n";
}

} // namespace

mlir::LogicalResult weft::emitRISCVHeader(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output) {
  llvm::SmallVector<KernelOp> kernels;
  module.walk([&](KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.empty()) {
    module.emitError("Weft generated header requires at least one kernel");
    return mlir::failure();
  }

  llvm::StringSet<> publicSymbols;
  llvm::StringSet<> metadataPrefixes;
  for (KernelOp kernel : kernels) {
    llvm::StringRef symbol = kernel.getSymName();
    if (!isPortableABIIdentifier(symbol))
      return kernel.emitError(
          "generated symbol must be a portable non-keyword C/C++ identifier");
    if (!publicSymbols.insert(symbol).second)
      return kernel.emitError("generated header contains duplicate C entry symbols");
    llvm::StringSet<> parameterNames;
    for (auto [index, nameAttr] : llvm::enumerate(kernel.getArgNames())) {
      llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
      if (!isPortableABIIdentifier(name))
        return kernel.emitError(
            "generated parameter must be a portable non-keyword C/C++ identifier");
      if (!parameterNames.insert(name).second)
        return kernel.emitError(
            "generated header contains duplicate C parameter names");
      mlir::Type type = kernel.getBody().front().getArgument(index).getType();
      if (!mlir::isa<PtrType>(type))
        continue;
      std::string prefix = macroName(symbol.str() + "_" + name.str());
      if (!metadataPrefixes.insert(prefix).second)
        return kernel.emitError(
            "generated pointer metadata names collide after C macro projection");
    }
    for (StorageOp storage : kernel.getBody().front().getOps<StorageOp>()) {
      std::optional<unsigned> pointerIndex =
          entryArgumentIndex(kernel, storage.getPointer());
      if (!pointerIndex)
        return storage.emitError(
            "generated storage metadata must bind an entry pointer");
      std::string base = symbol.str() + "__" +
                         argumentName(kernel, *pointerIndex).str();
      for (unsigned extent = 0; extent < storage.getShape().size(); ++extent) {
        std::string helper = base + "_extent_" + std::to_string(extent);
        if (!publicSymbols.insert(helper).second)
          return storage.emitError(
              "storage extent query collides with a public symbol");
      }
      if (!publicSymbols.insert(base + "_elements").second)
        return storage.emitError(
            "storage element query collides with a public symbol");
    }
  }

  std::string guard = "WEFT_GENERATED";
  for (KernelOp kernel : kernels)
    guard += "_" + macroName(kernel.getSymName());
  guard += "_H";
  output << "#ifndef " << guard << "\n#define " << guard << "\n\n";
  output << "#include <stdbool.h>\n#include <stddef.h>\n#include <stdint.h>\n\n";
  output << "#ifdef __cplusplus\n#define WEFT_GENERATED_RESTRICT __restrict__\n"
            "extern \"C\" {\n#else\n#define WEFT_GENERATED_RESTRICT restrict\n#endif\n\n";
  for (KernelOp kernel : kernels) {
    mlir::FailureOr<std::string> declaration = kernelDeclaration(kernel);
    if (mlir::failed(declaration))
      return mlir::failure();
    output << *declaration << "\n";
  }
  output << "\n#ifdef __cplusplus\n}\n#endif\n"
            "#undef WEFT_GENERATED_RESTRICT\n\n";
  output << "#ifndef WEFT_GENERATED_STORAGE_CLASS_DEFINED\n"
            "#define WEFT_GENERATED_STORAGE_CLASS_DEFINED\n"
            "typedef enum {\n"
            "  WEFT_STORAGE_EXTERNAL = 0,\n"
            "  WEFT_STORAGE_PERSISTENT = 1,\n"
            "  WEFT_STORAGE_WORKSPACE = 2\n"
            "} weft_storage_class;\n"
            "#endif\n\n";

  for (KernelOp kernel : kernels) {
    mlir::Block &entry = kernel.getBody().front();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments()))
      if (auto pointer = mlir::dyn_cast<PtrType>(argument.getType());
          pointer && pointer.getStorageClass() == "external")
        emitPointerMetadata(kernel, index, pointer, output);
    for (StorageOp storage : entry.getOps<StorageOp>())
      if (mlir::failed(emitStorageMetadata(kernel, storage, options, output)))
        return mlir::failure();
  }
  output << "#endif\n";
  return mlir::success();
}
