#include "Weft/Target/RISCVLowering.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace weft;
using namespace weft::kernel;

enum class CValueKind {
  Scalar,
  Pointer,
  Coordinate,
  F32Vector,
  Mask,
  Tuple,
};

struct CValue {
  mlir::Type type;
  CValueKind kind = CValueKind::Scalar;
  std::string spelling;
  bool lanePointer = false;
  std::vector<CValue> fields;
};

std::string sanitize(llvm::StringRef input) {
  std::string result;
  result.reserve(input.size() + 1);
  for (char character : input) {
    unsigned char byte = static_cast<unsigned char>(character);
    result.push_back(std::isalnum(byte) || character == '_' ? character : '_');
  }
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

mlir::Type elementType(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    type = masked.getValueType();
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  return type;
}

bool isF16(mlir::Type type) {
  auto floating = mlir::dyn_cast<mlir::FloatType>(type);
  return floating && floating.getWidth() == 16;
}

std::string scalarCType(mlir::Type type) {
  if (type.isIndex())
    return "size_t";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    std::string prefix = integer.isUnsigned() ? "uint" : "int";
    return prefix + std::to_string(integer.getWidth()) + "_t";
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floating.getWidth() == 16)
      return "_Float16";
    if (floating.getWidth() == 32)
      return "float";
    if (floating.getWidth() == 64)
      return "double";
  }
  return {};
}

std::string pointerCType(PtrType pointer) {
  std::string element = scalarCType(pointer.getElementType());
  if (element.empty())
    return {};
  return (pointer.getAccess() == "read" ? "const " : "") + element + " *";
}

std::string floatLiteral(mlir::FloatAttr attribute) {
  std::ostringstream stream;
  stream << std::setprecision(17) << attribute.getValueAsDouble();
  std::string result = stream.str();
  if (result.find_first_of(".eE") == std::string::npos)
    result += ".0";
  if (mlir::cast<mlir::FloatType>(attribute.getType()).getWidth() <= 32)
    result += "f";
  return result;
}

std::string constantLiteral(mlir::Attribute attribute) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (integer.getType().isInteger(1))
      return integer.getValue().isZero() ? "false" : "true";
    return std::to_string(integer.getInt());
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attribute))
    return floatLiteral(floating);
  return {};
}

bool isTrue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  return value.getType().isInteger(1) && integer && !integer.getValue().isZero();
}

bool isFloatConstant(mlir::Value value, double expected) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto floating = constant
                      ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                      : mlir::FloatAttr{};
  return floating && floating.getValueAsDouble() == expected;
}

bool isTupleField(mlir::Value value, mlir::BlockArgument tuple,
                  int64_t index) {
  auto get = value.getDefiningOp<TupleGetOp>();
  return get && get.getInput() == tuple && get.getIndex() == index;
}

bool matchesScaledSummaryTerm(mlir::Value value, mlir::BlockArgument state,
                              mlir::Value maximum) {
  auto multiply = value.getDefiningOp<BinaryOp>();
  if (!multiply || multiply.getKind() != "mul")
    return false;
  mlir::Value stateSum;
  mlir::Value exponential;
  if (isTupleField(multiply.getLhs(), state, 1)) {
    stateSum = multiply.getLhs();
    exponential = multiply.getRhs();
  } else if (isTupleField(multiply.getRhs(), state, 1)) {
    stateSum = multiply.getRhs();
    exponential = multiply.getLhs();
  } else {
    return false;
  }
  (void)stateSum;
  auto exp = exponential.getDefiningOp<UnaryOp>();
  if (!exp || exp.getKind() != "exp")
    return false;
  auto subtract = exp.getInput().getDefiningOp<BinaryOp>();
  return subtract && subtract.getKind() == "sub" &&
         isTupleField(subtract.getLhs(), state, 0) &&
         subtract.getRhs() == maximum;
}

bool isOnlineSoftmaxSummary(SummaryFoldOp op) {
  if (op.getOrder() != "preserve" || !isTrue(op.getWhere()))
    return false;
  auto resultType = mlir::dyn_cast<TupleType>(op.getResult().getType());
  if (!resultType || resultType.getTypes().size() != 2 ||
      !resultType.getTypes()[0].isF32() || !resultType.getTypes()[1].isF32())
    return false;

  auto identity = op.getIdentity().getDefiningOp<TupleOp>();
  if (!identity || identity.getNumOperands() != 2 ||
      !identity.getOperand(0).getDefiningOp<SpecialValueOp>() ||
      identity.getOperand(0).getDefiningOp<SpecialValueOp>().getKind() !=
          "neg_inf" ||
      !isFloatConstant(identity.getOperand(1), 0.0))
    return false;

  mlir::Block &lift = op.getLift().front();
  auto liftYield = mlir::cast<YieldOp>(lift.getTerminator());
  auto lifted = liftYield.getOperand(0).getDefiningOp<TupleOp>();
  if (!lifted || lifted.getNumOperands() != 2 ||
      lifted.getOperand(0) != lift.getArgument(0) ||
      !isFloatConstant(lifted.getOperand(1), 1.0))
    return false;

  mlir::Block &merge = op.getMerge().front();
  auto mergeYield = mlir::cast<YieldOp>(merge.getTerminator());
  auto merged = mergeYield.getOperand(0).getDefiningOp<TupleOp>();
  if (!merged || merged.getNumOperands() != 2)
    return false;
  auto maximum = merged.getOperand(0).getDefiningOp<BinaryOp>();
  if (!maximum || maximum.getKind() != "max")
    return false;
  bool maxOperandsMatch =
      (isTupleField(maximum.getLhs(), merge.getArgument(0), 0) &&
       isTupleField(maximum.getRhs(), merge.getArgument(1), 0)) ||
      (isTupleField(maximum.getLhs(), merge.getArgument(1), 0) &&
       isTupleField(maximum.getRhs(), merge.getArgument(0), 0));
  if (!maxOperandsMatch)
    return false;
  auto sum = merged.getOperand(1).getDefiningOp<BinaryOp>();
  if (!sum || sum.getKind() != "add")
    return false;
  return (matchesScaledSummaryTerm(sum.getLhs(), merge.getArgument(0),
                                   maximum.getResult()) &&
          matchesScaledSummaryTerm(sum.getRhs(), merge.getArgument(1),
                                   maximum.getResult())) ||
         (matchesScaledSummaryTerm(sum.getLhs(), merge.getArgument(1),
                                   maximum.getResult()) &&
          matchesScaledSummaryTerm(sum.getRhs(), merge.getArgument(0),
                                   maximum.getResult()));
}

class KernelEmitter {
public:
  KernelEmitter(KernelOp kernel, const RISCVLoweringOptions &options,
                llvm::raw_ostream &output)
      : kernel(kernel), options(options), output(output) {}

  mlir::LogicalResult emit() {
    mlir::Block &body = kernel.getBody().front();
    llvm::SmallVector<std::string> parameters;
    auto names = kernel.getArgNames();
    auto kinds = kernel.getArgKinds();
    for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
      llvm::StringRef kind =
          mlir::cast<mlir::StringAttr>(kinds[index]).getValue();
      if (kind == "constexpr")
        continue;
      std::string name = sanitize(
          mlir::cast<mlir::StringAttr>(names[index]).getValue());
      mlir::Type type = argument.getType();
      std::string cType;
      CValueKind valueKind = CValueKind::Scalar;
      if (auto pointer = mlir::dyn_cast<PtrType>(type)) {
        cType = pointerCType(pointer);
        valueKind = CValueKind::Pointer;
        if (pointer.getNoAlias())
          cType += " restrict";
      } else {
        cType = scalarCType(type);
      }
      if (cType.empty())
        return kernel.emitError(
            "RISC-V intrinsic C ABI has an unsupported argument type");
      parameters.push_back(cType + " " + name);
      values[argument] = CValue{type, valueKind, name};
    }

    mlir::Type returnType = kernel.getReturnType();
    std::string returnTypeSpelling = mlir::isa<mlir::NoneType>(returnType)
                                         ? "void"
                                         : scalarCType(returnType);
    if (returnTypeSpelling.empty())
      return kernel.emitError(
          "RISC-V intrinsic C ABI has an unsupported return type");
    line(returnTypeSpelling + " " + sanitize(kernel.getSymName()) + "(" +
         llvm::join(parameters, ", ") + ") {");
    ++indent;
    for (mlir::Operation &operation : body) {
      if (auto returnOp = mlir::dyn_cast<ReturnOp>(operation)) {
        if (returnOp.getNumOperands() == 0)
          line("return;");
        else
          line("return " + require(returnOp.getOperand(0)).spelling + ";");
        continue;
      }
      if (mlir::failed(emitOperation(&operation)))
        return mlir::failure();
    }
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

private:
  KernelOp kernel;
  const RISCVLoweringOptions &options;
  llvm::raw_ostream &output;
  llvm::DenseMap<mlir::Value, CValue> values;
  llvm::DenseSet<mlir::Operation *> consumed;
  unsigned indent = 0;
  unsigned nextValue = 0;
  unsigned nextLoop = 0;
  bool inVLA = false;
  std::string activeVL;

  void line(llvm::Twine text) {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix) {
    return "__weft_" + prefix.str() + std::to_string(nextValue++);
  }

  CValue require(mlir::Value value) const {
    auto found = values.find(value);
    return found == values.end() ? CValue{} : found->second;
  }

  std::string expression(mlir::Value value) const {
    if (auto constant = value.getDefiningOp<ConstantOp>())
      return constantLiteral(constant.getValue());
    if (auto special = value.getDefiningOp<SpecialValueOp>()) {
      if (special.getKind() == "neg_inf")
        return "-INFINITY";
      if (special.getKind() == "pos_inf")
        return "INFINITY";
      if (special.getKind() == "nan")
        return "NAN";
    }
    return require(value).spelling;
  }

  mlir::LogicalResult emitOperation(mlir::Operation *operation) {
    if (consumed.contains(operation))
      return mlir::success();
    if (auto op = mlir::dyn_cast<ConstantOp>(operation))
      return emitConstant(op);
    if (auto op = mlir::dyn_cast<MetaValueOp>(operation))
      return emitMeta(op);
    if (auto op = mlir::dyn_cast<ForOp>(operation))
      return emitFor(op);
    if (auto op = mlir::dyn_cast<VLAOp>(operation))
      return emitVLA(op);
    if (auto op = mlir::dyn_cast<PtrAddOp>(operation))
      return emitPtrAdd(op);
    if (auto op = mlir::dyn_cast<UnaryOp>(operation))
      return emitUnary(op);
    if (auto op = mlir::dyn_cast<BinaryOp>(operation))
      return emitBinary(op);
    if (auto op = mlir::dyn_cast<CompareOp>(operation))
      return emitCompare(op);
    if (auto op = mlir::dyn_cast<CastOp>(operation))
      return emitCast(op);
    if (auto op = mlir::dyn_cast<SelectOp>(operation))
      return emitSelect(op);
    if (auto op = mlir::dyn_cast<TupleOp>(operation))
      return emitTuple(op);
    if (auto op = mlir::dyn_cast<TupleGetOp>(operation))
      return emitTupleGet(op);
    if (auto op = mlir::dyn_cast<SpecialValueOp>(operation))
      return emitSpecial(op);
    if (auto op = mlir::dyn_cast<InvalidOp>(operation)) {
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Scalar, {}};
      return mlir::success();
    }
    if (auto op = mlir::dyn_cast<LoadOp>(operation))
      return emitLoad(op);
    if (auto op = mlir::dyn_cast<StoreOp>(operation))
      return emitStore(op);
    if (auto op = mlir::dyn_cast<ReduceOp>(operation))
      return emitReduce(op);
    if (auto op = mlir::dyn_cast<SummaryFoldOp>(operation))
      return op.emitError(
          "summary_fold must be lowered by its enclosing VLA region");
    if (mlir::isa<YieldOp, ReturnOp>(operation))
      return mlir::success();
    return operation->emitError(
        "RISC-V target lowering does not implement this Kernel IR primitive");
  }

  mlir::LogicalResult emitConstant(ConstantOp op) {
    std::string literal = constantLiteral(op.getValue());
    if (literal.empty())
      return op.emitError("RISC-V target cannot spell constant");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, literal};
    return mlir::success();
  }

  mlir::LogicalResult emitMeta(MetaValueOp op) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(op.getInput());
    if (!argument)
      return op.emitError("meta value must reference a kernel argument");
    auto names = kernel.getArgNames();
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(
                               names[argument.getArgNumber()])
                               .getValue();
    auto binding = options.metaBindings.find(name);
    if (binding == options.metaBindings.end())
      return op.emitError() << "missing --meta binding for " << name;
    std::string value = std::to_string(binding->second);
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, value};
    return mlir::success();
  }

  mlir::LogicalResult emitFor(ForOp op) {
    llvm::SmallVector<CValue> carried;
    for (auto [result, initial] :
         llvm::zip(op.getResults(), op.getOperands().drop_front(3))) {
      CValue init = require(initial);
      if (init.spelling.empty() || init.kind != CValueKind::Scalar)
        return op.emitError(
            "ordered range currently carries only scalar values");
      CValue value{result.getType(), CValueKind::Scalar, fresh("carry")};
      std::string type = scalarCType(result.getType());
      line(type + " " + value.spelling + " = " + init.spelling + ";");
      values[result] = value;
      carried.push_back(value);
    }
    CValue lower = require(op.getLower());
    CValue upper = require(op.getUpper());
    CValue step = require(op.getStep());
    if (lower.spelling.empty() || upper.spelling.empty() ||
        step.spelling.empty())
      return op.emitError("ordered range has unavailable bounds");
    std::string induction = "__weft_i" + std::to_string(nextLoop++);
    line("for (size_t " + induction + " = " + lower.spelling + "; " +
         induction + " < " + upper.spelling + "; " + induction + " += " +
         step.spelling + ") {");
    ++indent;
    mlir::Block &body = op.getBody().front();
    values[body.getArgument(0)] =
        CValue{body.getArgument(0).getType(), CValueKind::Scalar, induction};
    for (auto [argument, value] :
         llvm::zip(body.getArguments().drop_front(), carried))
      values[argument] = value;
    for (mlir::Operation &nested : body.without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (auto [destination, yielded] :
         llvm::zip(carried, yield.getOperands())) {
      CValue value = require(yielded);
      if (value.spelling.empty())
        return yield.emitError("ordered range yielded an unavailable value");
      line(destination.spelling + " = " + value.spelling + ";");
    }
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitVLA(VLAOp op) {
    if (inVLA)
      return op.emitError("nested VLA regions are not supported");
    if (!options.target.hasRVV)
      return op.emitError(
          "VLA requires RVV on the selected target; no scalar fallback exists");
    if (mlir::succeeded(tryEmitSoftmaxEnvelope(op)))
      return mlir::success();
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Operation *, CValue> aggregates;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        if (!isOnlineSoftmaxSummary(summary))
          return summary.emitError(
              "RISC-V target does not implement this summary algebra");
        auto identity = summary.getIdentity().getDefiningOp<TupleOp>();
        CValue maximum{mlir::Float32Type::get(kernel.getContext()),
                       CValueKind::Scalar, fresh("summary_max")};
        CValue sum{mlir::Float32Type::get(kernel.getContext()),
                   CValueKind::Scalar, fresh("summary_sum")};
        line("float " + maximum.spelling + " = " +
             expression(identity.getOperand(0)) + ";");
        line("float " + sum.spelling + " = " +
             expression(identity.getOperand(1)) + ";");
        CValue aggregate{summary.getResult().getType(), CValueKind::Tuple, {}};
        aggregate.fields = {maximum, sum};
        aggregates[summary.getOperation()] = aggregate;
        values[summary.getResult()] = aggregate;
        continue;
      }
      auto reduce = mlir::dyn_cast<ReduceOp>(nested);
      if (!reduce)
        continue;
      std::string identity = expression(reduce.getIdentity());
      if (identity.empty())
        return reduce.emitError("VLA reduction identity is unavailable");
      CValue aggregate{reduce.getResult().getType(), CValueKind::Scalar,
                       fresh("reduce")};
      std::string type = scalarCType(reduce.getResult().getType());
      line(type + " " + aggregate.spelling + " = " + identity + ";");
      aggregates[reduce.getOperation()] = aggregate;
      values[reduce.getResult()] = aggregate;
    }

    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (begin.spelling.empty() || end.spelling.empty())
      return op.emitError("VLA bounds are unavailable");
    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end.spelling +
         " - " + strip + ");");

    bool previousInVLA = inVLA;
    std::string previousVL = activeVL;
    inVLA = true;
    activeVL = vl;
    values[body.getArgument(0)] =
        CValue{body.getArgument(0).getType(), CValueKind::Coordinate, strip};
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
        if (mlir::failed(emitVectorReduce(reduce,
                                          aggregates[reduce.getOperation()])))
          return mlir::failure();
        continue;
      }
      if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        if (mlir::failed(emitOnlineSoftmaxSummary(
                summary, aggregates[summary.getOperation()])))
          return mlir::failure();
        continue;
      }
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    }
    line(strip + " += " + vl + ";");
    inVLA = previousInVLA;
    activeVL = previousVL;
    --indent;
    line("}");

    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (auto [result, yielded] :
         llvm::zip(op.getResults(), yield.getOperands())) {
      CValue value = require(yielded);
      if ((value.spelling.empty() && value.kind != CValueKind::Tuple) ||
          (value.kind != CValueKind::Scalar &&
           value.kind != CValueKind::Tuple))
        return op.emitError("VLA result is not a materialized scalar aggregate");
      value.type = result.getType();
      values[result] = value;
    }
    return mlir::success();
  }

  mlir::LogicalResult emitPtrAdd(PtrAddOp op) {
    CValue base = require(op.getBase());
    CValue offset = require(op.getOffset());
    if (base.spelling.empty() || offset.spelling.empty())
      return op.emitError("pointer addition has an unavailable operand");
    if (base.kind != CValueKind::Pointer)
      return op.emitError("pointer addition base is not a pointer");
    bool lanePointer = offset.kind == CValueKind::Coordinate;
    if (inVLA && offset.kind != CValueKind::Coordinate && base.lanePointer)
      lanePointer = true;
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Pointer,
               "(" + base.spelling + " + " + offset.spelling + ")",
               lanePointer};
    return mlir::success();
  }

  std::optional<std::string> pointerBase(mlir::Value value,
                                         mlir::Value coordinate) {
    if (value == coordinate)
      return std::nullopt;
    if (auto pointer = value.getDefiningOp<PtrAddOp>()) {
      if (pointer.getOffset() == coordinate)
        return pointerBase(pointer.getBase(), coordinate);
      std::optional<std::string> base =
          pointerBase(pointer.getBase(), coordinate);
      std::string offset = expression(pointer.getOffset());
      if (!base || offset.empty())
        return std::nullopt;
      return "(" + *base + " + " + offset + ")";
    }
    CValue materialized = require(value);
    if (materialized.kind != CValueKind::Pointer ||
        materialized.spelling.empty())
      return std::nullopt;
    return materialized.spelling;
  }

  bool sameBound(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == rhs)
      return true;
    std::string lhsExpression = expression(lhs);
    std::string rhsExpression = expression(rhs);
    return !lhsExpression.empty() && lhsExpression == rhsExpression;
  }

  mlir::LogicalResult tryEmitSoftmaxEnvelope(VLAOp producer) {
    SummaryFoldOp summary;
    for (mlir::Operation &operation :
         producer.getBody().front().without_terminator()) {
      if (auto candidate = mlir::dyn_cast<SummaryFoldOp>(operation)) {
        if (summary)
          return mlir::failure();
        summary = candidate;
      }
    }
    if (!summary || !isOnlineSoftmaxSummary(summary) ||
        producer.getNumResults() != 1)
      return mlir::failure();

    TupleGetOp maximumGet;
    TupleGetOp sumGet;
    for (mlir::Operation *user : producer.getResult(0).getUsers()) {
      auto get = mlir::dyn_cast<TupleGetOp>(user);
      if (!get)
        return mlir::failure();
      if (get.getIndex() == 0 && !maximumGet)
        maximumGet = get;
      else if (get.getIndex() == 1 && !sumGet)
        sumGet = get;
      else
        return mlir::failure();
    }
    if (!maximumGet || !sumGet || maximumGet.getResult().use_empty() ||
        sumGet.getResult().use_empty())
      return mlir::failure();

    VLAOp consumer;
    for (mlir::Operation *user : maximumGet.getResult().getUsers()) {
      auto parent = user->getParentOfType<VLAOp>();
      if (!parent || (consumer && consumer != parent))
        return mlir::failure();
      consumer = parent;
    }
    for (mlir::Operation *user : sumGet.getResult().getUsers())
      if (user->getParentOfType<VLAOp>() != consumer)
        return mlir::failure();
    if (!consumer || consumer->getBlock() != producer->getBlock() ||
        !sameBound(producer.getBegin(), consumer.getBegin()) ||
        !sameBound(producer.getEnd(), consumer.getEnd()))
      return mlir::failure();

    LoadOp consumerLoad;
    StoreOp consumerStore;
    for (mlir::Operation &operation :
         consumer.getBody().front().without_terminator()) {
      if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
        if (consumerLoad)
          return mlir::failure();
        consumerLoad = load;
      }
      if (auto store = mlir::dyn_cast<StoreOp>(operation)) {
        if (consumerStore)
          return mlir::failure();
        consumerStore = store;
      }
    }
    if (!consumerLoad || !consumerStore || !isTrue(consumerLoad.getWhere()) ||
        !isTrue(consumerStore.getWhere()))
      return mlir::failure();
    auto divide = consumerStore.getValue().getDefiningOp<BinaryOp>();
    if (!divide || divide.getKind() != "div" ||
        divide.getRhs() != sumGet.getResult())
      return mlir::failure();
    auto exponential = divide.getLhs().getDefiningOp<UnaryOp>();
    auto subtract = exponential
                        ? exponential.getInput().getDefiningOp<BinaryOp>()
                        : BinaryOp{};
    if (!exponential || exponential.getKind() != "exp" || !subtract ||
        subtract.getKind() != "sub" ||
        subtract.getLhs() != consumerLoad.getResult() ||
        subtract.getRhs() != maximumGet.getResult())
      return mlir::failure();
    auto producerLoad = summary.getInput().getDefiningOp<LoadOp>();
    if (!producerLoad || !isTrue(producerLoad.getWhere()))
      return mlir::failure();

    mlir::Value producerCoordinate =
        producer.getBody().front().getArgument(0);
    mlir::Value consumerCoordinate =
        consumer.getBody().front().getArgument(0);
    std::optional<std::string> input =
        pointerBase(producerLoad.getPointer(), producerCoordinate);
    std::optional<std::string> consumerInput =
        pointerBase(consumerLoad.getPointer(), consumerCoordinate);
    std::optional<std::string> outputPointer =
        pointerBase(consumerStore.getPointer(), consumerCoordinate);
    if (!input || !consumerInput || !outputPointer || *input != *consumerInput)
      return mlir::failure();

    std::string begin = expression(producer.getBegin());
    std::string end = expression(producer.getEnd());
    if (begin.empty() || end.empty())
      return mlir::failure();
    std::string maximum = fresh("softmax_max");
    std::string strip = fresh("softmax_i");
    std::string vl = fresh("vl");
    std::string inputVector = fresh("softmax_x");
    std::string seed = fresh("softmax_seed");
    std::string partial = fresh("softmax_partial");
    line("float " + maximum + " = -INFINITY;");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + inputVector + " = __riscv_vle32_v_f32m2(" +
         *input + " + " + strip + ", " + vl + ");");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" + maximum +
         ", 1);");
    line("vfloat32m1_t " + partial +
         " = __riscv_vfredmax_vs_f32m2_f32m1(" + inputVector + ", " + seed +
         ", " + vl + ");");
    line(maximum + " = __riscv_vfmv_f_s_f32m1_f32(" + partial + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    std::string sum = fresh("softmax_sum");
    std::string shifted = fresh("softmax_shifted");
    std::string exponentials = fresh("softmax_exp");
    std::string sumSeed = fresh("softmax_sum_seed");
    std::string sumPartial = fresh("softmax_sum_partial");
    line("float " + sum + " = 0.0f;");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + inputVector + " = __riscv_vle32_v_f32m2(" +
         *input + " + " + strip + ", " + vl + ");");
    line("vfloat32m2_t " + shifted + " = __riscv_vfsub_vf_f32m2(" +
         inputVector + ", " + maximum + ", " + vl + ");");
    line("vfloat32m2_t " + exponentials + " = __weft_exp_f32m2(" + shifted +
         ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + *outputPointer + " + " + strip + ", " +
         exponentials + ", " + vl + ");");
    line("vfloat32m1_t " + sumSeed + " = __riscv_vfmv_v_f_f32m1(" + sum +
         ", 1);");
    line("vfloat32m1_t " + sumPartial +
         " = __riscv_vfredusum_vs_f32m2_f32m1(" + exponentials + ", " +
         sumSeed + ", " + vl + ");");
    line(sum + " = __riscv_vfmv_f_s_f32m1_f32(" + sumPartial + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    std::string inverse = fresh("softmax_inverse");
    std::string outputVector = fresh("softmax_y");
    line("const float " + inverse + " = 1.0f / " + sum + ";");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + outputVector + " = __riscv_vle32_v_f32m2(" +
         *outputPointer + " + " + strip + ", " + vl + ");");
    line(outputVector + " = __riscv_vfmul_vf_f32m2(" + outputVector + ", " +
         inverse + ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + *outputPointer + " + " + strip + ", " +
         outputVector + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    CValue maxValue{mlir::Float32Type::get(kernel.getContext()),
                    CValueKind::Scalar, maximum};
    CValue sumValue{mlir::Float32Type::get(kernel.getContext()),
                    CValueKind::Scalar, sum};
    CValue tuple{producer.getResult(0).getType(), CValueKind::Tuple, {}};
    tuple.fields = {maxValue, sumValue};
    values[summary.getResult()] = tuple;
    values[producer.getResult(0)] = tuple;
    consumed.insert(maximumGet.getOperation());
    consumed.insert(sumGet.getOperation());
    consumed.insert(consumer.getOperation());
    return mlir::success();
  }

  std::string scalarBinary(llvm::StringRef kind, llvm::StringRef lhs,
                           llvm::StringRef rhs) {
    llvm::StringRef spelling = llvm::StringSwitch<llvm::StringRef>(kind)
                                   .Case("add", "+")
                                   .Case("sub", "-")
                                   .Case("mul", "*")
                                   .Case("div", "/")
                                   .Case("mod", "%")
                                   .Case("and", "&")
                                   .Case("or", "|")
                                   .Case("xor", "^")
                                   .Case("shl", "<<")
                                   .Case("shr", ">>")
                                   .Default("");
    if (!spelling.empty())
      return "(" + lhs.str() + " " + spelling.str() + " " + rhs.str() + ")";
    if (kind == "max")
      return "fmaxf(" + lhs.str() + ", " + rhs.str() + ")";
    if (kind == "min")
      return "fminf(" + lhs.str() + ", " + rhs.str() + ")";
    return {};
  }

  mlir::LogicalResult emitBinary(BinaryOp op) {
    CValue lhs = require(op.getLhs());
    CValue rhs = require(op.getRhs());
    bool lhsVector = lhs.kind == CValueKind::F32Vector;
    bool rhsVector = rhs.kind == CValueKind::F32Vector;
    if (!lhsVector && !rhsVector) {
      std::string expression =
          scalarBinary(op.getKind(), lhs.spelling, rhs.spelling);
      if (expression.empty())
        return op.emitError("RISC-V scalar lowering does not implement binary kind");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Scalar, expression};
      return mlir::success();
    }
    if (!inVLA || (!elementType(op.getResult().getType()).isF32()))
      return op.emitError(
          "RVV pointwise lowering currently requires VLA f32 values");
    std::string intrinsic;
    std::string first = lhs.spelling;
    std::string second = rhs.spelling;
    if (lhsVector && rhsVector) {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vv_f32m2")
                      .Case("sub", "__riscv_vfsub_vv_f32m2")
                      .Case("mul", "__riscv_vfmul_vv_f32m2")
                      .Case("div", "__riscv_vfdiv_vv_f32m2")
                      .Case("max", "__riscv_vfmax_vv_f32m2")
                      .Case("min", "__riscv_vfmin_vv_f32m2")
                      .Default("");
    } else if (lhsVector) {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vf_f32m2")
                      .Case("sub", "__riscv_vfsub_vf_f32m2")
                      .Case("mul", "__riscv_vfmul_vf_f32m2")
                      .Case("div", "__riscv_vfdiv_vf_f32m2")
                      .Case("max", "__riscv_vfmax_vf_f32m2")
                      .Case("min", "__riscv_vfmin_vf_f32m2")
                      .Default("");
    } else {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vf_f32m2")
                      .Case("sub", "__riscv_vfrsub_vf_f32m2")
                      .Case("mul", "__riscv_vfmul_vf_f32m2")
                      .Case("div", "__riscv_vfrdiv_vf_f32m2")
                      .Case("max", "__riscv_vfmax_vf_f32m2")
                      .Case("min", "__riscv_vfmin_vf_f32m2")
                      .Default("");
      std::swap(first, second);
    }
    if (intrinsic.empty())
      return op.emitError("RVV pointwise lowering does not implement binary kind");
    std::string name = fresh("v");
    line("vfloat32m2_t " + name + " = " + intrinsic + "(" + first + ", " +
         second + ", " + activeVL + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::F32Vector, name};
    return mlir::success();
  }

  mlir::LogicalResult emitUnary(UnaryOp op) {
    CValue input = require(op.getInput());
    if (input.kind == CValueKind::F32Vector) {
      std::string name = fresh("v");
      if (op.getKind() == "neg")
        line("vfloat32m2_t " + name + " = __riscv_vfneg_v_f32m2(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "abs")
        line("vfloat32m2_t " + name + " = __riscv_vfabs_v_f32m2(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "exp")
        line("vfloat32m2_t " + name + " = __weft_exp_f32m2(" +
             input.spelling + ", " + activeVL + ");");
      else
        return op.emitError(
            "RVV pointwise lowering does not implement unary kind");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      return mlir::success();
    }
    std::string expression;
    if (op.getKind() == "neg")
      expression = "(-" + input.spelling + ")";
    else if (op.getKind() == "exp")
      expression = "expf(" + input.spelling + ")";
    else if (op.getKind() == "sqrt")
      expression = "sqrtf(" + input.spelling + ")";
    else if (op.getKind() == "rsqrt")
      expression = "(1.0f / sqrtf(" + input.spelling + "))";
    else if (op.getKind() == "abs")
      expression = "fabsf(" + input.spelling + ")";
    else
      return op.emitError(
          "RISC-V scalar lowering does not implement unary kind");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, expression};
    return mlir::success();
  }

  mlir::LogicalResult emitCompare(CompareOp op) {
    CValue lhs = require(op.getLhs());
    CValue rhs = require(op.getRhs());
    if (lhs.kind != CValueKind::Scalar || rhs.kind != CValueKind::Scalar)
      return op.emitError("RVV predicate lowering is not implemented yet");
    llvm::StringRef spelling =
        llvm::StringSwitch<llvm::StringRef>(op.getPredicate())
            .Case("eq", "==")
            .Case("ne", "!=")
            .Case("lt", "<")
            .Case("le", "<=")
            .Case("gt", ">")
            .Case("ge", ">=")
            .Default("");
    if (spelling.empty())
      return op.emitError("comparison predicate is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               "(" + lhs.spelling + " " + spelling.str() + " " +
                   rhs.spelling + ")"};
    return mlir::success();
  }

  mlir::LogicalResult emitCast(CastOp op) {
    CValue input = require(op.getInput());
    if (input.kind != CValueKind::Scalar)
      return op.emitError("RVV cast lowering is not implemented yet");
    std::string target = scalarCType(elementType(op.getResult().getType()));
    if (target.empty())
      return op.emitError("cast target type is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               "((" + target + ")(" + input.spelling + "))"};
    return mlir::success();
  }

  mlir::LogicalResult emitSelect(SelectOp op) {
    CValue predicate = require(op.getPredicate());
    CValue trueValue = require(op.getTrueValue());
    CValue falseValue = require(op.getFalseValue());
    if (predicate.kind != CValueKind::Scalar ||
        trueValue.kind != CValueKind::Scalar ||
        falseValue.kind != CValueKind::Scalar)
      return op.emitError("RVV select lowering is not implemented yet");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               "(" + predicate.spelling + " ? " + trueValue.spelling + " : " +
                   falseValue.spelling + ")"};
    return mlir::success();
  }

  mlir::LogicalResult emitTuple(TupleOp op) {
    CValue tuple{op.getResult().getType(), CValueKind::Tuple, {}};
    for (mlir::Value operand : op.getOperands()) {
      CValue field = require(operand);
      if (field.spelling.empty())
        return op.emitError("tuple field is unavailable");
      tuple.fields.push_back(std::move(field));
    }
    values[op.getResult()] = std::move(tuple);
    return mlir::success();
  }

  mlir::LogicalResult emitTupleGet(TupleGetOp op) {
    CValue tuple = require(op.getInput());
    if (tuple.kind != CValueKind::Tuple || op.getIndex() < 0 ||
        static_cast<size_t>(op.getIndex()) >= tuple.fields.size())
      return op.emitError("tuple field is unavailable");
    CValue field = tuple.fields[op.getIndex()];
    field.type = op.getResult().getType();
    values[op.getResult()] = field;
    return mlir::success();
  }

  mlir::LogicalResult emitSpecial(SpecialValueOp op) {
    std::string value;
    if (op.getKind() == "neg_inf")
      value = "-INFINITY";
    else if (op.getKind() == "pos_inf")
      value = "INFINITY";
    else if (op.getKind() == "nan")
      value = "NAN";
    else
      return op.emitError("special value kind is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, value};
    return mlir::success();
  }

  mlir::LogicalResult emitLoad(LoadOp op) {
    CValue pointer = require(op.getPointer());
    if (pointer.kind != CValueKind::Pointer || pointer.spelling.empty())
      return op.emitError("load pointer is unavailable");
    if (inVLA && pointer.lanePointer) {
      if (!elementType(op.getResult().getType()).isF32() ||
          !isTrue(op.getWhere()) ||
          !mlir::isa<mlir::NoneType>(op.getOther().getType()))
        return op.emitError(
            "RVV load currently requires an all-active f32 VLA value");
      std::string name = fresh("load");
      line("vfloat32m2_t " + name + " = __riscv_vle32_v_f32m2(" +
           pointer.spelling + ", " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      return mlir::success();
    }
    CValue where = require(op.getWhere());
    CValue other = require(op.getOther());
    std::string read = "(*" + pointer.spelling + ")";
    std::string expression;
    if (isTrue(op.getWhere()))
      expression = read;
    else if (!other.spelling.empty())
      expression = "(" + where.spelling + " ? " + read + " : " +
                   other.spelling + ")";
    else
      return op.emitError("masked scalar load requires an explicit other value");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, expression};
    return mlir::success();
  }

  mlir::LogicalResult emitStore(StoreOp op) {
    CValue pointer = require(op.getPointer());
    CValue value = require(op.getValue());
    if (pointer.kind != CValueKind::Pointer || pointer.spelling.empty())
      return op.emitError("store pointer is unavailable");
    if (inVLA && pointer.lanePointer) {
      if (value.kind != CValueKind::F32Vector || !isTrue(op.getWhere()))
        return op.emitError(
            "RVV store currently requires an all-active f32 VLA value");
      line("__riscv_vse32_v_f32m2(" + pointer.spelling + ", " +
           value.spelling + ", " + activeVL + ");");
      return mlir::success();
    }
    CValue where = require(op.getWhere());
    if (value.kind != CValueKind::Scalar)
      return op.emitError("scalar store requires a scalar value");
    if (isTrue(op.getWhere()))
      line("*" + pointer.spelling + " = " + value.spelling + ";");
    else
      line("if (" + where.spelling + ") *" + pointer.spelling + " = " +
           value.spelling + ";");
    return mlir::success();
  }

  mlir::LogicalResult emitReduce(ReduceOp op) {
    if (inVLA)
      return op.emitError("VLA reduction must be owned by its VLA lowering");
    return op.emitError(
        "block reduction has no RISC-V lowering yet; no scalar fallback exists");
  }

  mlir::LogicalResult emitVectorReduce(ReduceOp op,
                                       const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (input.kind != CValueKind::F32Vector || op.getAxis() != -1 ||
        !op.getResult().getType().isF32() || !isTrue(op.getWhere()))
      return op.emitError(
          "RVV reduction requires an all-active f32 VLA input");
    std::string seed = fresh("seed");
    std::string partial = fresh("partial");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
         aggregate.spelling + ", 1);");
    if (op.getKind() == "add")
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredusum_vs_f32m2_f32m1(" + input.spelling + ", " +
           seed + ", " + activeVL + ");");
    else if (op.getKind() == "max")
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredmax_vs_f32m2_f32m1(" + input.spelling + ", " +
           seed + ", " + activeVL + ");");
    else
      return op.emitError("RVV reduction kind is unsupported");
    line(aggregate.spelling + " = __riscv_vfmv_f_s_f32m1_f32(" + partial +
         ");");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitOnlineSoftmaxSummary(
      SummaryFoldOp op, const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (!isOnlineSoftmaxSummary(op) ||
        input.kind != CValueKind::F32Vector ||
        aggregate.kind != CValueKind::Tuple || aggregate.fields.size() != 2)
      return op.emitError(
          "online summary requires an all-active f32 VLA input");
    const CValue &maximum = aggregate.fields[0];
    const CValue &sum = aggregate.fields[1];
    std::string maxSeed = fresh("summary_max_seed");
    std::string maxVector = fresh("summary_max_vector");
    std::string stripMaximum = fresh("summary_strip_max");
    std::string shifted = fresh("summary_shifted");
    std::string exponentials = fresh("summary_exp");
    std::string sumSeed = fresh("summary_sum_seed");
    std::string sumVector = fresh("summary_sum_vector");
    std::string stripSum = fresh("summary_strip_sum");
    std::string mergedMaximum = fresh("summary_merged_max");
    line("vfloat32m1_t " + maxSeed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + maxVector +
         " = __riscv_vfredmax_vs_f32m2_f32m1(" + input.spelling + ", " +
         maxSeed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + maxVector + ");");
    line("vfloat32m2_t " + shifted + " = __riscv_vfsub_vf_f32m2(" +
         input.spelling + ", " + stripMaximum + ", " + activeVL + ");");
    line("vfloat32m2_t " + exponentials + " = __weft_exp_f32m2(" +
         shifted + ", " + activeVL + ");");
    line("vfloat32m1_t " + sumSeed +
         " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
    line("vfloat32m1_t " + sumVector +
         " = __riscv_vfredusum_vs_f32m2_f32m1(" + exponentials + ", " +
         sumSeed + ", " + activeVL + ");");
    line("const float " + stripSum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + sumVector + ");");
    line("const float " + mergedMaximum + " = fmaxf(" + maximum.spelling +
         ", " + stripMaximum + ");");
    line(sum.spelling + " = " + sum.spelling + " * expf(" +
         maximum.spelling + " - " + mergedMaximum + ") + " + stripSum +
         " * expf(" + stripMaximum + " - " + mergedMaximum + ");");
    line(maximum.spelling + " = " + mergedMaximum + ";");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }
};

void emitPrelude(llvm::raw_ostream &output, bool usesExp) {
  output << R"c(#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

)c";
  if (!usesExp)
    return;
  output << R"c(static inline vfloat32m2_t __weft_exp_f32m2(
    vfloat32m2_t x, size_t vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
  const vfloat32m2_t z =
      __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b0 =
      __riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl);
  const vfloat32m2_t b =
      __riscv_vfnmsac_vf_f32m2(b0, 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t z_bits = __riscv_vreinterpret_v_f32m2_u32m2(z);
  const vuint32m2_t e = __riscv_vsll_vx_u32m2(z_bits, 23, vl);
  const vuint32m2_t k_bits =
      __riscv_vadd_vx_u32m2(e, UINT32_C(0x3f800000), vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(k_bits);
  const vfloat32m2_t abs_n = __riscv_vfabs_v_f32m2(n, vl);
  const vbool16_t extreme =
      __riscv_vmfgt_vf_f32m2_b16(abs_n, 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j0 =
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl);
  const vfloat32m2_t j1 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl),
      0x1.555e66p-3f, b, vl);
  const vfloat32m2_t j2 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl),
      0x1.0e4020p-7f, b, vl);
  const vfloat32m2_t j3 = __riscv_vfmacc_vv_f32m2(j1, j2, u, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(j0, j3, u, vl);
  const vfloat32m2_t fast = __riscv_vfmacc_vv_f32m2(k, k, j, vl);
  if (__riscv_vcpop_m_b16(extreme, vl) == 0)
    return fast;

  const vbool16_t negative =
      __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(
      __riscv_vmv_v_x_u32m2(0, vl), UINT32_C(0x82000000), negative, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(d, UINT32_C(0x7f000000), vl));
  const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t corrected = __riscv_vfmul_vv_f32m2(
      __riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl);
  const vfloat32m2_t bounded =
      __riscv_vmerge_vvm_f32m2(fast, corrected, extreme, vl);
  const vbool16_t overflow = __riscv_vmfgt_vf_f32m2_b16(
      __riscv_vfabs_v_f32m2(n, vl), 192.0f, vl);
  return __riscv_vmerge_vvm_f32m2(
      bounded, __riscv_vfmul_vv_f32m2(s1, s1, vl), overflow, vl);
}

)c";
}

} // namespace

mlir::LogicalResult weft::lowerToRISCVIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output) {
  if (!options.target.hasRVV)
    return module.emitError(
        "the intrinsic C target requires RVV; no fallback backend is installed");
  bool usesExp = false;
  module.walk([&](UnaryOp op) { usesExp |= op.getKind() == "exp"; });
  emitPrelude(output, usesExp);

  llvm::SmallVector<KernelOp> kernels;
  for (KernelOp kernel : module.getOps<KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("RISC-V lowering requires a Weft kernel");
  for (KernelOp kernel : kernels) {
    KernelEmitter emitter(kernel, options, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
  }
  return mlir::success();
}
