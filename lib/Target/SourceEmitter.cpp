#include "Weft/Target/SourceEmitter.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace weft;
using namespace weft::execution;
using namespace weft::kernel;

struct ValueInfo {
  mlir::Type type;
  std::string value;
  std::string validity;
  llvm::SmallVector<std::string> shape;
  std::vector<ValueInfo> fields;

  bool isMasked() const { return !validity.empty(); }
  bool isShaped() const { return !shape.empty(); }
};

mlir::Type unwrapMasked(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return masked.getValueType();
  return type;
}

mlir::Type elementType(mlir::Type type) {
  type = unwrapMasked(type);
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  return type;
}

bool isNone(mlir::Type type) { return mlir::isa<mlir::NoneType>(type); }

bool isF16(mlir::Type type) {
  auto floating = mlir::dyn_cast<mlir::FloatType>(type);
  return floating && floating.getWidth() == 16;
}

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

std::string scalarCType(mlir::Type type) {
  if (type.isIndex())
    return "std::size_t";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    std::string prefix = integer.isUnsigned() ? "std::uint" : "std::int";
    return prefix + std::to_string(integer.getWidth()) + "_t";
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floating.getWidth() == 16 || floating.getWidth() == 32)
      return "float";
    if (floating.getWidth() == 64)
      return "double";
  }
  return {};
}

std::string pointerCType(PtrType pointer) {
  mlir::Type pointee = pointer.getElementType();
  std::string element = isF16(pointee) ? "std::uint16_t" : scalarCType(pointee);
  if (element.empty())
    return {};
  bool readOnly = pointer.getAccess() == "read";
  return (readOnly ? "const " : "") + element + " *";
}

std::string bareCType(mlir::Type type) {
  type = unwrapMasked(type);
  mlir::Type element = elementType(type);
  if (auto pointer = mlir::dyn_cast<PtrType>(element))
    return pointerCType(pointer);
  if (auto tuple = mlir::dyn_cast<TupleType>(element)) {
    std::string result = "std::tuple<";
    for (auto [index, field] : llvm::enumerate(tuple.getTypes())) {
      if (index)
        result += ", ";
      result += bareCType(field);
    }
    return result + ">";
  }
  return scalarCType(element);
}

std::string valueCType(const ValueInfo &info) {
  std::string scalar = bareCType(info.type);
  if (scalar.empty())
    return {};
  if (!info.isShaped())
    return scalar;
  return "weft_runtime::Tensor<" + scalar + ">";
}

std::string floatLiteral(mlir::FloatAttr attribute) {
  std::ostringstream stream;
  stream << std::setprecision(17) << attribute.getValueAsDouble();
  std::string result = stream.str();
  if (result.find_first_of(".eE") == std::string::npos)
    result += ".0";
  auto type = mlir::cast<mlir::FloatType>(attribute.getType());
  if (type.getWidth() <= 32)
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

std::string shapeInitializer(llvm::ArrayRef<std::string> shape) {
  return "{" + llvm::join(shape, ", ") + "}";
}

llvm::SmallVector<std::string>
broadcastShape(llvm::ArrayRef<ValueInfo> operands) {
  size_t rank = 0;
  for (const ValueInfo &operand : operands)
    rank = std::max(rank, operand.shape.size());
  llvm::SmallVector<std::string> result(rank, "1");
  for (const ValueInfo &operand : operands) {
    size_t padding = rank - operand.shape.size();
    for (auto [axis, dimension] : llvm::enumerate(operand.shape)) {
      std::string &selected = result[padding + axis];
      if (selected == "1" || dimension != "1")
        selected = dimension;
    }
  }
  return result;
}

class KernelEmitter {
public:
  KernelEmitter(KernelOp kernel, PlanOp plan, llvm::raw_ostream &output)
      : kernel(kernel), plan(plan), output(output) {
    for (mlir::Operation &record : plan.getBody().front().without_terminator()) {
      auto anchor = record.getAttrOfType<mlir::IntegerAttr>("anchor");
      if (anchor)
        records[anchor.getInt()] = &record;
    }
  }

  mlir::LogicalResult emit() {
    if (mlir::failed(requireSupportedPlan()))
      return mlir::failure();

    mlir::Block &body = kernel.getBody().front();
    llvm::SmallVector<std::string> parameters;
    auto names = kernel.getArgNames();
    auto kinds = kernel.getArgKinds();
    for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
      auto kind = mlir::cast<mlir::StringAttr>(kinds[index]).getValue();
      if (kind == "constexpr")
        continue;
      std::string name = sanitize(mlir::cast<mlir::StringAttr>(names[index]).getValue());
      mlir::Type type = argument.getType();
      std::string cType;
      if (auto pointer = mlir::dyn_cast<PtrType>(type))
        cType = pointerCType(pointer);
      else
        cType = scalarCType(type);
      if (cType.empty())
        return kernel.emitError("source ABI has an unsupported argument type");
      if (auto pointer = mlir::dyn_cast<PtrType>(type);
          pointer && pointer.getNoAlias())
        parameters.push_back(cType + " __restrict " + name);
      else
        parameters.push_back(cType + " " + name);
      values[argument] = ValueInfo{type, name, {}, {}};
    }

    mlir::Type returnType = kernel.getReturnType();
    std::string returnCType = isNone(returnType) ? "void" : scalarCType(returnType);
    if (returnCType.empty())
      return kernel.emitError("source ABI has an unsupported return type");
    line("extern \"C\" " + returnCType + " " +
         sanitize(kernel.getSymName()) + "(" + llvm::join(parameters, ", ") + ") {");
    ++indent;
    for (mlir::Operation &operation : body) {
      if (auto returnOp = mlir::dyn_cast<ReturnOp>(&operation)) {
        if (returnOp.getNumOperands() == 0)
          line("return;");
        else
          line("return " + lookup(returnOp.getOperand(0)).value + ";");
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
  PlanOp plan;
  llvm::raw_ostream &output;
  llvm::DenseMap<mlir::Value, ValueInfo> values;
  llvm::DenseMap<int64_t, mlir::Operation *> records;
  unsigned nextValue = 0;
  unsigned nextLoop = 0;
  unsigned indent = 0;

  void line(llvm::Twine text) {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix = "v") {
    return "__weft_" + prefix.str() + std::to_string(nextValue++);
  }

  ValueInfo lookup(mlir::Value value) const {
    auto found = values.find(value);
    if (found == values.end())
      return {};
    return found->second;
  }

  mlir::LogicalResult requireSupportedPlan() {
    for (mlir::Operation &record : plan.getBody().front().without_terminator()) {
      if (mlir::isa<MetaBindingOp>(record))
        continue;
      if (auto provider = record.getAttrOfType<mlir::StringAttr>("provider");
          !provider || (provider.getValue() != "scalar" && provider.getValue() != "rvv"))
        return record.emitError("C++ source emitter received an unknown selected provider");
    }
    return mlir::success();
  }

  mlir::Operation *recordFor(mlir::Operation *operation) {
    auto anchor = operation->getAttrOfType<mlir::IntegerAttr>(kCanonicalAnchorAttr);
    if (!anchor)
      return nullptr;
    auto found = records.find(anchor.getInt());
    return found == records.end() ? nullptr : found->second;
  }

  mlir::LogicalResult requireStrategy(mlir::Operation *operation,
                                      llvm::StringRef strategy) {
    mlir::Operation *record = recordFor(operation);
    if (!record)
      return operation->emitError("has no selected realization record");
    auto provider = record->getAttrOfType<mlir::StringAttr>("provider");
    auto selected = record->getAttrOfType<mlir::StringAttr>("strategy");
    llvm::StringRef expectedProvider = strategy.starts_with("rvv_") ? "rvv" : "scalar";
    if (!provider || provider.getValue() != expectedProvider || !selected ||
        selected.getValue() != strategy)
      return operation->emitError()
             << "source emitter does not implement selected realization "
             << (selected ? selected.getValue() : "<missing>");
    return mlir::success();
  }

  mlir::LogicalResult requireScalarAxis(mlir::Operation *operation) {
    auto record = mlir::dyn_cast_or_null<AxisPlanOp>(recordFor(operation));
    if (!record || record.getProvider() != "scalar" ||
        record.getRealization() != "scalar")
      return operation->emitError("source emitter requires selected scalar axis realization");
    return mlir::success();
  }

  std::string literalExpression(mlir::Value value) {
    if (auto constant = value.getDefiningOp<ConstantOp>())
      return constantLiteral(constant.getValue());
    if (auto special = value.getDefiningOp<SpecialValueOp>()) {
      if (special.getKind() == "neg_inf")
        return "-std::numeric_limits<" + scalarCType(special.getResult().getType()) + ">::infinity()";
      if (special.getKind() == "pos_inf")
        return "std::numeric_limits<" + scalarCType(special.getResult().getType()) + ">::infinity()";
      if (special.getKind() == "nan")
        return "std::numeric_limits<" + scalarCType(special.getResult().getType()) + ">::quiet_NaN()";
    }
    if (auto tuple = value.getDefiningOp<TupleOp>()) {
      llvm::SmallVector<std::string> fields;
      for (mlir::Value field : tuple.getOperands())
        fields.push_back(literalExpression(field));
      if (llvm::any_of(fields, [](const std::string &field) { return field.empty(); }))
        return {};
      return "std::make_tuple(" + llvm::join(fields, ", ") + ")";
    }
    ValueInfo info = lookup(value);
    if (!info.value.empty() && !info.isShaped())
      return info.value;
    return {};
  }

  llvm::SmallVector<std::string> explicitShape(mlir::Operation *operation,
                                                unsigned firstExtent) {
    llvm::SmallVector<std::string> result;
    for (unsigned index = firstExtent; index < operation->getNumOperands(); ++index)
      result.push_back(lookup(operation->getOperand(index)).value);
    return result;
  }

  std::string elementAccess(const ValueInfo &info, llvm::StringRef index,
                            llvm::StringRef outputName) {
    if (!info.isShaped())
      return info.value;
    return info.value + ".data[weft_runtime::broadcast_index(" + index.str() +
           ", " + outputName.str() + ".shape, " + info.value + ".shape)]";
  }

  std::string validityAccess(const ValueInfo &info, llvm::StringRef index,
                             llvm::StringRef outputName) {
    if (!info.isMasked())
      return "true";
    if (!info.isShaped())
      return info.validity;
    return info.validity + ".data[weft_runtime::broadcast_index(" + index.str() +
           ", " + outputName.str() + ".shape, " + info.validity + ".shape)]";
  }

  mlir::LogicalResult emitOperation(mlir::Operation *operation) {
    if (auto op = mlir::dyn_cast<ConstantOp>(operation))
      return emitConstant(op);
    if (auto op = mlir::dyn_cast<MetaValueOp>(operation))
      return emitMeta(op);
    if (auto op = mlir::dyn_cast<ForOp>(operation))
      return emitFor(op);
    if (auto op = mlir::dyn_cast<IfOp>(operation))
      return emitIf(op);
    if (auto op = mlir::dyn_cast<VLAOp>(operation))
      return emitVLA(op);
    if (auto op = mlir::dyn_cast<BlockAxisOp>(operation))
      return emitBlockAxis(op);
    if (auto op = mlir::dyn_cast<FullOp>(operation))
      return emitFull(op);
    if (auto op = mlir::dyn_cast<ExpandDimsOp>(operation))
      return emitExpandDims(op);
    if (auto op = mlir::dyn_cast<BroadcastToOp>(operation))
      return emitBroadcastTo(op);
    if (auto op = mlir::dyn_cast<ReshapeOp>(operation))
      return emitReshape(op);
    if (auto op = mlir::dyn_cast<TransposeOp>(operation))
      return emitTranspose(op);
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
    if (auto op = mlir::dyn_cast<BitcastOp>(operation))
      return emitBitcast(op);
    if (auto op = mlir::dyn_cast<SelectOp>(operation))
      return emitSelect(op);
    if (auto op = mlir::dyn_cast<TupleOp>(operation))
      return emitTuple(op);
    if (auto op = mlir::dyn_cast<TupleGetOp>(operation))
      return emitTupleGet(op);
    if (auto op = mlir::dyn_cast<SpecialValueOp>(operation))
      return emitSpecial(op);
    if (auto op = mlir::dyn_cast<InvalidOp>(operation)) {
      values[op.getResult()] = ValueInfo{op.getResult().getType(), {}, {}, {}};
      return mlir::success();
    }
    if (auto op = mlir::dyn_cast<LoadOp>(operation))
      return emitLoad(op);
    if (auto op = mlir::dyn_cast<StoreOp>(operation))
      return emitStore(op);
    if (auto op = mlir::dyn_cast<ValidOp>(operation))
      return emitValid(op);
    if (auto op = mlir::dyn_cast<FillOp>(operation))
      return emitFill(op);
    if (auto op = mlir::dyn_cast<ReduceOp>(operation))
      return emitReduce(op);
    if (auto op = mlir::dyn_cast<ContractOp>(operation))
      return emitContract(op);
    if (mlir::isa<YieldOp, ReturnOp>(operation))
      return mlir::success();
    return operation->emitError("scalar source provider does not implement this canonical primitive");
  }

  mlir::LogicalResult emitConstant(ConstantOp op) {
    std::string literal = constantLiteral(op.getValue());
    if (literal.empty())
      return op.emitError("source emitter cannot spell constant");
    std::string name = fresh();
    std::string type = scalarCType(op.getResult().getType());
    if (type.empty())
      return op.emitError("source emitter cannot represent constant type");
    line("[[maybe_unused]] const " + type + " " + name + " = " + literal + ";");
    values[op.getResult()] = ValueInfo{op.getResult().getType(), name, {}, {}};
    return mlir::success();
  }

  mlir::LogicalResult emitMeta(MetaValueOp op) {
    auto record = mlir::dyn_cast_or_null<MetaBindingOp>(recordFor(op));
    if (!record)
      return op.emitError("meta value has no selected binding");
    std::string name = fresh("meta");
    std::string type = scalarCType(op.getResult().getType());
    line("constexpr " + type + " " + name + " = " +
         std::to_string(record.getValue()) + ";");
    values[op.getResult()] = ValueInfo{op.getResult().getType(), name, {}, {}};
    return mlir::success();
  }

  ValueInfo makeResult(mlir::Value result,
                       llvm::ArrayRef<std::string> shape,
                       llvm::StringRef valueName = {}) {
    ValueInfo info{result.getType(),
                   valueName.empty() ? fresh() : valueName.str(), {}, {shape.begin(), shape.end()}};
    if (mlir::isa<MaskedType>(result.getType()))
      info.validity = fresh("valid");
    populateTupleFields(info);
    return info;
  }

  void populateTupleFields(ValueInfo &info) {
    auto tuple = mlir::dyn_cast<TupleType>(unwrapMasked(info.type));
    if (!tuple)
      return;
    for (mlir::Type fieldType : tuple.getTypes()) {
      ValueInfo field{fieldType, {}, {}, {}};
      populateTupleFields(field);
      info.fields.push_back(std::move(field));
    }
  }

  void declareResult(const ValueInfo &info, bool initialize = false) {
    std::string type = valueCType(info);
    if (info.isShaped())
      line(type + " " + info.value + "(" + shapeInitializer(info.shape) + ");");
    else
      line(type + " " + info.value + (initialize ? "{};" : ";"));
    if (info.isMasked()) {
      if (info.isShaped())
        line("weft_runtime::Tensor<bool> " + info.validity + "(" +
             shapeInitializer(info.shape) + ");");
      else
        line("bool " + info.validity + " = false;");
    }
  }

  mlir::LogicalResult emitFor(ForOp op) {
    llvm::SmallVector<ValueInfo> carried;
    for (auto [result, initial] : llvm::zip(op.getResults(), op.getOperands().drop_front(3))) {
      ValueInfo init = lookup(initial);
      ValueInfo info{result.getType(), fresh("carried"), {}, init.shape};
      if (init.isMasked())
        info.validity = fresh("carried_valid");
      line(valueCType(info) + " " + info.value + " = " + init.value + ";");
      if (info.isMasked())
        line((info.isShaped() ? "weft_runtime::Tensor<bool> " : "bool ") +
             info.validity + " = " + init.validity + ";");
      values[result] = info;
      carried.push_back(info);
    }
    std::string induction = "__weft_i" + std::to_string(nextLoop++);
    line("for (std::size_t " + induction + " = " + lookup(op.getLower()).value +
         "; " + induction + " < " + lookup(op.getUpper()).value + "; " +
         induction + " += " + lookup(op.getStep()).value + ") {");
    ++indent;
    mlir::Block &body = op.getBody().front();
    values[body.getArgument(0)] = ValueInfo{body.getArgument(0).getType(), induction, {}, {}};
    for (auto [argument, info] : llvm::zip(body.getArguments().drop_front(), carried))
      values[argument] = info;
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (mlir::Operation &nested : body.without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    for (auto [info, yielded] : llvm::zip(carried, yield.getOperands())) {
      ValueInfo value = lookup(yielded);
      line(info.value + " = " + value.value + ";");
      if (info.isMasked())
        line(info.validity + " = " + value.validity + ";");
    }
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitIf(IfOp op) {
    llvm::SmallVector<ValueInfo> results;
    for (mlir::Value result : op.getResults()) {
      ValueInfo info = makeResult(result, {});
      declareResult(info, true);
      values[result] = info;
      results.push_back(info);
    }
    line("if (" + lookup(op.getCondition()).value + ") {");
    ++indent;
    auto thenYield = mlir::cast<YieldOp>(op.getThenRegion().front().getTerminator());
    for (mlir::Operation &nested : op.getThenRegion().front().without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    for (auto [result, yielded] : llvm::zip(results, thenYield.getOperands()))
      line(result.value + " = " + lookup(yielded).value + ";");
    --indent;
    line("} else {");
    ++indent;
    auto elseYield = mlir::cast<YieldOp>(op.getElseRegion().front().getTerminator());
    for (mlir::Operation &nested : op.getElseRegion().front().without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    for (auto [result, yielded] : llvm::zip(results, elseYield.getOperands()))
      line(result.value + " = " + lookup(yielded).value + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitVLA(VLAOp op) {
    auto axis = mlir::dyn_cast_or_null<AxisPlanOp>(recordFor(op));
    if (!axis)
      return op.emitError("VLA has no selected axis realization");
    if (axis.getProvider() == "rvv")
      return emitRVVVLA(op, axis);
    if (mlir::failed(requireScalarAxis(op)))
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Operation *, ValueInfo> aggregates;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (!mlir::isa<ReduceOp, SummaryFoldOp>(nested))
        continue;
      mlir::Value identity = nested.getOperand(1);
      std::string expression = literalExpression(identity);
      if (expression.empty())
        return nested.emitError("scalar VLA aggregate identity must be loop-invariant");
      mlir::Value result = nested.getResult(0);
      ValueInfo info = makeResult(result, {} , fresh("aggregate"));
      line(valueCType(info) + " " + info.value + " = " + expression + ";");
      aggregates[&nested] = info;
      values[result] = info;
    }

    std::string induction = "__weft_vla" + std::to_string(nextLoop++);
    line("for (std::size_t " + induction + " = " + lookup(op.getBegin()).value +
         "; " + induction + " < " + lookup(op.getEnd()).value + "; ++" +
         induction + ") {");
    ++indent;
    values[body.getArgument(0)] =
        ValueInfo{body.getArgument(0).getType(), induction, {}, {}};
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto reduce = mlir::dyn_cast<ReduceOp>(&nested)) {
        if (mlir::failed(emitVLAReduce(reduce, aggregates[&nested])))
          return mlir::failure();
        continue;
      }
      if (auto summary = mlir::dyn_cast<SummaryFoldOp>(&nested)) {
        if (mlir::failed(emitVLASummary(summary, aggregates[&nested])))
          return mlir::failure();
        continue;
      }
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    }
    --indent;
    line("}");

    for (mlir::Operation &nested : body.without_terminator()) {
      auto summary = mlir::dyn_cast<SummaryFoldOp>(&nested);
      if (!summary)
        continue;
      ValueInfo aggregate = aggregates[&nested];
      mlir::Block &finalize = summary.getFinalize().front();
      values[finalize.getArgument(0)] = aggregate;
      line("{");
      ++indent;
      for (mlir::Operation &operation : finalize.without_terminator())
        if (mlir::failed(emitOperation(&operation)))
          return mlir::failure();
      ValueInfo finalized = lookup(mlir::cast<YieldOp>(finalize.getTerminator()).getOperand(0));
      if (aggregate.value != finalized.value)
        line(aggregate.value + " = " + finalized.value + ";");
      --indent;
      line("}");
    }

    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (auto [result, yielded] : llvm::zip(op.getResults(), yield.getOperands())) {
      ValueInfo info = lookup(yielded);
      info.type = result.getType();
      values[result] = info;
    }
    return mlir::success();
  }

  struct RVVInfo {
    enum class Kind { Coordinate, Pointer, Vector } kind;
    mlir::Type type;
    std::string value;
  };

  mlir::LogicalResult emitRVVVLA(VLAOp op, AxisPlanOp axis) {
    if (axis.getRealization() != "rvv" || axis.getSew() != 32 ||
        axis.getLmul() != "m1")
      return op.emitError("RVV VLA provider implements e32m1 strips");
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Value, RVVInfo> rvvValues;
    llvm::DenseMap<mlir::Operation *, ValueInfo> aggregates;
    for (mlir::Operation &nested : body.without_terminator()) {
      auto reduce = mlir::dyn_cast<ReduceOp>(nested);
      if (!reduce)
        continue;
      std::string identity = literalExpression(reduce.getIdentity());
      if (identity.empty())
        return reduce.emitError("RVV reduction identity must be loop-invariant");
      ValueInfo aggregate = makeResult(reduce.getResult(), {}, fresh("rvv_reduce"));
      line(valueCType(aggregate) + " " + aggregate.value + " = " + identity + ";");
      aggregates[reduce.getOperation()] = aggregate;
      values[reduce.getResult()] = aggregate;
    }
    std::string lane = "__weft_rvv_i" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    line("for (std::size_t " + lane + " = " + lookup(op.getBegin()).value +
         "; " + lane + " < " + lookup(op.getEnd()).value + ";) {");
    ++indent;
    line("const std::size_t " + vl + " = __riscv_vsetvl_e32m1(" +
         lookup(op.getEnd()).value + " - " + lane + ");");
    rvvValues[body.getArgument(0)] =
        RVVInfo{RVVInfo::Kind::Coordinate, body.getArgument(0).getType(), lane};

    for (mlir::Operation &nested : body.without_terminator()) {
      if (mlir::isa<ConstantOp, InvalidOp>(nested)) {
        if (mlir::failed(emitOperation(&nested)))
          return mlir::failure();
        continue;
      }
      if (auto pointer = mlir::dyn_cast<PtrAddOp>(nested)) {
        if (mlir::isa<RegionType>(unwrapMasked(pointer.getResult().getType()))) {
          auto coordinate = rvvValues.find(pointer.getOffset());
          ValueInfo base = lookup(pointer.getBase());
          if (coordinate == rvvValues.end() ||
              coordinate->second.kind != RVVInfo::Kind::Coordinate ||
              base.value.empty() || base.isShaped())
            return pointer.emitError("RVV unit-stride pointer must be scalar base plus VLA coordinate");
          rvvValues[pointer.getResult()] =
              RVVInfo{RVVInfo::Kind::Pointer, pointer.getResult().getType(),
                      "(" + base.value + " + " + lane + ")"};
        } else if (mlir::failed(emitOperation(&nested))) {
          return mlir::failure();
        }
        continue;
      }
      if (auto binary = mlir::dyn_cast<BinaryOp>(nested)) {
        if (mlir::isa<RegionType>(unwrapMasked(binary.getResult().getType()))) {
          if (mlir::failed(emitRVVBinary(binary, rvvValues, vl)))
            return mlir::failure();
        } else if (mlir::failed(emitOperation(&nested))) {
          return mlir::failure();
        }
        continue;
      }
      if (auto load = mlir::dyn_cast<LoadOp>(nested)) {
        if (mlir::failed(emitRVVLoad(load, rvvValues, vl)))
          return mlir::failure();
        continue;
      }
      if (auto store = mlir::dyn_cast<StoreOp>(nested)) {
        if (mlir::failed(emitRVVStore(store, rvvValues, vl)))
          return mlir::failure();
        continue;
      }
      if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
        if (mlir::failed(emitRVVReduce(reduce, aggregates[reduce.getOperation()],
                                       rvvValues, vl)))
          return mlir::failure();
        continue;
      }
      return nested.emitError("selected RVV VLA contains an unsupported primitive");
    }
    line(lane + " += " + vl + ";");
    --indent;
    line("}");
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (auto [result, yielded] : llvm::zip(op.getResults(), yield.getOperands())) {
      ValueInfo info = lookup(yielded);
      if (info.value.empty())
        return op.emitError("RVV VLA yielded value has no materialized aggregate");
      info.type = result.getType();
      values[result] = info;
    }
    return mlir::success();
  }

  mlir::LogicalResult emitRVVLoad(LoadOp op,
                                  llvm::DenseMap<mlir::Value, RVVInfo> &rvvValues,
                                  llvm::StringRef vl) {
    if (mlir::failed(requireStrategy(op, "rvv_unit_stride")))
      return mlir::failure();
    auto pointer = rvvValues.find(op.getPointer());
    if (pointer == rvvValues.end() ||
        pointer->second.kind != RVVInfo::Kind::Pointer ||
        !elementType(op.getResult().getType()).isF32())
      return op.emitError("RVV unit-stride load requires a region f32 pointer");
    std::string name = fresh("rvv");
    line("vfloat32m1_t " + name + " = __riscv_vle32_v_f32m1(" +
         pointer->second.value + ", " + vl.str() + ");");
    rvvValues[op.getResult()] =
        RVVInfo{RVVInfo::Kind::Vector, op.getResult().getType(), name};
    return mlir::success();
  }

  mlir::LogicalResult emitRVVBinary(
      BinaryOp op, llvm::DenseMap<mlir::Value, RVVInfo> &rvvValues,
      llvm::StringRef vl) {
    auto lhsVector = rvvValues.find(op.getLhs());
    auto rhsVector = rvvValues.find(op.getRhs());
    bool lhsIsVector = lhsVector != rvvValues.end() &&
                       lhsVector->second.kind == RVVInfo::Kind::Vector;
    bool rhsIsVector = rhsVector != rvvValues.end() &&
                       rhsVector->second.kind == RVVInfo::Kind::Vector;
    if (!lhsIsVector && !rhsIsVector)
      return op.emitError("RVV pointwise operation has no vector operand");
    ValueInfo lhsScalar = lookup(op.getLhs());
    ValueInfo rhsScalar = lookup(op.getRhs());
    std::string lhs = lhsIsVector ? lhsVector->second.value : lhsScalar.value;
    std::string rhs = rhsIsVector ? rhsVector->second.value : rhsScalar.value;
    if (lhs.empty() || rhs.empty())
      return op.emitError("RVV pointwise operand is unavailable");

    std::string intrinsic;
    llvm::StringRef kind = op.getKind();
    if (lhsIsVector && rhsIsVector) {
      intrinsic = llvm::StringSwitch<std::string>(kind)
                      .Case("add", "__riscv_vfadd_vv_f32m1")
                      .Case("sub", "__riscv_vfsub_vv_f32m1")
                      .Case("mul", "__riscv_vfmul_vv_f32m1")
                      .Case("div", "__riscv_vfdiv_vv_f32m1")
                      .Default("");
    } else if (lhsIsVector) {
      intrinsic = llvm::StringSwitch<std::string>(kind)
                      .Case("add", "__riscv_vfadd_vf_f32m1")
                      .Case("sub", "__riscv_vfsub_vf_f32m1")
                      .Case("mul", "__riscv_vfmul_vf_f32m1")
                      .Case("div", "__riscv_vfdiv_vf_f32m1")
                      .Default("");
    } else {
      intrinsic = llvm::StringSwitch<std::string>(kind)
                      .Case("add", "__riscv_vfadd_vf_f32m1")
                      .Case("mul", "__riscv_vfmul_vf_f32m1")
                      .Case("sub", "__riscv_vfrsub_vf_f32m1")
                      .Case("div", "__riscv_vfrdiv_vf_f32m1")
                      .Default("");
      std::swap(lhs, rhs);
    }
    if (intrinsic.empty())
      return op.emitError("RVV pointwise provider does not implement binary kind");
    std::string name = fresh("rvv");
    line("vfloat32m1_t " + name + " = " + intrinsic + "(" + lhs + ", " +
         rhs + ", " + vl.str() + ");");
    rvvValues[op.getResult()] =
        RVVInfo{RVVInfo::Kind::Vector, op.getResult().getType(), name};
    return mlir::success();
  }

  mlir::LogicalResult emitRVVStore(
      StoreOp op, llvm::DenseMap<mlir::Value, RVVInfo> &rvvValues,
      llvm::StringRef vl) {
    if (mlir::failed(requireStrategy(op, "rvv_unit_stride")))
      return mlir::failure();
    auto pointer = rvvValues.find(op.getPointer());
    auto value = rvvValues.find(op.getValue());
    if (pointer == rvvValues.end() || value == rvvValues.end() ||
        pointer->second.kind != RVVInfo::Kind::Pointer ||
        value->second.kind != RVVInfo::Kind::Vector)
      return op.emitError("RVV unit-stride store requires region pointer and vector value");
    line("__riscv_vse32_v_f32m1(" + pointer->second.value + ", " +
         value->second.value + ", " + vl.str() + ");");
    return mlir::success();
  }

  mlir::LogicalResult emitRVVReduce(
      ReduceOp op, const ValueInfo &aggregate,
      llvm::DenseMap<mlir::Value, RVVInfo> &rvvValues,
      llvm::StringRef vl) {
    if (mlir::failed(requireStrategy(op, "rvv_tree")))
      return mlir::failure();
    auto input = rvvValues.find(op.getInput());
    if (input == rvvValues.end() ||
        input->second.kind != RVVInfo::Kind::Vector ||
        op.getAxis() != -1 || op.getKind() != "add" ||
        !elementType(op.getInput().getType()).isF32())
      return op.emitError("RVV tree reduction requires an active-axis f32 add");
    std::string seed = fresh("rvv_seed");
    std::string partial = fresh("rvv_partial");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
         aggregate.value + ", " + vl.str() + ");");
    line("vfloat32m1_t " + partial +
         " = __riscv_vfredusum_vs_f32m1_f32m1(" + input->second.value +
         ", " + seed + ", " + vl.str() + ");");
    line(aggregate.value + " = __riscv_vfmv_f_s_f32m1_f32(" + partial + ");");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitVLAReduce(ReduceOp op, const ValueInfo &aggregate) {
    if (mlir::failed(requireStrategy(op, "scalar_linear")))
      return mlir::failure();
    ValueInfo input = lookup(op.getInput());
    if (input.isShaped() || op.getAxis() != -1)
      return op.emitError("scalar VLA provider expects the active VLA axis reduction");
    ValueInfo where = lookup(op.getWhere());
    std::string condition = where.value;
    if (input.isMasked())
      condition = "(" + condition + " && " + input.validity + ")";
    line("if (" + condition + ") " + aggregate.value + " = " +
         combineExpression(op.getKind(), aggregate.value, input.value) + ";");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitVLASummary(SummaryFoldOp op,
                                     const ValueInfo &aggregate) {
    if (mlir::failed(requireStrategy(op, "scalar_linear")))
      return mlir::failure();
    ValueInfo input = lookup(op.getInput());
    if (input.isShaped())
      return op.emitError("scalar VLA summary expects one scalarized logical point");
    ValueInfo where = lookup(op.getWhere());
    std::string condition = where.value;
    if (input.isMasked())
      condition = "(" + condition + " && " + input.validity + ")";
    line("if (" + condition + ") {");
    ++indent;
    mlir::Block &lift = op.getLift().front();
    ValueInfo liftedInput = input;
    liftedInput.type = lift.getArgument(0).getType();
    liftedInput.validity.clear();
    values[lift.getArgument(0)] = liftedInput;
    for (mlir::Operation &operation : lift.without_terminator())
      if (mlir::failed(emitOperation(&operation)))
        return mlir::failure();
    ValueInfo lifted = lookup(mlir::cast<YieldOp>(lift.getTerminator()).getOperand(0));

    mlir::Block &merge = op.getMerge().front();
    values[merge.getArgument(0)] = aggregate;
    values[merge.getArgument(1)] = lifted;
    for (mlir::Operation &operation : merge.without_terminator())
      if (mlir::failed(emitOperation(&operation)))
        return mlir::failure();
    ValueInfo merged = lookup(mlir::cast<YieldOp>(merge.getTerminator()).getOperand(0));
    line(aggregate.value + " = " + merged.value + ";");
    --indent;
    line("}");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitBlockAxis(BlockAxisOp op) {
    if (mlir::failed(requireScalarAxis(op)))
      return mlir::failure();
    ValueInfo info = makeResult(op.getResult(), {lookup(op.getExtent()).value});
    declareResult(info);
    std::string index = fresh("lane");
    line("for (std::size_t " + index + " = 0; " + index + " < " +
         info.value + ".data.size(); ++" + index + ") " + info.value +
         ".data[" + index + "] = " + lookup(op.getOffset()).value + " + " + index + ";");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitFull(FullOp op) {
    llvm::SmallVector<std::string> shape = explicitShape(op, 1);
    ValueInfo info = makeResult(op.getResult(), shape);
    declareResult(info);
    line("std::fill(" + info.value + ".data.begin(), " + info.value +
         ".data.end(), " + lookup(op.getValue()).value + ");");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitExpandDims(ExpandDimsOp op) {
    ValueInfo input = lookup(op.getInput());
    if (!input.isShaped())
      return op.emitError("expand_dims input has no materialized logical shape");
    llvm::SmallVector<std::string> shape = input.shape;
    shape.insert(shape.begin() + op.getAxis(), "1");
    ValueInfo info = makeResult(op.getResult(), shape);
    line(valueCType(info) + " " + info.value + " = " + input.value + ";");
    line(info.value + ".shape = " + shapeInitializer(shape) + ";");
    if (info.isMasked()) {
      line("weft_runtime::Tensor<bool> " + info.validity + " = " + input.validity + ";");
      line(info.validity + ".shape = " + shapeInitializer(shape) + ";");
    }
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitBroadcastTo(BroadcastToOp op) {
    ValueInfo input = lookup(op.getInput());
    llvm::SmallVector<std::string> shape = explicitShape(op, 1);
    return emitPointwiseCopy(op.getResult(), shape, input);
  }

  mlir::LogicalResult emitReshape(ReshapeOp op) {
    ValueInfo input = lookup(op.getInput());
    llvm::SmallVector<std::string> shape = explicitShape(op, 1);
    ValueInfo info = makeResult(op.getResult(), shape);
    if (!input.isShaped())
      return op.emitError("reshape input has no materialized logical shape");
    line(valueCType(info) + " " + info.value + " = " + input.value + ";");
    line(info.value + ".shape = " + shapeInitializer(shape) + ";");
    if (info.isMasked()) {
      line("weft_runtime::Tensor<bool> " + info.validity + " = " + input.validity + ";");
      line(info.validity + ".shape = " + shapeInitializer(shape) + ";");
    }
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitTranspose(TransposeOp op) {
    ValueInfo input = lookup(op.getInput());
    if (!input.isShaped())
      return op.emitError("transpose input has no materialized logical shape");
    auto permutation = op.getPermutation();
    llvm::SmallVector<std::string> shape;
    for (int64_t axis : permutation)
      shape.push_back(input.shape[axis]);
    ValueInfo info = makeResult(op.getResult(), shape);
    declareResult(info);
    std::string index = fresh("element");
    std::string spelling = "{";
    for (auto [position, axis] : llvm::enumerate(permutation)) {
      if (position)
        spelling += ", ";
      spelling += std::to_string(axis);
    }
    spelling += "}";
    line("for (std::size_t " + index + " = 0; " + index + " < " + info.value +
         ".data.size(); ++" + index + ") {");
    ++indent;
    std::string sourceIndex = "weft_runtime::transpose_input_index(" + index +
                              ", " + info.value + ".shape, " + input.value +
                              ".shape, " + spelling + ")";
    line(info.value + ".data[" + index + "] = " + input.value + ".data[" + sourceIndex + "];" );
    if (info.isMasked())
      line(info.validity + ".data[" + index + "] = " + input.validity + ".data[" + sourceIndex + "];" );
    --indent;
    line("}");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitPointwiseCopy(mlir::Value result,
                                        llvm::ArrayRef<std::string> shape,
                                        const ValueInfo &input) {
    ValueInfo info = makeResult(result, shape);
    if (shape.empty()) {
      line(valueCType(info) + " " + info.value + " = " + input.value + ";");
      if (info.isMasked())
        line("bool " + info.validity + " = " + input.validity + ";");
    } else {
      declareResult(info);
      std::string index = fresh("element");
      line("for (std::size_t " + index + " = 0; " + index + " < " + info.value +
           ".data.size(); ++" + index + ") {");
      ++indent;
      line(info.value + ".data[" + index + "] = " +
           elementAccess(input, index, info.value) + ";");
      if (info.isMasked())
        line(info.validity + ".data[" + index + "] = " +
             validityAccess(input, index, info.value) + ";");
      --indent;
      line("}");
    }
    values[result] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitPtrAdd(PtrAddOp op) {
    ValueInfo base = lookup(op.getBase());
    ValueInfo offset = lookup(op.getOffset());
    llvm::SmallVector<ValueInfo> operands{base, offset};
    llvm::SmallVector<std::string> shape = broadcastShape(operands);
    return emitPointwise(op.getResult(), shape, operands,
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return "(" + elementAccess(base, index, outputName) + " + " +
             elementAccess(offset, index, outputName) + ")";
    });
  }

  template <typename Builder>
  mlir::LogicalResult emitPointwise(mlir::Value result,
                                    llvm::ArrayRef<std::string> shape,
                                    llvm::ArrayRef<ValueInfo> operands,
                                    Builder expression) {
    ValueInfo info = makeResult(result, shape);
    if (shape.empty()) {
      std::string valid = "true";
      for (const ValueInfo &operand : operands)
        if (operand.isMasked())
          valid = "(" + valid + " && " + operand.validity + ")";
      if (info.isMasked()) {
        line("bool " + info.validity + " = " + valid + ";");
        line(valueCType(info) + " " + info.value + "{};");
        line("if (" + info.validity + ") " + info.value + " = " +
             expression("0", info.value) + ";");
      } else {
        line(valueCType(info) + " " + info.value + " = " +
             expression("0", info.value) + ";");
      }
    } else {
      declareResult(info);
      std::string index = fresh("element");
      line("for (std::size_t " + index + " = 0; " + index + " < " + info.value +
           ".data.size(); ++" + index + ") {");
      ++indent;
      if (info.isMasked()) {
        std::string valid = "true";
        for (const ValueInfo &operand : operands)
          if (operand.isMasked())
            valid = "(" + valid + " && " + validityAccess(operand, index, info.value) + ")";
        line(info.validity + ".data[" + index + "] = " + valid + ";");
        line("if (" + info.validity + ".data[" + index + "]) " + info.value +
             ".data[" + index + "] = " + expression(index, info.value) + ";");
      } else {
        line(info.value + ".data[" + index + "] = " + expression(index, info.value) + ";");
      }
      --indent;
      line("}");
    }
    values[result] = info;
    return mlir::success();
  }

  std::string unaryExpression(llvm::StringRef kind, llvm::StringRef input) {
    if (kind == "neg")
      return "(-" + input.str() + ")";
    if (kind == "exp")
      return "std::exp(" + input.str() + ")";
    if (kind == "sqrt")
      return "std::sqrt(" + input.str() + ")";
    if (kind == "rsqrt")
      return "(1.0f / std::sqrt(" + input.str() + "))";
    if (kind == "abs")
      return "std::abs(" + input.str() + ")";
    return {};
  }

  std::string combineExpression(llvm::StringRef kind, llvm::StringRef lhs,
                                llvm::StringRef rhs) {
    if (kind == "add") return "(" + lhs.str() + " + " + rhs.str() + ")";
    if (kind == "sub") return "(" + lhs.str() + " - " + rhs.str() + ")";
    if (kind == "mul") return "(" + lhs.str() + " * " + rhs.str() + ")";
    if (kind == "div") return "(" + lhs.str() + " / " + rhs.str() + ")";
    if (kind == "mod") return "(" + lhs.str() + " % " + rhs.str() + ")";
    if (kind == "and") return "(" + lhs.str() + " & " + rhs.str() + ")";
    if (kind == "or") return "(" + lhs.str() + " | " + rhs.str() + ")";
    if (kind == "xor") return "(" + lhs.str() + " ^ " + rhs.str() + ")";
    if (kind == "shl") return "(" + lhs.str() + " << " + rhs.str() + ")";
    if (kind == "shr") return "(" + lhs.str() + " >> " + rhs.str() + ")";
    if (kind == "max") return "std::max(" + lhs.str() + ", " + rhs.str() + ")";
    if (kind == "min") return "std::min(" + lhs.str() + ", " + rhs.str() + ")";
    return {};
  }

  mlir::LogicalResult emitUnary(UnaryOp op) {
    if (op.getKind() != "neg" &&
        mlir::failed(requireStrategy(op, "scalar_libm")))
      return mlir::failure();
    ValueInfo input = lookup(op.getInput());
    std::string probe = unaryExpression(op.getKind(), "x");
    if (probe.empty())
      return op.emitError("scalar source provider does not implement unary kind");
    return emitPointwise(op.getResult(), input.shape, {input},
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return unaryExpression(op.getKind(), elementAccess(input, index, outputName));
    });
  }

  mlir::LogicalResult emitBinary(BinaryOp op) {
    ValueInfo lhs = lookup(op.getLhs());
    ValueInfo rhs = lookup(op.getRhs());
    llvm::SmallVector<ValueInfo> operands{lhs, rhs};
    llvm::SmallVector<std::string> shape = broadcastShape(operands);
    auto build = [&](llvm::StringRef lhsExpression, llvm::StringRef rhsExpression) {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType(op.getResult().getType()));
      if (integer && integer.getWidth() == 1) {
        if (op.getKind() == "and")
          return "(" + lhsExpression.str() + " && " + rhsExpression.str() + ")";
        if (op.getKind() == "or")
          return "(" + lhsExpression.str() + " || " + rhsExpression.str() + ")";
        if (op.getKind() == "xor")
          return "(" + lhsExpression.str() + " != " + rhsExpression.str() + ")";
      }
      return combineExpression(op.getKind(), lhsExpression, rhsExpression);
    };
    if (build("a", "b").empty())
      return op.emitError("scalar source provider does not implement binary kind");
    return emitPointwise(op.getResult(), shape, operands,
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return build(elementAccess(lhs, index, outputName),
                   elementAccess(rhs, index, outputName));
    });
  }

  std::string compareExpression(llvm::StringRef predicate, llvm::StringRef lhs,
                                llvm::StringRef rhs) {
    llvm::StringRef spelling;
    if (predicate == "eq") spelling = "==";
    else if (predicate == "ne") spelling = "!=";
    else if (predicate == "lt") spelling = "<";
    else if (predicate == "le") spelling = "<=";
    else if (predicate == "gt") spelling = ">";
    else if (predicate == "ge") spelling = ">=";
    else return {};
    return "(" + lhs.str() + " " + spelling.str() + " " + rhs.str() + ")";
  }

  mlir::LogicalResult emitCompare(CompareOp op) {
    ValueInfo lhs = lookup(op.getLhs());
    ValueInfo rhs = lookup(op.getRhs());
    llvm::SmallVector<ValueInfo> operands{lhs, rhs};
    llvm::SmallVector<std::string> shape = broadcastShape(operands);
    if (compareExpression(op.getPredicate(), "a", "b").empty())
      return op.emitError("scalar source provider does not implement comparison predicate");
    return emitPointwise(op.getResult(), shape, operands,
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return compareExpression(op.getPredicate(), elementAccess(lhs, index, outputName),
                               elementAccess(rhs, index, outputName));
    });
  }

  mlir::LogicalResult emitCast(CastOp op) {
    ValueInfo input = lookup(op.getInput());
    std::string target = scalarCType(elementType(op.getResult().getType()));
    if (target.empty())
      return op.emitError("scalar source provider cannot represent cast target");
    return emitPointwise(op.getResult(), input.shape, {input},
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return "static_cast<" + target + ">(" + elementAccess(input, index, outputName) + ")";
    });
  }

  std::string bitcastExpression(mlir::Type sourceType, mlir::Type targetType,
                                llvm::StringRef input) {
    sourceType = elementType(sourceType);
    targetType = elementType(targetType);
    if (isF16(targetType))
      return "weft_runtime::half_to_float(static_cast<std::uint16_t>(" + input.str() + "))";
    if (isF16(sourceType))
      return "weft_runtime::float_to_half(" + input.str() + ")";
    return "weft_runtime::bit_cast<" + scalarCType(targetType) + ">(" + input.str() + ")";
  }

  mlir::LogicalResult emitBitcast(BitcastOp op) {
    ValueInfo input = lookup(op.getInput());
    return emitPointwise(op.getResult(), input.shape, {input},
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return bitcastExpression(op.getInput().getType(), op.getResult().getType(),
                               elementAccess(input, index, outputName));
    });
  }

  mlir::LogicalResult emitSelect(SelectOp op) {
    ValueInfo predicate = lookup(op.getPredicate());
    ValueInfo trueValue = lookup(op.getTrueValue());
    ValueInfo falseValue = lookup(op.getFalseValue());
    llvm::SmallVector<ValueInfo> operands{predicate, trueValue, falseValue};
    llvm::SmallVector<std::string> shape = broadcastShape(operands);
    return emitPointwise(op.getResult(), shape, operands,
                         [&](llvm::StringRef index, llvm::StringRef outputName) {
      return "(" + elementAccess(predicate, index, outputName) + " ? " +
             elementAccess(trueValue, index, outputName) + " : " +
             elementAccess(falseValue, index, outputName) + ")";
    });
  }

  mlir::LogicalResult emitTuple(TupleOp op) {
    llvm::SmallVector<std::string> fields;
    std::vector<ValueInfo> fieldInfos;
    for (mlir::Value operand : op.getOperands()) {
      ValueInfo info = lookup(operand);
      if (info.isMasked())
        return op.emitError("scalar source provider does not place masked values in tuples");
      fields.push_back(info.value);
      fieldInfos.push_back(info);
    }
    ValueInfo info = makeResult(op.getResult(), {});
    line("[[maybe_unused]] auto " + info.value + " = std::make_tuple(" +
         llvm::join(fields, ", ") + ");");
    info.fields = std::move(fieldInfos);
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitTupleGet(TupleGetOp op) {
    ValueInfo input = lookup(op.getInput());
    if (op.getIndex() < 0 || static_cast<size_t>(op.getIndex()) >= input.fields.size())
      return op.emitError("tuple field shape information is unavailable");
    ValueInfo info = input.fields[op.getIndex()];
    info.type = op.getResult().getType();
    info.value = fresh();
    line("auto " + info.value + " = std::get<" +
         std::to_string(op.getIndex()) + ">(" + input.value + ");");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitSpecial(SpecialValueOp op) {
    std::string expression = literalExpression(op.getResult());
    if (expression.empty())
      return op.emitError("scalar source provider cannot spell special value");
    ValueInfo info = makeResult(op.getResult(), {});
    line("[[maybe_unused]] " + valueCType(info) + " " + info.value + " = " + expression + ";");
    values[op.getResult()] = info;
    return mlir::success();
  }

  PtrType pointerType(const ValueInfo &info) {
    return mlir::dyn_cast<PtrType>(elementType(info.type));
  }

  std::string loadExpression(llvm::StringRef pointer, mlir::Type type) {
    if (isF16(type))
      return "weft_runtime::half_to_float(*" + pointer.str() + ")";
    return "(*" + pointer.str() + ")";
  }

  mlir::LogicalResult emitLoad(LoadOp op) {
    if (mlir::failed(requireStrategy(op, "scalar_direct")))
      return mlir::failure();
    ValueInfo pointer = lookup(op.getPointer());
    ValueInfo where = lookup(op.getWhere());
    ValueInfo other = lookup(op.getOther());
    PtrType pointerTy = pointerType(pointer);
    if (!pointerTy)
      return op.emitError("load pointer has no pointer element type");
    llvm::SmallVector<ValueInfo> shapes{pointer, where};
    if (!isNone(op.getOther().getType()))
      shapes.push_back(other);
    llvm::SmallVector<std::string> shape = broadcastShape(shapes);
    ValueInfo info = makeResult(op.getResult(), shape);
    auto emitCondition = [&](llvm::StringRef index, llvm::StringRef outputName) {
      std::string condition = elementAccess(where, index, outputName);
      if (pointer.isMasked())
        condition = "(" + condition + " && " + validityAccess(pointer, index, outputName) + ")";
      return condition;
    };
    if (shape.empty()) {
      std::string condition = emitCondition("0", info.value);
      std::string read = loadExpression(pointer.value, pointerTy.getElementType());
      if (info.isMasked()) {
        line("bool " + info.validity + " = " + condition + ";");
        line(valueCType(info) + " " + info.value + "{};");
        line("if (" + info.validity + ") " + info.value + " = " + read + ";");
      } else {
        line(valueCType(info) + " " + info.value + " = " + condition + " ? " +
             read + " : " + other.value + ";");
      }
    } else {
      declareResult(info);
      std::string index = fresh("element");
      line("for (std::size_t " + index + " = 0; " + index + " < " + info.value +
           ".data.size(); ++" + index + ") {");
      ++indent;
      std::string condition = emitCondition(index, info.value);
      std::string pointerExpr = elementAccess(pointer, index, info.value);
      std::string read = loadExpression(pointerExpr, pointerTy.getElementType());
      if (info.isMasked()) {
        line(info.validity + ".data[" + index + "] = " + condition + ";");
        line("if (" + info.validity + ".data[" + index + "]) " + info.value +
             ".data[" + index + "] = " + read + ";");
      } else {
        line(info.value + ".data[" + index + "] = " + condition + " ? " + read +
             " : " + elementAccess(other, index, info.value) + ";");
      }
      --indent;
      line("}");
    }
    values[op.getResult()] = info;
    return mlir::success();
  }

  std::string storeStatement(llvm::StringRef pointer, llvm::StringRef value,
                             mlir::Type type) {
    if (isF16(type))
      return "*" + pointer.str() + " = weft_runtime::float_to_half(" + value.str() + ");";
    return "*" + pointer.str() + " = " + value.str() + ";";
  }

  mlir::LogicalResult emitStore(StoreOp op) {
    if (mlir::failed(requireStrategy(op, "scalar_direct")))
      return mlir::failure();
    ValueInfo pointer = lookup(op.getPointer());
    ValueInfo value = lookup(op.getValue());
    ValueInfo where = lookup(op.getWhere());
    PtrType pointerTy = pointerType(pointer);
    if (!pointerTy)
      return op.emitError("store pointer has no pointer element type");
    llvm::SmallVector<ValueInfo> operands{pointer, value, where};
    llvm::SmallVector<std::string> shape = broadcastShape(operands);
    if (shape.empty()) {
      std::string condition = where.value;
      if (value.isMasked())
        condition = "(" + condition + " && " + value.validity + ")";
      line("if (" + condition + ") " +
           storeStatement(pointer.value, value.value, pointerTy.getElementType()));
      return mlir::success();
    }
    std::string shapeName = fresh("store_shape");
    line("const std::vector<std::size_t> " + shapeName + " = " + shapeInitializer(shape) + ";");
    std::string index = fresh("element");
    line("for (std::size_t " + index + " = 0; " + index + " < weft_runtime::element_count(" +
         shapeName + "); ++" + index + ") {");
    ++indent;
    auto access = [&](const ValueInfo &info) {
      if (!info.isShaped())
        return info.value;
      return info.value + ".data[weft_runtime::broadcast_index(" + index + ", " +
             shapeName + ", " + info.value + ".shape)]";
    };
    auto valid = [&](const ValueInfo &info) {
      if (!info.isMasked())
        return std::string("true");
      if (!info.isShaped())
        return info.validity;
      return info.validity + ".data[weft_runtime::broadcast_index(" + index + ", " +
             shapeName + ", " + info.validity + ".shape)]";
    };
    std::string condition = "(" + access(where) + " && " + valid(value) + " && " + valid(pointer) + ")";
    line("if (" + condition + ") " +
         storeStatement(access(pointer), access(value), pointerTy.getElementType()));
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitValid(ValidOp op) {
    ValueInfo input = lookup(op.getInput());
    if (!input.isMasked())
      return op.emitError("valid operand has no materialized validity");
    ValueInfo info{op.getResult().getType(), input.validity, {}, input.shape};
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitFill(FillOp op) {
    ValueInfo input = lookup(op.getInput());
    ValueInfo fill = lookup(op.getFillValue());
    if (!input.isMasked())
      return op.emitError("fill operand has no materialized validity");
    ValueInfo info = makeResult(op.getResult(), input.shape);
    if (input.shape.empty()) {
      line(valueCType(info) + " " + info.value + " = " + input.validity + " ? " +
           input.value + " : " + fill.value + ";");
    } else {
      declareResult(info);
      std::string index = fresh("element");
      line("for (std::size_t " + index + " = 0; " + index + " < " + info.value +
           ".data.size(); ++" + index + ") " + info.value + ".data[" + index +
           "] = " + input.validity + ".data[" + index + "] ? " + input.value +
           ".data[" + index + "] : " + elementAccess(fill, index, info.value) + ";");
    }
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitReduce(ReduceOp op) {
    if (mlir::failed(requireStrategy(op, "scalar_linear")))
      return mlir::failure();
    ValueInfo input = lookup(op.getInput());
    if (!input.isShaped())
      return op.emitError("non-VLA scalar reduction requires a materialized block");
    int64_t axis = op.getAxis();
    if (axis < 0 || static_cast<size_t>(axis) >= input.shape.size())
      return op.emitError("block reduction axis is outside the materialized shape");
    llvm::SmallVector<std::string> shape = input.shape;
    shape.erase(shape.begin() + axis);
    ValueInfo info = makeResult(op.getResult(), shape);
    std::string identity = lookup(op.getIdentity()).value;
    if (shape.empty())
      line(valueCType(info) + " " + info.value + " = " + identity + ";");
    else {
      declareResult(info);
      line("std::fill(" + info.value + ".data.begin(), " + info.value +
           ".data.end(), " + identity + ");");
    }
    ValueInfo where = lookup(op.getWhere());
    std::string index = fresh("reduce");
    line("for (std::size_t " + index + " = 0; " + index + " < " + input.value +
         ".data.size(); ++" + index + ") {");
    ++indent;
    std::string condition = elementAccess(where, index, input.value);
    if (input.isMasked())
      condition = "(" + condition + " && " + input.validity + ".data[" + index + "])";
    std::string target = info.value;
    if (!shape.empty())
      target += ".data[weft_runtime::remove_axis_index(" + index + ", " +
                input.value + ".shape, " + std::to_string(axis) + ")]";
    line("if (" + condition + ") " + target + " = " +
         combineExpression(op.getKind(), target, input.value + ".data[" + index + "]") + ";");
    --indent;
    line("}");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitContract(ContractOp op) {
    if (mlir::Operation *record = recordFor(op)) {
      auto provider = record->getAttrOfType<mlir::StringAttr>("provider");
      if (provider && provider.getValue() == "rvv")
        return emitRVVContract(op);
    }
    if (mlir::failed(requireStrategy(op, "scalar_nested")))
      return mlir::failure();
    ValueInfo lhs = lookup(op.getLhs());
    ValueInfo rhs = lookup(op.getRhs());
    ValueInfo init = lookup(op.getInit());
    auto lhsAxes = op.getLhsAxes();
    auto rhsAxes = op.getRhsAxes();
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2 || init.shape.size() != 2 ||
        lhsAxes.size() != 1 || rhsAxes.size() != 1 ||
        !op.getOutputOrder().empty())
      return op.emitError("scalar_nested provider currently requires rank-2 single-axis contraction");
    int64_t lhsAxis = lhsAxes.front();
    int64_t rhsAxis = rhsAxes.front();
    int64_t lhsFree = 1 - lhsAxis;
    int64_t rhsFree = 1 - rhsAxis;
    ValueInfo info{op.getResult().getType(), fresh("contract"), {}, init.shape};
    line(valueCType(info) + " " + info.value + " = " + init.value + ";");
    std::string m = fresh("m");
    std::string n = fresh("n");
    std::string k = fresh("k");
    line("for (std::size_t " + m + " = 0; " + m + " < " + lhs.value +
         ".shape[" + std::to_string(lhsFree) + "]; ++" + m + ") {");
    ++indent;
    line("for (std::size_t " + n + " = 0; " + n + " < " + rhs.value +
         ".shape[" + std::to_string(rhsFree) + "]; ++" + n + ") {");
    ++indent;
    line("for (std::size_t " + k + " = 0; " + k + " < " + lhs.value +
         ".shape[" + std::to_string(lhsAxis) + "]; ++" + k + ") {");
    ++indent;
    std::string lhsIndex = lhsAxis == 1
                               ? "(" + m + " * " + lhs.value + ".shape[1] + " + k + ")"
                               : "(" + k + " * " + lhs.value + ".shape[1] + " + m + ")";
    std::string rhsIndex = rhsAxis == 0
                               ? "(" + k + " * " + rhs.value + ".shape[1] + " + n + ")"
                               : "(" + n + " * " + rhs.value + ".shape[1] + " + k + ")";
    std::string condition = "(" + lookup(op.getWhereLhs()).value + " && " +
                            lookup(op.getWhereRhs()).value + ")";
    if (lhs.isMasked())
      condition = "(" + condition + " && " + lhs.validity + ".data[" + lhsIndex + "])";
    if (rhs.isMasked())
      condition = "(" + condition + " && " + rhs.validity + ".data[" + rhsIndex + "])";
    std::string outIndex = "(" + m + " * " + info.value + ".shape[1] + " + n + ")";
    line("if (" + condition + ") " + info.value + ".data[" + outIndex +
         "] += static_cast<" + scalarCType(op.getAccDtype()) + ">(" + lhs.value +
         ".data[" + lhsIndex + "]) * static_cast<" + scalarCType(op.getAccDtype()) +
         ">(" + rhs.value + ".data[" + rhsIndex + "]);" );
    --indent;
    line("}");
    --indent;
    line("}");
    --indent;
    line("}");
    values[op.getResult()] = info;
    return mlir::success();
  }

  mlir::LogicalResult emitRVVContract(ContractOp op) {
    if (mlir::failed(requireStrategy(op, "rvv_f16_f32_contract")))
      return mlir::failure();
    ValueInfo lhs = lookup(op.getLhs());
    ValueInfo rhs = lookup(op.getRhs());
    ValueInfo init = lookup(op.getInit());
    auto lhsAxes = op.getLhsAxes();
    auto rhsAxes = op.getRhsAxes();
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2 ||
        init.shape.size() != 2 || lhsAxes.size() != 1 ||
        rhsAxes.size() != 1 || lhsAxes.front() != 1 ||
        rhsAxes.front() != 0 || !op.getOutputOrder().empty() ||
        !elementType(op.getLhs().getType()).isF16() ||
        !elementType(op.getRhs().getType()).isF16() ||
        !op.getAccDtype().isF32() || !op.getOutDtype().isF32())
      return op.emitError("RVV f16/f32 contract provider requires [M,K]x[K,N]");

    ValueInfo info{op.getResult().getType(), fresh("rvv_contract"), {}, init.shape};
    line(valueCType(info) + " " + info.value + " = " + init.value + ";");
    std::string m = fresh("m");
    std::string n = fresh("n");
    std::string k = fresh("k");
    std::string lhsSlice = fresh("lhs_slice");
    std::string rhsSlice = fresh("rhs_slice");
    std::string strip = fresh("strip");
    std::string vl = fresh("vl");
    std::string lhsVector = fresh("lhs_vec");
    std::string rhsVector = fresh("rhs_vec");
    std::string product = fresh("product");
    std::string seed = fresh("seed");
    std::string partial = fresh("partial");
    std::string lhsWhere = lookup(op.getWhereLhs()).value;
    std::string rhsWhere = lookup(op.getWhereRhs()).value;
    line("for (std::size_t " + m + " = 0; " + m + " < " + lhs.value +
         ".shape[0]; ++" + m + ") {");
    ++indent;
    line("for (std::size_t " + n + " = 0; " + n + " < " + rhs.value +
         ".shape[1]; ++" + n + ") {");
    ++indent;
    line("std::vector<float> " + lhsSlice + "(" + lhs.value + ".shape[1], 0.0f);");
    line("std::vector<float> " + rhsSlice + "(" + rhs.value + ".shape[0], 0.0f);");
    line("for (std::size_t " + k + " = 0; " + k + " < " + lhs.value +
         ".shape[1]; ++" + k + ") {");
    ++indent;
    std::string lhsIndex = "(" + m + " * " + lhs.value + ".shape[1] + " + k + ")";
    std::string rhsIndex = "(" + k + " * " + rhs.value + ".shape[1] + " + n + ")";
    std::string lhsValid = lhs.isMasked()
                               ? lhs.validity + ".data[" + lhsIndex + "]"
                               : "true";
    std::string rhsValid = rhs.isMasked()
                               ? rhs.validity + ".data[" + rhsIndex + "]"
                               : "true";
    line("if (" + lhsWhere + " && " + lhsValid + ") " + lhsSlice + "[" + k +
         "] = static_cast<float>(" + lhs.value + ".data[" + lhsIndex + "]);");
    line("if (" + rhsWhere + " && " + rhsValid + ") " + rhsSlice + "[" + k +
         "] = static_cast<float>(" + rhs.value + ".data[" + rhsIndex + "]);");
    --indent;
    line("}");
    std::string outputIndex = "(" + m + " * " + info.value + ".shape[1] + " + n + ")";
    line("for (std::size_t " + strip + " = 0; " + strip + " < " + lhsSlice +
         ".size();) {");
    ++indent;
    line("const std::size_t " + vl + " = __riscv_vsetvl_e32m1(" + lhsSlice +
         ".size() - " + strip + ");");
    line("vfloat32m1_t " + lhsVector + " = __riscv_vle32_v_f32m1(" + lhsSlice +
         ".data() + " + strip + ", " + vl + ");");
    line("vfloat32m1_t " + rhsVector + " = __riscv_vle32_v_f32m1(" + rhsSlice +
         ".data() + " + strip + ", " + vl + ");");
    line("vfloat32m1_t " + product + " = __riscv_vfmul_vv_f32m1(" + lhsVector +
         ", " + rhsVector + ", " + vl + ");");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" + info.value +
         ".data[" + outputIndex + "], " + vl + ");");
    line("vfloat32m1_t " + partial +
         " = __riscv_vfredusum_vs_f32m1_f32m1(" + product + ", " + seed +
         ", " + vl + ");");
    line(info.value + ".data[" + outputIndex +
         "] = __riscv_vfmv_f_s_f32m1_f32(" + partial + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    --indent;
    line("}");
    --indent;
    line("}");
    values[op.getResult()] = info;
    return mlir::success();
  }
};

void emitPrelude(llvm::raw_ostream &output, bool usesRVV) {
  if (usesRVV)
    output << "#include <riscv_vector.h>\n";
  output << R"cpp(#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <vector>

namespace weft_runtime {

inline std::size_t element_count(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (std::size_t dimension : shape)
    result *= dimension;
  return result;
}

template <typename T> struct Tensor {
  std::vector<std::size_t> shape;
  std::vector<T> data;

  Tensor() = default;
  Tensor(std::initializer_list<std::size_t> dimensions)
      : shape(dimensions), data(element_count(shape)) {}
};

inline std::size_t broadcast_index(
    std::size_t output_index, const std::vector<std::size_t> &output_shape,
    const std::vector<std::size_t> &input_shape) {
  if (input_shape.empty())
    return 0;
  std::size_t input_index = 0;
  std::size_t input_stride = 1;
  for (std::size_t reverse = 0; reverse < input_shape.size(); ++reverse) {
    std::size_t input_axis = input_shape.size() - 1 - reverse;
    std::size_t output_axis = output_shape.size() - 1 - reverse;
    std::size_t coordinate = output_index % output_shape[output_axis];
    output_index /= output_shape[output_axis];
    if (input_shape[input_axis] != 1)
      input_index += coordinate * input_stride;
    input_stride *= input_shape[input_axis];
  }
  return input_index;
}

inline std::size_t remove_axis_index(
    std::size_t input_index, const std::vector<std::size_t> &input_shape,
    std::size_t removed_axis) {
  std::size_t result = 0;
  std::size_t result_stride = 1;
  for (std::size_t reverse = 0; reverse < input_shape.size(); ++reverse) {
    std::size_t axis = input_shape.size() - 1 - reverse;
    std::size_t coordinate = input_index % input_shape[axis];
    input_index /= input_shape[axis];
    if (axis == removed_axis)
      continue;
    result += coordinate * result_stride;
    result_stride *= input_shape[axis];
  }
  return result;
}

inline std::size_t transpose_input_index(
    std::size_t output_index, const std::vector<std::size_t> &output_shape,
    const std::vector<std::size_t> &input_shape,
    std::initializer_list<std::size_t> permutation_list) {
  std::vector<std::size_t> permutation(permutation_list);
  std::vector<std::size_t> output_coordinates(output_shape.size());
  for (std::size_t reverse = 0; reverse < output_shape.size(); ++reverse) {
    std::size_t axis = output_shape.size() - 1 - reverse;
    output_coordinates[axis] = output_index % output_shape[axis];
    output_index /= output_shape[axis];
  }
  std::vector<std::size_t> input_coordinates(input_shape.size());
  for (std::size_t axis = 0; axis < permutation.size(); ++axis)
    input_coordinates[permutation[axis]] = output_coordinates[axis];
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < input_shape.size(); ++axis)
    result = result * input_shape[axis] + input_coordinates[axis];
  return result;
}

template <typename To, typename From> To bit_cast(const From &input) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast width mismatch");
  To result;
  std::memcpy(&result, &input, sizeof(result));
  return result;
}

inline float half_to_float(std::uint16_t bits) {
  std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  std::uint32_t exponent = (bits >> 10) & 0x1fu;
  std::uint32_t mantissa = bits & 0x3ffu;
  std::uint32_t value;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffu;
      value = sign | static_cast<std::uint32_t>(127 - 15 - shift) << 23 |
              mantissa << 13;
    }
  } else if (exponent == 0x1fu) {
    value = sign | 0x7f800000u | mantissa << 13;
  } else {
    value = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
  }
  return bit_cast<float>(value);
}

inline std::uint16_t float_to_half(float input) {
  std::uint32_t bits = bit_cast<std::uint32_t>(input);
  std::uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<std::uint16_t>(sign);
    mantissa = (mantissa | 0x800000u) >> (1 - exponent);
    return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
  }
  if (exponent >= 31)
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                    ((mantissa + 0x1000u) >> 13));
}

} // namespace weft_runtime

)cpp";
}

} // namespace

mlir::LogicalResult weft::emitSelectedSource(mlir::ModuleOp module,
                                             llvm::raw_ostream &output) {
  bool usesRVV = false;
  module.walk([&](mlir::Operation *operation) {
    auto provider = operation->getAttrOfType<mlir::StringAttr>("provider");
    if (provider && provider.getValue() == "rvv")
      usesRVV = true;
  });
  emitPrelude(output, usesRVV);
  llvm::SmallVector<KernelOp> kernels;
  for (KernelOp kernel : module.getOps<KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("source emission requires a canonical kernel");
  for (KernelOp kernel : kernels) {
    PlanOp matched;
    for (PlanOp plan : module.getOps<PlanOp>())
      if (plan.getKernelRef() == kernel.getSymName()) {
        if (matched)
          return kernel.emitError("has more than one selected execution plan");
        matched = plan;
      }
    if (!matched)
      return kernel.emitError("has no selected execution plan");
    KernelEmitter emitter(kernel, matched, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
  }
  return mlir::success();
}
