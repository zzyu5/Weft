#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

using namespace weft;

namespace {

struct LinearIndex {
  llvm::DenseMap<mlir::Value, int64_t> terms;
  int64_t constant = 0;
  bool valid = true;
};

std::optional<int64_t> constantIndex(mlir::Value value) {
  return riscv_internal::constantInt(value);
}

void decomposeIndex(mlir::Value value, int64_t coefficient,
                    LinearIndex &result) {
  if (!result.valid)
    return;
  if (auto constant = constantIndex(value)) {
    result.constant += coefficient * *constant;
    return;
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    decomposeIndex(add.getLhs(), coefficient, result);
    decomposeIndex(add.getRhs(), coefficient, result);
    return;
  }
  if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
    decomposeIndex(sub.getLhs(), coefficient, result);
    decomposeIndex(sub.getRhs(), -coefficient, result);
    return;
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto lhs = constantIndex(multiply.getLhs())) {
      decomposeIndex(multiply.getRhs(), coefficient * *lhs, result);
      return;
    }
    if (auto rhs = constantIndex(multiply.getRhs())) {
      decomposeIndex(multiply.getLhs(), coefficient * *rhs, result);
      return;
    }
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
  }
  result.terms[value] += coefficient;
  if (result.terms[value] == 0)
    result.terms.erase(value);
}

bool sameTerms(const LinearIndex &lhs, const LinearIndex &rhs) {
  if (!lhs.valid || !rhs.valid || lhs.terms.size() != rhs.terms.size())
    return false;
  for (const auto &[value, coefficient] : lhs.terms) {
    auto found = rhs.terms.find(value);
    if (found == rhs.terms.end() || found->second != coefficient)
      return false;
  }
  return true;
}

struct ShapedAffineIndex {
  llvm::DenseMap<int64_t, int64_t> axisCoefficients;
  mlir::Value scalarBase;
  int64_t constant = 0;
  bool valid = true;
};

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
  if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
      (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()))
    return false;
  if (lhs > 0) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
        (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs))
      return false;
  } else {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
      return false;
  }
  result = lhs * rhs;
  return true;
}

void addConstant(ShapedAffineIndex &result, int64_t value) {
  int64_t next = 0;
  if (!checkedAdd(result.constant, value, next)) {
    result.valid = false;
    return;
  }
  result.constant = next;
}

void addAxisCoefficient(ShapedAffineIndex &result, int64_t axis,
                        int64_t coefficient) {
  int64_t next = 0;
  if (!checkedAdd(result.axisCoefficients.lookup(axis), coefficient, next)) {
    result.valid = false;
    return;
  }
  if (next == 0)
    result.axisCoefficients.erase(axis);
  else
    result.axisCoefficients[axis] = next;
}

void decomposeShapedIndex(mlir::Value value, int64_t coefficient,
                          ShapedAffineIndex &result, unsigned depth = 0) {
  if (!result.valid || depth > 24)
    return result.valid = false, void();
  if (auto constant = constantIndex(value)) {
    int64_t scaled = 0;
    if (!checkedMultiply(coefficient, *constant, scaled))
      return result.valid = false, void();
    addConstant(result, scaled);
    return;
  }

  auto shaped = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!shaped) {
    if (coefficient != 1 || result.scalarBase)
      return result.valid = false, void();
    result.scalarBase = value;
    return;
  }
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return result.valid = false, void();
    decomposeShapedIndex(conversion.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
    decomposeShapedIndex(cast.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    auto type = iota.getResult().getType();
    if (type.getShape().size() != 1 || type.getAxisIds().size() != 1 ||
        iota.getEnd() - iota.getStart() != type.getShape()[0])
      return result.valid = false, void();
    int64_t start = 0;
    if (!checkedMultiply(coefficient, iota.getStart(), start))
      return result.valid = false, void();
    addConstant(result, start);
    addAxisCoefficient(result, type.getAxisIds()[0], coefficient);
    return;
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    llvm::StringRef kind = binary.getKind();
    if (kind == "add" || kind == "sub") {
      decomposeShapedIndex(binary.getLhs(), coefficient, result, depth + 1);
      int64_t rhsCoefficient = coefficient;
      if (kind == "sub" &&
          !checkedMultiply(coefficient, -1, rhsCoefficient))
        return result.valid = false, void();
      decomposeShapedIndex(binary.getRhs(), rhsCoefficient, result, depth + 1);
      return;
    }
    if (kind == "mul") {
      if (auto lhs = constantIndex(binary.getLhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *lhs, scaled))
          return result.valid = false, void();
        decomposeShapedIndex(binary.getRhs(), scaled, result, depth + 1);
        return;
      }
      if (auto rhs = constantIndex(binary.getRhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *rhs, scaled))
          return result.valid = false, void();
        decomposeShapedIndex(binary.getLhs(), scaled, result, depth + 1);
        return;
      }
    }
  }
  result.valid = false;
}

int64_t knownMultiple(mlir::Value value, unsigned depth = 0) {
  if (depth > 24)
    return 1;
  if (auto constant = constantIndex(value))
    return *constant == std::numeric_limits<int64_t>::min()
               ? 1
               : std::abs(*constant);
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return knownMultiple(cast.getInput(), depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return knownMultiple(cast.getIn(), depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return knownMultiple(cast.getIn(), depth + 1);
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    if (binary.getKind() == "mul") {
      if (auto lhs = constantIndex(binary.getLhs())) {
        int64_t factor = *lhs == std::numeric_limits<int64_t>::min()
                             ? 1
                             : std::abs(*lhs);
        int64_t result = 0;
        return checkedMultiply(factor, knownMultiple(binary.getRhs(), depth + 1),
                               result)
                   ? result
                   : 1;
      }
      if (auto rhs = constantIndex(binary.getRhs())) {
        int64_t factor = *rhs == std::numeric_limits<int64_t>::min()
                             ? 1
                             : std::abs(*rhs);
        int64_t result = 0;
        return checkedMultiply(factor, knownMultiple(binary.getLhs(), depth + 1),
                               result)
                   ? result
                   : 1;
      }
    }
    if (binary.getKind() == "add" || binary.getKind() == "sub")
      return std::gcd(knownMultiple(binary.getLhs(), depth + 1),
                      knownMultiple(binary.getRhs(), depth + 1));
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto lhs = constantIndex(multiply.getLhs())) {
      int64_t result = 0;
      return checkedMultiply(std::abs(*lhs),
                             knownMultiple(multiply.getRhs(), depth + 1), result)
                 ? result
                 : 1;
    }
    if (auto rhs = constantIndex(multiply.getRhs())) {
      int64_t result = 0;
      return checkedMultiply(std::abs(*rhs),
                             knownMultiple(multiply.getLhs(), depth + 1), result)
                 ? result
                 : 1;
    }
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
    return std::gcd(knownMultiple(add.getLhs(), depth + 1),
                    knownMultiple(add.getRhs(), depth + 1));
  if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>())
    return std::gcd(knownMultiple(sub.getLhs(), depth + 1),
                    knownMultiple(sub.getRhs(), depth + 1));
  return 1;
}

bool isReadOnlyOrPure(mlir::Operation *operation) {
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
    llvm::StringRef effect = conversion.getConversion().getEffect();
    return effect == "pure" || effect == "read";
  }
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &instance) {
    return mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

bool canFoldReadConversions(
    riscv::ExtractOp first, riscv::ExtractOp second,
    llvm::ArrayRef<riscv::ConvertLayoutOp> firstConversions,
  llvm::ArrayRef<riscv::ConvertLayoutOp> secondConversions) {
  llvm::SmallVector<mlir::Operation *> reads;
  auto collectReads = [&](llvm::ArrayRef<riscv::ConvertLayoutOp> conversions) {
    for (riscv::ConvertLayoutOp conversion : conversions) {
      if (conversion.getConversion().getEffect() != "read")
        continue;
      if (conversion.getConversion().getKind() != "local_load" ||
          conversion->getBlock() != first->getBlock())
        return false;
      reads.push_back(conversion);
    }
    return true;
  };
  if (!collectReads(firstConversions) || !collectReads(secondConversions))
    return false;
  if (reads.empty())
    return true;

  mlir::Operation *begin = reads.front();
  for (mlir::Operation *read : reads)
    if (read->isBeforeInBlock(begin))
      begin = read;
  if (second->isBeforeInBlock(begin))
    return false;
  for (mlir::Operation *operation = begin; operation;
       operation = operation->getNextNode()) {
    if (!isReadOnlyOrPure(operation))
      return false;
    if (operation == second.getOperation())
      return true;
  }
  return false;
}

bool isLayerExtract(
    riscv::ExtractOp extract, riscv::FieldOp &field,
    riscv::PhysicalPointOp &point,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), conversions);
  field = riscv_internal::sourceField(source);
  point = extract.getIndices().size() == 1
              ? riscv_internal::stripRepresentationConversions(
                    extract.getIndices().front())
                    .getDefiningOp<riscv::PhysicalPointOp>()
              : riscv::PhysicalPointOp();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  int64_t domainSelectors = 0;
  for (mlir::Attribute selector : extract.getSelectors())
    domainSelectors +=
        mlir::cast<mlir::StringAttr>(selector).getValue() == "domain";
  if (!field || !point || !result || !input)
    return false;
  auto axis = llvm::find(result.getAxisIds().asArrayRef(),
                         point.getResult().getType().getDomain().getAxisId());
  const bool hasLayerAxis =
      result && axis != result.getAxisIds().asArrayRef().end() &&
      result.getShape()[static_cast<size_t>(
          axis - result.getAxisIds().asArrayRef().begin())] ==
          extract.getAccess().getLayerSize();
  llvm::StringRef validity = result.getLayout().getValidity();
  return domainSelectors == 1 &&
         extract.getAccess().getForm() == "indexed" &&
         extract.getAccess().getMapping() == "grouped_layered" &&
         extract.getAccess().getGroupSize() ==
             extract.getAccess().getLayerSize() * 2 &&
         result.getShape().size() == input.getShape().size() && hasLayerAxis &&
         result.getLayout().getCarrier() == "rvv" &&
         (validity == "full" || validity == "tail");
}

bool sameFieldEdge(riscv::FieldOp lhs, riscv::FieldOp rhs) {
  if (!lhs || !rhs || lhs.getOwner() != rhs.getOwner() ||
      lhs.getName() != rhs.getName() ||
      lhs.getResult().getType() != rhs.getResult().getType() ||
      lhs.getAccess() != rhs.getAccess())
    return false;
  riscv_internal::FieldFacts left = riscv_internal::fieldFacts(lhs);
  riscv_internal::FieldFacts right = riscv_internal::fieldFacts(rhs);
  return left.mapping == right.mapping &&
         left.scalarPerRecord == right.scalarPerRecord &&
         left.logicalRank == right.logicalRank &&
         left.group == right.group && left.layer == right.layer &&
         left.joinFields == right.joinFields &&
         left.joinLowBits == right.joinLowBits &&
         left.joinRole == right.joinRole && left.bitOffset == right.bitOffset &&
         left.storageBits == right.storageBits &&
         left.alignment == right.alignment && left.order == right.order;
}

bool isLayeredStreamField(riscv::FieldOp field) {
  auto value = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  auto element = value
                     ? mlir::dyn_cast<mlir::IntegerType>(value.getElementType())
                     : mlir::IntegerType();
  riscv::AccessAttr access = field.getAccess();
  const int64_t group = access ? access.getGroupSize() : 0;
  const int64_t layer = access ? access.getLayerSize() : 0;
  const int64_t layers = layer > 0 ? group / layer : 0;
  if (!value || !element || element.isSigned() || !access ||
      value.getLayout().getCarrier() != "rvv" ||
      access.getForm() != "indexed" ||
      access.getMapping() != "grouped_layered" ||
      group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers > 8 || access.getBitOffset() % 8)
    return false;
  bool hasStreamAxis = false;
  for (size_t index = 0; index < value.getShape().size(); ++index) {
    const int64_t extent = value.getShape()[index];
    const int64_t time = value.getLayout().getTimeFactors()[index];
    const int64_t lanes = value.getLayout().getLaneFactors()[index];
    const bool streamAxis = extent > 0 && extent % group == 0 && lanes > 1 &&
                            layer % lanes == 0 && time > 1 &&
                            time * lanes == extent;
    if (streamAxis) {
      if (hasStreamAxis)
        return false;
      hasStreamAxis = true;
    } else if (time != 1 || lanes != 1) {
      return false;
    }
  }
  return hasStreamAxis;
}

bool isProjectedLayeredExtract(
    riscv::ExtractOp extract, riscv::FieldOp &field,
    riscv::PhysicalPointOp &origin, int64_t &axis, int64_t &base,
    int64_t &stride, int64_t &repeat, int64_t &extent,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), conversions);
  field = source.getDefiningOp<riscv::FieldOp>();
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto pattern = extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
  if (!field || !input || !result || !pattern || pattern.size() != 3 ||
      input.getShape().size() != result.getShape().size() ||
      extract.getAccess().getForm() != "indexed" ||
      extract.getAccess().getMapping() != "grouped_layered")
    return false;

  size_t projection = extract.getSelectors().size();
  bool hasRegular = false;
  for (auto [position, selector] : llvm::enumerate(extract.getSelectors())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (name == "regular") {
      if (projection != extract.getSelectors().size())
        return false;
      projection = position;
      hasRegular = true;
    } else if (name != "all") {
      return false;
    }
  }
  if (!hasRegular || projection >= result.getShape().size() ||
      projection >= result.getAxisIds().size() ||
      !extract.getIndices().empty())
    return false;

  axis = result.getAxisIds()[projection];
  base = pattern[0];
  stride = pattern[1];
  repeat = pattern[2];
  extent = result.getShape()[projection];
  origin = riscv_internal::originPoint(field.getOwner(), axis);
  const int64_t sourceExtent = input.getShape()[projection];
  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  const int64_t lanes = result.getLayout().getLaneFactors()[projection];
  const int64_t time = result.getLayout().getTimeFactors()[projection];
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto element = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  if (!origin || base < 0 || stride != 1 || repeat != 1 || extent <= 0 ||
      sourceExtent <= 0 || group <= 0 || layer <= 0 || group % layer ||
      layers <= 1 || base % group || extent % group || lanes <= 1 ||
      layer % lanes || time <= 1 || time > std::numeric_limits<int64_t>::max() / lanes ||
      time * lanes != extent || !element || element.isSigned() ||
      element.getWidth() * layers > 8 ||
      base > sourceExtent - extent ||
      origin.getResult().getType().getDomain().getTail() != "exact" ||
      result.getLayout().getCarrier() != "rvv")
    return false;
  if (conversions &&
      !canFoldReadConversions(
          extract, extract, *conversions,
          llvm::ArrayRef<riscv::ConvertLayoutOp>{}))
    return false;
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    if (position == projection)
      continue;
    if (input.getShape()[position] != result.getShape()[position] ||
        result.getLayout().getTimeFactors()[position] != 1 ||
        result.getLayout().getLaneFactors()[position] != 1)
      return false;
  }
  return true;
}

struct CompleteLayeredExtract {
  riscv::ExtractOp extract;
  riscv::FieldOp field;
  riscv::PhysicalPointOp point;
  riscv::PhysicalPointOp parent;
  riscv::ValueType resultType;
  LinearIndex relative;
  llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
};

std::optional<CompleteLayeredExtract>
completeLayeredExtract(riscv::ExtractOp extract) {
  CompleteLayeredExtract result;
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &result.conversions);
  result.field = source.getDefiningOp<riscv::FieldOp>();
  result.point =
      extract.getIndices().size() == 1
          ? riscv_internal::stripRepresentationConversions(
                extract.getIndices().front())
                .getDefiningOp<riscv::PhysicalPointOp>()
          : riscv::PhysicalPointOp();
  result.parent = result.point
                      ? result.point.getParent().getDefiningOp<
                            riscv::PhysicalPointOp>()
                      : riscv::PhysicalPointOp();
  result.resultType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto fieldType = result.field
                       ? mlir::dyn_cast<riscv::ValueType>(
                             result.field.getResult().getType())
                       : riscv::ValueType();
  auto element = fieldType
                     ? mlir::dyn_cast<mlir::IntegerType>(
                           fieldType.getElementType())
                     : mlir::IntegerType();
  if (!result.field || !result.point || !result.parent || !result.resultType ||
      !fieldType || !element || element.isSigned() ||
      extract.getAccess().getForm() != "indexed" ||
      extract.getAccess().getMapping() != "grouped_layered" ||
      extract.getIndices().size() != 1 ||
      fieldType.getShape().size() != result.resultType.getShape().size())
    return std::nullopt;

  int64_t domainSelectors = 0;
  for (mlir::Attribute selector : extract.getSelectors()) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (name == "domain")
      ++domainSelectors;
    else if (name != "all")
      return std::nullopt;
  }
  const int64_t axis =
      result.point.getResult().getType().getDomain().getAxisId();
  auto position = llvm::find(result.resultType.getAxisIds().asArrayRef(), axis);
  if (domainSelectors != 1 ||
      position == result.resultType.getAxisIds().asArrayRef().end())
    return std::nullopt;
  const size_t axisPosition = static_cast<size_t>(
      position - result.resultType.getAxisIds().asArrayRef().begin());
  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  const int64_t lanes =
      result.resultType.getLayout().getLaneFactors()[axisPosition];
  auto parentPartition = result.parent.getPartition()
                             .getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (group <= 0 || layer <= 0 || group % layer || layers <= 1 || lanes <= 1 ||
      layer % lanes || element.getWidth() * layers != 8 ||
      extract.getAccess().getBitOffset() % 8 ||
      (extract.getAccess().getOrder() != "lo_first" &&
       extract.getAccess().getOrder() != "hi_first") ||
      result.resultType.getShape()[axisPosition] != lanes ||
      result.resultType.getLayout().getTimeFactors()[axisPosition] != 1 ||
      result.resultType.getLayout().getCarrier() != "rvv" || !parentPartition ||
      parentPartition.value() % group ||
      result.parent.getResult().getType().getDomain().getAxisId() != axis ||
      result.parent.getResult().getType().getDomain().getTail() != "exact")
    return std::nullopt;
  for (size_t index = 0; index < result.resultType.getShape().size(); ++index) {
    if (index == axisPosition)
      continue;
    if (fieldType.getShape()[index] != result.resultType.getShape()[index] ||
        result.resultType.getLayout().getTimeFactors()[index] != 1 ||
        result.resultType.getLayout().getLaneFactors()[index] != 1)
      return std::nullopt;
  }

  decomposeIndex(result.point.getBase(), 1, result.relative);
  decomposeIndex(result.parent.getBase(), -1, result.relative);
  if (!result.relative.valid || result.relative.constant < 0)
    return std::nullopt;
  result.extract = extract;
  return result;
}

bool sameCompleteLayeredGroup(CompleteLayeredExtract &lhs,
                              CompleteLayeredExtract &rhs) {
  return lhs.extract->getBlock() == rhs.extract->getBlock() &&
         sameFieldEdge(lhs.field, rhs.field) && lhs.parent == rhs.parent &&
         lhs.point.getActive() == rhs.point.getActive() &&
         lhs.point.getPartition() == rhs.point.getPartition() &&
         lhs.point.getResult().getType() == rhs.point.getResult().getType() &&
         lhs.resultType == rhs.resultType &&
         lhs.extract.getAccess() == rhs.extract.getAccess() &&
         sameTerms(lhs.relative, rhs.relative);
}

bool materializeReplicaStorageLoad(mlir::IRRewriter &rewriter,
                                   riscv::ExtractOp extract) {
  llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
  mlir::Value input = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &conversions);
  auto field = input.getDefiningOp<riscv::FieldOp>();
  auto fieldType = field
                       ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                       : riscv::ValueType();
  auto resultType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto kernel = extract->getParentOfType<riscv::KernelOp>();
  if (!field || !fieldType || !resultType || extract.getIndices().size() != 1 ||
      !kernel ||
      extract.getSelectors().size() != 1 ||
      mlir::cast<mlir::StringAttr>(extract.getSelectors()[0]).getValue() !=
          "gather" ||
      extract.getAccess().getForm() != "indexed" ||
      (extract.getAccess().getMapping() != "natural" &&
       extract.getAccess().getMapping() != "grouped_layered"))
    return false;

  ShapedAffineIndex affine;
  decomposeShapedIndex(extract.getIndices().front(), 1, affine);
  if (!affine.valid)
    return false;

  riscv::ConvertLayoutOp consumerConversion;
  riscv::ValueType selectedType = resultType;
  if (extract.getResult().hasOneUse()) {
    auto *user = *extract.getResult().getUsers().begin();
    auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(user);
    auto targetType =
        conversion
            ? mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType())
            : riscv::ValueType();
    if (conversion && conversion.getConversion().getEffect() == "pure" &&
        targetType &&
        targetType.getLayout().getCarrier() == "rvv" &&
        targetType.getElementType() == resultType.getElementType() &&
        targetType.getShape() == resultType.getShape() &&
        targetType.getAxisIds() == resultType.getAxisIds()) {
      // The typed geometry checks below are sufficient for both natural and
      // grouped/layered storage.  Refusing the latter here forces a narrow raw
      // window followed by register-to-lane packing even when the target lane
      // layout divides the declared group and layer exactly.
      consumerConversion = conversion;
      selectedType = targetType;
    }
  }
  const bool scalarReplicas = resultType.getLayout().getCarrier() == "scalar";
  if (selectedType == resultType && scalarReplicas)
    return false;
  if (selectedType.getLayout().getCarrier() != "rvv") {
    return false;
  }

  const bool layered =
      extract.getAccess().getMapping() == "grouped_layered";
  int64_t laneAxis = 0;
  size_t lanePosition = 0;
  llvm::SmallVector<size_t> lanePositions;
  int64_t lanes = 0;
  int64_t streams = 1;
  int64_t replicas = 1;
  for (size_t position = 0; position < selectedType.getShape().size(); ++position) {
    const int64_t time = selectedType.getLayout().getTimeFactors()[position];
    const int64_t lane = selectedType.getLayout().getLaneFactors()[position];
    const int64_t replica = selectedType.getLayout().getReplicaFactors()[position];
    const int64_t fragment =
        selectedType.getLayout().getFragmentFactors()[position];
    const int64_t local = selectedType.getLayout().getLocalFactors()[position];
    int64_t represented = 0;
    if (time <= 0 || lane <= 0 || replica <= 0 || fragment != 1 || local != 1 ||
        !checkedMultiply(time, lane, represented) ||
        !checkedMultiply(represented, replica, represented) ||
        represented != selectedType.getShape()[position])
      return false;
    if (lane > 1) {
      if (replica != 1)
        return false;
      lanePositions.push_back(position);
      laneAxis = selectedType.getAxisIds()[position];
      lanePosition = position;
    } else {
      if ((time > 1 && replica > 1) ||
          (selectedType.getShape()[position] > 1 &&
           !affine.axisCoefficients.contains(selectedType.getAxisIds()[position])))
        return false;
    }
    if (!checkedMultiply(streams, time, streams) ||
        !checkedMultiply(replicas, replica, replicas))
      return false;
  }
  lanes = 1;
  for (size_t position : lanePositions)
    if (!checkedMultiply(lanes,
                         selectedType.getLayout().getLaneFactors()[position],
                         lanes))
      return false;
  if (!laneAxis || affine.axisCoefficients.lookup(laneAxis) != 1 ||
      streams <= 0 || replicas <= 0 ||
      streams > std::numeric_limits<int64_t>::max() / replicas)
    return false;
  for (const auto &[axis, coefficient] : affine.axisCoefficients) {
    auto found = llvm::find(selectedType.getAxisIds().asArrayRef(), axis);
    if (found == selectedType.getAxisIds().asArrayRef().end() || coefficient <= 0)
      return false;
  }

  // Several logical axes may map into one RVV register only when the typed
  // affine storage coordinates prove a contiguous row-major interval.  For a
  // layered field the subsequent group/layer divisibility checks additionally
  // prove that this wider logical interval can be decoded from whole raw
  // windows without crossing an encoded layer boundary.
  if (lanePositions.size() > 1) {
    int64_t expectedStride = 1;
    for (int64_t position =
             static_cast<int64_t>(selectedType.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t lane =
          selectedType.getLayout().getLaneFactors()[position];
      if (lane <= 1)
        continue;
      const int64_t axis = selectedType.getAxisIds()[position];
      if (affine.axisCoefficients.lookup(axis) != expectedStride ||
          !checkedMultiply(expectedStride, lane, expectedStride))
        return false;
    }
  }

  auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(), laneAxis);
  const bool gatherAxis =
      fieldAxis == fieldType.getAxisIds().asArrayRef().end();
  if (gatherAxis &&
      (fieldType.getShape().size() != 1 ||
       affine.axisCoefficients.lookup(laneAxis) != 1))
    return false;
  const int64_t fieldExtent =
      gatherAxis
          ? fieldType.getShape()[0]
          : fieldType.getShape()[static_cast<size_t>(
                fieldAxis - fieldType.getAxisIds().asArrayRef().begin())];
  if (fieldExtent <= 0)
    return false;

  llvm::SmallVector<int64_t> partOffsets;
  int64_t logicalSpan = 0;
  const int64_t parts = streams * replicas;
  for (int64_t part = 0; part < parts; ++part) {
    int64_t remainingStream = part % streams;
    int64_t remainingReplica = part / streams;
    llvm::SmallVector<int64_t> timeCoordinates(selectedType.getShape().size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(selectedType.getShape().size(), 0);
    for (int64_t position = static_cast<int64_t>(selectedType.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t time =
          selectedType.getLayout().getTimeFactors()[static_cast<size_t>(position)];
      const int64_t replica =
          selectedType.getLayout().getReplicaFactors()[static_cast<size_t>(position)];
      timeCoordinates[static_cast<size_t>(position)] = remainingStream % time;
      remainingStream /= time;
      replicaCoordinates[static_cast<size_t>(position)] =
          remainingReplica % replica;
      remainingReplica /= replica;
    }
    if (remainingStream || remainingReplica)
      return false;
    int64_t logicalOffset = 0;
    for (size_t position = 0; position < selectedType.getShape().size(); ++position) {
      const int64_t time = selectedType.getLayout().getTimeFactors()[position];
      const int64_t replica =
          selectedType.getLayout().getReplicaFactors()[position];
      int64_t coordinate = 0;
      const int64_t lane =
          selectedType.getLayout().getLaneFactors()[position];
      if (lane > 1) {
        if (!checkedMultiply(timeCoordinates[position], lane, coordinate))
          return false;
      } else {
        if (!checkedMultiply(timeCoordinates[position], replica, coordinate) ||
            !checkedAdd(coordinate, replicaCoordinates[position], coordinate))
          return false;
      }
      if (time == 1 && replica == 1 && position != lanePosition)
        continue;
      int64_t contribution = 0;
      if (!checkedMultiply(
              coordinate,
              affine.axisCoefficients.lookup(selectedType.getAxisIds()[position]),
              contribution) ||
          !checkedAdd(logicalOffset, contribution, logicalOffset))
        return false;
    }
    if (logicalOffset < 0 || logicalOffset % lanes)
      return false;
    partOffsets.push_back(logicalOffset);
    int64_t end = 0;
    if (!checkedAdd(logicalOffset, lanes, end))
      return false;
    logicalSpan = std::max(logicalSpan, end);
  }
  if (logicalSpan <= 0 || logicalSpan > fieldExtent || affine.constant < 0 ||
      (affine.scalarBase && knownMultiple(affine.scalarBase) % logicalSpan))
    return false;

  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  if (layered &&
      (group <= 0 || layer <= 0 || group % layer || layer % lanes ||
       group % logicalSpan ||
       logicalSpan % layer || affine.constant != 0))
    return false;
  if (!layered && !affine.scalarBase &&
      affine.constant > fieldExtent - logicalSpan)
    return false;
  if (affine.scalarBase && affine.constant != 0)
    return false;

  llvm::DenseMap<int64_t, int64_t> windowIds;
  llvm::SmallVector<int64_t> windowOffsets;
  llvm::SmallVector<int64_t> windowForPart;
  llvm::SmallVector<int64_t> layerForPart;
  for (int64_t offset : partOffsets) {
    const int64_t windowOffset = layered ? offset % layer : offset;
    auto [found, inserted] = windowIds.try_emplace(
        windowOffset, static_cast<int64_t>(windowOffsets.size()));
    if (inserted)
      windowOffsets.push_back(windowOffset);
    windowForPart.push_back(found->second);
    layerForPart.push_back(layered ? offset / layer : 0);
  }

  rewriter.setInsertionPoint(extract);
  mlir::Value scalarBase = affine.scalarBase;
  if (!scalarBase)
    scalarBase = rewriter.create<mlir::arith::ConstantIndexOp>(extract.getLoc(),
                                                               affine.constant);
  const int64_t logicalBaseMultiple =
      affine.scalarBase
          ? std::max<int64_t>(1, knownMultiple(affine.scalarBase))
          : (affine.constant == 0
                 ? logicalSpan
                 : std::max<int64_t>(1, std::abs(affine.constant)));
  const int64_t resultGroups = selectedType.getLayout().getRegisterGroups();
  const int64_t temporaryGroups =
      std::max<int64_t>(1,
                        (selectedType.getLayout().getLmulEighths() + 7) / 8);
  const bool tail = selectedType.getLayout().getValidity() == "tail";
  llvm::StringRef instruction =
      layered ? "rvv.replica-storage-load.layered"
              : "rvv.replica-storage-load.natural";
  auto load = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
      extract.getLoc(), selectedType, field.getResult(), scalarBase,
      logicalBaseMultiple, laneAxis, logicalSpan,
      rewriter.getDenseI64ArrayAttr(windowOffsets),
      rewriter.getDenseI64ArrayAttr(windowForPart),
      rewriter.getDenseI64ArrayAttr(layerForPart), extract.getAccess(),
      riscv_internal::leaf(rewriter, "rvv", "replica-storage-load",
                           instruction, instruction, 0, resultGroups,
                           temporaryGroups, 0, "none",
                           tail ? "agnostic" : "exact"));
  riscv_internal::copyOrigin(extract, load);
  if (auto canonical = extract->getAttr("canonical_op"))
    load->setAttr("canonical_op", canonical);
  mlir::Value replacement = load.getResult();
  if (consumerConversion) {
    consumerConversion.getResult().replaceAllUsesWith(replacement);
    rewriter.eraseOp(consumerConversion);
  }
  extract.getResult().replaceAllUsesWith(replacement);
  rewriter.eraseOp(extract);
  for (riscv::ConvertLayoutOp conversion : llvm::reverse(conversions))
    if (conversion.getResult().use_empty())
      rewriter.eraseOp(conversion);
  return true;
}

void eraseDeadPureProducers(mlir::IRRewriter &rewriter,
                            llvm::ArrayRef<mlir::Value> roots) {
  llvm::SmallVector<mlir::Operation *> worklist;
  llvm::SmallVector<mlir::Operation *> candidates;
  llvm::DenseSet<mlir::Operation *> seen;
  for (mlir::Value root : roots)
    if (mlir::Operation *definition = root.getDefiningOp())
      worklist.push_back(definition);
  while (!worklist.empty()) {
    mlir::Operation *operation = worklist.pop_back_val();
    if (!operation || !seen.insert(operation).second ||
        operation->getNumRegions() != 0)
      continue;
    if (!mlir::isMemoryEffectFree(operation))
      continue;
    candidates.push_back(operation);
    for (mlir::Value operand : operation->getOperands())
      if (mlir::Operation *definition = operand.getDefiningOp())
        worklist.push_back(definition);
  }

  llvm::DenseSet<mlir::Operation *> erased;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation *operation : llvm::reverse(candidates)) {
      if (erased.contains(operation) ||
          llvm::any_of(operation->getResults(),
                       [](mlir::Value result) { return !result.use_empty(); }))
        continue;
      rewriter.eraseOp(operation);
      erased.insert(operation);
      changed = true;
    }
  }
}

class ShareRISCVLayeredWindowsPass
    : public mlir::PassWrapper<ShareRISCVLayeredWindowsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-share-layered-windows";
  }
  llvm::StringRef getDescription() const override {
    return "Represent adjacent packed layers as one shared RVV storage window";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<riscv::FieldOp> fields;
    getOperation().walk(
        [&](riscv::FieldOp field) { fields.push_back(field); });
    for (riscv::FieldOp field : fields) {
      if (!isLayeredStreamField(field))
        continue;
      auto type = mlir::cast<riscv::ValueType>(field.getResult().getType());
      const int64_t resultGroups = type.getLayout().getRegisterGroups();
      const int64_t temporaryGroups =
          std::max<int64_t>(1,
                            (type.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = type.getLayout().getValidity() == "tail";
      rewriter.setInsertionPointAfter(field);
      auto stream = rewriter.create<riscv::RVVLayeredStreamOp>(
          field.getLoc(), type, field.getResult(), field.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "layered-stream",
                               "rvv.layered-stream", "rvv.layered-stream", 0,
                               resultGroups, temporaryGroups, 0, "none",
                               tail ? "agnostic" : "exact"));
      if (auto origin = field->getAttr("source_origin"))
        stream->setAttr("source_origin", origin);
      stream->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.field"));
      field.getResult().replaceAllUsesExcept(stream.getResult(),
                                             stream.getOperation());
    }

    llvm::SmallVector<riscv::ExtractOp> projectedExtracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { projectedExtracts.push_back(extract); });
    for (riscv::ExtractOp extract : projectedExtracts) {
      riscv::FieldOp field;
      riscv::PhysicalPointOp origin;
      llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
      int64_t axis = 0;
      int64_t base = 0;
      int64_t stride = 0;
      int64_t repeat = 0;
      int64_t extent = 0;
      if (!isProjectedLayeredExtract(extract, field, origin, axis, base, stride,
                                     repeat, extent, &conversions))
        continue;
      auto type = mlir::cast<riscv::ValueType>(extract.getResult().getType());
      const int64_t resultGroups = type.getLayout().getRegisterGroups();
      const int64_t temporaryGroups =
          std::max<int64_t>(1, (type.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = type.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(extract);
      auto stream = rewriter.create<riscv::RVVProjectedLayeredStreamOp>(
          extract.getLoc(), type, field.getResult(), origin.getResult(), axis,
          base, stride, repeat, extent, extract.getAccess(),
          riscv_internal::leaf(
              rewriter, "rvv", "projected-layered-stream",
              "rvv.projected-layered-stream", "rvv.projected-layered-stream",
              mlir::cast<riscv::ValueType>(field.getResult().getType())
                  .getLayout()
                  .getRegisterGroups(),
              resultGroups, temporaryGroups, 0, "none",
              tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(extract, stream);
      if (auto canonical = extract->getAttr("canonical_op"))
        stream->setAttr("canonical_op", canonical);
      extract.getResult().replaceAllUsesWith(stream.getResult());
      rewriter.eraseOp(extract);
      for (riscv::ConvertLayoutOp conversion : llvm::reverse(conversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
    }

    llvm::SmallVector<riscv::ExtractOp> replicaExtracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { replicaExtracts.push_back(extract); });
    llvm::SmallVector<mlir::Value> replacedReplicaIndices;
    for (riscv::ExtractOp extract : replicaExtracts) {
      llvm::SmallVector<mlir::Value> indices(extract.getIndices().begin(),
                                             extract.getIndices().end());
      if (materializeReplicaStorageLoad(rewriter, extract))
        replacedReplicaIndices.append(indices.begin(), indices.end());
    }
    eraseDeadPureProducers(rewriter, replacedReplicaIndices);

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    llvm::DenseSet<mlir::Operation *> consumed;

    // A complete grouped/layered logical group is represented by one raw byte
    // window per lane-width position and one typed decode per storage layer.
    // The relation is derived from Field storage geometry and the child point's
    // affine coordinate relative to its typed parent point.  Equivalent
    // consumers therefore share the same physical load without relying on
    // source operation adjacency or a kernel-specific closure.
    for (riscv::ExtractOp first : extracts) {
      if (consumed.contains(first))
        continue;
      auto firstInfo = completeLayeredExtract(first);
      if (!firstInfo)
        continue;
      const int64_t group = first.getAccess().getGroupSize();
      const int64_t layer = first.getAccess().getLayerSize();
      auto axisIt = llvm::find(
          firstInfo->resultType.getAxisIds().asArrayRef(),
          firstInfo->point.getResult().getType().getDomain().getAxisId());
      if (axisIt == firstInfo->resultType.getAxisIds().asArrayRef().end())
        continue;
      const size_t axisPosition = static_cast<size_t>(
          axisIt - firstInfo->resultType.getAxisIds().asArrayRef().begin());
      const int64_t lanes =
          firstInfo->resultType.getLayout().getLaneFactors()[axisPosition];
      const int64_t layers = group / layer;
      const int64_t windowsPerLayer = layer / lanes;
      const int64_t slots = layers * windowsPerLayer;
      const int64_t groupBase =
          (firstInfo->relative.constant / group) * group;

      llvm::SmallVector<llvm::SmallVector<CompleteLayeredExtract, 1>, 8>
          bySlot(static_cast<size_t>(slots));
      for (riscv::ExtractOp candidate : extracts) {
        if (consumed.contains(candidate))
          continue;
        auto info = completeLayeredExtract(candidate);
        if (!info || !sameCompleteLayeredGroup(*firstInfo, *info))
          continue;
        const int64_t delta = info->relative.constant - groupBase;
        if (delta < 0 || delta >= group || delta % lanes)
          continue;
        bySlot[static_cast<size_t>(delta / lanes)].push_back(std::move(*info));
      }
      if (llvm::any_of(bySlot, [](const auto &slot) { return slot.empty(); }))
        continue;

      mlir::Operation *insertion = bySlot.front().front().extract;
      bool foldable = true;
      for (auto &slot : bySlot) {
        for (CompleteLayeredExtract &info : slot) {
          if (info.extract->isBeforeInBlock(insertion))
            insertion = info.extract;
          foldable &= canFoldReadConversions(
              bySlot.front().front().extract, info.extract,
              bySlot.front().front().conversions, info.conversions);
        }
      }
      if (!foldable)
        continue;

      CompleteLayeredExtract &leader = bySlot.front().front();
      rewriter.setInsertionPoint(insertion);
      auto groupValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          first.getLoc(), group);
      auto relativeBase = rewriter.create<mlir::arith::SubIOp>(
          first.getLoc(), leader.point.getBase(), leader.parent.getBase());
      auto groupIndex = rewriter.create<mlir::arith::DivUIOp>(
          first.getLoc(), relativeBase, groupValue);
      mlir::Value windowGroup = groupIndex;
      if (windowsPerLayer != 1) {
        auto windowsValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            first.getLoc(), windowsPerLayer);
        windowGroup = rewriter.create<mlir::arith::MulIOp>(
            first.getLoc(), groupIndex, windowsValue);
      }

      auto fieldType = mlir::cast<riscv::ValueType>(
          leader.field.getResult().getType());
      auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(),
                                  leader.point.getResult()
                                      .getType()
                                      .getDomain()
                                      .getAxisId());
      if (fieldAxis == fieldType.getAxisIds().asArrayRef().end())
        continue;
      const int64_t projectionExtent =
          fieldType.getShape()[static_cast<size_t>(
              fieldAxis - fieldType.getAxisIds().asArrayRef().begin())];
      const int64_t rawGroups =
          firstInfo->resultType.getLayout().getRegisterGroups();
      auto storageType = riscv::LayeredWindowType::get(
          rewriter.getContext(), fieldType, leader.resultType,
          leader.point.getResult().getType().getDomain().getAxisId(), layers,
          windowsPerLayer, rawGroups);
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 2> decoded(
          static_cast<size_t>(windowsPerLayer));
      for (int64_t window = 0; window < windowsPerLayer; ++window) {
        mlir::Value windowIndex = windowGroup;
        if (window) {
          auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(
              first.getLoc(), window);
          windowIndex = rewriter.create<mlir::arith::AddIOp>(
              first.getLoc(), windowGroup, offset);
        }
        auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
            first.getLoc(), storageType, leader.field.getResult(),
            leader.parent.getResult(), windowIndex,
            leader.point.getResult().getType().getDomain().getAxisId(), 0,
            projectionExtent, first.getAccess(),
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-load",
                "rvv.layered-storage-load", "rvv.layered-storage-load",
                fieldType.getLayout().getRegisterGroups(), rawGroups, rawGroups,
                0, "none",
                leader.resultType.getLayout().getValidity() == "tail"
                    ? "agnostic"
                    : "exact"));
        riscv_internal::copyOrigin(first, storage);
        for (int64_t storageLayer = 0; storageLayer < layers; ++storageLayer) {
          auto value = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
              first.getLoc(), leader.resultType, storage.getResult(),
              storageLayer,
              riscv_internal::leaf(
                  rewriter, "rvv", "layered-storage-decode",
                  "rvv.layered-storage-decode", "rvv.layered-storage-decode",
                  rawGroups,
                  leader.resultType.getLayout().getRegisterGroups(), 1, 0,
                  "none",
                  leader.resultType.getLayout().getValidity() == "tail"
                      ? "agnostic"
                      : "exact"));
          riscv_internal::copyOrigin(first, value);
          decoded[static_cast<size_t>(window)].push_back(value.getResult());
        }
      }

      for (int64_t storageLayer = 0; storageLayer < layers; ++storageLayer) {
        for (int64_t window = 0; window < windowsPerLayer; ++window) {
          const int64_t slot =
              (storageLayer * layer + window * lanes) / lanes;
          for (CompleteLayeredExtract &info :
               bySlot[static_cast<size_t>(slot)]) {
            info.extract.getResult().replaceAllUsesWith(
                decoded[static_cast<size_t>(window)]
                       [static_cast<size_t>(storageLayer)]);
            consumed.insert(info.extract);
          }
        }
      }
      for (auto &slot : bySlot) {
        for (CompleteLayeredExtract &info : slot) {
          rewriter.eraseOp(info.extract);
          for (riscv::ConvertLayoutOp conversion :
               llvm::reverse(info.conversions))
            if (conversion.getResult().use_empty())
              rewriter.eraseOp(conversion);
          if (info.field != leader.field &&
              info.field.getResult().use_empty())
            rewriter.eraseOp(info.field);
        }
      }
    }

    for (riscv::ExtractOp first : extracts) {
      if (consumed.contains(first))
        continue;
      riscv::FieldOp firstField;
      riscv::PhysicalPointOp firstPoint;
      llvm::SmallVector<riscv::ConvertLayoutOp> firstConversions;
      if (!isLayerExtract(first, firstField, firstPoint, &firstConversions))
        continue;
      const int64_t layer = first.getAccess().getLayerSize();

      LinearIndex firstIndex;
      decomposeIndex(firstPoint.getBase(), 1, firstIndex);
      if (!firstIndex.valid)
        continue;

      riscv::ExtractOp second;
      riscv::FieldOp secondField;
      llvm::SmallVector<riscv::ConvertLayoutOp> secondConversions;
      for (riscv::ExtractOp candidate : extracts) {
        if (candidate == first || consumed.contains(candidate) ||
            candidate->getBlock() != first->getBlock() ||
            first->isBeforeInBlock(candidate) == false)
          continue;
        riscv::PhysicalPointOp candidatePoint;
        riscv::FieldOp candidateField;
        llvm::SmallVector<riscv::ConvertLayoutOp> candidateConversions;
        if (!isLayerExtract(candidate, candidateField, candidatePoint,
                            &candidateConversions) ||
            candidate.getResult().getType() != first.getResult().getType() ||
            candidate.getAccess() != first.getAccess() ||
            !sameFieldEdge(candidateField, firstField) ||
            candidatePoint.getParent() != firstPoint.getParent() ||
            candidatePoint.getActive() != firstPoint.getActive() ||
            candidatePoint.getPartition() != firstPoint.getPartition() ||
            candidatePoint.getResult().getType() !=
                firstPoint.getResult().getType())
          continue;
        LinearIndex candidateIndex;
        decomposeIndex(candidatePoint.getBase(), 1, candidateIndex);
        if (!sameTerms(firstIndex, candidateIndex) ||
            candidateIndex.constant - firstIndex.constant != layer)
          continue;
        second = candidate;
        secondField = candidateField;
        secondConversions = std::move(candidateConversions);
        break;
      }
      if (!second)
        continue;
      if (!canFoldReadConversions(first, second, firstConversions,
                                  secondConversions))
        continue;

      auto firstType = mlir::cast<riscv::ValueType>(first.getResult().getType());
      const int64_t groups = firstType.getLayout().getRegisterGroups();
      const bool tail = firstType.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(first);
      auto shared = rewriter.create<riscv::RVVLayeredWindowOp>(
          first.getLoc(),
          mlir::TypeRange{first.getResult().getType(), second.getResult().getType()},
          firstField.getResult(), firstPoint.getResult(), first.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "layered-window",
                               "rvv.layered-window", "rvv.layered-window", 0,
                               groups * 2, groups, 0, "none",
                               tail ? "agnostic" : "exact"));
      if (auto origin = first->getAttr("source_origin"))
        shared->setAttr("source_origin", origin);
      shared->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.extract"));
      first.getResult().replaceAllUsesWith(shared.getFirst());
      second.getResult().replaceAllUsesWith(shared.getSecond());
      consumed.insert(first);
      consumed.insert(second);
      rewriter.eraseOp(first);
      rewriter.eraseOp(second);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(firstConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(secondConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      if (secondField.getResult().use_empty())
        rewriter.eraseOp(secondField);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createShareRISCVLayeredWindowsPass() {
  return std::make_unique<ShareRISCVLayeredWindowsPass>();
}
