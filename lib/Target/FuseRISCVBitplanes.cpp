#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

bool isBinaryWithConstant(riscv::BinaryOp operation, llvm::StringRef kind,
                          int64_t expected, mlir::Value &other) {
  if (!operation || operation.getKind() != kind)
    return false;
  if (auto value = riscv_internal::constantInt(operation.getRhs());
      value == expected) {
    other = operation.getLhs();
    return true;
  }
  if ((kind == "and" || kind == "or" || kind == "add" || kind == "mul") &&
      riscv_internal::constantInt(operation.getLhs()) == expected) {
    other = operation.getRhs();
    return true;
  }
  return false;
}

struct BitplaneRelation {
  mlir::Value low;
  riscv::FieldOp plane;
  int64_t insertBit = 0;
  llvm::StringRef instruction;
};

struct BitmaskRelation {
  riscv::FieldOp field;
  riscv::PhysicalPointOp origin;
  riscv::PhysicalPointOp point;
};

std::optional<BitmaskRelation>
matchBitmaskDecode(riscv::ConvertLayoutOp conversion) {
  if (!conversion || conversion.getConversion().getEffect() != "pure" ||
      conversion.getConversion().getKind() != "time_to_lane")
    return std::nullopt;
  auto extract = conversion.getInput().getDefiningOp<riscv::ExtractOp>();
  auto field = extract ? extract.getInput().getDefiningOp<riscv::FieldOp>()
                       : riscv::FieldOp();
  auto result = mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
  auto element = result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                        : mlir::IntegerType();
  if (!extract || !field || !result || !element || element.isSigned() ||
      element.getWidth() != 1 || result.getLayout().getCarrier() != "rvv" ||
      extract.getIndices().size() != 1)
    return std::nullopt;

  riscv::PhysicalPointOp point;
  size_t cursor = 0;
  for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (cursor >= extract.getIndices().size() || selector != "domain" || point)
      return std::nullopt;
    point = extract.getIndices()[cursor++].getDefiningOp<riscv::PhysicalPointOp>();
  }
  auto origin = point
                    ? point.getParent().getDefiningOp<riscv::PhysicalPointOp>()
                    : riscv::PhysicalPointOp();
  if (!point || !origin || cursor != extract.getIndices().size())
    return std::nullopt;

  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.mapping != "grouped_layered" || facts.group <= 0 ||
      facts.layer <= 0 || facts.group != facts.layer * 8 ||
      facts.bitOffset % 8 || facts.order != "lo_first")
    return std::nullopt;
  return BitmaskRelation{field, origin, point};
}

std::optional<BitplaneRelation>
matchLogicalBitplane(riscv::BinaryOp merge, mlir::Value low,
                     riscv::BinaryOp shiftHigh, int64_t insertBit) {
  mlir::Value planeValue =
      riscv_internal::stripRepresentationConversions(shiftHigh.getLhs());
  if (auto widen = planeValue.getDefiningOp<riscv::WidenOp>())
    planeValue = riscv_internal::stripRepresentationConversions(widen.getInput());
  else if (auto cast = planeValue.getDefiningOp<riscv::CastOp>())
    planeValue = riscv_internal::stripRepresentationConversions(cast.getInput());
  else
    return std::nullopt;

  auto plane = riscv_internal::sourceField(planeValue);
  auto lowType = mlir::dyn_cast<riscv::ValueType>(low.getType());
  auto resultType = mlir::dyn_cast<riscv::ValueType>(merge.getResult().getType());
  auto planeType = plane
                       ? mlir::dyn_cast<riscv::ValueType>(plane.getResult().getType())
                       : riscv::ValueType();
  auto lowInteger = lowType ? mlir::dyn_cast<mlir::IntegerType>(
                                  riscv_internal::logicalElement(lowType))
                            : mlir::IntegerType();
  auto planeInteger = planeType ? mlir::dyn_cast<mlir::IntegerType>(
                                      riscv_internal::logicalElement(planeType))
                                : mlir::IntegerType();
  if (!plane || !lowType || !resultType || !planeType || !lowInteger ||
      lowInteger.isSigned() || lowInteger.getWidth() != 8 || !planeInteger ||
      planeInteger.isSigned() || planeInteger.getWidth() != 1 ||
      lowType != resultType || lowType.getLayout().getCarrier() != "rvv" ||
      lowType.getShape() != planeType.getShape() ||
      lowType.getAxisIds() != planeType.getAxisIds())
    return std::nullopt;

  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(plane);
  if (facts.mapping != "grouped_layered" || facts.group != 8 ||
      facts.layer != 1 || facts.logicalRank <= 0 ||
      facts.logicalRank > static_cast<int64_t>(planeType.getAxisIds().size()))
    return std::nullopt;

  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(lowType.getLayout().getAxisIds().asArrayRef(),
                 lowType.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  auto logicalAxes = planeType.getAxisIds().asArrayRef().take_back(
      static_cast<size_t>(facts.logicalRank));
  if (laneAxes != 1 || logicalAxes.size() != 1 ||
      laneAxis != logicalAxes.front() || lowType.getLayout().getVl() <= 0 ||
      lowType.getLayout().getVl() % 8)
    return std::nullopt;
  auto owner = mlir::dyn_cast<riscv::ValueType>(plane.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(
                              riscv_internal::logicalElement(owner))
                        : kernel::EncodingType();
  if (!encoding || riscv_internal::interleaveRows(plane, encoding) != 0)
    return std::nullopt;
  return BitplaneRelation{low, plane, insertBit,
                          "rvv.bitplane-merge.mask"};
}

std::optional<BitplaneRelation> matchBitplane(riscv::BinaryOp merge,
                                              mlir::Value low,
                                              mlir::Value high) {
  if (!merge || merge.getKind() != "or")
    return std::nullopt;
  auto shiftHigh = riscv_internal::stripRepresentationConversions(high)
                       .getDefiningOp<riscv::BinaryOp>();
  if (!shiftHigh || shiftHigh.getKind() != "shl")
    return std::nullopt;
  auto insertBit = riscv_internal::constantInt(shiftHigh.getRhs());
  if (!insertBit || *insertBit <= 0 || *insertBit >= 8)
    return std::nullopt;

  if (auto logical = matchLogicalBitplane(merge, low, shiftHigh, *insertBit))
    return logical;

  mlir::Value shifted;
  if (!isBinaryWithConstant(
          riscv_internal::stripRepresentationConversions(shiftHigh.getLhs())
              .getDefiningOp<riscv::BinaryOp>(),
                            "and", 1, shifted))
    return std::nullopt;
  auto variableShift = riscv_internal::stripRepresentationConversions(shifted)
                           .getDefiningOp<riscv::BinaryOp>();
  if (!variableShift || variableShift.getKind() != "shr")
    return std::nullopt;
  auto extract =
      riscv_internal::stripRepresentationConversions(variableShift.getLhs())
          .getDefiningOp<riscv::ExtractOp>();
  auto position =
      riscv_internal::stripRepresentationConversions(variableShift.getRhs())
          .getDefiningOp<riscv::BinaryOp>();
  size_t gatherSelectors = 0;
  if (extract)
    for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "gather")
        ++gatherSelectors;
      else if (selector != "all")
        return std::nullopt;
    }
  if (!extract || !position || position.getKind() != "mod" ||
      riscv_internal::constantInt(position.getRhs()) != 8 ||
      gatherSelectors != 1 || extract.getIndices().size() != 1)
    return std::nullopt;

  auto byteIndex =
      riscv_internal::stripRepresentationConversions(extract.getIndices().front())
          .getDefiningOp<riscv::BinaryOp>();
  if (!byteIndex || byteIndex.getKind() != "div" ||
      riscv_internal::constantInt(byteIndex.getRhs()) != 8 ||
      riscv_internal::stripRepresentationConversions(byteIndex.getLhs()) !=
          riscv_internal::stripRepresentationConversions(position.getLhs()))
    return std::nullopt;
  auto coordinates =
      riscv_internal::stripRepresentationConversions(byteIndex.getLhs())
          .getDefiningOp<riscv::IotaOp>();
  if (!coordinates || coordinates.getStart() != 0)
    return std::nullopt;

  auto plane = riscv_internal::sourceField(extract.getInput());
  auto lowType = mlir::dyn_cast<riscv::ValueType>(low.getType());
  auto resultType = mlir::dyn_cast<riscv::ValueType>(merge.getResult().getType());
  auto planeType = plane
                       ? mlir::dyn_cast<riscv::ValueType>(plane.getResult().getType())
                       : riscv::ValueType();
  auto coordinateType =
      mlir::dyn_cast<riscv::ValueType>(coordinates.getResult().getType());
  if (!plane || !lowType || !resultType || !planeType ||
      !coordinateType || coordinateType.getAxisIds().size() != 1 ||
      lowType != resultType || lowType.getLayout().getCarrier() != "rvv" ||
      planeType.getLayout().getCarrier() != "local" ||
      coordinates.getEnd() - coordinates.getStart() !=
          riscv_internal::staticProduct(coordinateType.getShape()).value_or(-1))
    return std::nullopt;
  auto coordinateAxis = llvm::find(resultType.getAxisIds().asArrayRef(),
                                   coordinateType.getAxisIds()[0]);
  if (coordinateAxis == resultType.getAxisIds().asArrayRef().end() ||
      resultType.getShape()[static_cast<size_t>(
          coordinateAxis - resultType.getAxisIds().asArrayRef().begin())] !=
          coordinateType.getShape()[0])
    return std::nullopt;

  auto lowInteger =
      mlir::dyn_cast<mlir::IntegerType>(riscv_internal::logicalElement(lowType));
  auto planeInteger = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(planeType));
  if (!lowInteger || lowInteger.isSigned() || lowInteger.getWidth() != 8 ||
      !planeInteger || planeInteger.isSigned() || planeInteger.getWidth() != 8 ||
      plane.getAccess().getMapping() != "natural" ||
      plane.getAccess().getBitOffset() % 8)
    return std::nullopt;

  if (resultType.getAxisIds() != planeType.getAxisIds() ||
      resultType.getShape().size() != planeType.getShape().size())
    return std::nullopt;
  const size_t packedPosition = static_cast<size_t>(
      coordinateAxis - resultType.getAxisIds().asArrayRef().begin());
  for (size_t index = 0; index < resultType.getShape().size(); ++index) {
    const int64_t lowExtent = resultType.getShape()[index];
    const int64_t planeExtent = planeType.getShape()[index];
    if (index == packedPosition) {
      if (lowExtent <= 0 || planeExtent <= 0 ||
          planeExtent > std::numeric_limits<int64_t>::max() / 8 ||
          planeExtent * 8 != lowExtent)
        return std::nullopt;
    } else if (lowExtent != planeExtent) {
      return std::nullopt;
    }
  }
  auto owner = mlir::dyn_cast<riscv::ValueType>(plane.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(
                              riscv_internal::logicalElement(owner))
                        : kernel::EncodingType();
  if (!encoding || riscv_internal::interleaveRows(plane, encoding) != 0)
    return std::nullopt;

  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(lowType.getLayout().getAxisIds().asArrayRef(),
                 lowType.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  if (laneAxes != 1)
    return std::nullopt;
  const int64_t packedAxis = coordinateType.getAxisIds()[0];
  llvm::StringRef instruction;
  if (laneAxis == packedAxis && lowType.getLayout().getVl() > 0 &&
      lowType.getLayout().getVl() % 8 == 0) {
    instruction = "rvv.bitplane-merge.mask";
  } else if (laneAxis != packedAxis) {
    auto packed = llvm::find(lowType.getAxisIds().asArrayRef(), packedAxis);
    const size_t position = static_cast<size_t>(
        packed - lowType.getAxisIds().asArrayRef().begin());
    bool otherTime = false;
    for (auto [axis, time] :
         llvm::zip(lowType.getAxisIds().asArrayRef(),
                   lowType.getLayout().getTimeFactors().asArrayRef()))
      otherTime |= axis != packedAxis && time != 1;
    if (packed == lowType.getAxisIds().asArrayRef().end() || otherTime ||
        lowType.getLayout().getLaneFactors()[position] != 1 ||
        lowType.getLayout().getReplicaFactors()[position] != 1 ||
        lowType.getLayout().getFragmentFactors()[position] != 1 ||
        lowType.getLayout().getLocalFactors()[position] != 1 ||
        lowType.getLayout().getTimeFactors()[position] !=
            lowType.getShape()[position])
      return std::nullopt;
    instruction = "rvv.bitplane-merge.strided";
  } else {
    return std::nullopt;
  }

  return BitplaneRelation{low, plane, *insertBit, instruction};
}

void eraseDeadTree(mlir::Value value, mlir::IRRewriter &rewriter) {
  mlir::Operation *operation = value.getDefiningOp();
  const bool pureLayoutConversion =
      operation && mlir::isa<riscv::ConvertLayoutOp>(operation) &&
      mlir::cast<riscv::ConvertLayoutOp>(operation)
              .getConversion()
              .getEffect() == "pure";
  if (!operation ||
      (!mlir::isMemoryEffectFree(operation) && !pureLayoutConversion) ||
      llvm::any_of(operation->getResults(),
                   [](mlir::Value result) { return !result.use_empty(); }))
    return;
  llvm::SmallVector<mlir::Value> operands(operation->getOperands());
  rewriter.eraseOp(operation);
  for (mlir::Value operand : operands)
    eraseDeadTree(operand, rewriter);
}

class FuseRISCVBitplanesPass
    : public mlir::PassWrapper<FuseRISCVBitplanesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-fuse-bitplanes";
  }
  llvm::StringRef getDescription() const override {
    return "Fuse typed packed-bit extraction with its RVV value consumer";
  }

  void runOnOperation() override {
    llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
    getOperation().walk([&](riscv::ConvertLayoutOp operation) {
      conversions.push_back(operation);
    });
    mlir::IRRewriter rewriter(&getContext());
    for (riscv::ConvertLayoutOp conversion : conversions) {
      auto relation = matchBitmaskDecode(conversion);
      if (!relation)
        continue;
      auto result = conversion.getResult().getType();
      const int64_t resultGroups = result.getLayout().getRegisterGroups();
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (result.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = result.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(conversion);
      auto decoded = rewriter.create<riscv::RVVBitmaskDecodeOp>(
          conversion.getLoc(), result, relation->field.getResult(),
          relation->origin.getResult(), relation->point.getResult(),
          relation->field.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "bitmask-decode",
                               "rvv.bitmask-decode", "rvv.bitmask-decode", 0,
                               resultGroups, temporaryGroups, 0, "none",
                               tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(conversion, decoded);
      decoded->setAttr("canonical_op",
                       rewriter.getStringAttr("weft_kernel.extract"));
      mlir::Value oldInput = conversion.getInput();
      rewriter.replaceOp(conversion, decoded.getResult());
      eraseDeadTree(oldInput, rewriter);
    }

    llvm::SmallVector<riscv::BinaryOp> merges;
    getOperation().walk([&](riscv::BinaryOp operation) {
      if (operation.getKind() == "or")
        merges.push_back(operation);
    });
    for (riscv::BinaryOp merge : merges) {
      auto relation = matchBitplane(merge, merge.getLhs(), merge.getRhs());
      if (!relation)
        relation = matchBitplane(merge, merge.getRhs(), merge.getLhs());
      if (!relation)
        continue;
      rewriter.setInsertionPoint(merge);
      auto fused = rewriter.create<riscv::RVVBitplaneMergeOp>(
          merge.getLoc(), merge.getResult().getType(), relation->low,
          relation->plane.getResult(), relation->insertBit,
          riscv_internal::leaf(rewriter, "rvv", "bitplane-merge",
                               relation->instruction, relation->instruction, 0,
                               0, 1, 0, "none", "agnostic",
                               {relation->insertBit}));
      riscv_internal::copyOrigin(merge, fused);
      fused->setAttr("canonical_op",
                     rewriter.getStringAttr("weft_kernel.binary"));
      llvm::SmallVector<mlir::Value> oldOperands(merge->getOperands());
      rewriter.replaceOp(merge, fused.getResult());
      for (mlir::Value operand : oldOperands)
        eraseDeadTree(operand, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createFuseRISCVBitplanesPass() {
  return std::make_unique<FuseRISCVBitplanesPass>();
}
