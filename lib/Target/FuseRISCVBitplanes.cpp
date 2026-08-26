#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

std::optional<int64_t> integerConstant(mlir::Value value) {
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return integerConstant(cast.getInput());
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

mlir::Value stripLayoutConversions(mlir::Value value) {
  while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    value = conversion.getInput();
  return value;
}

bool isBinaryWithConstant(riscv::BinaryOp operation, llvm::StringRef kind,
                          int64_t expected, mlir::Value &other) {
  if (!operation || operation.getKind() != kind)
    return false;
  if (auto value = integerConstant(operation.getRhs()); value == expected) {
    other = operation.getLhs();
    return true;
  }
  if ((kind == "and" || kind == "or" || kind == "add" || kind == "mul") &&
      integerConstant(operation.getLhs()) == expected) {
    other = operation.getRhs();
    return true;
  }
  return false;
}

struct BitplaneRelation {
  mlir::Value low;
  riscv::FieldOp plane;
  int64_t insertBit = 0;
};

std::optional<BitplaneRelation> matchBitplane(riscv::BinaryOp merge,
                                              mlir::Value low,
                                              mlir::Value high) {
  if (!merge || merge.getKind() != "or")
    return std::nullopt;
  auto shiftHigh = high.getDefiningOp<riscv::BinaryOp>();
  if (!shiftHigh || shiftHigh.getKind() != "shl")
    return std::nullopt;
  auto insertBit = integerConstant(shiftHigh.getRhs());
  if (!insertBit || *insertBit <= 0 || *insertBit >= 8)
    return std::nullopt;

  mlir::Value shifted;
  if (!isBinaryWithConstant(shiftHigh.getLhs().getDefiningOp<riscv::BinaryOp>(),
                            "and", 1, shifted))
    return std::nullopt;
  auto variableShift = shifted.getDefiningOp<riscv::BinaryOp>();
  if (!variableShift || variableShift.getKind() != "shr")
    return std::nullopt;
  auto extract = variableShift.getLhs().getDefiningOp<riscv::ExtractOp>();
  auto position = variableShift.getRhs().getDefiningOp<riscv::BinaryOp>();
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
      integerConstant(position.getRhs()) != 8 ||
      gatherSelectors != 1 || extract.getIndices().size() != 1)
    return std::nullopt;

  auto byteIndex = extract.getIndices().front().getDefiningOp<riscv::BinaryOp>();
  if (!byteIndex || byteIndex.getKind() != "div" ||
      integerConstant(byteIndex.getRhs()) != 8 ||
      byteIndex.getLhs() != position.getLhs())
    return std::nullopt;
  auto coordinates =
      stripLayoutConversions(byteIndex.getLhs()).getDefiningOp<riscv::IotaOp>();
  if (!coordinates || coordinates.getStart() != 0)
    return std::nullopt;

  mlir::Value planeInput = extract.getInput();
  while (auto conversion =
             planeInput.getDefiningOp<riscv::ConvertLayoutOp>())
    planeInput = conversion.getInput();
  auto plane = planeInput.getDefiningOp<riscv::FieldOp>();
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
  const size_t packedAxis = static_cast<size_t>(
      coordinateAxis - resultType.getAxisIds().asArrayRef().begin());
  for (size_t index = 0; index < resultType.getShape().size(); ++index) {
    const int64_t lowExtent = resultType.getShape()[index];
    const int64_t planeExtent = planeType.getShape()[index];
    if (index == packedAxis) {
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

  return BitplaneRelation{low, plane, *insertBit};
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
    llvm::SmallVector<riscv::BinaryOp> merges;
    getOperation().walk([&](riscv::BinaryOp operation) {
      if (operation.getKind() == "or")
        merges.push_back(operation);
    });
    mlir::IRRewriter rewriter(&getContext());
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
                               "rvv.bitplane-merge", "rvv.bitplane-merge", 0,
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
