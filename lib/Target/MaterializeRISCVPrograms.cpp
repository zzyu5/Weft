#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

void copyProvenance(mlir::Operation *source, mlir::Operation *target) {
  for (llvm::StringRef name : {"canonical_op", "source_origin"})
    if (mlir::Attribute attribute = source->getAttr(name))
      target->setAttr(name, attribute);
}

int64_t factorProduct(mlir::DenseI64ArrayAttr factors) {
  int64_t result = 1;
  for (int64_t factor : factors.asArrayRef()) {
    if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
      return -1;
    result *= factor;
  }
  return result;
}

llvm::SmallVector<int64_t> representedShape(riscv::LayoutAttr layout) {
  llvm::SmallVector<int64_t> shape;
  for (size_t position = 0; position < layout.getAxisIds().size(); ++position) {
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             layout.getTimeFactors(), layout.getLaneFactors(),
             layout.getReplicaFactors(), layout.getFragmentFactors(),
             layout.getLocalFactors()}) {
      const int64_t factor = factors[position];
      if (factor <= 0 ||
          extent > std::numeric_limits<int64_t>::max() / factor)
        return {};
      extent *= factor;
    }
    shape.push_back(extent);
  }
  return shape;
}

riscv::LayoutAttr singleStreamLayout(mlir::Builder &builder,
                                     riscv::LayoutAttr source) {
  llvm::SmallVector<int64_t> time(source.getAxisIds().size(), 1);
  const int64_t replicas = factorProduct(source.getReplicaFactors());
  const int64_t groupsPerVector = (source.getLmulEighths() + 7) / 8;
  if (replicas <= 0 || groupsPerVector <= 0)
    return {};
  return riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      riscv_internal::integers(builder, time), source.getLaneFactors(),
      source.getReplicaFactors(), source.getFragmentFactors(),
      source.getLocalFactors(), source.getSew(), source.getLmulEighths(),
      source.getVl(), groupsPerVector * replicas, "tail");
}

riscv::ValueType valueForLayout(mlir::Builder &builder, mlir::Type element,
                                riscv::LayoutAttr layout) {
  llvm::SmallVector<int64_t> shape = representedShape(layout);
  if (shape.empty() || shape.size() != layout.getAxisIds().size())
    return {};
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               layout.getAxisIds(), layout);
}

riscv::ValueType contractOperandType(mlir::Builder &builder,
                                     riscv::MemDescType memory,
                                     mlir::Type element,
                                     riscv::LayoutAttr operand,
                                     riscv::LayoutAttr accumulator,
                                     int64_t reductionAxis) {
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane;
  llvm::SmallVector<int64_t> replica;
  llvm::SmallVector<int64_t> fragment;
  llvm::SmallVector<int64_t> local;
  for (int64_t axis : memory.getAxisIds().asArrayRef()) {
    riscv::LayoutAttr source = axis == reductionAxis ? operand : accumulator;
    auto found = llvm::find(source.getAxisIds().asArrayRef(), axis);
    if (found == source.getAxisIds().asArrayRef().end())
      return {};
    const size_t position = static_cast<size_t>(
        found - source.getAxisIds().asArrayRef().begin());
    time.push_back(source.getTimeFactors()[position]);
    lane.push_back(source.getLaneFactors()[position]);
    replica.push_back(source.getReplicaFactors()[position]);
    fragment.push_back(source.getFragmentFactors()[position]);
    local.push_back(source.getLocalFactors()[position]);
  }
  int64_t replicas = 1;
  for (int64_t factor : replica) {
    if (factor <= 0 ||
        replicas > std::numeric_limits<int64_t>::max() / factor)
      return {};
    replicas *= factor;
  }
  const int64_t groupsPerVector = (operand.getLmulEighths() + 7) / 8;
  if (replicas <= 0 || groupsPerVector <= 0)
    return {};
  auto axes = memory.getAxisIds();
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axes,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local),
      riscv_internal::logicalBitWidth(element), operand.getLmulEighths(),
      operand.getVl(), groupsPerVector * replicas, "tail");
  return valueForLayout(builder, element, layout);
}

llvm::SmallVector<mlir::Value>
freeAxisPoints(mlir::Value source, riscv::ValueType type,
               int64_t reductionAxis) {
  llvm::SmallVector<mlir::Value> points;
  llvm::DenseSet<int64_t> seen;
  for (auto [position, axis] :
       llvm::enumerate(type.getAxisIds().asArrayRef())) {
    if (axis == reductionAxis ||
        type.getLayout().getReplicaFactors()[position] <= 1 ||
        !seen.insert(axis).second)
      continue;
    auto point = riscv_internal::originPoint(source, axis);
    if (!point)
      return {};
    points.push_back(point.getResult());
  }
  return points;
}

class MaterializeRISCVProgramsPass final
    : public mlir::PassWrapper<MaterializeRISCVProgramsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVProgramsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-materialize-programs";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize transient RISC-V macro programs into explicit control and SSA";
  }

  void runOnOperation() final {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    llvm::SmallVector<riscv::EncodedLocalPackOp> packs;
    getOperation().walk(
        [&](riscv::EncodedLocalPackOp operation) { packs.push_back(operation); });

    for (riscv::EncodedLocalPackOp operation : packs) {
      auto rowPoint = operation.getRowPoint().getDefiningOp<riscv::PhysicalPointOp>();
      auto recordPoint =
          operation.getRecordPoint().getDefiningOp<riscv::PhysicalPointOp>();
      riscv::LocalPackPlanAttr plan = operation.getPlan();
      if (!rowPoint || !recordPoint) {
        operation.emitError(
            "encoded local pack program requires explicit physical row and record points");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(operation);
      auto guard = rewriter.create<riscv::IndexMultipleGuardOp>(
          operation.getLoc(), recordPoint.getActive(), plan.getRecordElements(),
          riscv_internal::leaf(rewriter, "scalar", "index-guard",
                               "scalar.index-multiple-guard",
                               "scalar.index-multiple-guard", 0, 0, 0));
      copyProvenance(operation, guard);

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 1);
      mlir::Value rowsPerGroup =
          rewriter.create<mlir::arith::ConstantIndexOp>(
              operation.getLoc(), plan.getInterleaveRows());
      mlir::Value elementsPerRecord =
          rewriter.create<mlir::arith::ConstantIndexOp>(
              operation.getLoc(), plan.getRecordElements());
      mlir::Value bytesPerRecord =
          rewriter.create<mlir::arith::ConstantIndexOp>(
              operation.getLoc(), plan.getRecordBytes());
      mlir::Value rowGroups = rewriter.create<mlir::arith::CeilDivUIOp>(
          operation.getLoc(), rowPoint.getActive(), rowsPerGroup);
      mlir::Value blockCount = rewriter.create<mlir::arith::DivUIOp>(
          operation.getLoc(), recordPoint.getActive(), elementsPerRecord);

      auto rowLoop = rewriter.create<mlir::scf::ForOp>(
          operation.getLoc(), zero, rowGroups, one);
      rowLoop->setAttr("weft.riscv.direction",
                       rewriter.getStringAttr("ascending"));
      copyProvenance(operation, rowLoop);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value rowBase = rewriter.create<mlir::arith::MulIOp>(
          operation.getLoc(), rowLoop.getInductionVar(), rowsPerGroup);
      mlir::Value rowsRemaining = rewriter.create<mlir::arith::SubIOp>(
          operation.getLoc(), rowPoint.getActive(), rowBase);
      mlir::Value transferVL = rewriter.create<mlir::arith::MinUIOp>(
          operation.getLoc(), rowsRemaining, rowsPerGroup);

      auto blockLoop = rewriter.create<mlir::scf::ForOp>(
          operation.getLoc(), zero, blockCount, one);
      blockLoop->setAttr("weft.riscv.direction",
                         rewriter.getStringAttr("ascending"));
      copyProvenance(operation, blockLoop);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      auto byteLoop = rewriter.create<mlir::scf::ForOp>(
          operation.getLoc(), zero, bytesPerRecord, one);
      byteLoop->setAttr("weft.riscv.direction",
                        rewriter.getStringAttr("ascending"));
      copyProvenance(operation, byteLoop);
      rewriter.setInsertionPointToStart(byteLoop.getBody());
      auto transfer =
          rewriter.create<riscv::RVVEncodedLocalPackTransferOp>(
              operation.getLoc(), operation.getInput(), operation.getStorage(),
              rowLoop.getInductionVar(), rowBase, blockLoop.getInductionVar(),
              blockCount, byteLoop.getInductionVar(), transferVL, plan,
              operation.getTransferLayout(),
              riscv_internal::leaf(
                  rewriter, "transfer", "local-pack-transfer",
                  "rvv.local-pack.transfer.interleave",
                  "rvv.local-pack.transfer.interleave", 0, 0,
                  operation.getTransferLayout().getRegisterGroups()));
      copyProvenance(operation, transfer);

      rewriter.setInsertionPointAfter(rowLoop);
      auto binding = rewriter.create<riscv::EncodedLocalBindOp>(
          operation.getLoc(), operation.getResult().getType(),
          operation.getInput(), operation.getStorage(), blockCount, plan);
      copyProvenance(operation, binding);
      operation.getResult().replaceAllUsesWith(binding.getResult());
      rewriter.eraseOp(operation);
    }

    llvm::SmallVector<riscv::RVVStreamReduceOp> streamReductions;
    getOperation().walk([&](riscv::RVVStreamReduceOp operation) {
      streamReductions.push_back(operation);
    });
    for (riscv::RVVStreamReduceOp operation : streamReductions) {
      riscv::LayoutAttr layout =
          singleStreamLayout(rewriter, operation.getInputLayout());
      riscv::ValueType streamType = valueForLayout(
          rewriter, mlir::Float32Type::get(rewriter.getContext()), layout);
      const int64_t logicalAxis = layout && !layout.getAxisIds().empty()
                                      ? layout.getAxisIds()[0]
                                      : 0;
      const int64_t streams = factorProduct(
          operation.getInputLayout().getTimeFactors());
      if (!layout || !streamType || streams <= 1 || logicalAxis <= 0) {
        operation.emitError(
            "stream reduction has no legal single-stream lane representation");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(operation);
      llvm::SmallVector<mlir::Value> initial;
      for (mlir::Attribute kindAttribute : operation.getKinds()) {
        llvm::StringRef kind =
            mlir::cast<mlir::StringAttr>(kindAttribute).getValue();
        double seedValue = kind == "max"
                               ? -std::numeric_limits<double>::infinity()
                               : kind == "min"
                                     ? std::numeric_limits<double>::infinity()
                                     : 0.0;
        auto seed = rewriter.create<riscv::ConstantOp>(
            operation.getLoc(), mlir::Float32Type::get(rewriter.getContext()),
            rewriter.getF32FloatAttr(seedValue));
        auto splat = rewriter.create<riscv::RVVSplatOp>(
            operation.getLoc(), streamType, seed,
            riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                                 "rvv.splat", 0,
                                 layout.getRegisterGroups()));
        copyProvenance(operation, seed);
        copyProvenance(operation, splat);
        initial.push_back(splat.getResult());
      }
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 1);
      mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), streams);
      mlir::Value lanes = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), layout.getVl());
      mlir::Value valid = rewriter.create<mlir::arith::ConstantIntOp>(
          operation.getLoc(), 1, 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          operation.getLoc(), zero, upper, one, initial);
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      copyProvenance(operation, loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value offset = rewriter.create<mlir::arith::MulIOp>(
          operation.getLoc(), loop.getInductionVar(), lanes);
      auto load = rewriter.create<riscv::RVVStreamLoadOp>(
          operation.getLoc(), streamType, operation.getSource(), offset, lanes,
          valid, mlir::ValueRange{}, logicalAxis, false,
          operation.getAccess(),
          riscv_internal::leaf(rewriter, "transfer", "stream-load",
                               "rvv.stream-load", "rvv.stream-load", 0,
                               layout.getRegisterGroups(), 0, 0, "none",
                               "exact"));
      copyProvenance(operation, load);
      llvm::SmallVector<mlir::Value> next;
      for (auto [index, kindAttribute] :
           llvm::enumerate(operation.getKinds())) {
        llvm::StringRef kind =
            mlir::cast<mlir::StringAttr>(kindAttribute).getValue();
        auto step = rewriter.create<riscv::RVVStreamReduceStepOp>(
            operation.getLoc(), streamType, load.getResult(),
            loop.getRegionIterArg(index), lanes, kind,
            riscv_internal::leaf(
                rewriter, "rvv", "stream-reduce-step",
                "rvv.stream-reduce-step", "rvv.stream-reduce-step",
                2 * layout.getRegisterGroups(), layout.getRegisterGroups()));
        copyProvenance(operation, step);
        next.push_back(step.getResult());
      }
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<mlir::scf::YieldOp>(operation.getLoc(), next);

      rewriter.setInsertionPointAfter(loop);
      llvm::SmallVector<mlir::Value> results;
      for (auto [index, kindAttribute] :
           llvm::enumerate(operation.getKinds())) {
        llvm::StringRef kind =
            mlir::cast<mlir::StringAttr>(kindAttribute).getValue();
        auto finalize = rewriter.create<riscv::RVVStreamFinalizeOp>(
            operation.getLoc(), operation.getResult(index).getType(),
            loop.getResult(index), logicalAxis, kind,
            riscv_internal::leaf(
                rewriter, "rvv", "stream-finalize", "rvv.stream-finalize",
                "rvv.stream-finalize", layout.getRegisterGroups(), 0, 1));
        copyProvenance(operation, finalize);
        results.push_back(finalize.getResult());
      }
      for (auto [source, target] :
           llvm::zip(operation.getResults(), results))
        source.replaceAllUsesWith(target);
      rewriter.eraseOp(operation);
    }

    llvm::SmallVector<riscv::RVVStreamDotOp> streamDots;
    getOperation().walk(
        [&](riscv::RVVStreamDotOp operation) { streamDots.push_back(operation); });
    for (riscv::RVVStreamDotOp operation : streamDots) {
      riscv::LayoutAttr layout =
          singleStreamLayout(rewriter, operation.getInputLayout());
      riscv::ValueType streamType = valueForLayout(
          rewriter, mlir::Float32Type::get(rewriter.getContext()), layout);
      const int64_t logicalAxis = layout && !layout.getAxisIds().empty()
                                      ? layout.getAxisIds()[0]
                                      : 0;
      if (!layout || !streamType || logicalAxis <= 0) {
        operation.emitError(
            "stream dot has no legal single-stream lane representation");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(operation);
      auto zeroValue = rewriter.create<riscv::ConstantOp>(
          operation.getLoc(), mlir::Float32Type::get(rewriter.getContext()),
          rewriter.getF32FloatAttr(0.0));
      auto accumulator = rewriter.create<riscv::RVVSplatOp>(
          operation.getLoc(), streamType, zeroValue,
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                               "rvv.splat", 0,
                               layout.getRegisterGroups()));
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 0);
      mlir::Value lanes = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), layout.getVl());
      mlir::Value valid = rewriter.create<mlir::arith::ConstantIntOp>(
          operation.getLoc(), 1, 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          operation.getLoc(), lower, operation.getExtent(), lanes,
          mlir::ValueRange{accumulator.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      copyProvenance(operation, zeroValue);
      copyProvenance(operation, accumulator);
      copyProvenance(operation, loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value remaining = rewriter.create<mlir::arith::SubIOp>(
          operation.getLoc(), operation.getExtent(), loop.getInductionVar());
      mlir::Value active = rewriter.create<mlir::arith::MinUIOp>(
          operation.getLoc(), remaining, lanes);
      auto lhs = rewriter.create<riscv::RVVStreamLoadOp>(
          operation.getLoc(), streamType, operation.getLhs(),
          loop.getInductionVar(), active, valid, mlir::ValueRange{}, logicalAxis,
          false, operation.getLhsAccess(),
          riscv_internal::leaf(rewriter, "transfer", "stream-load",
                               "rvv.stream-load", "rvv.stream-load", 0,
                               layout.getRegisterGroups(), 0, 0, "none",
                               "exact"));
      auto rhs = rewriter.create<riscv::RVVStreamLoadOp>(
          operation.getLoc(), streamType, operation.getRhs(),
          loop.getInductionVar(), active, valid, mlir::ValueRange{}, logicalAxis,
          false, operation.getRhsAccess(),
          riscv_internal::leaf(rewriter, "transfer", "stream-load",
                               "rvv.stream-load", "rvv.stream-load", 0,
                               layout.getRegisterGroups(), 0, 0, "none",
                               "exact"));
      auto step = rewriter.create<riscv::RVVStreamDotStepOp>(
          operation.getLoc(), streamType, lhs.getResult(), rhs.getResult(),
          loop.getRegionIterArg(0), active,
          riscv_internal::leaf(
              rewriter, "rvv", "stream-dot-step", "rvv.stream-dot-step",
              "rvv.stream-dot-step", 3 * layout.getRegisterGroups(),
              layout.getRegisterGroups()));
      copyProvenance(operation, lhs);
      copyProvenance(operation, rhs);
      copyProvenance(operation, step);
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<mlir::scf::YieldOp>(operation.getLoc(), step.getResult());
      rewriter.setInsertionPointAfter(loop);
      auto finalize = rewriter.create<riscv::RVVStreamFinalizeOp>(
          operation.getLoc(), operation.getResult().getType(), loop.getResult(0),
          logicalAxis, "add",
          riscv_internal::leaf(
              rewriter, "rvv", "stream-finalize", "rvv.stream-finalize",
              "rvv.stream-finalize", layout.getRegisterGroups(), 0, 1));
      copyProvenance(operation, finalize);
      operation.getResult().replaceAllUsesWith(finalize.getResult());
      rewriter.eraseOp(operation);
    }

    llvm::SmallVector<riscv::RVVStreamContractOp> streamContracts;
    getOperation().walk([&](riscv::RVVStreamContractOp operation) {
      streamContracts.push_back(operation);
    });
    for (riscv::RVVStreamContractOp operation : streamContracts) {
      const bool widening =
          operation.getLeaf().getInstruction() == "rvv.stream-widen-contract";
      mlir::Type operandElement = mlir::Float32Type::get(rewriter.getContext());
      if (widening)
        operandElement = mlir::Float16Type::get(rewriter.getContext());
      riscv::ValueType accumulatorType = valueForLayout(
          rewriter, mlir::Float32Type::get(rewriter.getContext()),
          operation.getAccumulatorLayout());
      riscv::ValueType lhsType = contractOperandType(
          rewriter, operation.getLhs().getType(), operandElement,
          operation.getOperandLayout(), operation.getAccumulatorLayout(),
          operation.getReductionAxis());
      riscv::ValueType rhsType = contractOperandType(
          rewriter, operation.getRhs().getType(), operandElement,
          operation.getOperandLayout(), operation.getAccumulatorLayout(),
          operation.getReductionAxis());
      llvm::SmallVector<mlir::Value> lhsPoints = freeAxisPoints(
          operation.getLhs(), lhsType, operation.getReductionAxis());
      llvm::SmallVector<mlir::Value> rhsPoints = freeAxisPoints(
          operation.getRhs(), rhsType, operation.getReductionAxis());
      auto requiredPoints = [&](riscv::ValueType type) {
        if (!type)
          return int64_t{-1};
        int64_t count = 0;
        for (auto [position, axis] :
             llvm::enumerate(type.getAxisIds().asArrayRef()))
          if (axis != operation.getReductionAxis() &&
              type.getLayout().getReplicaFactors()[position] > 1)
            ++count;
        return count;
      };
      if (!accumulatorType || !lhsType || !rhsType ||
          static_cast<int64_t>(lhsPoints.size()) != requiredPoints(lhsType) ||
          static_cast<int64_t>(rhsPoints.size()) != requiredPoints(rhsType)) {
        operation.emitError(
            "stream contract has no explicit accumulator, operand, or free-axis point representation")
            << "; accumulator=" << accumulatorType << "; lhs=" << lhsType
            << "; rhs=" << rhsType << "; lhs_points=" << lhsPoints.size()
            << "/" << requiredPoints(lhsType) << "; rhs_points="
            << rhsPoints.size() << "/" << requiredPoints(rhsType);
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(operation);
      auto zeroValue = rewriter.create<riscv::ConstantOp>(
          operation.getLoc(), mlir::Float32Type::get(rewriter.getContext()),
          rewriter.getF32FloatAttr(0.0));
      copyProvenance(operation, zeroValue);

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), 0);
      mlir::Value lanes = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), operation.getOperandLayout().getVl());
      mlir::Value stepSize = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation.getLoc(), operation.getUnroll() *
                                  operation.getOperandLayout().getVl());
      mlir::Value trueValue = rewriter.create<mlir::arith::ConstantIntOp>(
          operation.getLoc(), 1, 1);
      mlir::Value fullTile = trueValue;
      llvm::DenseSet<int64_t> checkedAxes;
      llvm::SmallVector<mlir::Value> allPoints(lhsPoints.begin(),
                                               lhsPoints.end());
      llvm::append_range(allPoints, rhsPoints);
      for (mlir::Value pointValue : allPoints) {
        auto point = pointValue.getDefiningOp<riscv::PhysicalPointOp>();
        const int64_t axis =
            mlir::cast<riscv::PointType>(pointValue.getType())
                .getDomain()
                .getAxisId();
        if (!point || !checkedAxes.insert(axis).second)
          continue;
        auto found = llvm::find(
            operation.getAccumulatorLayout().getAxisIds().asArrayRef(), axis);
        if (found ==
            operation.getAccumulatorLayout().getAxisIds().asArrayRef().end()) {
          operation.emitError(
              "stream contract free-axis point is absent from its accumulator layout");
          failed = true;
          break;
        }
        const size_t position = static_cast<size_t>(
            found - operation.getAccumulatorLayout().getAxisIds().asArrayRef().begin());
        mlir::Value expected = rewriter.create<mlir::arith::ConstantIndexOp>(
            operation.getLoc(),
            operation.getAccumulatorLayout().getReplicaFactors()[position]);
        mlir::Value exact = rewriter.create<mlir::arith::CmpIOp>(
            operation.getLoc(), mlir::arith::CmpIPredicate::eq,
            point.getActive(), expected);
        fullTile = rewriter.create<mlir::arith::AndIOp>(
            operation.getLoc(), fullTile, exact);
      }
      if (failed)
        continue;
      mlir::Value remainder = rewriter.create<mlir::arith::RemUIOp>(
          operation.getLoc(), operation.getExtent(), stepSize);
      mlir::Value exactExtent = rewriter.create<mlir::arith::CmpIOp>(
          operation.getLoc(), mlir::arith::CmpIPredicate::eq, remainder, zero);
      fullTile = rewriter.create<mlir::arith::AndIOp>(
          operation.getLoc(), fullTile, exactExtent);

      auto branch = rewriter.create<mlir::scf::IfOp>(
          operation.getLoc(), mlir::TypeRange{accumulatorType}, fullTile, true);
      copyProvenance(operation, branch);
      auto buildLoop = [&](mlir::Region &region,
                           bool guarded) -> mlir::Value {
        rewriter.setInsertionPointToStart(&region.front());
        auto accumulator = rewriter.create<riscv::RVVSplatOp>(
            operation.getLoc(), accumulatorType, zeroValue,
            riscv_internal::leaf(
                rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                accumulatorType.getLayout().getRegisterGroups()));
        copyProvenance(operation, accumulator);
        auto loop = rewriter.create<mlir::scf::ForOp>(
            operation.getLoc(), zero, operation.getExtent(), stepSize,
            mlir::ValueRange{accumulator.getResult()});
        loop->setAttr("weft.riscv.direction",
                      rewriter.getStringAttr("ascending"));
        loop->setAttr("weft.riscv.system_unroll",
                      rewriter.getStringAttr("disable"));
        copyProvenance(operation, loop);
        rewriter.setInsertionPointToStart(loop.getBody());
        mlir::Value carried = loop.getRegionIterArg(0);
        for (int64_t strip = 0; strip < operation.getUnroll(); ++strip) {
          mlir::Value offset = loop.getInductionVar();
          if (strip) {
            mlir::Value delta = rewriter.create<mlir::arith::ConstantIndexOp>(
                operation.getLoc(),
                strip * operation.getOperandLayout().getVl());
            offset = rewriter.create<mlir::arith::AddIOp>(
                operation.getLoc(), offset, delta);
          }
          mlir::Value valid = trueValue;
          mlir::Value active = lanes;
          if (guarded) {
            valid = rewriter.create<mlir::arith::CmpIOp>(
                operation.getLoc(), mlir::arith::CmpIPredicate::ult, offset,
                operation.getExtent());
            auto activeChoice = rewriter.create<mlir::scf::IfOp>(
                operation.getLoc(), mlir::TypeRange{rewriter.getIndexType()},
                valid, true);
            rewriter.setInsertionPointToStart(
                &activeChoice.getThenRegion().front());
            mlir::Value remaining = rewriter.create<mlir::arith::SubIOp>(
                operation.getLoc(), operation.getExtent(), offset);
            mlir::Value tail = rewriter.create<mlir::arith::MinUIOp>(
                operation.getLoc(), remaining, lanes);
            rewriter.create<mlir::scf::YieldOp>(operation.getLoc(), tail);
            rewriter.setInsertionPointToStart(
                &activeChoice.getElseRegion().front());
            rewriter.create<mlir::scf::YieldOp>(operation.getLoc(), zero);
            rewriter.setInsertionPointAfter(activeChoice);
            active = activeChoice.getResult(0);
          }
          auto lhs = rewriter.create<riscv::RVVStreamLoadOp>(
              operation.getLoc(), lhsType, operation.getLhs(), offset, active,
              valid, lhsPoints, operation.getReductionAxis(), guarded,
              operation.getLhsAccess(),
              riscv_internal::leaf(
                  rewriter, "transfer", "stream-load", "rvv.stream-load",
                  "rvv.stream-load", 0,
                  lhsType.getLayout().getRegisterGroups(), 0, 0, "none",
                  guarded ? "agnostic" : "exact"));
          auto rhs = rewriter.create<riscv::RVVStreamLoadOp>(
              operation.getLoc(), rhsType, operation.getRhs(), offset, active,
              valid, rhsPoints, operation.getReductionAxis(), guarded,
              operation.getRhsAccess(),
              riscv_internal::leaf(
                  rewriter, "transfer", "stream-load", "rvv.stream-load",
                  "rvv.stream-load", 0,
                  rhsType.getLayout().getRegisterGroups(), 0, 0, "none",
                  guarded ? "agnostic" : "exact"));
          llvm::StringRef instruction =
              widening ? "rvv.stream-widen-contract-step"
                       : "rvv.stream-contract-step";
          auto step = rewriter.create<riscv::RVVStreamContractStepOp>(
              operation.getLoc(), accumulatorType, lhs.getResult(),
              rhs.getResult(), carried, active, operation.getReductionAxis(),
              riscv_internal::leaf(
                  rewriter, "rvv", "stream-contract-step", instruction,
                  instruction,
                  lhsType.getLayout().getRegisterGroups() +
                      rhsType.getLayout().getRegisterGroups() +
                      accumulatorType.getLayout().getRegisterGroups(),
                  accumulatorType.getLayout().getRegisterGroups()));
          copyProvenance(operation, lhs);
          copyProvenance(operation, rhs);
          copyProvenance(operation, step);
          carried = step.getResult();
        }
        rewriter.setInsertionPointToEnd(loop.getBody());
        rewriter.create<mlir::scf::YieldOp>(operation.getLoc(), carried);
        rewriter.setInsertionPointAfter(loop);
        rewriter.create<mlir::scf::YieldOp>(operation.getLoc(),
                                            loop.getResult(0));
        return loop.getResult(0);
      };
      buildLoop(branch.getThenRegion(), false);
      buildLoop(branch.getElseRegion(), true);

      rewriter.setInsertionPointAfter(branch);
      auto finalize = rewriter.create<riscv::RVVStreamFinalizeOp>(
          operation.getLoc(), operation.getResult().getType(),
          branch.getResult(0), operation.getReductionAxis(), "add",
          riscv_internal::leaf(
              rewriter, "rvv", "stream-finalize", "rvv.stream-finalize",
              "rvv.stream-finalize",
              accumulatorType.getLayout().getRegisterGroups(), 0, 1));
      copyProvenance(operation, finalize);
      operation.getResult().replaceAllUsesWith(finalize.getResult());
      rewriter.eraseOp(operation);
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVProgramsPass() {
  return std::make_unique<MaterializeRISCVProgramsPass>();
}
