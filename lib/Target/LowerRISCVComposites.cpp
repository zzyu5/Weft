#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <initializer_list>
#include <limits>

using namespace weft;

namespace {

int64_t product(mlir::DenseI64ArrayAttr values) {
  return riscv_internal::staticProduct(values.asArrayRef()).value_or(-1);
}

int64_t product(std::initializer_list<int64_t> values) {
  return riscv_internal::staticProduct(values).value_or(-1);
}

int64_t sum(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs)
    return -1;
  return lhs + rhs;
}

int64_t resultParts(riscv::LayoutAttr layout) {
  return product(
      {product(layout.getTimeFactors()), product(layout.getReplicaFactors())});
}

riscv::AccessAttr accessOf(mlir::Value value) {
  for (mlir::Operation *operation = value.getDefiningOp(); operation;) {
    if (auto access = operation->getAttrOfType<riscv::AccessAttr>("access")) {
      if (!(access.getForm() == "local" && access.getMapping() == "dense"))
        return access;
    }
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation)) {
      value = extract.getInput();
      operation = value.getDefiningOp();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
      value = conversion.getInput();
      operation = value.getDefiningOp();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(operation)) {
      value = materialize.getInput();
      operation = value.getDefiningOp();
      continue;
    }
    break;
  }
  return {};
}

void copyIdentity(mlir::Operation *source, mlir::Operation *target) {
  if (auto origin = source->getAttr("source_origin"))
    target->setAttr("source_origin", origin);
  if (auto canonical = source->getAttr("canonical_op"))
    target->setAttr("canonical_op", canonical);
}

riscv::LayoutAttr fragmentLayout(mlir::Builder &builder, riscv::ValueType value,
                                 llvm::ArrayRef<int64_t> physicalShape) {
  auto axes = value.getAxisIds().asArrayRef();
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  llvm::SmallVector<int64_t> fragment(physicalShape.begin(), physicalShape.end());
  return riscv::LayoutAttr::get(
      builder.getContext(), "ime", value.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, one),
      std::max<int64_t>(8, riscv_internal::logicalBitWidth(value)), 0, 1, 0,
      value.getLayout().getValidity());
}

class LowerRISCVCompositesPass
    : public mlir::PassWrapper<LowerRISCVCompositesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-lower-composites";
  }
  llvm::StringRef getDescription() const override {
    return "Rewrite composites into typed RVV/IME/local physical operations";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    lowerMaterializations(rewriter, failed);
    lowerLocalStates(rewriter, failed);
    lowerLocalCommits(rewriter, failed);
    lowerLoops(rewriter, failed);
    closeOrderedLoops(failed);
    lowerGroupedReductions(rewriter, failed);
    lowerEncodedDots(rewriter, failed);
    lowerContracts(rewriter, failed);
    eraseDeadComposites(rewriter);
    closeConversions(rewriter, failed);
    if (failed)
      signalPassFailure();
  }

private:
  mlir::FailureOr<std::pair<mlir::Value, int64_t>>
  elementCount(mlir::IRRewriter &rewriter, mlir::Operation *anchor,
               riscv::ValueType value) const {
    mlir::Value count = rewriter.create<mlir::arith::ConstantIndexOp>(
        anchor->getLoc(), 1);
    int64_t staticCount = 1;
    for (auto [extent, axis] :
         llvm::zip(value.getShape().asArrayRef(),
                   value.getAxisIds().asArrayRef())) {
      mlir::Value dimension;
      if (extent > 0) {
        dimension = rewriter.create<mlir::arith::ConstantIndexOp>(
            anchor->getLoc(), extent);
        if (staticCount > 0) {
          staticCount = product({staticCount, extent});
          if (staticCount < 0)
            return mlir::failure();
        }
      } else {
        dimension = fullAxisExtent(anchor, axis);
        staticCount = -1;
      }
      if (!dimension)
        return mlir::failure();
      count = rewriter.create<mlir::arith::MulIOp>(anchor->getLoc(), count,
                                                  dimension);
    }
    return std::make_pair(count, staticCount);
  }

  void lowerLocalStates(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::NewOp> states;
    getOperation().walk([&](riscv::NewOp state) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType());
          value && value.getLayout().getCarrier() == "local")
        states.push_back(state);
    });
    for (riscv::NewOp state : states) {
      auto value = mlir::cast<riscv::ValueType>(state.getResult().getType());
      const int64_t elementBytes =
          std::max<int64_t>(1, riscv_internal::logicalBitWidth(value) / 8);
      rewriter.setInsertionPoint(state);
      auto count = elementCount(rewriter, state, value);
      if (mlir::failed(count)) {
        state.emitError("local state has no explicit runtime extent for every axis");
        failed = true;
        continue;
      }
      mlir::Value bytes = rewriter.create<mlir::arith::MulIOp>(
          state.getLoc(), count->first,
          rewriter.create<mlir::arith::ConstantIndexOp>(state.getLoc(),
                                                        elementBytes));
      int64_t staticBytes =
          count->second < 0 ? -1 : product({count->second, elementBytes});
      if (count->second >= 0 && staticBytes < 0) {
        state.emitError("local state byte size overflows the target index domain");
        failed = true;
        continue;
      }
      mlir::Value allocationBytes = bytes;
      mlir::Value logicalElements = count->first;
      if (staticBytes >= 0) {
        allocationBytes = rewriter.create<mlir::arith::ConstantIndexOp>(
            state.getLoc(), staticBytes);
        logicalElements = rewriter.create<mlir::arith::ConstantIndexOp>(
            state.getLoc(), count->second);
      }
      auto storageType = riscv::LocalType::get(
          rewriter.getContext(), value.getElementType(), value.getShape(),
          value.getAxisIds(), staticBytes, std::max<int64_t>(elementBytes, 8),
          state.getBirthId(), "handoff", state.getOwnerDomainId(),
          state.getBirthId(), state.getLifetimeEndDomainId(),
          "local-state-array");
      auto allocation = rewriter.create<riscv::LocalAllocOp>(
          state.getLoc(), storageType, allocationBytes);
      auto binding = rewriter.create<riscv::LocalBindOp>(
          state.getLoc(), value, allocation.getResult(), logicalElements);
      copyIdentity(state, allocation);
      copyIdentity(state, binding);
      if (state.getInitialized()) {
        mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
            state.getLoc(), 0);
        mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
            state.getLoc(), 1);
        auto loop = rewriter.create<mlir::scf::ForOp>(
            state.getLoc(), lower, count->first, step);
        loop->setAttr("weft.riscv.direction",
                      rewriter.getStringAttr("ascending"));
        rewriter.setInsertionPointToStart(loop.getBody());
        auto store = rewriter.create<riscv::LocalStoreOp>(
            state.getLoc(), state.getInitial(), binding.getResult(),
            loop.getInductionVar(),
            riscv_internal::leaf(rewriter, "transfer", "local-store",
                                 "local.store.element",
                                 "local.store.element", 0, 0, 0, 0));
        copyIdentity(state, store);
      }
      state.getResult().replaceAllUsesWith(binding.getResult());
      rewriter.eraseOp(state);
    }
  }

  void lowerLocalCommits(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::StoreOp> stores;
    getOperation().walk([&](riscv::StoreOp store) {
      auto value = mlir::dyn_cast<riscv::ValueType>(store.getValue().getType());
      if (value && value.getLayout().getCarrier() == "local")
        stores.push_back(store);
    });
    for (riscv::StoreOp store : stores) {
      auto value = mlir::cast<riscv::ValueType>(store.getValue().getType());
      auto region = store.getRegion().getType();
      auto encoding =
          mlir::dyn_cast<kernel::EncodingType>(region.getEncoding());
      if (value.getShape().size() != 1 || region.getShape().size() != 1 ||
          value.getShape() != region.getShape() ||
          value.getAxisIds() != region.getAxisIds() || !encoding ||
          encoding.getKind() != "dense" || region.getElements() != 1 ||
          region.getInterleaveRows() != 0) {
        store.emitError(
            "local commit currently requires one matching one-dimensional dense destination");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(store);
      auto count = elementCount(rewriter, store, value);
      if (mlir::failed(count)) {
        store.emitError("local commit has no explicit runtime element extent");
        failed = true;
        continue;
      }
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          store.getLoc(), 0);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          store.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          store.getLoc(), lower, count->first, step);
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      rewriter.setInsertionPointToStart(loop.getBody());
      auto scalarLoad = rewriter.create<riscv::LocalLoadOp>(
          store.getLoc(), value.getElementType(), store.getValue(),
          loop.getInductionVar(),
          riscv_internal::leaf(rewriter, "transfer", "local-load",
                               "local.load.element", "local.load.element", 0,
                               0, 0, 0));
      auto scalarDescriptor = riscv::MemDescType::get(
          rewriter.getContext(), region.getEncoding(),
          riscv_internal::integers(rewriter, {}),
          riscv_internal::integers(rewriter, {}),
          riscv_internal::integers(rewriter, {}),
          riscv_internal::integers(rewriter, {}), region.getAlignment(),
          "slice", region.getAccess(), region.getAliasSet(),
          region.getLayoutIdentity(), region.getStorageBits(),
          region.getElements(), region.getInterleaveRows());
      auto elementSlice = rewriter.create<riscv::SliceOp>(
          store.getLoc(), scalarDescriptor, store.getRegion(),
          mlir::ValueRange{loop.getInductionVar()},
          rewriter.getArrayAttr({rewriter.getStringAttr("index")}));
      auto scalarAccess = riscv::AccessAttr::get(
          rewriter.getContext(), "unit", "dense", region.getAlignment(), 0,
          0, 0, 0, 0, 0, 0, 0, 0, "none");
      auto scalarStore = rewriter.create<riscv::StoreOp>(
          store.getLoc(), scalarLoad.getResult(), elementSlice.getResult(),
          scalarAccess,
          riscv_internal::leaf(rewriter, "transfer", "store",
                               "scalar.store", "scalar.store", 0, 0, 0, 0));
      copyIdentity(store, scalarLoad);
      copyIdentity(store, elementSlice);
      copyIdentity(store, scalarStore);
      rewriter.eraseOp(store);
    }
  }

  void eraseDeadComposites(mlir::IRRewriter &rewriter) {
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<mlir::Operation *> dead;
      getOperation().walk<mlir::WalkOrder::PostOrder>(
          [&](mlir::Operation *operation) {
            if (mlir::isa<riscv::Fold2Op, riscv::MacGroupsOp,
                          riscv::DotOp, riscv::ContractOp,
                          riscv::OuterContractOp,
                          riscv::ConvertLayoutOp, riscv::LoadOp>(operation) &&
                llvm::all_of(operation->getResults(),
                             [](mlir::Value value) { return value.use_empty(); }))
              dead.push_back(operation);
          });
      for (mlir::Operation *operation : dead) {
        rewriter.eraseOp(operation);
        changed = true;
      }
    }
  }

  mlir::Value findOperandPoint(mlir::Value value, int64_t axis) const {
    llvm::SmallVector<mlir::Value> worklist{value};
    llvm::SmallPtrSet<mlir::Operation *, 8> visited;
    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (auto point = mlir::dyn_cast<riscv::PointType>(current.getType());
          point && point.getDomain().getAxisId() == axis)
        return current;
      mlir::Operation *definition = current.getDefiningOp();
      if (!definition || !visited.insert(definition).second)
        continue;
      for (mlir::Value operand : definition->getOperands())
        worklist.push_back(operand);
    }
    return {};
  }

  mlir::Value fullAxisExtent(mlir::Operation *operation, int64_t axis) const {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    if (!kernel || axis <= 0 ||
        axis > static_cast<int64_t>(kernel.getShapeSymbols().size()))
      return {};
    llvm::StringRef name =
        mlir::cast<mlir::StringAttr>(kernel.getShapeSymbols()[axis - 1])
            .getValue();
    mlir::Value extent;
    kernel.getBody().walk([&](riscv::SymbolOp symbol) {
      if (!extent && symbol.getName() == name)
        extent = symbol.getResult();
    });
    return extent;
  }

  riscv::ValueType projectedReductionType(mlir::Builder &builder,
                                          riscv::ValueType input,
                                          int64_t reductionAxis) const {
    llvm::SmallVector<int64_t> shape;
    llvm::SmallVector<int64_t> axes;
    llvm::SmallVector<int64_t> time;
    llvm::SmallVector<int64_t> lane;
    llvm::SmallVector<int64_t> replica;
    llvm::SmallVector<int64_t> fragment;
    llvm::SmallVector<int64_t> local;
    riscv::LayoutAttr source = input.getLayout();
    for (size_t index = 0; index < input.getAxisIds().size(); ++index) {
      if (input.getAxisIds()[index] == reductionAxis)
        continue;
      shape.push_back(input.getShape()[index]);
      axes.push_back(input.getAxisIds()[index]);
      time.push_back(source.getTimeFactors()[index]);
      lane.push_back(source.getLaneFactors()[index]);
      replica.push_back(source.getReplicaFactors()[index]);
      fragment.push_back(source.getFragmentFactors()[index]);
      local.push_back(source.getLocalFactors()[index]);
    }
    int64_t laneCount = 1;
    int64_t replicas = 1;
    for (int64_t factor : lane)
      laneCount *= factor;
    for (int64_t factor : replica)
      replicas *= factor;
    llvm::StringRef carrier = laneCount > 1 ? "rvv" : "scalar";
    int64_t lmul = carrier == "rvv" ? source.getLmulEighths() : 0;
    int64_t groups =
        carrier == "rvv" ? ((lmul + 7) / 8) * replicas : 0;
    auto axisAttr = riscv_internal::integers(builder, axes);
    auto layout = riscv::LayoutAttr::get(
        builder.getContext(), carrier, axisAttr,
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), source.getSew(), lmul,
        carrier == "rvv" ? laneCount : 1, groups, source.getValidity());
    return riscv::ValueType::get(builder.getContext(), input.getElementType(),
                                 riscv_internal::integers(builder, shape),
                                 axisAttr, layout);
  }

  riscv::LoadOp sourceLoad(mlir::Value value) const {
    while (mlir::Operation *definition = value.getDefiningOp()) {
      if (auto load = mlir::dyn_cast<riscv::LoadOp>(definition))
        return load;
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
        value = conversion.getInput();
        continue;
      }
      if (auto materialize =
              mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
        value = materialize.getInput();
        continue;
      }
      break;
    }
    return {};
  }

  mlir::FailureOr<mlir::Value>
  projectLoadedOperand(mlir::IRRewriter &rewriter, mlir::Value value,
                       riscv::ValueType projectedType, mlir::Value index,
                       int64_t reductionAxis, mlir::Operation *origin) const {
    riscv::LoadOp load = sourceLoad(value);
    if (!load)
      return mlir::Value();
    auto memory = load.getRegion().getType();
    auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
    if (encoding.getKind() != "dense") {
      origin->emitError(
          "a projected load inside a regular contraction requires dense storage");
      return mlir::failure();
    }
    llvm::SmallVector<int64_t> strides;
    llvm::SmallVector<int64_t> origins;
    llvm::SmallVector<int64_t> expectedAxes;
    llvm::SmallVector<mlir::Attribute> selectors;
    bool foundReduction = false;
    for (size_t position = 0; position < memory.getAxisIds().size(); ++position) {
      int64_t axis = memory.getAxisIds()[position];
      if (axis == reductionAxis) {
        if (foundReduction) {
          origin->emitError("projected load reduction axis is not unique");
          return mlir::failure();
        }
        foundReduction = true;
        selectors.push_back(rewriter.getStringAttr("index"));
        continue;
      }
      selectors.push_back(rewriter.getStringAttr("all"));
      expectedAxes.push_back(axis);
      strides.push_back(memory.getStrides()[position]);
      origins.push_back(memory.getOrigins()[position]);
    }
    if (!foundReduction ||
        projectedType.getAxisIds().asArrayRef() !=
            llvm::ArrayRef<int64_t>(expectedAxes)) {
      origin->emitError(
          "projected load result does not preserve every free memory axis");
      return mlir::failure();
    }
    auto descriptor = riscv::MemDescType::get(
        rewriter.getContext(), memory.getEncoding(), projectedType.getShape(),
        projectedType.getAxisIds(), riscv_internal::integers(rewriter, strides),
        riscv_internal::integers(rewriter, origins), memory.getAlignment(),
        "slice", memory.getAccess(), memory.getAliasSet(),
        memory.getLayoutIdentity(), memory.getStorageBits(), memory.getElements(),
        memory.getInterleaveRows());
    auto slice = rewriter.create<riscv::SliceOp>(
        origin->getLoc(), descriptor, load.getRegion(), mlir::ValueRange{index},
        rewriter.getArrayAttr(selectors));
    auto access =
        riscv_internal::denseAccess(rewriter, descriptor, projectedType);
    if (access.getForm() == "indexed" &&
        !origin->getParentOfType<riscv::KernelOp>()
             .getTarget()
             .getHasIndexedMemory()) {
      origin->emitError(
          "projected contraction load requires indexed memory unsupported by target");
      return mlir::failure();
    }
    auto projected = rewriter.create<riscv::LoadOp>(
        origin->getLoc(), projectedType, slice.getResult(), access,
        riscv_internal::leaf(rewriter, "transfer", "load",
                             ("rvv.load." + access.getForm()).str(),
                             ("rvv.load." + access.getForm()).str(), 0, 0));
    copyIdentity(origin, slice);
    copyIdentity(origin, projected);
    return projected.getResult();
  }

  void lowerLoops(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::LoopOp> loops;
    getOperation().walk<mlir::WalkOrder::PreOrder>(
        [&](riscv::LoopOp loop) { loops.push_back(loop); });
    for (riscv::LoopOp loop : loops) {
      auto domain = loop.getDomain().getDefiningOp<riscv::DomainOp>();
      if (!domain || loop.getBody().empty()) {
        loop.emitError("physical Level has no explicit domain definition/body");
        failed = true;
        continue;
      }
      auto domainType = loop.getDomain().getType();
      mlir::Value parentPoint = loop.getParentPoint();
      mlir::Value origin;
      mlir::Value total = domain.getExtent();
      rewriter.setInsertionPoint(loop);
      origin = rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);
      auto parentType =
          mlir::cast<riscv::PointType>(parentPoint.getType()).getDomain();
      if (parentType.getRelation() != "root" &&
          parentType.getAxisId() == domainType.getAxisId()) {
        auto point = parentPoint.getDefiningOp<riscv::PhysicalPointOp>();
        if (!point) {
          loop.emitError(
              "same-axis physical Level lost its parent point entity");
          failed = true;
          continue;
        }
        origin = point.getBase();
        total = point.getActive();
      }
      mlir::Value lower =
          rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);
      auto physical = rewriter.create<mlir::scf::ForOp>(
          loop.getLoc(), lower, total, domain.getPartition(), loop.getCarried());
      if (auto source = loop->getAttr("source_origin"))
        physical->setAttr("source_origin", source);
      physical->setAttr(
          "weft.riscv.level",
          riscv::LevelAttr::get(
              rewriter.getContext(), domainType.getDomainId(),
              domainType.getAxisId(), domainType.getRelation(),
              loop.getStateBirthCount(), loop.getStagedBirthCount(),
              loop.getHandoffCount(), "ascending"));

      mlir::Block &sourceBody = loop.getBody().front();
      mlir::Block *targetBody = physical.getBody();
      rewriter.setInsertionPointToStart(targetBody);
      mlir::Value base = rewriter.create<mlir::arith::AddIOp>(
          loop.getLoc(), origin, physical.getInductionVar());
      mlir::Value remaining = rewriter.create<mlir::arith::SubIOp>(
          loop.getLoc(), total, physical.getInductionVar());
      mlir::Value active = rewriter.create<mlir::arith::MinUIOp>(
          loop.getLoc(), remaining, domain.getPartition());
      auto point = rewriter.create<riscv::PhysicalPointOp>(
          loop.getLoc(), sourceBody.getArgument(0).getType(), parentPoint, base,
          active, domain.getPartition());
      sourceBody.getArgument(0).replaceAllUsesWith(point.getResult());
      if (sourceBody.getNumArguments() !=
          physical.getRegionIterArgs().size() + 1) {
        loop.emitError("physical Level birth/handoff arity was not preserved");
        failed = true;
        continue;
      }
      for (auto [source, target] : llvm::zip(
               sourceBody.getArguments().drop_front(),
               physical.getRegionIterArgs()))
        source.replaceAllUsesWith(target);
      auto sourceYield = mlir::cast<riscv::YieldOp>(sourceBody.getTerminator());
      llvm::SmallVector<mlir::Value> yielded(sourceYield.getValues());
      if (!targetBody->empty())
        if (auto existing =
                mlir::dyn_cast<mlir::scf::YieldOp>(targetBody->back()))
          rewriter.eraseOp(existing);
      for (mlir::Operation &nested : llvm::make_early_inc_range(sourceBody)) {
        if (&nested == sourceYield.getOperation())
          continue;
        nested.moveBefore(targetBody, targetBody->end());
      }
      rewriter.setInsertionPointToEnd(targetBody);
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), yielded);
      rewriter.eraseOp(sourceYield);
      loop.getResults().replaceAllUsesWith(physical.getResults());
      rewriter.eraseOp(loop);
    }
  }

  void closeOrderedLoops(bool &failed) {
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.direction") ||
          loop->hasAttr("weft.riscv.level"))
        return;
      auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!step || step.value() == 0) {
        loop.emitError(
            "ordered scalar loop requires one non-zero compile-time direction");
        failed = true;
        return;
      }
      loop->setAttr("weft.riscv.direction",
                    mlir::StringAttr::get(loop.getContext(),
                                          step.value() > 0 ? "ascending"
                                                           : "descending"));
    });
  }

  void lowerMaterializations(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::MaterializeOp> operations;
    getOperation().walk(
        [&](riscv::MaterializeOp operation) { operations.push_back(operation); });
    for (riscv::MaterializeOp operation : operations) {
      if (operation.getPlacement() == "local") {
        operation.emitError(
            "local materialize reached composite lowering without an explicit local object");
        failed = true;
        continue;
      }
      if (operation.getPlacement() == "reload") {
        if (auto load = operation.getInput().getDefiningOp<riscv::LoadOp>()) {
          rewriter.setInsertionPoint(operation);
          auto staged = rewriter.create<riscv::StagedViewOp>(
              operation.getLoc(), load.getRegion().getType(), load.getRegion(),
              operation.getSchema(),
              operation.getOwnerDomainId(), operation.getBirthId(),
              operation.getLifetimeEndDomainId());
          copyIdentity(operation, staged);
          llvm::SmallVector<riscv::ExtractOp> extracts;
          for (mlir::Operation *user : operation.getResult().getUsers())
            if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
                extract && extract.getInput() == operation.getResult())
              extracts.push_back(extract);
          for (riscv::ExtractOp extract : extracts) {
            auto result = mlir::dyn_cast<riscv::ValueType>(
                extract.getResult().getType());
            auto source = load.getRegion().getType();
            if (!result) {
              extract.emitError(
                  "reload projection requires one shaped physical result");
              failed = true;
              continue;
            }
            llvm::SmallVector<int64_t> strides;
            llvm::SmallVector<int64_t> origins;
            for (int64_t axis : result.getAxisIds().asArrayRef()) {
              auto found = llvm::find(source.getAxisIds().asArrayRef(), axis);
              if (found == source.getAxisIds().asArrayRef().end()) {
                extract.emitError(
                    "reload projection introduced an axis absent from its memory descriptor");
                failed = true;
                break;
              }
              size_t index = static_cast<size_t>(
                  found - source.getAxisIds().asArrayRef().begin());
              strides.push_back(source.getStrides()[index]);
              origins.push_back(source.getOrigins()[index]);
            }
            if (failed)
              continue;
            auto descriptor = riscv::MemDescType::get(
                rewriter.getContext(), source.getEncoding(), result.getShape(),
                result.getAxisIds(), riscv_internal::integers(rewriter, strides),
                riscv_internal::integers(rewriter, origins),
                source.getAlignment(), "slice", source.getAccess(),
                source.getAliasSet(), source.getLayoutIdentity(),
                source.getStorageBits(), source.getElements(),
                source.getInterleaveRows());
            rewriter.setInsertionPoint(extract);
            auto slice = rewriter.create<riscv::SliceOp>(
                extract.getLoc(), descriptor, staged.getResult(),
                extract.getIndices(), extract.getSelectors());
            copyIdentity(extract, slice);
            llvm::StringRef form = extract.getAccess().getForm();
            auto leaf = riscv_internal::leaf(
                rewriter, "transfer", "load", ("rvv.load." + form).str(),
                ("rvv.load." + form).str(), 0, 0);
            auto projected = rewriter.create<riscv::LoadOp>(
                extract.getLoc(), result, slice.getResult(),
                extract.getAccess(), leaf);
            copyIdentity(extract, projected);
            extract.getResult().replaceAllUsesWith(projected.getResult());
            rewriter.eraseOp(extract);
          }
          if (failed)
            continue;
          llvm::SmallVector<mlir::OpOperand *> uses;
          for (mlir::OpOperand &use : operation.getResult().getUses())
            uses.push_back(&use);
          for (mlir::OpOperand *use : uses) {
            rewriter.setInsertionPoint(use->getOwner());
            auto cloned = rewriter.create<riscv::LoadOp>(
                operation.getLoc(), operation.getResult().getType(),
                staged.getResult(), load.getAccess(), load.getLeaf());
            copyIdentity(load, cloned);
            use->set(cloned.getResult());
          }
          rewriter.eraseOp(operation);
          if (load.getResult().use_empty())
            rewriter.eraseOp(load);
          continue;
        }
        // A non-memory pure producer cannot be represented as a lazy view.
        // Keep one explicit SSA value and let resource materialization decide
        // whether that computed value must later spill.
      }
      rewriter.setInsertionPoint(operation);
      mlir::Value input = operation.getInput();
      if (input.getType() != operation.getResult().getType()) {
        auto source = mlir::dyn_cast<riscv::ValueType>(input.getType());
        auto target =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
        if (!source || !target) {
          operation.emitError(
              "register materialization cannot convert a non-shaped representation");
          failed = true;
          continue;
        }
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            operation.getLoc(), target, input,
            riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                             target.getLayout()),
            riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, conversion);
        input = conversion.getResult();
      }
      auto materialized = rewriter.create<riscv::RegisterMaterializeOp>(
          operation.getLoc(), operation.getResult().getType(),
          input, operation.getOwnerDomainId(),
          operation.getBirthId(), operation.getLifetimeEndDomainId(), "share");
      copyIdentity(operation, materialized);
      operation.getResult().replaceAllUsesWith(materialized.getResult());
      rewriter.eraseOp(operation);
    }
  }

  void lowerGroupedReductions(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp operation) { reductions.push_back(operation); });
    for (riscv::ReduceOp reduce : reductions) {
      auto widen = reduce.getInput().getDefiningOp<riscv::WidenOp>();
      riscv::ConvertLayoutOp bridge;
      mlir::Value macValue = widen ? widen.getInput() : mlir::Value();
      if (macValue)
        bridge = macValue.getDefiningOp<riscv::ConvertLayoutOp>();
      if (bridge)
        macValue = bridge.getInput();
      auto mac = macValue ? macValue.getDefiningOp<riscv::MacGroupsOp>()
                          : riscv::MacGroupsOp();
      if (!widen || !mac)
        continue;
      auto implementation = mac->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if (!implementation || implementation.getEngine() != "rvv" ||
          implementation.getFamily() != "grouped-mac" ||
          implementation.getOperation() != "rvv.vwmaccsu.typed" ||
          implementation.getParameters().asArrayRef() !=
              llvm::ArrayRef<int64_t>({static_cast<int64_t>(mac.getGroup())})) {
        mac.emitError(
            "grouped MAC reduction has no selected mixed-sign widening implementation");
        failed = true;
        continue;
      }
      mlir::Value lhs = mac.getLhs();
      mlir::Value rhs = mac.getRhs();
      while (auto conversion = lhs.getDefiningOp<riscv::ConvertLayoutOp>())
        lhs = conversion.getInput();
      while (auto conversion = rhs.getDefiningOp<riscv::ConvertLayoutOp>())
        rhs = conversion.getInput();
      riscv::AccessAttr lhsAccess = accessOf(lhs);
      riscv::AccessAttr rhsAccess = accessOf(rhs);
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      if (!lhsAccess || !rhsAccess || !resultType) {
        reduce.emitError(
            "grouped MAC reduction has no typed operand access/result contract");
        failed = true;
        continue;
      }
      int64_t reductionAxis =
          mlir::cast<riscv::ValueType>(widen.getResult().getType())
              .getAxisIds()[reduce.getAxis()];
      riscv::ScheduleAttr schedule = mac.getSchedule();
      auto lhsLayout =
          mlir::cast<riscv::ValueType>(mac.getLhs().getType()).getLayout();
      auto partialLayout = riscv::LayoutAttr::get(
          rewriter.getContext(), lhsLayout.getCarrier(), lhsLayout.getAxisIds(),
          lhsLayout.getTimeFactors(), lhsLayout.getLaneFactors(),
          lhsLayout.getReplicaFactors(), lhsLayout.getFragmentFactors(),
          lhsLayout.getLocalFactors(), 16,
          std::min<int64_t>(64, lhsLayout.getLmulEighths() * 2),
          lhsLayout.getVl(),
          std::max<int64_t>(1, lhsLayout.getRegisterGroups() * 2),
          lhsLayout.getValidity());
      mlir::Value point = findOperandPoint(lhs, reductionAxis);
      if (!point)
        point = findOperandPoint(rhs, reductionAxis);
      auto physicalPoint =
          point ? point.getDefiningOp<riscv::PhysicalPointOp>()
                : riscv::PhysicalPointOp();
      mlir::Value active = physicalPoint ? physicalPoint.getActive()
                                         : fullAxisExtent(reduce, reductionAxis);
      if (!active) {
        reduce.emitError(
            "grouped MAC reduction has no explicit physical reduction extent");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(reduce);
      mlir::Type element = resultType.getElementType();
      mlir::TypedAttr zero = mlir::isa<mlir::FloatType>(element)
                                 ? mlir::cast<mlir::TypedAttr>(
                                       rewriter.getFloatAttr(element, 0.0))
                                 : mlir::cast<mlir::TypedAttr>(
                                       rewriter.getIntegerAttr(element, 0));
      auto scalarZero =
          rewriter.create<riscv::ConstantOp>(reduce.getLoc(), element, zero);
      auto accumulator = rewriter.create<riscv::RVVSplatOp>(
          reduce.getLoc(), resultType, scalarZero,
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                               "rvv.splat", 0,
                               resultType.getLayout().getRegisterGroups()));
      mlir::Value groupSize = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), mac.getGroup());
      mlir::Value groupCount = rewriter.create<mlir::arith::CeilDivUIOp>(
          reduce.getLoc(), active, groupSize);
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 0);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), schedule.getUnroll());
      auto loop = rewriter.create<mlir::scf::ForOp>(
          reduce.getLoc(), lower, groupCount, step,
          mlir::ValueRange{accumulator.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.schedule", schedule);
      auto windowType = riscv::WindowType::get(
          rewriter.getContext(), "grouped-mac", lhs.getType(), rhs.getType(),
          resultType, reductionAxis, schedule.getUnroll(), mac.getGroup(),
          resultParts(resultType.getLayout()), partialLayout, resultType.getLayout(),
          product({std::max<int64_t>(1, lhsLayout.getRegisterGroups()),
                   schedule.getUnroll(), static_cast<int64_t>(mac.getGroup())}));
      rewriter.setInsertionPointToStart(loop.getBody());
      auto load = rewriter.create<riscv::RVVGroupedMacLoadOp>(
          reduce.getLoc(), windowType, lhs, rhs, loop.getInductionVar(), active,
          mac.getGroup(), schedule.getUnroll(), reductionAxis, partialLayout,
          lhsAccess, rhsAccess,
          riscv_internal::leaf(
              rewriter, "rvv", "grouped-mac-load",
              "rvv.grouped-mac-load.u8-s8",
              "rvv.grouped-mac-load.u8-s8", 0,
              windowType.getResourceGroups(),
              0, 0, "none", "agnostic",
              {static_cast<int64_t>(mac.getGroup()), schedule.getUnroll(),
               reductionAxis}));
      auto stepOp = rewriter.create<riscv::RVVGroupedMacStepOp>(
          reduce.getLoc(), resultType, load.getResult(),
          loop.getRegionIterArg(0),
          riscv_internal::leaf(
              rewriter, "rvv", "grouped-mac-step",
              "rvv.vwmaccsu.vx.grouped", "rvv.vwmaccsu.vx.grouped",
              windowType.getResourceGroups(),
              resultType.getLayout().getRegisterGroups(),
              partialLayout.getRegisterGroups(), 0, "none", "agnostic",
              {static_cast<int64_t>(mac.getGroup()), schedule.getUnroll(),
               reductionAxis}));
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<mlir::scf::YieldOp>(reduce.getLoc(), stepOp.getResult());
      copyIdentity(reduce, load);
      copyIdentity(reduce, stepOp);
      reduce.getResult().replaceAllUsesWith(loop.getResult(0));
      rewriter.eraseOp(reduce);
      if (widen.getResult().use_empty())
        rewriter.eraseOp(widen);
      if (bridge && bridge.getResult().use_empty())
        rewriter.eraseOp(bridge);
      if (mac.getResult().use_empty())
        rewriter.eraseOp(mac);
    }
  }

  void lowerEncodedDots(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::DotOp> dots;
    getOperation().walk([&](riscv::DotOp operation) {
      dots.push_back(operation);
    });
    for (riscv::DotOp dot : dots) {
      auto implementation = dot->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if (!implementation || implementation.getEngine() != "rvv" ||
          implementation.getFamily() != "encoded-contract" ||
          implementation.getOperation() != "rvv.vmacc.decoded-u8-s8")
        continue;
      mlir::Value lhsSource = dot.getLhs();
      if (auto conversion = lhsSource.getDefiningOp<riscv::ConvertLayoutOp>())
        lhsSource = conversion.getInput();
      auto lhsField = lhsSource.getDefiningOp<riscv::FieldOp>();
      mlir::Value rhsSource = dot.getRhs();
      llvm::SmallVector<riscv::ConvertLayoutOp> rhsConversions;
      while (auto conversion =
                 rhsSource.getDefiningOp<riscv::ConvertLayoutOp>()) {
        rhsConversions.push_back(conversion);
        rhsSource = conversion.getInput();
      }
      auto rhsFold = rhsSource.getDefiningOp<riscv::Fold2Op>();
      mlir::Value rhsFieldValue = rhsFold ? rhsFold.getInput() : mlir::Value();
      while (auto conversion =
                 rhsFieldValue.getDefiningOp<riscv::ConvertLayoutOp>())
        rhsFieldValue = conversion.getInput();
      auto rhsField = rhsFieldValue
                          ? rhsFieldValue.getDefiningOp<riscv::FieldOp>()
                          : riscv::FieldOp();
      if (!lhsField || !rhsFold || !rhsField)
        continue;
      auto lhsType = mlir::cast<riscv::ValueType>(dot.getLhs().getType());
      auto foldInput =
          mlir::cast<riscv::ValueType>(rhsFold.getInput().getType());
      auto foldResult =
          mlir::cast<riscv::ValueType>(rhsFold.getResult().getType());
      int64_t reductionAxis = dot.getOver().empty() ? 0 : dot.getOver()[0];
      int64_t groups = 0;
      for (auto [axis, extent] : llvm::zip(lhsType.getAxisIds().asArrayRef(),
                                           lhsType.getShape().asArrayRef()))
        if (axis == reductionAxis)
          groups = extent;
      int64_t pairs = foldResult.getShape().size() == 1 &&
                              foldResult.getShape()[0] > 0
                          ? foldInput.getShape()[0] / foldResult.getShape()[0]
                          : 0;
      auto resultType = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      if (reductionAxis <= 0 || groups <= 0 || pairs <= 0 || !resultType) {
        dot.emitError("encoded dot has no closed typed reduction contract");
        failed = true;
        continue;
      }
      mlir::Value lhs = dot.getLhs();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      auto desiredLhsLayout = riscv_internal::projectLayout(
          rewriter, mlir::cast<riscv::ValueType>(lhs.getType()),
          resultType.getLayout(), kernel.getTarget());
      if (!desiredLhsLayout) {
        dot.emitError("encoded dot has no legal decoded lhs layout");
        failed = true;
        continue;
      }
      if (lhsType.getLayout() != desiredLhsLayout) {
        rewriter.setInsertionPoint(dot);
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            dot.getLoc(),
            mlir::cast<riscv::ValueType>(
                riscv_internal::withLayout(lhs.getType(), desiredLhsLayout)),
            lhs, riscv_internal::layoutConversion(
                     rewriter, lhsType.getLayout(), desiredLhsLayout),
            riscv_internal::unselectedLeaf(rewriter));
        lhs = conversion.getResult();
      }
      auto schedule = dot.getSchedule();
      rewriter.setInsertionPoint(dot);
      mlir::Type element = resultType.getElementType();
      mlir::TypedAttr zero = mlir::isa<mlir::FloatType>(element)
                                 ? mlir::cast<mlir::TypedAttr>(
                                       rewriter.getFloatAttr(element, 0.0))
                                 : mlir::cast<mlir::TypedAttr>(
                                       rewriter.getIntegerAttr(element, 0));
      auto scalarZero =
          rewriter.create<riscv::ConstantOp>(dot.getLoc(), element, zero);
      auto accumulator = rewriter.create<riscv::RVVSplatOp>(
          dot.getLoc(), resultType, scalarZero,
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                               "rvv.splat", 0,
                               resultType.getLayout().getRegisterGroups()));
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), 0);
      mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), groups);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), schedule.getUnroll());
      auto loop = rewriter.create<mlir::scf::ForOp>(
          dot.getLoc(), lower, upper, step,
          mlir::ValueRange{accumulator.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.schedule", schedule);
      auto windowType = riscv::WindowType::get(
          rewriter.getContext(), "encoded-dot", lhs.getType(),
          rhsField.getResult().getType(), resultType, reductionAxis, schedule.getUnroll(),
          pairs, resultParts(resultType.getLayout()), resultType.getLayout(),
          resultType.getLayout(),
          product({std::max<int64_t>(
                       1, mlir::cast<riscv::ValueType>(lhs.getType())
                              .getLayout().getRegisterGroups()),
                   schedule.getUnroll(),
                   resultParts(resultType.getLayout())}));
      rewriter.setInsertionPointToStart(loop.getBody());
      auto load = rewriter.create<riscv::RVVEncodedDotLoadOp>(
          dot.getLoc(), windowType, lhs, rhsField.getResult(),
          loop.getInductionVar(), upper, schedule.getUnroll(), pairs,
          reductionAxis, lhsField.getAccess(), rhsField.getAccess(),
          riscv_internal::leaf(
              rewriter, "rvv", "encoded-dot-load",
              "rvv.encoded-dot-load.i16-pairs",
              "rvv.encoded-dot-load.i16-pairs", 0,
              windowType.getResourceGroups(),
              0, 0, "none", "agnostic",
              {reductionAxis, schedule.getUnroll(), pairs}));
      auto stepOp = rewriter.create<riscv::RVVEncodedDotStepOp>(
          dot.getLoc(), resultType, load.getResult(),
          loop.getRegionIterArg(0),
          riscv_internal::leaf(
              rewriter, "rvv", "encoded-dot-step",
              "rvv.vzext-vmacc.vx", "rvv.vzext-vmacc.vx",
              windowType.getResourceGroups(),
              resultType.getLayout().getRegisterGroups(), 1, 0, "none",
              "agnostic", {reductionAxis, schedule.getUnroll(), pairs}));
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<mlir::scf::YieldOp>(dot.getLoc(), stepOp.getResult());
      copyIdentity(dot, load);
      copyIdentity(dot, stepOp);
      dot.getResult().replaceAllUsesWith(loop.getResult(0));
      rewriter.eraseOp(dot);
      for (riscv::ConvertLayoutOp conversion : rhsConversions)
        if (conversion && conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      if (rhsFold.getResult().use_empty())
        rewriter.eraseOp(rhsFold);
    }
  }

  void lowerContracts(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<mlir::Operation *> contracts;
    getOperation().walk([&](mlir::Operation *operation) {
      if (mlir::isa<riscv::DotOp, riscv::ContractOp,
                    riscv::OuterContractOp>(operation))
        contracts.push_back(operation);
    });
    for (mlir::Operation *operation : contracts) {
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
      auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
      auto accType = operation->getAttrOfType<mlir::TypeAttr>("acc_type");
      auto laneOperand =
          operation->getAttrOfType<mlir::StringAttr>("lane_operand");
      auto laneMemory =
          operation->getAttrOfType<mlir::StringAttr>("lane_memory_form");
      auto schedule = operation->getAttrOfType<riscv::ScheduleAttr>("schedule");
      if (!implementation || !resultType || !over || !laneOperand || !laneMemory ||
          !schedule) {
        operation->emitError("contract has no complete typed physical contract");
        failed = true;
        continue;
      }
      if (accType &&
          riscv_internal::logicalElement(resultType) != accType.getValue()) {
        operation->emitError(
            "contract accumulator type disagrees with its physical result element");
        failed = true;
        continue;
      }
      if (schedule.getUnroll() <= 0 || schedule.getPipelineDepth() <= 0 ||
          schedule.getBufferCount() < schedule.getPipelineDepth()) {
        operation->emitError("contract schedule is not a legal physical binding");
        failed = true;
        continue;
      }
      // This realization has no explicit load-window/compute-window cluster.
      // Silently rewriting a requested depth would make the parameter fictive;
      // only the window realizations lowered above currently implement depth 2.
      if (schedule.getPipelineDepth() != 1) {
        operation->emitError(
            "regular register-resident contract has no implemented multi-stage local pipeline");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(operation);
      if (implementation.getEngine() == "ime") {
        auto parameters = implementation.getParameters().asArrayRef();
        riscv::FragmentCapabilityAttr capability;
        for (mlir::Attribute candidate :
             operation->getParentOfType<riscv::KernelOp>()
                 .getTarget()
                 .getFragments()) {
          auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(candidate);
          if (fragment.getInstruction() == implementation.getOperation()) {
            capability = fragment;
            break;
          }
        }
        if (parameters.size() != 3 || !capability) {
          operation->emitError("IME selection has no fragment shape/resources");
          failed = true;
          continue;
        }
        auto lhsType =
            mlir::cast<riscv::ValueType>(operation->getOperand(0).getType());
        auto rhsType =
            mlir::cast<riscv::ValueType>(operation->getOperand(1).getType());
        llvm::SmallVector<int64_t, 2> lhsShape = {
            capability.getMFactor(), capability.getKFactor()};
        llvm::SmallVector<int64_t, 2> rhsShape = {
            capability.getKFactor(), capability.getNFactor()};
        llvm::SmallVector<int64_t, 2> resultShape = {
            capability.getMFactor(), capability.getNFactor()};
        auto makeFragment = [&](riscv::ValueType value, llvm::StringRef role,
                                riscv::FragmentPackingAttr packing,
                                llvm::ArrayRef<int64_t> shape) {
          const int64_t resources =
              role == "lhs" ? capability.getLhsResourceGroups()
              : role == "rhs" ? capability.getRhsResourceGroups()
                              : capability.getAccumulatorResourceGroups();
          return riscv::FragmentType::get(
              value.getContext(), implementation.getOperation(),
              role, packing, value.getElementType(),
              riscv_internal::integers(rewriter, shape), value.getAxisIds(),
              fragmentLayout(rewriter, value, shape), resources);
        };
        riscv::FragmentType lhsFragment =
            makeFragment(lhsType, "lhs", capability.getLhsPacking(), lhsShape);
        riscv::FragmentType rhsFragment =
            makeFragment(rhsType, "rhs", capability.getRhsPacking(), rhsShape);
        riscv::FragmentType resultFragment = makeFragment(
            resultType, "accumulator", capability.getAccumulatorPacking(),
            resultShape);
        auto packLeaf = [&](llvm::StringRef role, riscv::ValueType value,
                            riscv::FragmentPackingAttr packing,
                            riscv::FragmentType fragment) {
          const int64_t elements = product(fragment.getShape());
          const int64_t logicalBytes =
              product({elements,
                       std::max<int64_t>(
                           1, (riscv_internal::logicalBitWidth(value) + 7) / 8)});
          const int64_t packedBytes =
              product({elements, packing.getStorageBits() / 8});
          return riscv_internal::leaf(
              rewriter, "ime", "fragment-pack", ("ime.pack." + role).str(),
              ("ime.pack." + role).str(), 0, 0, 1, 0, "none", "exact",
              parameters, sum(logicalBytes, packedBytes));
        };
        riscv::AccessAttr lhsAccess = accessOf(operation->getOperand(0));
        riscv::AccessAttr rhsAccess = accessOf(operation->getOperand(1));
        if (!lhsAccess || !rhsAccess) {
          operation->emitError("IME operands have no typed memory access contract");
          failed = true;
          continue;
        }
        auto lhsPack = rewriter.create<riscv::IMEPackOp>(
            operation->getLoc(), lhsFragment, operation->getOperand(0), "lhs",
            capability.getLhsPacking(), lhsAccess,
            packLeaf("lhs", lhsType, capability.getLhsPacking(), lhsFragment));
        auto rhsPack = rewriter.create<riscv::IMEPackOp>(
            operation->getLoc(), rhsFragment, operation->getOperand(1), "rhs",
            capability.getRhsPacking(), rhsAccess,
            packLeaf("rhs", rhsType, capability.getRhsPacking(), rhsFragment));
        const int64_t mmaLocalBytes =
            product({product(resultFragment.getShape()),
                     std::max<int64_t>(
                         1, (riscv_internal::logicalBitWidth(resultType) + 7) / 8)});
        auto mmaLeaf = riscv_internal::leaf(
            rewriter, "ime", "fragment-mma", implementation.getOperation(),
            implementation.getOperation(), 0, 0, 0, 0, "none", "exact",
            parameters, mmaLocalBytes);
        auto mma = rewriter.create<riscv::IMEFragmentMMAOp>(
            operation->getLoc(), resultFragment, lhsPack.getResult(),
            rhsPack.getResult(), capability.getMmaGroups(),
            capability.getMmaChunks(), capability.getClobbers(),
            capability.getVolatileAsm(), capability.getMemoryClobber(), mmaLeaf);
        auto unpackLeaf = riscv_internal::leaf(
            rewriter, "ime", "fragment-unpack", "ime.unpack.rvv",
            "ime.unpack.rvv", 0, 0, 1, 0);
        auto unpack = rewriter.create<riscv::IMEUnpackOp>(
            operation->getLoc(), resultType, mma.getResult(),
            riscv_internal::conversion(rewriter, "fragment_to_rvv", "handoff",
                                       1),
            unpackLeaf);
        copyIdentity(operation, mma);
        copyIdentity(operation, unpack);
        operation->getResult(0).replaceAllUsesWith(unpack.getResult());
        rewriter.eraseOp(operation);
        continue;
      }

      int64_t reductionAxis = over.asArrayRef().front();
      mlir::Value point = findOperandPoint(operation->getOperand(0), reductionAxis);
      if (!point)
        point = findOperandPoint(operation->getOperand(1), reductionAxis);
      auto physicalPoint =
          point ? point.getDefiningOp<riscv::PhysicalPointOp>()
                : riscv::PhysicalPointOp();
      mlir::Value reductionUpper =
          physicalPoint ? physicalPoint.getActive()
                        : fullAxisExtent(operation, reductionAxis);
      if (!reductionUpper) {
        operation->emitError("contract reduction axis has no physical extent");
        failed = true;
        continue;
      }
      mlir::Type element = resultType.getElementType();
      mlir::TypedAttr zero;
      if (auto floating = mlir::dyn_cast<mlir::FloatType>(element))
        zero = rewriter.getFloatAttr(floating, 0.0);
      else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
        zero = rewriter.getIntegerAttr(integer, 0);
      else {
        operation->emitError("contract accumulator element is not numeric");
        failed = true;
        continue;
      }
      auto scalarZero = rewriter.create<riscv::ConstantOp>(
          operation->getLoc(), element, zero);
      auto zeroLeaf = riscv_internal::leaf(
          rewriter, "rvv", "splat", "rvv.splat", "rvv.splat",
          0, resultType.getLayout().getRegisterGroups());
      auto accumulator = rewriter.create<riscv::RVVSplatOp>(
          operation->getLoc(), resultType, scalarZero, zeroLeaf);
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation->getLoc(), 0);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          operation->getLoc(), schedule.getUnroll());
      auto reductionLoop = rewriter.create<mlir::scf::ForOp>(
          operation->getLoc(), lower, reductionUpper, step,
          mlir::ValueRange{accumulator.getResult()});
      reductionLoop->setAttr("weft.riscv.direction",
                             rewriter.getStringAttr("ascending"));
      rewriter.setInsertionPointToStart(reductionLoop.getBody());
      llvm::StringRef stepInstruction;
      if (implementation.getOperation() == "rvv.vfmacc")
        stepInstruction = "rvv.vfmacc.vf";
      else if (implementation.getOperation() == "rvv.vmacc")
        stepInstruction = "rvv.vmacc.vx";
      else if (implementation.getFamily() == "encoded-contract" &&
               implementation.getOperation() ==
                   "rvv.vmacc.decoded-u8-s8")
        stepInstruction = "rvv.vmacc.decoded-u8-s8";
      else {
        operation->emitError(
            "RVV contract implementation has no exact terminal step instruction");
        failed = true;
        rewriter.eraseOp(reductionLoop);
        rewriter.eraseOp(accumulator);
        rewriter.eraseOp(scalarZero);
        continue;
      }
      auto stepLeaf = riscv_internal::leaf(
          rewriter, "rvv", "contract-step", stepInstruction,
          stepInstruction,
          0, 0, 0, 0, "none", "agnostic",
          over.asArrayRef());
      mlir::Value laneValue = laneOperand.getValue() == "rhs"
                                  ? operation->getOperand(1)
                                  : operation->getOperand(0);
      riscv::LayoutAttr resultLayout = resultType.getLayout();
      auto kernel = operation->getParentOfType<riscv::KernelOp>();
      auto laneType = mlir::dyn_cast<riscv::ValueType>(laneValue.getType());
      auto laneLoadLayout =
          laneType ? riscv_internal::projectLayout(
                         rewriter, laneType, resultLayout, kernel.getTarget())
                   : riscv::LayoutAttr();
      if (!laneLoadLayout || laneLoadLayout.getCarrier() != "rvv") {
        operation->emitError(
            "contract lane operand has no legal projected RVV representation");
        failed = true;
        rewriter.eraseOp(reductionLoop);
        rewriter.eraseOp(accumulator);
        rewriter.eraseOp(scalarZero);
        continue;
      }
      mlir::Value lhs = operation->getOperand(0);
      mlir::Value rhs = operation->getOperand(1);
      if (implementation.getFamily() != "encoded-contract" &&
          laneType.getLayout() != laneLoadLayout) {
        auto convertedType = mlir::cast<riscv::ValueType>(
            riscv_internal::withLayout(laneType, laneLoadLayout));
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            operation->getLoc(), convertedType, laneValue,
            riscv_internal::layoutConversion(rewriter, laneType.getLayout(),
                                             laneLoadLayout),
            riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, conversion);
        laneValue = conversion.getResult();
        if (laneOperand.getValue() == "rhs")
          rhs = laneValue;
        else
          lhs = laneValue;
      }
      auto currentLaneType =
          mlir::dyn_cast<riscv::ValueType>(laneValue.getType());
      auto sourceInteger = currentLaneType
                               ? mlir::dyn_cast<mlir::IntegerType>(
                                     currentLaneType.getElementType())
                               : mlir::IntegerType();
      auto resultInteger = mlir::dyn_cast<mlir::IntegerType>(element);
      if (implementation.getFamily() != "encoded-contract" && sourceInteger &&
          resultInteger &&
          std::max<unsigned>(8, sourceInteger.getWidth()) <
              resultInteger.getWidth()) {
        auto widenedSeed = riscv::ValueType::get(
            rewriter.getContext(), resultInteger, currentLaneType.getShape(),
            currentLaneType.getAxisIds(), currentLaneType.getLayout());
        auto widenedLayout = riscv_internal::projectLayout(
            rewriter, widenedSeed, resultLayout, kernel.getTarget());
        unsigned sourceWidth = std::max<unsigned>(8, sourceInteger.getWidth());
        unsigned factor = resultInteger.getWidth() / sourceWidth;
        if (!widenedLayout || widenedLayout.getCarrier() != "rvv" ||
            resultInteger.getWidth() % sourceWidth ||
            (factor != 2 && factor != 4 && factor != 8)) {
          operation->emitError(
              "integer contract lane operand has no legal widening representation");
          failed = true;
          rewriter.eraseOp(reductionLoop);
          rewriter.eraseOp(accumulator);
          rewriter.eraseOp(scalarZero);
          continue;
        }
        auto widenedType = riscv::ValueType::get(
            rewriter.getContext(), resultInteger, currentLaneType.getShape(),
            currentLaneType.getAxisIds(), widenedLayout);
        std::string instruction =
            std::string(sourceInteger.isSigned() ? "rvv.sext.vf" :
                                                   "rvv.zext.vf") +
            std::to_string(factor);
        auto widen = rewriter.create<riscv::WidenOp>(
            operation->getLoc(), widenedType, laneValue,
            riscv_internal::leaf(
                rewriter, "rvv", "widen", instruction, instruction,
                currentLaneType.getLayout().getRegisterGroups(),
                widenedLayout.getRegisterGroups()));
        copyIdentity(operation, widen);
        laneValue = widen.getResult();
        if (laneOperand.getValue() == "rhs")
          rhs = laneValue;
        else
          lhs = laneValue;
      }
      riscv::AccessAttr lhsAccess = accessOf(lhs);
      riscv::AccessAttr rhsAccess = accessOf(rhs);
      if (implementation.getFamily() == "encoded-contract" &&
          (!lhsAccess || !rhsAccess)) {
        operation->emitError(
            "encoded contraction operands have no typed storage mappings");
        failed = true;
        rewriter.eraseOp(reductionLoop);
        rewriter.eraseOp(accumulator);
        rewriter.eraseOp(scalarZero);
        continue;
      }

      bool contractFailed = false;
      mlir::Value carried = reductionLoop.getRegionIterArg(0);
      for (int64_t offset = 0; offset < schedule.getUnroll(); ++offset) {
        mlir::Value index = reductionLoop.getInductionVar();
        if (offset) {
          mlir::Value delta = rewriter.create<mlir::arith::ConstantIndexOp>(
              operation->getLoc(), offset);
          index = rewriter.create<mlir::arith::AddIOp>(operation->getLoc(), index,
                                                       delta);
        }
        mlir::Value inBounds = rewriter.create<mlir::arith::CmpIOp>(
            operation->getLoc(), mlir::arith::CmpIPredicate::ult, index,
            reductionUpper);
        auto guarded = rewriter.create<mlir::scf::IfOp>(
            operation->getLoc(), mlir::TypeRange{resultType}, inBounds, true);
        rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());

        mlir::Value next;
        if (implementation.getFamily() == "encoded-contract") {
          auto encodedStep =
              rewriter.create<riscv::RVVEncodedContractStepOp>(
                  operation->getLoc(), resultType, lhs, rhs, carried, index,
                  reductionAxis, laneOperand.getValue(), lhsAccess, rhsAccess,
                  laneLoadLayout,
                  riscv_internal::leaf(
                      rewriter, "rvv", "encoded-contract-step",
                      "rvv.vmacc.decoded-u8-s8",
                      "rvv.vmacc.decoded-u8-s8", 0, 0, 0, 0, "none",
                      "agnostic", over.asArrayRef()));
          copyIdentity(operation, encodedStep);
          next = encodedStep.getResult();
        } else {
          mlir::Value stepLhs = lhs;
          mlir::Value stepRhs = rhs;
          mlir::Value stepLane = laneOperand.getValue() == "rhs" ? stepRhs
                                                                  : stepLhs;
          auto stepLaneType =
              mlir::dyn_cast<riscv::ValueType>(stepLane.getType());
          if (stepLaneType && llvm::is_contained(
                                  stepLaneType.getAxisIds().asArrayRef(),
                                  reductionAxis)) {
            auto projectedType = projectedReductionType(
                rewriter, stepLaneType, reductionAxis);
            auto loaded = projectLoadedOperand(rewriter, stepLane, projectedType,
                                               index, reductionAxis, operation);
            if (mlir::failed(loaded)) {
              failed = true;
              contractFailed = true;
              break;
            }
            if (*loaded) {
              stepLane = *loaded;
            } else {
              auto projected =
                  rewriter.create<riscv::ProjectReductionOperandOp>(
                      operation->getLoc(), projectedType, stepLane, index,
                      reductionAxis,
                      riscv_internal::leaf(
                          rewriter, "transfer", "reduction-projection",
                          "rvv.project-reduction-operand",
                          "rvv.project-reduction-operand", 0, 0, 1, 0));
              stepLane = projected.getResult();
            }
            if (laneOperand.getValue() == "rhs")
              stepRhs = stepLane;
            else
              stepLhs = stepLane;
          }
          const unsigned repeatedIndex =
              laneOperand.getValue() == "rhs" ? 0 : 1;
          mlir::Value repeated = operation->getOperand(repeatedIndex);
          auto repeatedType =
              mlir::dyn_cast<riscv::ValueType>(repeated.getType());
          const bool registerRepeated =
              repeatedType && repeatedType.getLayout().getCarrier() == "rvv" &&
              llvm::is_contained(repeatedType.getAxisIds().asArrayRef(),
                                 reductionAxis);
          if (registerRepeated || !accessOf(repeated)) {
            if (!repeatedType) {
              operation->emitError(
                  "computed contraction operand has no shaped physical value");
              failed = true;
              contractFailed = true;
              break;
            }
            auto projectedType = projectedReductionType(
                rewriter, repeatedType, reductionAxis);
            auto loaded = projectLoadedOperand(rewriter, repeated, projectedType,
                                               index, reductionAxis, operation);
            if (mlir::failed(loaded)) {
              failed = true;
              contractFailed = true;
              break;
            }
            mlir::Value projectedValue;
            if (*loaded) {
              projectedValue = *loaded;
            } else {
              auto projected =
                  rewriter.create<riscv::ProjectReductionOperandOp>(
                      operation->getLoc(), projectedType, repeated, index,
                      reductionAxis,
                      riscv_internal::leaf(
                          rewriter, "transfer", "reduction-projection",
                          "rvv.project-reduction-operand",
                          "rvv.project-reduction-operand", 0, 0, 1, 0));
              projectedValue = projected.getResult();
            }
            if (repeatedIndex == 0)
              stepLhs = projectedValue;
            else
              stepRhs = projectedValue;
          }
          auto contractStep = rewriter.create<riscv::RVVContractStepOp>(
              operation->getLoc(), resultType, stepLhs, stepRhs, carried, index,
              reductionAxis, laneOperand.getValue(),
              stepLane == laneValue ? laneMemory.getValue() : "register",
              mlir::cast<riscv::ValueType>(stepLane.getType()).getLayout(),
              stepLeaf);
          copyIdentity(operation, contractStep);
          next = contractStep.getResult();
        }
        rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), next);
        rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
        rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), carried);
        rewriter.setInsertionPointAfter(guarded);
        carried = guarded.getResult(0);
      }
      if (contractFailed) {
        rewriter.eraseOp(reductionLoop);
        rewriter.eraseOp(accumulator);
        rewriter.eraseOp(scalarZero);
        continue;
      }
      rewriter.setInsertionPointToEnd(reductionLoop.getBody());
      rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), carried);
      operation->getResult(0).replaceAllUsesWith(reductionLoop.getResult(0));
      rewriter.eraseOp(operation);
    }
  }

  void closeConversions(mlir::IRRewriter &rewriter, bool &failed) {
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      llvm::StringRef kind = conversion.getConversion().getKind();
      llvm::StringRef instruction;
      if (kind == "splat")
        instruction = "rvv.splat";
      else if (kind == "extract")
        instruction = "rvv.extract";
      else if (kind == "local_load")
        instruction = "rvv.local-load";
      else if (kind == "local_store")
        instruction = "rvv.local-store";
      else if (kind == "tuple")
        instruction = "rvv.tuple-convert";
      else if (kind == "register_to_lane")
        instruction = "rvv.register-to-lane";
      else if (kind == "lane_to_register")
        instruction = "rvv.lane-to-register";
      else if (kind == "reshape")
        instruction = "rvv.layout-reshape";
      else {
        conversion.emitError()
            << "no terminal leaf implements typed layout conversion kind '"
            << kind << "'";
        failed = true;
        return;
      }
      conversion.setLeafAttr(riscv_internal::leaf(
          rewriter, "rvv", "layout-conversion", instruction, instruction,
          conversion.getLeaf().getOperandGroups(),
          conversion.getLeaf().getResultGroups(),
          conversion.getConversion().getTemporaryGroups(), 0));
      conversion->removeAttr("implementation");
    });
  }

};

} // namespace

std::unique_ptr<mlir::Pass> weft::createLowerRISCVCompositesPass() {
  return std::make_unique<LowerRISCVCompositesPass>();
}
