#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <iterator>
#include <limits>

using namespace weft;

namespace {

int64_t groups(mlir::Type type) {
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::WindowType>(type))
    return window.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::LayeredWindowType>(type))
    return window.getResourceGroups();
  if (auto partials = mlir::dyn_cast<riscv::PartialSetType>(type))
    return partials.getResourceGroups();
  if (auto layout = riscv_internal::layoutOf(type))
    return layout.getRegisterGroups();
  return 0;
}

struct Interval {
  unsigned begin = 0;
  unsigned end = 0;
  int64_t groups = 0;
  bool fragment = false;
  mlir::Value value;
};

struct Peak {
  int64_t vector = 0;
  int64_t fragment = 0;
  unsigned count = 0;
  mlir::Block *block = nullptr;
  unsigned point = 0;
  bool before = true;

  int64_t total() const { return vector + fragment; }
};

mlir::Operation *ancestorInBlock(mlir::Operation *operation,
                                 mlir::Block *block) {
  while (operation && operation->getBlock() != block)
    operation = operation->getParentOp();
  return operation;
}

llvm::SmallVector<Interval> intervalsFor(mlir::Block &block) {
  llvm::DenseMap<mlir::Operation *, unsigned> position;
  unsigned next = 0;
  for (mlir::Operation &operation : block)
    position[&operation] = next++;

  llvm::SmallVector<Interval> intervals;
  for (mlir::BlockArgument argument : block.getArguments()) {
    const int64_t valueGroups = groups(argument.getType());
    if (!valueGroups || argument.use_empty())
      continue;
    Interval interval{0, 0, valueGroups,
                      mlir::isa<riscv::FragmentType>(argument.getType()),
                      argument};
    for (mlir::OpOperand &use : argument.getUses()) {
      mlir::Operation *owner = ancestorInBlock(use.getOwner(), &block);
      if (owner)
        interval.end = std::max(interval.end, position.lookup(owner));
    }
    intervals.push_back(interval);
  }
  for (mlir::Operation &operation : block) {
    for (mlir::Value result : operation.getResults()) {
      const int64_t valueGroups = groups(result.getType());
      if (!valueGroups)
        continue;
      Interval interval{position.lookup(&operation), position.lookup(&operation),
                        valueGroups,
                        mlir::isa<riscv::FragmentType>(result.getType()), result};
      for (mlir::OpOperand &use : result.getUses()) {
        mlir::Operation *owner = ancestorInBlock(use.getOwner(), &block);
        if (owner)
          interval.end = std::max(interval.end, position.lookup(owner));
      }
      intervals.push_back(interval);
    }
  }
  return intervals;
}

void recordPeak(Peak &peak, int64_t vectors, int64_t fragments,
                mlir::Block *block, unsigned point, bool before) {
  const int64_t total = vectors + fragments;
  if (total > peak.total()) {
    peak.vector = vectors;
    peak.fragment = fragments;
    peak.count = 1;
    peak.block = block;
    peak.point = point;
    peak.before = before;
    return;
  }
  if (total == peak.total())
    ++peak.count;
}

Peak analyzeBlock(mlir::Block &block) {
  llvm::SmallVector<mlir::Operation *> operations;
  for (mlir::Operation &operation : block)
    operations.push_back(&operation);
  llvm::SmallVector<Interval> intervals = intervalsFor(block);

  Peak peak;
  for (auto [point, operation] : llvm::enumerate(operations)) {
    int64_t beforeVectors = 0;
    int64_t beforeFragments = 0;
    int64_t afterVectors = 0;
    int64_t afterFragments = 0;
    for (const Interval &interval : intervals) {
      if (interval.begin < point && point <= interval.end) {
        if (interval.fragment)
          beforeFragments += interval.groups;
        else
          beforeVectors += interval.groups;
      }
      if (interval.begin <= point && point < interval.end) {
        if (interval.fragment)
          afterFragments += interval.groups;
        else
          afterVectors += interval.groups;
      }
    }
    int64_t issueVectors = beforeVectors;
    int64_t issueFragments = beforeFragments;
    if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf")) {
      issueVectors += leaf.getTemporaryGroups();
      issueFragments += leaf.getFragmentGroups();
    }

    // Regions execute under this operation's live set. Their peaks are
    // alternatives in time, not values made simultaneous by a syntax walk.
    // Captures are already represented by intervals whose use maps here.
    Peak childPeak;
    int64_t childVectors = 0;
    int64_t childFragments = 0;
    for (mlir::Region &region : operation->getRegions()) {
      for (mlir::Block &child : region) {
        Peak candidate = analyzeBlock(child);
        int64_t entryVectors = 0;
        int64_t entryFragments = 0;
        for (mlir::BlockArgument argument : child.getArguments()) {
          if (mlir::isa<riscv::FragmentType>(argument.getType()))
            entryFragments += groups(argument.getType());
          else
            entryVectors += groups(argument.getType());
        }
        const int64_t candidateVectors =
            std::max<int64_t>(0, beforeVectors - entryVectors) +
            candidate.vector;
        const int64_t candidateFragments =
            std::max<int64_t>(0, beforeFragments - entryFragments) +
            candidate.fragment;
        if (candidateVectors + candidateFragments >
            childVectors + childFragments) {
          childPeak = candidate;
          childVectors = candidateVectors;
          childFragments = candidateFragments;
        }
      }
    }
    if (childPeak.total()) {
      const int64_t totalVectors = childVectors;
      const int64_t totalFragments = childFragments;
      if (totalVectors + totalFragments > peak.total()) {
        peak.vector = totalVectors;
        peak.fragment = totalFragments;
        peak.count = childPeak.count;
        peak.block = childPeak.block;
        peak.point = childPeak.point;
        peak.before = childPeak.before;
      } else if (totalVectors + totalFragments == peak.total()) {
        peak.count += std::max<unsigned>(1, childPeak.count);
      }
    }
    recordPeak(peak, issueVectors, issueFragments, &block, point, true);
    recordPeak(peak, afterVectors, afterFragments, &block, point, false);
  }
  return peak;
}

void closeLeafResources(riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  kernel.walk([&](mlir::Operation *operation) {
    auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
    if (!leaf)
      return;
    int64_t operandGroups = 0;
    int64_t resultGroups = 0;
    for (mlir::Value operand : operation->getOperands())
      operandGroups += groups(operand.getType());
    for (mlir::Value result : operation->getResults())
      resultGroups += groups(result.getType());
    operation->setAttr(
        "leaf", riscv_internal::leaf(
                    rewriter, leaf.getEngine(), leaf.getFamily(),
                    leaf.getInstruction(), leaf.getSpelling(), operandGroups,
                    resultGroups, leaf.getTemporaryGroups(),
                    leaf.getFragmentGroups(), leaf.getMask(), leaf.getTail(),
                    leaf.getParameters(), leaf.getLocalBytes()));
  });
}

mlir::LogicalResult closeDynamicLocalGuards(riscv::KernelOp kernel,
                                            mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::LocalAllocOp> dynamic;
  int64_t staticBytes = 0;
  kernel.walk([&](mlir::Operation *operation) {
    if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf"))
      staticBytes += leaf.getLocalBytes();
  });
  kernel.walk([&](riscv::LocalAllocOp allocation) {
    int64_t bytes = allocation.getResult().getType().getSizeBytes();
    if (bytes < 0)
      dynamic.push_back(allocation);
    else
      staticBytes += bytes;
  });
  if (dynamic.empty())
    return mlir::success();
  const int64_t budget = kernel.getTarget().getMaxPrivateStackBytes();
  if (staticBytes >= budget) {
    kernel.emitError(
        "static local storage leaves no target capacity for dynamic objects");
    return mlir::failure();
  }
  const int64_t perObjectLimit =
      (budget - staticBytes) / static_cast<int64_t>(dynamic.size());
  if (perObjectLimit <= 0) {
    kernel.emitError("dynamic local storage cannot receive a positive target bound");
    return mlir::failure();
  }
  for (riscv::LocalAllocOp allocation : dynamic) {
    riscv::LocalCapacityGuardOp guard;
    if (allocation->getIterator() != allocation->getBlock()->begin())
      guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(
          &*std::prev(allocation->getIterator()));
    auto exactLeaf = riscv_internal::leaf(
        rewriter, "scalar", "local-capacity",
        "scalar.local-capacity-guard", "scalar.local-capacity-guard", 0, 0);
    if (guard && guard.getBirthId() ==
                     allocation.getResult().getType().getBirthId()) {
      guard.setLimitBytes(perObjectLimit);
      guard.setLeafAttr(exactLeaf);
      continue;
    }
    rewriter.setInsertionPoint(allocation);
    rewriter.create<riscv::LocalCapacityGuardOp>(
        allocation.getLoc(), allocation.getSizeBytes(), perObjectLimit,
        allocation.getResult().getType().getBirthId(), exactLeaf);
  }
  return mlir::success();
}

mlir::FailureOr<int64_t> localStorageBytes(riscv::KernelOp kernel) {
  int64_t bytes = 0;
  mlir::LogicalResult complete = mlir::success();
  kernel.walk([&](riscv::LocalAllocOp allocation) {
    const int64_t allocationBytes = allocation.getResult().getType().getSizeBytes();
    if (allocationBytes < 0) {
      if (allocation->getIterator() == allocation->getBlock()->begin()) {
        allocation.emitError("dynamic local allocation is not guarded");
        complete = mlir::failure();
        return;
      }
      auto guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(
          &*std::prev(allocation->getIterator()));
      if (!guard || guard.getSizeBytes() != allocation.getSizeBytes() ||
          guard.getBirthId() != allocation.getResult().getType().getBirthId()) {
        allocation.emitError(
            "dynamic local allocation has no unique typed capacity guard");
        complete = mlir::failure();
        return;
      }
      bytes += guard.getLimitBytes();
      return;
    }
    // Conservative until local-slot lifetime coloring becomes a real pass:
    // every declared object receives distinct storage.
    bytes += allocationBytes;
  });
  if (mlir::failed(complete))
    return mlir::failure();
  kernel.walk([&](mlir::Operation *operation) {
    if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf"))
      bytes += leaf.getLocalBytes();
  });
  return bytes;
}

bool canPreserveProducerResult(mlir::Operation *operation) {
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!interface || operation->getNumRegions())
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  // Spill preserves the read result; it never clones or moves the producer.
  return llvm::all_of(effects, [](const auto &effect) {
    return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

mlir::LogicalResult rebindFieldReadProjections(
    riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::FieldReadOp> reads;
  kernel.walk([&](riscv::FieldReadOp read) { reads.push_back(read); });
  for (auto read : reads) {
    llvm::SmallVector<mlir::Value> captured;
    auto field = riscv::fieldReadProjection(read.getInput(), &captured);
    if (!field || captured.size() != read.getIndices().size())
      return read.emitError("spill rewriting lost the field address operand structure");
    if (llvm::equal(captured, read.getIndices()))
      continue;
    for (auto [before, after] : llvm::zip(captured, read.getIndices()))
      if (before.getType() != after.getType())
        return read.emitError("spill rewriting changed a field address operand type");

    // The read owns its use-site reloads. Clone only the pure address
    // projections, binding their index slots to this read's actual operands.
    llvm::SmallVector<riscv::ExtractOp> projections;
    mlir::Value value = read.getInput();
    while (auto extract = value.getDefiningOp<riscv::ExtractOp>()) {
      projections.push_back(extract);
      value = extract.getInput();
    }
    rewriter.setInsertionPoint(read);
    value = field.getResult();
    size_t cursor = captured.size();
    for (auto extract : llvm::reverse(projections)) {
      const size_t count = extract.getIndices().size();
      cursor -= count;
      auto copy = mlir::cast<riscv::ExtractOp>(rewriter.clone(*extract));
      copy.getInputMutable().assign(value);
      copy.getIndicesMutable().assign(read.getIndices().slice(cursor, count));
      value = copy.getResult();
    }
    read.getInputMutable().assign(value);
    for (auto extract : projections) {
      if (!extract.getResult().use_empty())
        break;
      llvm::SmallVector<riscv::ReloadOp> oldReloads;
      for (mlir::Value index : extract.getIndices())
        if (auto reload = index.getDefiningOp<riscv::ReloadOp>();
            reload && !llvm::is_contained(oldReloads, reload))
          oldReloads.push_back(reload);
      rewriter.eraseOp(extract);
      for (auto reload : oldReloads)
        if (reload.getResult().use_empty())
          rewriter.eraseOp(reload);
    }
  }
  return mlir::success();
}

bool shortenSharedReload(Peak &peak, mlir::IRRewriter &rewriter) {
  if (!peak.block || peak.point >= peak.block->getOperations().size())
    return false;
  mlir::Operation *at = &*std::next(peak.block->begin(), peak.point);
  riscv::ReloadOp selected;
  int64_t selectedGroups = 0;
  for (const Interval &interval : intervalsFor(*peak.block)) {
    auto reload = interval.value.getDefiningOp<riscv::ReloadOp>();
    if (!reload || reload->getBlock() != peak.block || interval.fragment ||
        interval.begin >= peak.point || interval.end <= peak.point ||
        interval.groups <= selectedGroups)
      continue;
    // A private slot with one preceding spill and only reload users is immutable
    // here. Splitting its reads does not rematerialize the numerical producer.
    if (!reload.getSlot().getDefiningOp<riscv::LocalAllocOp>())
      continue;
    riscv::SpillOp writer;
    bool immutable = true;
    for (mlir::Operation *user : reload.getSlot().getUsers()) {
      if (auto spill = mlir::dyn_cast<riscv::SpillOp>(user)) {
        if (writer || spill.getSlot() != reload.getSlot()) {
          immutable = false;
          break;
        }
        writer = spill;
      } else if (!mlir::isa<riscv::ReloadOp>(user)) {
        immutable = false;
        break;
      }
    }
    if (!immutable || !writer || writer->getBlock() != peak.block ||
        !writer->isBeforeInBlock(reload))
      continue;
    bool before = false, after = false, usedAtPeak = false, sameBlock = true;
    for (mlir::Operation *user : reload.getResult().getUsers()) {
      if (user->getBlock() != peak.block) {
        sameBlock = false;
        break;
      }
      usedAtPeak |= user == at;
      before |= user->isBeforeInBlock(at) || (!peak.before && user == at);
      after |= at->isBeforeInBlock(user);
    }
    if (!sameBlock || !before || !after || (peak.before && usedAtPeak))
      continue;
    selected = reload;
    selectedGroups = interval.groups;
  }
  if (!selected)
    return false;
  llvm::SmallVector<mlir::Operation *> consumers;
  for (mlir::Operation &operation : *peak.block)
    if (llvm::is_contained(operation.getOperands(), selected.getResult()))
      consumers.push_back(&operation);
  rewriter.moveOpBefore(selected, consumers.front());
  for (mlir::Operation *consumer : llvm::drop_begin(consumers)) {
    rewriter.setInsertionPoint(consumer);
    auto copy = rewriter.create<riscv::ReloadOp>(
        consumer->getLoc(), selected.getResult().getType(), selected.getSlot(),
        selected.getLeaf());
    riscv_internal::copyOrigin(selected, copy);
    consumer->replaceUsesOfWith(selected.getResult(), copy.getResult());
  }
  return true;
}

const Interval *selectVictim(mlir::Block &block, unsigned peakPoint,
                             bool peakBefore,
                             llvm::SmallVectorImpl<Interval> &intervals) {
  const Interval *victim = nullptr;
  int64_t bestScore = 0;
  mlir::Operation *peakOperation =
      peakPoint < block.getOperations().size()
          ? &*std::next(block.begin(), peakPoint)
          : nullptr;
  for (const Interval &interval : intervals) {
    if (interval.fragment || interval.end <= peakPoint ||
        riscv::fieldReadProjection(interval.value))
      continue;
    mlir::Operation *definition = interval.value.getDefiningOp();
    if (definition && interval.begin >= peakPoint)
      continue;
    auto valueType = mlir::dyn_cast<riscv::ValueType>(interval.value.getType());
    if ((definition && definition->getBlock() != &block) || !valueType ||
        valueType.getLayout().getCarrier() == "local" ||
        (definition &&
         (mlir::isa<riscv::NewOp, riscv::ReloadOp, riscv::RVVSplatOp>(definition) ||
          !canPreserveProducerResult(definition))))
      continue;

    bool sameBlock = !interval.value.use_empty();
    bool usedAtPeak = false;
    for (mlir::OpOperand &use : interval.value.getUses()) {
      sameBlock &= use.getOwner()->getBlock() == &block;
      usedAtPeak |= use.getOwner() == peakOperation;
    }
    // A use-site reload replaces the original at that exact point and cannot
    // reduce its peak. Only spill a value which is merely resident here.
    if (!sameBlock || (peakBefore && usedAtPeak))
      continue;
    const int64_t score =
        interval.groups * static_cast<int64_t>(interval.end - interval.begin);
    if (score > bestScore) {
      victim = &interval;
      bestScore = score;
    }
  }
  return victim;
}

class CloseRISCVLeafResourcesPass
    : public mlir::PassWrapper<CloseRISCVLeafResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-close-leaf-resources";
  }
  llvm::StringRef getDescription() const override {
    return "Close selected leaf operand/result resource fields before physical CSE";
  }
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>())
      closeLeafResources(kernel, rewriter);
  }
};

class MaterializeRISCVResourcesPass
    : public mlir::PassWrapper<MaterializeRISCVResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-resources";
  }
  llvm::StringRef getDescription() const override {
    return "Compute region-aware physical SSA liveness and materialize legal spills";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>()) {
      if (kernel.getResourcesMaterialized())
        continue;
      int64_t spillIdentity = 1;
      kernel.walk([&](riscv::LocalAllocOp allocation) {
        auto type = allocation.getResult().getType();
        spillIdentity = std::max({spillIdentity, type.getAliasSet() + 1,
                                  type.getBirthId() + 1});
      });
      unsigned materializableValues = 0;
      kernel.walk([&](mlir::Operation *operation) {
        for (mlir::Value result : operation->getResults())
          if (groups(result.getType()) > 0)
            ++materializableValues;
      });

      Peak previous;
      bool havePrevious = false;
      bool closed = false;
      for (unsigned attempt = 0; attempt <= materializableValues; ++attempt) {
        closeLeafResources(kernel, rewriter);
        Peak peak = analyzeBlock(kernel.getBody().front());
        if (mlir::failed(closeDynamicLocalGuards(kernel, rewriter))) {
          failed = true;
          break;
        }
        auto localBytes = localStorageBytes(kernel);
        if (mlir::failed(localBytes)) {
          failed = true;
          break;
        }
        kernel.setVectorRegisterPeak(peak.vector);
        kernel.setFragmentRegisterPeak(peak.fragment);
        kernel.setLocalStorageBytes(*localBytes);
        if (*localBytes > kernel.getTarget().getMaxPrivateStackBytes()) {
          kernel.emitError()
              << "physical local storage requires " << *localBytes
              << " bytes, exceeding target budget "
              << kernel.getTarget().getMaxPrivateStackBytes();
          failed = true;
          break;
        }
        if (peak.total() <= kernel.getTarget().getVectorRegisters()) {
          closed = true;
          break;
        }
        if (shortenSharedReload(peak, rewriter)) {
          if (mlir::failed(rebindFieldReadProjections(kernel, rewriter))) {
            failed = true;
            break;
          }
          // Added read sites make peak.count incomparable with the preceding
          // spill iteration. The original SSA candidate bound still applies.
          havePrevious = false;
          continue;
        }
        if (havePrevious &&
            (peak.total() > previous.total() ||
             (peak.total() == previous.total() && peak.count >= previous.count))) {
          auto diagnostic = kernel.emitError()
              << "resource spill made no progress: " << peak.vector
              << " vector and " << peak.fragment
              << " fragment groups remain simultaneously live";
          if (peak.block) {
            llvm::SmallVector<Interval> intervals = intervalsFor(*peak.block);
            for (const Interval &interval : intervals)
              if (interval.begin <= peak.point && peak.point <= interval.end)
                diagnostic << " [" << interval.value << ": " << interval.groups
                           << " groups, interval " << interval.begin << ".."
                           << interval.end << "]";
          }
          failed = true;
          break;
        }
        previous = peak;
        havePrevious = true;
        if (!peak.block) {
          kernel.emitError("resource peak has no physical block identity");
          failed = true;
          break;
        }
        llvm::SmallVector<Interval> intervals = intervalsFor(*peak.block);
        const Interval *victim =
            selectVictim(*peak.block, peak.point, peak.before, intervals);
        if (!victim) {
          mlir::Operation *at =
              peak.point < peak.block->getOperations().size()
                  ? &*std::next(peak.block->begin(), peak.point)
                  : nullptr;
          auto diagnostic = kernel.emitError()
              << "physical program requires " << peak.vector << " vector and "
              << peak.fragment << " fragment groups "
              << (peak.before ? "before " : "after ")
              << (at ? at->getName().getStringRef() : llvm::StringRef("<block-end>"))
              << ", exceeding target budget "
              << kernel.getTarget().getVectorRegisters()
              << "; no same-block resident value has a profitable spill";
          for (const Interval &interval : intervals)
            if (interval.begin <= peak.point && peak.point <= interval.end)
              diagnostic << " [" << interval.value << ": " << interval.groups
                         << " groups, interval " << interval.begin << ".."
                         << interval.end << "]";
          failed = true;
          break;
        }

        mlir::Value value = victim->value;
        auto valueType = mlir::cast<riscv::ValueType>(value.getType());
        mlir::Operation *definition = value.getDefiningOp();
        mlir::Location location =
            definition ? definition->getLoc()
                       : peak.block->getParentOp()->getLoc();
        int64_t streamParts = 1;
        bool spillSizeOverflow = false;
        for (int64_t factor :
             valueType.getLayout().getTimeFactors().asArrayRef()) {
          if (factor <= 0 ||
              streamParts > std::numeric_limits<int64_t>::max() / factor) {
            spillSizeOverflow = true;
            break;
          }
          streamParts *= factor;
        }
        const int64_t vlenBytes = kernel.getTarget().getVlenBits() / 8;
        if (spillSizeOverflow || vlenBytes <= 0 ||
            victim->groups > std::numeric_limits<int64_t>::max() / streamParts ||
            victim->groups * streamParts >
                std::numeric_limits<int64_t>::max() / vlenBytes) {
          kernel.emitError("physical spill storage size is not representable");
          failed = true;
          break;
        }
        // Register groups count simultaneously-live replicas.  A value with a
        // time decomposition also owns one sequential RVV part per time
        // coordinate; an explicit spill must preserve every part even though
        // those parts do not contribute simultaneously to the register peak.
        const int64_t sizeBytes = victim->groups * streamParts * vlenBytes;
        const int64_t alignment =
            std::max<int64_t>(16, kernel.getTarget().getVlenBits() / 8);
        int64_t ownerDomain = 0;
        for (mlir::Operation *parent =
                 definition ? definition->getParentOp()
                            : peak.block->getParentOp();
             parent;
             parent = parent->getParentOp())
          if (auto level =
                  parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level")) {
            ownerDomain = level.getDomainId();
            break;
          }
        auto slotType = riscv::LocalType::get(
            rewriter.getContext(), valueType.getElementType(), valueType.getShape(),
            valueType.getAxisIds(), sizeBytes, alignment, spillIdentity, "spill",
            ownerDomain, spillIdentity, ownerDomain, "rvv-register-spill");
        if (definition)
          rewriter.setInsertionPoint(definition);
        else
          rewriter.setInsertionPointToStart(peak.block);
        mlir::Value allocationBytes =
            rewriter.create<mlir::arith::ConstantIndexOp>(location, sizeBytes);
        auto slot = rewriter.create<riscv::LocalAllocOp>(
            location, slotType, allocationBytes);
        if (definition)
          rewriter.setInsertionPointAfter(definition);
        else
          rewriter.setInsertionPointAfter(slot);
        auto spillLeaf = riscv_internal::leaf(
            rewriter, "transfer", "spill", "rvv.spill", "rvv.spill",
            victim->groups, 0, 0, 0, "none", "exact");
        auto spill = rewriter.create<riscv::SpillOp>(
            location, value, slot.getResult(), spillLeaf);
        llvm::SmallVector<mlir::OpOperand *> uses;
        for (mlir::OpOperand &use : value.getUses())
          if (use.getOwner() != spill.getOperation())
            uses.push_back(&use);
        auto reloadLeaf = riscv_internal::leaf(
            rewriter, "transfer", "reload", "rvv.reload", "rvv.reload", 0,
            victim->groups, 0, 0, "none", "exact");
        llvm::DenseMap<mlir::Operation *, mlir::Value> reloads;
        for (mlir::OpOperand *use : uses) {
          auto found = reloads.find(use->getOwner());
          if (found != reloads.end()) {
            use->set(found->second);
            continue;
          }
          rewriter.setInsertionPoint(use->getOwner());
          auto reload = rewriter.create<riscv::ReloadOp>(
              use->getOwner()->getLoc(), valueType, slot.getResult(), reloadLeaf);
          use->set(reload.getResult());
          reloads[use->getOwner()] = reload.getResult();
        }
        if (mlir::failed(rebindFieldReadProjections(kernel, rewriter))) {
          failed = true;
          break;
        }
        ++spillIdentity;
      }
      if (!closed && !failed) {
        kernel.emitError(
            "resource materialization exhausted every physical SSA spill candidate");
        failed = true;
      } else if (closed) {
        kernel.setResourcesMaterialized(true);
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

int64_t weft::riscv_internal::physicalRegisterPressure(riscv::KernelOp kernel) {
  return analyzeBlock(kernel.getBody().front()).total();
}

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVResourcesPass() {
  return std::make_unique<MaterializeRISCVResourcesPass>();
}

std::unique_ptr<mlir::Pass> weft::createCloseRISCVLeafResourcesPass() {
  return std::make_unique<CloseRISCVLeafResourcesPass>();
}
