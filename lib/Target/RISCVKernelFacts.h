#ifndef WEFT_LIB_TARGET_RISCVKERNELFACTS_H
#define WEFT_LIB_TARGET_RISCVKERNELFACTS_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace weft::riscv_internal {

enum class LaneRelation {
  Independent,
  UnitStride,
  Strided,
  Indexed,
  NonAffine,
};

enum class AffineScalarExpressionKind {
  Constant,
  Value,
  Add,
  Subtract,
  Multiply,
};

struct AffineScalarExpression {
  AffineScalarExpressionKind kind = AffineScalarExpressionKind::Constant;
  int64_t constant = 0;
  mlir::Value value;
  std::shared_ptr<const AffineScalarExpression> lhs;
  std::shared_ptr<const AffineScalarExpression> rhs;
};

struct CoordinateInvariantPointerBaseFact {
  mlir::Value value;
  llvm::SmallVector<mlir::Operation *, 4> hoistedOperations;
};

struct MemoryAxisFact {
  LaneRelation relation = LaneRelation::Independent;
  std::optional<AffineScalarExpression> laneStride;
  mlir::Value indexedOffset;
  std::optional<CoordinateInvariantPointerBaseFact> pointerBase;
};

struct InterleavedMemoryFact {
  mlir::Value root;
  mlir::Value pointer;
  llvm::DenseMap<mlir::Value, int64_t> invariantTerms;
  unsigned field = 0;
  unsigned fields = 0;
  int64_t coordinateScale = 0;
};

struct LogicalAxisFact {
  mlir::Value coordinate;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value extent;
  int64_t identity = 0;
  bool vla = false;
  llvm::SmallVector<mlir::Operation *> orderedParents;
};

struct ValueUseFact {
  mlir::Value value;
  mlir::Operation *reloadSource = nullptr;
  llvm::SmallVector<mlir::Value> logicalAxes;
  llvm::SmallVector<mlir::Value> axisDependencies;
  llvm::SmallVector<mlir::Operation *> consumers;
  unsigned definitionOrdinal = 0;
  unsigned lastUseOrdinal = 0;
  bool multipleConsumers = false;
  bool crossesRegion = false;
  bool controlCarried = false;
};

struct MemoryAccessFact {
  mlir::Operation *operation = nullptr;
  mlir::Value pointer;
  mlir::Value root;
  mlir::Value predicate;
  mlir::Type elementType;
  bool write = false;
  llvm::DenseMap<mlir::Value, MemoryAxisFact> axes;
  llvm::DenseMap<mlir::Value, InterleavedMemoryFact> interleaved;
};

struct IndexedMemoryGroupFact {
  mlir::Value coordinate;
  mlir::Value indexedOffset;
  unsigned elementBytes = 0;
  llvm::SmallVector<mlir::Operation *, 4> accesses;
};

struct InterleavedMemoryGroupFact {
  mlir::Value coordinate;
  mlir::Value base;
  int64_t coordinateScale = 0;
  unsigned fields = 0;
  mlir::Type elementType;
  bool write = false;
  llvm::SmallVector<mlir::Operation *, 4> accesses;
};

struct KernelPhysicalFacts {
  llvm::DenseMap<int64_t, mlir::Value> blockAxes;
  llvm::DenseMap<mlir::Value, LogicalAxisFact> axes;
  llvm::DenseMap<mlir::Value, ValueUseFact> values;
  llvm::DenseMap<mlir::Operation *, MemoryAccessFact> memory;
  llvm::SmallVector<IndexedMemoryGroupFact, 4> indexedMemoryGroups;
  llvm::SmallVector<InterleavedMemoryGroupFact, 4> interleavedMemoryGroups;
  llvm::DenseMap<mlir::Operation *, unsigned> operationOrdinals;
};

/// MLIR analysis owned by one canonical KernelOp. Construction is the only
/// producer of target-independent axis, use, and memory facts for a RISC-V
/// compilation invocation.
class KernelPhysicalFactsAnalysis {
public:
  explicit KernelPhysicalFactsAnalysis(weft::kernel::KernelOp kernel);

  bool succeeded() const { return valid; }
  const KernelPhysicalFacts &getFacts() const { return facts; }

private:
  KernelPhysicalFacts facts;
  bool valid = false;
};

struct StructuredProductFacts {
  llvm::SmallVector<mlir::Value> lhsAxes;
  llvm::SmallVector<mlir::Value> rhsAxes;
  llvm::SmallVector<mlir::Value> resultAxes;
  llvm::SmallVector<mlir::Value> lhsFreeAxes;
  llvm::SmallVector<mlir::Value> rhsFreeAxes;
  llvm::SmallVector<mlir::Value> reductionAxes;
  llvm::SmallVector<mlir::Value> lhsBroadcastAxes;
  llvm::SmallVector<mlir::Value> rhsBroadcastAxes;
};

std::optional<StructuredProductFacts>
analyzeStructuredProductFacts(const KernelPhysicalFacts &facts, mlir::Value lhs,
                              mlir::Value rhs, mlir::Value result);

LaneRelation classifyLaneRelation(mlir::Value value, mlir::Value coordinate);
mlir::Value findIndexedOffset(mlir::Value pointer, mlir::Value coordinate);
bool valueDependsOn(mlir::Value value, mlir::Value target);
std::optional<int64_t> getConstantIndexValue(mlir::Value value);
bool haveSameInterleavedInvariantAddress(const InterleavedMemoryFact &lhs,
                                         const InterleavedMemoryFact &rhs);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVKERNELFACTS_H
