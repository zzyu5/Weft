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

struct MemoryAxisFact {
  LaneRelation relation = LaneRelation::Independent;
  std::optional<AffineScalarExpression> laneStride;
  mlir::Value indexedOffset;
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
};

struct KernelPhysicalFacts {
  llvm::DenseMap<int64_t, mlir::Value> blockAxes;
  llvm::DenseMap<mlir::Value, LogicalAxisFact> axes;
  llvm::DenseMap<mlir::Value, ValueUseFact> values;
  llvm::DenseMap<mlir::Operation *, MemoryAccessFact> memory;
  llvm::DenseMap<mlir::Operation *, unsigned> operationOrdinals;
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

LaneRelation classifyLaneRelation(mlir::Value value, mlir::Value coordinate);
mlir::Value findIndexedOffset(mlir::Value pointer, mlir::Value coordinate);

mlir::LogicalResult
analyzeKernelPhysicalFacts(weft::kernel::KernelOp kernel,
                           KernelPhysicalFacts &facts);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVKERNELFACTS_H
