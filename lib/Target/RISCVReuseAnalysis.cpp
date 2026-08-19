#include "RISCVReuseAnalysis.h"

namespace weft::riscv_internal {
namespace {

using namespace weft::kernel;

bool dependsOnAxis(const KernelPhysicalFacts &facts, mlir::Value value,
                   mlir::Value axis) {
  auto found = facts.values.find(value);
  return found != facts.values.end() &&
         llvm::is_contained(found->second.axisDependencies, axis);
}

LoadOp reloadableLoad(const KernelPhysicalFacts &facts, mlir::Value value) {
  auto found = facts.values.find(value);
  return found == facts.values.end() || !found->second.reloadSource
             ? LoadOp{}
             : mlir::dyn_cast<LoadOp>(found->second.reloadSource);
}

const MemoryAccessFact *memoryFact(const KernelPhysicalFacts &facts,
                                   mlir::Operation *operation) {
  auto found = facts.memory.find(operation);
  return found == facts.memory.end() ? nullptr : &found->second;
}

LaneRelation memoryRelation(const MemoryAccessFact &access, mlir::Value axis) {
  auto found = access.axes.find(axis);
  return found == access.axes.end() ? LaneRelation::Independent
                                    : found->second.relation;
}

} // namespace

LocalScheduleDependenceFacts analyzeLocalScheduleDependences(
    const KernelPhysicalFacts &facts,
    llvm::ArrayRef<mlir::Value> primitiveOperands, mlir::Value reductionAxis,
    mlir::Value accumulator, mlir::Value result) {
  LocalScheduleDependenceFacts dependence;
  if (primitiveOperands.empty() || !reductionAxis || !accumulator || !result)
    return dependence;
  for (mlir::Value operand : primitiveOperands) {
    LoadOp load = reloadableLoad(facts, operand);
    LocalOperandDependenceFacts operandFacts;
    const MemoryAccessFact *access =
        load ? memoryFact(facts, load.getOperation()) : nullptr;
    const LaneRelation relation =
        access ? memoryRelation(*access, reductionAxis)
               : LaneRelation::NonAffine;
    if (relation == LaneRelation::UnitStride)
      operandFacts.memoryMode = PhysicalMemoryMode::UnitStride;
    else if (relation == LaneRelation::Strided)
      operandFacts.memoryMode = PhysicalMemoryMode::Strided;
    else if (relation == LaneRelation::Indexed)
      operandFacts.memoryMode = PhysicalMemoryMode::Indexed;
    operandFacts.advancesIteration =
        load ? dependsOnAxis(facts, load.getPointer(), reductionAxis)
             : dependsOnAxis(facts, operand, reductionAxis);
    operandFacts.feedsPrimitive =
        !load || valueDependsOn(operand, load.getResult());
    operandFacts.addressDependsOnAccumulator =
        load ? valueDependsOn(load.getPointer(), accumulator)
             : valueDependsOn(operand, accumulator);
    operandFacts.predicateDependsOnAccumulator =
        load && valueDependsOn(load.getWhere(), accumulator);
    auto value = facts.values.find(operand);
    if (value != facts.values.end()) {
      operandFacts.consumerCount = value->second.consumers.size();
      operandFacts.crossesControl =
          value->second.crossesRegion || value->second.controlCarried;
    }
    dependence.operands.push_back(operandFacts);
  }
  auto resultFacts = facts.values.find(result);
  if (resultFacts != facts.values.end()) {
    dependence.resultConsumerCount = resultFacts->second.consumers.size();
    dependence.resultCrossesControl = resultFacts->second.crossesRegion ||
                                      resultFacts->second.controlCarried;
  }
  return dependence;
}

} // namespace weft::riscv_internal
