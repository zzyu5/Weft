#ifndef WEFT_LIB_TARGET_RISCVPHYSICALSUPPORT_H
#define WEFT_LIB_TARGET_RISCVPHYSICALSUPPORT_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace weft::riscv_internal {

struct FieldFacts {
  llvm::StringRef mapping = "opaque";
  bool scalarPerRecord = false;
  int64_t logicalRank = 0;
  int64_t group = 0;
  int64_t layer = 0;
  int64_t joinFields = 0;
  int64_t joinLowBits = 0;
  int64_t joinRole = 0;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
  int64_t alignment = 1;
  llvm::StringRef order = "none";
};

struct IntegerRange {
  int64_t minimum;
  int64_t maximum;
};

struct WidenDotLaneSlicePlan {
  int64_t reductionLanes = 0;
  int64_t reductionStreams = 0;
  int64_t sliceLmulEighths = 0;
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> parts;
};

mlir::DenseI64ArrayAttr integers(mlir::Builder &builder,
                                 llvm::ArrayRef<int64_t> values);
mlir::ArrayAttr strings(mlir::Builder &builder,
                        llvm::ArrayRef<std::string> values);
mlir::DictionaryAttr dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values);
std::optional<int64_t> integer(mlir::DictionaryAttr dictionary,
                               llvm::StringRef name);
std::optional<llvm::StringRef> string(mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name);
mlir::ArrayAttr array(mlir::DictionaryAttr dictionary, llvm::StringRef name);

riscv::LayoutAttr unassignedLayout(mlir::Builder &builder,
                                   llvm::ArrayRef<int64_t> axes);
riscv::LayoutAttr scalarLayout(mlir::Builder &builder, mlir::Type element,
                               llvm::ArrayRef<int64_t> shape,
                               llvm::ArrayRef<int64_t> axes,
                               llvm::StringRef validity = "full");
riscv::LayoutAttr projectLayout(mlir::Builder &builder, riscv::ValueType source,
                                riscv::LayoutAttr reference,
                                riscv::TargetAttr target);
riscv::AccessAttr unassignedAccess(mlir::Builder &builder);
riscv::AccessAttr denseAccess(mlir::Builder &builder,
                              riscv::MemDescType memory,
                              mlir::Type physicalValue);
riscv::ConversionAttr conversion(mlir::Builder &builder,
                                 llvm::StringRef kind,
                                 llvm::StringRef effect = "pure",
                                 int64_t temporaryGroups = 0);
riscv::ConversionAttr layoutConversion(mlir::Builder &builder,
                                       riscv::LayoutAttr source,
                                       riscv::LayoutAttr target);
riscv::ScheduleAttr schedule(mlir::Builder &builder, int64_t unroll,
                             int64_t pipelineDepth,
                             int64_t prefetchDistance = 0);
riscv::ImplementationAttr implementation(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters = {});
riscv::LeafAttr unselectedLeaf(mlir::Builder &builder);
riscv::LeafAttr leaf(mlir::Builder &builder, llvm::StringRef engine,
                     llvm::StringRef family, llvm::StringRef instruction,
                     llvm::StringRef spelling, int64_t operandGroups,
                     int64_t resultGroups, int64_t temporaryGroups = 0,
                     int64_t fragmentGroups = 0,
                     llvm::StringRef mask = "none",
                     llvm::StringRef tail = "exact",
                     llvm::ArrayRef<int64_t> parameters = {},
                     int64_t localBytes = 0);
riscv::TargetAttr target(mlir::Builder &builder,
                         const RISCVTargetProfile &profile);

mlir::Type logicalElement(mlir::Type type);
llvm::ArrayRef<int64_t> logicalShape(mlir::Type type);
llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type);
unsigned logicalBitWidth(mlir::Type type);
riscv::LayoutAttr layoutOf(mlir::Type type);
mlir::Type withLayout(mlir::Type type, riscv::LayoutAttr layout);
int64_t physicalExtent(mlir::Value value, int64_t axis);
std::string terminalInstruction(mlir::Operation *operation);
llvm::StringRef layeredStorageDecodeInstruction(int64_t shiftAmount,
                                                int64_t maskValue);

std::optional<int64_t> staticProduct(llvm::ArrayRef<int64_t> values);
std::optional<WidenDotLaneSlicePlan>
planWidenDotLaneSlices(riscv::ValueType operand, riscv::ValueType result,
                       llvm::ArrayRef<int64_t> reductionAxes,
                       riscv::TargetAttr target);
std::string printType(mlir::Type type);

riscv::EncodingDeclOp findEncoding(mlir::Operation *operation,
                                   llvm::StringRef family);
llvm::StringRef baseEncodingFamily(mlir::Operation *operation,
                                   kernel::EncodingType encoding);
int64_t interleaveRows(mlir::Operation *operation,
                       kernel::EncodingType encoding);
FieldFacts fieldFacts(riscv::FieldOp operation);

std::optional<int64_t> constantInt(mlir::Value value);
std::optional<IntegerRange> integerRange(mlir::Value value,
                                         unsigned depth = 0);
std::optional<int64_t> maximumMagnitude(mlir::Value value);
mlir::Value stripRepresentationConversions(
    mlir::Value value,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr);
riscv::FieldOp sourceField(mlir::Value value);
riscv::LoadOp sourceLoad(mlir::Value value);
riscv::AccessAttr accessOf(mlir::Value value);
riscv::PhysicalPointOp originPoint(mlir::Value value, int64_t axis);
std::optional<riscv::StorageWindowPlanAttr>
storageWindowPlan(mlir::Builder &builder, riscv::FieldOp field,
                  int64_t reductionAxis, int64_t projectionBase,
                  int64_t projectionStride, int64_t projectionRepeat,
                  int64_t projectionExtent, int64_t offsetAlignment);

void copyOrigin(mlir::Operation *source, mlir::Operation *target);

} // namespace weft::riscv_internal

#endif
