#ifndef WEFT_LIB_TARGET_RISCVPLANNINGSUPPORT_H
#define WEFT_LIB_TARGET_RISCVPLANNINGSUPPORT_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::riscv_internal {

mlir::ArrayAttr strings(mlir::Builder &builder,
                        llvm::ArrayRef<std::string> values);
mlir::DenseI64ArrayAttr integers(mlir::Builder &builder,
                                 llvm::ArrayRef<int64_t> values);
mlir::DictionaryAttr dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values);
mlir::DictionaryAttr set(mlir::DictionaryAttr source, llvm::StringRef name,
                         mlir::Attribute value);
mlir::DictionaryAttr targetFacts(mlir::Builder &builder,
                                 const RISCVTargetProfile &target);

std::string printType(mlir::Type type);
std::string printAttribute(mlir::Attribute attribute);
unsigned logicalBitWidth(mlir::Type type);
llvm::ArrayRef<int64_t> logicalShape(mlir::Type type);
llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type);
mlir::Type logicalElement(mlir::Type type);

std::optional<int64_t> integer(mlir::DictionaryAttr dictionary,
                               llvm::StringRef name);
std::optional<llvm::StringRef> string(mlir::DictionaryAttr dictionary,
                                     llvm::StringRef name);
mlir::ArrayAttr array(mlir::DictionaryAttr dictionary, llvm::StringRef name);

kernel::KernelOp findKernel(mlir::ModuleOp module,
                            mlir::FlatSymbolRefAttr symbol);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVPLANNINGSUPPORT_H
