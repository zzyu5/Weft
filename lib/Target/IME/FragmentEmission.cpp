#include "FragmentEmission.h"
#include "Weft/Target/RISCVFragment.h"

namespace weft::riscv_emission {
namespace {
std::string emitSpacemitIME1(riscv::IMEFragmentMMAOp operation,
                            llvm::StringRef lhs, llvm::StringRef rhs,
                            const FragmentEmission &emission) {
  auto fresh = emission.fresh;
  unsigned indent = 0;
  auto line = [&](const std::string &text) {
    emission.line(std::string(indent * 2, ' ') + text);
  };
  const std::string output = fresh("ime_acc_fragment");
  const std::string lhsPointer = fresh("ime_lhs");
  const std::string rhsPointer = fresh("ime_rhs");
  const std::string lowPointer = fresh("ime_acc_low");
  const std::string highPointer = fresh("ime_acc_high");
  line("_Alignas(32) int32_t " + output + "[16] = {0};");
  line("const int8_t *" + lhsPointer + " = (const int8_t *)" +
       lhs.str() + ";");
  line("const int8_t *" + rhsPointer + " = (const int8_t *)" +
       rhs.str() + ";");
  line("int32_t *" + lowPointer + " = " + output + ";");
  line("int32_t *" + highPointer + " = " + output + " + 8;");
  line("__asm__ volatile(");
  ++indent;
  line("\"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
  line("\"vle8.v v0, (%[lhs])\\n\\t\"");
  line("\"vle8.v v1, (%[rhs])\\n\\t\"");
  line("\"vmv.v.i v2, 0\\n\\t\"");
  line("\"vmv.v.i v3, 0\\n\\t\"");
  line("\"vmadot v2, v0, v1\\n\\t\"");
  line("\"vsetvli t0, zero, e32, m1, ta, ma\\n\\t\"");
  line("\"vse32.v v2, (%[low])\\n\\t\"");
  line("\"vse32.v v3, (%[high])\"");
  line(":");
  line(": [lhs] \"r\"(" + lhsPointer + "), [rhs] \"r\"(" + rhsPointer +
       "), [low] \"r\"(" + lowPointer + "), [high] \"r\"(" +
       highPointer + ")");
  std::string clobbers = "\"memory\"";
  for (mlir::Attribute attribute : operation.getAsmClobbers())
    clobbers += ", \"" +
                mlir::cast<mlir::StringAttr>(attribute).getValue().str() +
                "\"";
  line(": " + clobbers + ");");
  --indent;

  return output;
}
} // namespace

mlir::FailureOr<std::string> emitFragmentMMA(
    riscv::IMEFragmentMMAOp operation, llvm::StringRef lhs, llvm::StringRef rhs,
    const FragmentEmission &emission) {
  const auto *contract = findRISCVFragmentCapability(operation.getLeaf().getInstruction());
  if (!contract) {
    operation.emitOpError("selected fragment instruction has no terminal implementation");
    return mlir::failure();
  }
  switch (contract->instruction) {
  case RISCVFragmentInstruction::SpacemitIME1I8MMA:
    return emitSpacemitIME1(operation, lhs, rhs, emission);
  }
  llvm_unreachable("unhandled registered fragment instruction");
}
} // namespace weft::riscv_emission
