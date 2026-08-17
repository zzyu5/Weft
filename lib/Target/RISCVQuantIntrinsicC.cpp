#include "RISCVIntrinsicC.h"
#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace weft::riscv_internal {

#include "RISCVQuantCodebooks.inc"

namespace {

void emitI64Table(llvm::raw_ostream &output, llvm::StringRef name,
                  const int64_t *values, size_t count) {
  output << "static const int64_t " << name << "[" << count << "] = {\n";
  for (size_t index = 0; index < count; ++index) {
    if (index % 4 == 0)
      output << "  ";
    output << values[index];
    if (index + 1 != count)
      output << ", ";
    output << (index % 4 == 3 ? "\n" : "");
  }
  if (count % 4 != 0)
    output << "\n";
  output << "};\n\n";
}

void emitI32Table(llvm::raw_ostream &output, llvm::StringRef name,
                  const int32_t *values, size_t count) {
  output << "static const int32_t " << name << "[" << count << "] = {\n";
  for (size_t index = 0; index < count; ++index) {
    if (index % 8 == 0)
      output << "  ";
    output << values[index];
    if (index + 1 != count)
      output << ", ";
    output << (index % 8 == 7 ? "\n" : "");
  }
  if (count % 8 != 0)
    output << "\n";
  output << "};\n\n";
}

} // namespace

bool emitQuantLocalImplementations(llvm::raw_ostream &output,
                                   const SelectedLocalImplementations &selected,
                                   std::string &unsupportedSymbol) {
  bool q6Register32 = false, q6Register64 = false, q6Strip = false;
  bool packedI3L32 = false, packedI3L64 = false;
  const LocalImplementation *iq2Implementation = nullptr;
  const LocalImplementation *iq3Implementation = nullptr;
  const LocalImplementation *iq1Implementation = nullptr;
  bool supported = true;
  selected.forEach([&](const LocalImplementation &implementation) {
    if (!supported)
      return;
    const std::string &symbol = implementation.helperSymbol;
    auto match = [&](const char *expected, bool &flag) {
      if (symbol != expected)
        return false;
      flag = true;
      return true;
    };
    switch (implementation.primitive) {
    case LocalPrimitiveKind::PackedI4I8:
      supported = emitPackedI4LocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI5I8:
      supported = emitPackedI5LocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI3GroupedI8:
      supported = match("__weft_packed_i3_grouped_i8_register_l32_e8m2",
                        packedI3L32) ||
                  match("__weft_packed_i3_grouped_i8_register_l64_e8m2",
                        packedI3L64);
      break;
    case LocalPrimitiveKind::Base3TernaryI8:
      supported = emitBase3TernaryLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI2TernaryI8:
      supported =
          emitPackedI2TernaryLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::SignedCodebook8I8:
    case LocalPrimitiveKind::SignedCodebook4I8:
      supported =
          emitSignedCodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedU9U7CodebookI8:
      supported =
          emitPackedU9U7CodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedU11GridDeltaI8:
      supported =
          emitPackedU11GridDeltaLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::NibbleCodebookI8:
      supported =
          emitNibbleCodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::IQ2SI8:
      iq2Implementation = &implementation;
      break;
    case LocalPrimitiveKind::IQ3SI8:
      iq3Implementation = &implementation;
      break;
    case LocalPrimitiveKind::IQ1MI8:
      iq1Implementation = &implementation;
      break;
    case LocalPrimitiveKind::Q6KI8:
      supported = match("__weft_q6_k_i8_register_l32_e8m2", q6Register32) ||
                  match("__weft_q6_k_i8_register_l64_e8m2", q6Register64) ||
                  match("__weft_q6_k_i8_strip", q6Strip);
      break;
    default:
      return;
    }
    if (!supported)
      unsupportedSymbol = symbol.empty() ? "<unnamed local implementation>"
                                         : symbol;
  });
  if (!supported)
    return false;
  if (!q6Register32 && !q6Register64 && !q6Strip && !packedI3L32 &&
      !packedI3L64 && !iq2Implementation && !iq3Implementation &&
      !iq1Implementation)
    return true;

  if (q6Strip) {
    output << R"c(static inline __attribute__((always_inline, unused)) int32_t
__weft_i8_dot(const int8_t *lhs, const int8_t *rhs, size_t count) {
  int32_t result = 0;
  size_t offset = 0;
  while (offset < count) {
    const size_t vl = __riscv_vsetvl_e8m2(count - offset);
    const vint8m2_t left = __riscv_vle8_v_i8m2(lhs + offset, vl);
    const vint8m2_t right = __riscv_vle8_v_i8m2(rhs + offset, vl);
    const vint16m4_t products = __riscv_vwmul_vv_i16m4(left, right, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    result += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m4_i32m1(products, zero, vl));
    offset += vl;
  }
  return result;
}
)c";
  }
  if (iq1Implementation)
    emitI64Table(output, "__weft_iq1_m_grid", __weft_iq1_m_grid, 2048);
  if (iq2Implementation)
    emitI64Table(output, "__weft_iq2_s_grid", __weft_iq2_s_grid, 1024);
  if (iq3Implementation)
    emitI32Table(output, "__weft_iq3_s_grid", __weft_iq3_s_grid, 512);

  if (iq2Implementation &&
      !emitIQ2LocalImplementation(output, *iq2Implementation)) {
    unsupportedSymbol = iq2Implementation->helperSymbol;
    return false;
  }
  if (iq3Implementation &&
      !emitIQ3LocalImplementation(output, *iq3Implementation)) {
    unsupportedSymbol = iq3Implementation->helperSymbol;
    return false;
  }
  if (iq1Implementation &&
      !emitIQ1LocalImplementation(output, *iq1Implementation)) {
    unsupportedSymbol = iq1Implementation->helperSymbol;
    return false;
  }
  emitQ6LocalImplementations(output, q6Register32, q6Register64, q6Strip);
  emitPackedI3GroupedLocalImplementations(output, packedI3L32, packedI3L64);
  return true;
}

} // namespace weft::riscv_internal
