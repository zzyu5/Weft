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
  bool iq2Register32 = false, iq2Register64 = false, iq2Strip = false;
  bool iq3Register64 = false, iq3Strip = false;
  bool iq1Register32 = false, iq1Register64 = false, iq1Strip = false;
  bool q6Register32 = false, q6Register64 = false, q6Strip = false;
  bool packedI3L32 = false, packedI3L64 = false;
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
      supported = match("__weft_iq2_s_i8_register_l32_e8m2", iq2Register32) ||
                  match("__weft_iq2_s_i8_register_l64_e8m2", iq2Register64) ||
                  match("__weft_iq2_s_i8_strip", iq2Strip);
      break;
    case LocalPrimitiveKind::IQ3SI8:
      supported = match("__weft_iq3_s_i8_register_l64_e8m2", iq3Register64) ||
                  match("__weft_iq3_s_i8_strip", iq3Strip);
      break;
    case LocalPrimitiveKind::IQ1MI8:
      supported = match("__weft_iq1_m_i8_register_l32_e8m2", iq1Register32) ||
                  match("__weft_iq1_m_i8_register_l64_e8m2", iq1Register64) ||
                  match("__weft_iq1_m_i8_strip", iq1Strip);
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
  if (!iq2Register32 && !iq2Register64 && !iq2Strip &&
      !iq3Register64 && !iq3Strip && !iq1Register32 &&
      !iq1Register64 && !iq1Strip && !q6Register32 &&
      !q6Register64 && !q6Strip && !packedI3L32 && !packedI3L64)
    return true;

  if (iq2Strip || iq3Strip || iq1Strip || q6Strip) {
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
  if (iq1Register32) {
    output << R"c(static inline __attribute__((always_inline, unused)) vuint16m1_t
__weft_get_u16m2_u16m1(vuint16m2_t value, size_t segment) {
  return segment == 0 ? __riscv_vget_v_u16m2_u16m1(value, 0)
                      : __riscv_vget_v_u16m2_u16m1(value, 1);
}

static inline __attribute__((always_inline, unused)) vint8m2_t
__weft_get_i8m4_i8m2(vint8m4_t value, size_t segment) {
  return segment == 0 ? __riscv_vget_v_i8m4_i8m2(value, 0)
                      : __riscv_vget_v_i8m4_i8m2(value, 1);
}

static inline __attribute__((always_inline, unused)) vint8m2_t
__weft_get_i8m8_i8m2(vint8m8_t value, size_t segment) {
  if (segment == 0)
    return __riscv_vget_v_i8m8_i8m2(value, 0);
  if (segment == 1)
    return __riscv_vget_v_i8m8_i8m2(value, 1);
  if (segment == 2)
    return __riscv_vget_v_i8m8_i8m2(value, 2);
  return __riscv_vget_v_i8m8_i8m2(value, 3);
}
)c";
  }

  if (iq1Register32 || iq1Register64 || iq1Strip)
    emitI64Table(output, "__weft_iq1_m_grid", __weft_iq1_m_grid, 2048);
  if (iq2Register32 || iq2Register64 || iq2Strip)
    emitI64Table(output, "__weft_iq2_s_grid", __weft_iq2_s_grid, 1024);
  if (iq3Register64 || iq3Strip)
    emitI32Table(output, "__weft_iq3_s_grid", __weft_iq3_s_grid, 512);

  emitIQ2LocalImplementations(output, iq2Register32, iq2Register64, iq2Strip);
  emitIQ3LocalImplementations(output, iq3Register64, iq3Strip);
  emitIQ1LocalImplementations(output, iq1Register32, iq1Register64, iq1Strip);
  emitQ6LocalImplementations(output, q6Register32, q6Register64, q6Strip);
  emitPackedI3GroupedLocalImplementations(output, packedI3L32, packedI3L64);
  return true;
}

} // namespace weft::riscv_internal
