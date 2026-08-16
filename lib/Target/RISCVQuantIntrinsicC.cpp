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

void emitQuantLocalImplementations(llvm::raw_ostream &output,
                                   const SelectedLocalImplementations &selected) {
  auto reg = [&](LocalImplementationLeaf leaf, unsigned lanes,
                 RVVVectorShape shape) {
    return selected.contains(leaf, lanes, shape);
  };
  auto strip = [&](LocalImplementationLeaf leaf) {
    return selected.contains(leaf);
  };
  const bool iq2Register32 =
      reg(LocalImplementationLeaf::RVVIQ2SI8Register, 32, {8, 16});
  const bool iq2Register64 =
      reg(LocalImplementationLeaf::RVVIQ2SI8Register, 64, {8, 16});
  const bool iq2Strip = strip(LocalImplementationLeaf::RVVIQ2SI8Strip);
  const bool iq3Register64 =
      reg(LocalImplementationLeaf::RVVIQ3SI8Register, 64, {8, 16});
  const bool iq3Strip = strip(LocalImplementationLeaf::RVVIQ3SI8Strip);
  const bool iq1Register32 =
      reg(LocalImplementationLeaf::RVVIQ1MI8Register, 32, {8, 16});
  const bool iq1Register64 =
      reg(LocalImplementationLeaf::RVVIQ1MI8Register, 64, {8, 16});
  const bool iq1Strip = strip(LocalImplementationLeaf::RVVIQ1MI8Strip);
  const bool q6Register32 =
      reg(LocalImplementationLeaf::RVVQ6KI8Register, 32, {8, 16});
  const bool q6Register64 =
      reg(LocalImplementationLeaf::RVVQ6KI8Register, 64, {8, 16});
  const bool q6Strip = strip(LocalImplementationLeaf::RVVQ6KI8Strip);
  const bool packedI4E8M2 =
      reg(LocalImplementationLeaf::RVVPackedI4I8Register, 32, {8, 16});
  const bool packedI4E8M1 =
      reg(LocalImplementationLeaf::RVVPackedI4I8Register, 32, {8, 8});
  const bool packedI5E8M2 =
      reg(LocalImplementationLeaf::RVVPackedI5I8Register, 32, {8, 16});
  const bool packedI5E8M1 =
      reg(LocalImplementationLeaf::RVVPackedI5I8Register, 32, {8, 8});
  const bool packedI3L32 =
      reg(LocalImplementationLeaf::RVVPackedI3GroupedI8Register, 32,
          {8, 16});
  const bool packedI3L64 =
      reg(LocalImplementationLeaf::RVVPackedI3GroupedI8Register, 64,
          {8, 16});
  const bool nibbleE8M2 =
      reg(LocalImplementationLeaf::RVVNibbleCodebookI8Register, 32,
          {8, 16});
  const bool nibbleE8M1 =
      reg(LocalImplementationLeaf::RVVNibbleCodebookI8Register, 32, {8, 8});
  const bool base3E8M2 =
      reg(LocalImplementationLeaf::RVVBase3TernaryI8Register, 32, {8, 16});
  const bool base3E8M1 =
      reg(LocalImplementationLeaf::RVVBase3TernaryI8Register, 32, {8, 8});
  const bool packedI2E8M2 =
      reg(LocalImplementationLeaf::RVVPackedI2TernaryI8Register, 32,
          {8, 16});
  const bool packedI2E8M1 =
      reg(LocalImplementationLeaf::RVVPackedI2TernaryI8Register, 32, {8, 8});
  const bool signed8E8M2 =
      reg(LocalImplementationLeaf::RVVSignedCodebook8I8Register, 32,
          {8, 16});
  const bool signed8E8M1 =
      reg(LocalImplementationLeaf::RVVSignedCodebook8I8Register, 32,
          {8, 8});
  const bool signed4E8M2 =
      reg(LocalImplementationLeaf::RVVSignedCodebook4I8Register, 32,
          {8, 16});
  const bool signed4E8M1 =
      reg(LocalImplementationLeaf::RVVSignedCodebook4I8Register, 32,
          {8, 8});
  const bool packedU9U7E8M2 =
      reg(LocalImplementationLeaf::RVVPackedU9U7CodebookI8Register, 32,
          {8, 16});
  const bool packedU9U7E8M1 =
      reg(LocalImplementationLeaf::RVVPackedU9U7CodebookI8Register, 32,
          {8, 8});
  const bool packedU11E8M2 =
      reg(LocalImplementationLeaf::RVVPackedU11GridDeltaI8Register, 32,
          {8, 16});
  const bool packedU11E8M1 =
      reg(LocalImplementationLeaf::RVVPackedU11GridDeltaI8Register, 32,
          {8, 8});
  if (!iq2Register32 && !iq2Register64 && !iq2Strip &&
      !iq3Register64 && !iq3Strip && !iq1Register32 &&
      !iq1Register64 && !iq1Strip && !q6Register32 &&
      !q6Register64 && !q6Strip && !packedI4E8M2 && !packedI4E8M1 &&
      !packedI5E8M2 && !packedI5E8M1 && !packedI3L32 && !packedI3L64 &&
      !nibbleE8M2 && !nibbleE8M1 && !base3E8M2 && !base3E8M1 &&
      !packedI2E8M2 && !packedI2E8M1 && !signed8E8M2 &&
      !signed8E8M1 && !signed4E8M2 && !signed4E8M1 &&
      !packedU9U7E8M2 && !packedU9U7E8M1 && !packedU11E8M2 &&
      !packedU11E8M1)
    return;

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
  emitPackedI4LocalImplementations(output, packedI4E8M2, packedI4E8M1);
  emitPackedI5LocalImplementations(output, packedI5E8M2, packedI5E8M1);
  emitPackedI3GroupedLocalImplementations(output, packedI3L32, packedI3L64);
  emitNibbleCodebookLocalImplementations(output, nibbleE8M2, nibbleE8M1);
  emitTernaryLocalImplementations(output, base3E8M2, base3E8M1,
                                  packedI2E8M2, packedI2E8M1);
  emitSignedCodebookLocalImplementations(
      output, signed8E8M2, signed8E8M1, signed4E8M2, signed4E8M1);
  emitPackedU9U7CodebookLocalImplementations(
      output, packedU9U7E8M2, packedU9U7E8M1);
  emitPackedU11GridDeltaLocalImplementations(
      output, packedU11E8M2, packedU11E8M1);
}

} // namespace weft::riscv_internal
