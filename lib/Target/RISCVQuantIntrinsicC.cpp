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

void emitQuantIntrinsicCLeaves(llvm::raw_ostream &output,
                               const SelectedIntrinsicCLeaves &leaves) {
  bool iq2FixedLanes32 =
      leaves.contains(IntrinsicCLeaf::IQ2SI8FixedLanes32);
  bool iq2FixedLanes64 =
      leaves.contains(IntrinsicCLeaf::IQ2SI8FixedLanes64);
  bool iq2Scalable = leaves.contains(IntrinsicCLeaf::IQ2SI8Scalable);
  bool iq3FixedLanes64 =
      leaves.contains(IntrinsicCLeaf::IQ3SI8FixedLanes64);
  bool iq3Scalable = leaves.contains(IntrinsicCLeaf::IQ3SI8Scalable);
  bool iq1FixedLanes32 =
      leaves.contains(IntrinsicCLeaf::IQ1MI8FixedLanes32);
  bool iq1FixedLanes64 =
      leaves.contains(IntrinsicCLeaf::IQ1MI8FixedLanes64);
  bool iq1Scalable = leaves.contains(IntrinsicCLeaf::IQ1MI8Scalable);
  bool q6FixedLanes32 =
      leaves.contains(IntrinsicCLeaf::Q6KI8FixedLanes32);
  bool q6FixedLanes64 =
      leaves.contains(IntrinsicCLeaf::Q6KI8FixedLanes64);
  bool q6Scalable = leaves.contains(IntrinsicCLeaf::Q6KI8Scalable);
  bool packedI5VLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedI5I8VLEN128);
  bool packedI5VLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedI5I8VLEN256);
  if (!iq2FixedLanes32 && !iq2FixedLanes64 && !iq2Scalable &&
      !iq3FixedLanes64 && !iq3Scalable && !iq1FixedLanes32 &&
      !iq1FixedLanes64 && !iq1Scalable && !q6FixedLanes32 &&
      !q6FixedLanes64 && !q6Scalable && !packedI5VLEN128 &&
      !packedI5VLEN256)
    return;

  if (iq2Scalable || iq3Scalable || iq1Scalable || q6Scalable) {
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
  if (iq1FixedLanes32) {
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

  if (iq1FixedLanes32 || iq1FixedLanes64 || iq1Scalable)
    emitI64Table(output, "__weft_iq1_m_grid", __weft_iq1_m_grid, 2048);
  if (iq2FixedLanes32 || iq2FixedLanes64 || iq2Scalable)
    emitI64Table(output, "__weft_iq2_s_grid", __weft_iq2_s_grid, 1024);
  if (iq3FixedLanes64 || iq3Scalable)
    emitI32Table(output, "__weft_iq3_s_grid", __weft_iq3_s_grid, 512);

  emitIQ2IntrinsicCLeaves(output, iq2FixedLanes32, iq2FixedLanes64,
                          iq2Scalable);
  emitIQ3IntrinsicCLeaves(output, iq3FixedLanes64, iq3Scalable);
  emitIQ1IntrinsicCLeaves(output, iq1FixedLanes32, iq1FixedLanes64,
                          iq1Scalable);
  emitQ6IntrinsicCLeaves(output, q6FixedLanes32, q6FixedLanes64,
                         q6Scalable);
  emitPackedI5IntrinsicCLeaves(output, packedI5VLEN128, packedI5VLEN256);
}

} // namespace weft::riscv_internal
