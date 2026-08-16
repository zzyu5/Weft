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
  bool packedI4VLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedI4I8VLEN128);
  bool packedI4VLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedI4I8VLEN256);
  bool packedI5VLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedI5I8VLEN128);
  bool packedI5VLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedI5I8VLEN256);
  bool packedI3GroupedVLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedI3GroupedI8VLEN128);
  bool packedI3GroupedVLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedI3GroupedI8VLEN256);
  bool nibbleCodebookVLEN128 =
      leaves.contains(IntrinsicCLeaf::NibbleCodebookI8VLEN128);
  bool nibbleCodebookVLEN256 =
      leaves.contains(IntrinsicCLeaf::NibbleCodebookI8VLEN256);
  bool base3VLEN128 =
      leaves.contains(IntrinsicCLeaf::Base3TernaryI8VLEN128);
  bool base3VLEN256 =
      leaves.contains(IntrinsicCLeaf::Base3TernaryI8VLEN256);
  bool packedI2VLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedI2TernaryI8VLEN128);
  bool packedI2VLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedI2TernaryI8VLEN256);
  bool signedCodebook8VLEN128 =
      leaves.contains(IntrinsicCLeaf::SignedCodebook8I8VLEN128);
  bool signedCodebook8VLEN256 =
      leaves.contains(IntrinsicCLeaf::SignedCodebook8I8VLEN256);
  bool signedCodebook4VLEN128 =
      leaves.contains(IntrinsicCLeaf::SignedCodebook4I8VLEN128);
  bool signedCodebook4VLEN256 =
      leaves.contains(IntrinsicCLeaf::SignedCodebook4I8VLEN256);
  bool packedU9U7CodebookVLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedU9U7CodebookI8VLEN128);
  bool packedU9U7CodebookVLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedU9U7CodebookI8VLEN256);
  bool packedU11GridDeltaVLEN128 =
      leaves.contains(IntrinsicCLeaf::PackedU11GridDeltaI8VLEN128);
  bool packedU11GridDeltaVLEN256 =
      leaves.contains(IntrinsicCLeaf::PackedU11GridDeltaI8VLEN256);
  if (!iq2FixedLanes32 && !iq2FixedLanes64 && !iq2Scalable &&
      !iq3FixedLanes64 && !iq3Scalable && !iq1FixedLanes32 &&
      !iq1FixedLanes64 && !iq1Scalable && !q6FixedLanes32 &&
      !q6FixedLanes64 && !q6Scalable && !packedI4VLEN128 &&
      !packedI4VLEN256 && !packedI5VLEN128 && !packedI5VLEN256 &&
      !packedI3GroupedVLEN128 && !packedI3GroupedVLEN256 &&
      !nibbleCodebookVLEN128 && !nibbleCodebookVLEN256 &&
      !base3VLEN128 && !base3VLEN256 &&
      !packedI2VLEN128 && !packedI2VLEN256 &&
      !signedCodebook8VLEN128 && !signedCodebook8VLEN256 &&
      !signedCodebook4VLEN128 && !signedCodebook4VLEN256 &&
      !packedU9U7CodebookVLEN128 && !packedU9U7CodebookVLEN256 &&
      !packedU11GridDeltaVLEN128 && !packedU11GridDeltaVLEN256)
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
  emitPackedI4IntrinsicCLeaves(output, packedI4VLEN128, packedI4VLEN256);
  emitPackedI5IntrinsicCLeaves(output, packedI5VLEN128, packedI5VLEN256);
  emitPackedI3GroupedIntrinsicCLeaves(
      output, packedI3GroupedVLEN128, packedI3GroupedVLEN256);
  emitNibbleCodebookIntrinsicCLeaves(output, nibbleCodebookVLEN128,
                                     nibbleCodebookVLEN256);
  emitTernaryIntrinsicCLeaves(output, base3VLEN128, base3VLEN256,
                              packedI2VLEN128, packedI2VLEN256);
  emitSignedCodebookIntrinsicCLeaves(
      output, signedCodebook8VLEN128, signedCodebook8VLEN256,
      signedCodebook4VLEN128, signedCodebook4VLEN256);
  emitPackedU9U7CodebookIntrinsicCLeaves(
      output, packedU9U7CodebookVLEN128, packedU9U7CodebookVLEN256);
  emitPackedU11GridDeltaIntrinsicCLeaves(
      output, packedU11GridDeltaVLEN128, packedU11GridDeltaVLEN256);
}

} // namespace weft::riscv_internal
