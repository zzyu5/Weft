#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace weft::riscv_internal {

namespace {

bool emitPackedDotLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation,
    bool hasHighBits) {
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  if (!isPackedDotLocalImplementationMapping(implementation) || !reduction ||
      implementation.helperSymbol.empty())
    return false;

  const RVVVectorShape shape = implementation.mapping.laneShape;
  const RVVVectorShape decodeShape = implementation.valueShapes[1];
  const RVVVectorShape widened{16, shape.lmulEighths * 2};
  const std::string unsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, shape);
  const std::string signedType =
      rvvVectorType(RVVElementCategory::SignedInteger, shape);
  const std::string signedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, shape);
  const std::string unsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, shape);
  const std::string widenedType =
      rvvVectorType(RVVElementCategory::SignedInteger, widened);
  const std::string widenedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, widened);
  const std::string setVL = rvvSetVLIntrinsic(shape);
  const std::string decodeUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, decodeShape);
  const std::string decodeSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, decodeShape);
  const std::string decodeUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, decodeShape);
  const std::string decodeSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, decodeShape);
  const std::string decodeSetVL = rvvSetVLIntrinsic(decodeShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(shape);
  if (unsignedType.empty() || signedType.empty() || signedSuffix.empty() ||
      unsignedSuffix.empty() || widenedType.empty() || widenedSuffix.empty() ||
      setVL.empty() || decodeUnsignedType.empty() || decodeSignedType.empty() ||
      decodeUnsignedSuffix.empty() || decodeSignedSuffix.empty() ||
      decodeSetVL.empty() || (hasHighBits && !maskRatio))
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n";
  if (hasHighBits)
    output << "    const uint8_t *low_bits, const uint8_t *high_bits,\n";
  else
    output << "    const uint8_t *packed_codes,\n";
  output << "    const uint8_t *activation_bytes, int32_t zero_point,\n"
         << "    float dot_scale, float additive_bias, float init) {\n";

  if (reduction->laneFactor == 32) {
    output << "  const size_t vl16 = " << decodeSetVL << "(16);\n"
           << "  const " << decodeUnsignedType
           << " packed = __riscv_vle8_v_" << decodeUnsignedSuffix << "("
           << (hasHighBits ? "low_bits" : "packed_codes")
           << ", vl16);\n"
           << "  const " << decodeSignedType
           << " low = __riscv_vreinterpret_v_" << decodeUnsignedSuffix << "_"
           << decodeSignedSuffix << "(__riscv_vand_vx_" << decodeUnsignedSuffix
           << "(packed, UINT8_C(15), vl16));\n"
           << "  const " << decodeSignedType
           << " high = __riscv_vreinterpret_v_" << decodeUnsignedSuffix << "_"
           << decodeSignedSuffix << "(__riscv_vsrl_vx_" << decodeUnsignedSuffix
           << "(packed, 4, vl16));\n"
           << "  const size_t vl32 = " << setVL << "(32);\n";
    if (decodeShape == shape)
      output << "  " << signedType << " codes = __riscv_vslideup_vx_"
             << signedSuffix << "(low, high, 16, vl32);\n";
    else
      output << "  " << signedType << " codes = __riscv_vcreate_v_"
             << decodeSignedSuffix << "_" << signedSuffix << "(low, high);\n";
    if (hasHighBits)
      output << "  const " << rvvMaskType(*maskRatio)
             << " high_mask = __riscv_vlm_v_" << rvvMaskSuffix(*maskRatio)
             << "(high_bits, vl32);\n"
             << "  codes = __riscv_vor_vx_" << signedSuffix
             << "_mu(high_mask, codes, codes, INT8_C(16), vl32);\n";
    output << "  codes = __riscv_vsub_vx_" << signedSuffix
           << "(codes, zero_point, vl32);\n"
           << "  const " << signedType
           << " activation = __riscv_vle8_v_" << signedSuffix
           << "((const int8_t *)(const void *)activation_bytes, vl32);\n"
           << "  const " << widenedType << " products = __riscv_vwmul_vv_"
           << widenedSuffix << "(codes, activation, vl32);\n";
  } else {
    for (unsigned chunk = 0; chunk < reduction->registerFactor; ++chunk) {
      const unsigned start = chunk * reduction->laneFactor;
      const unsigned packedOffset = start % 16;
      const bool highNibble = start >= 16;
      output << "  const size_t vl" << chunk << " = " << setVL << "(16);\n"
             << "  const " << unsignedType << " packed" << chunk
             << " = __riscv_vle8_v_" << unsignedSuffix << "("
             << (hasHighBits ? "low_bits" : "packed_codes");
      if (packedOffset != 0)
        output << " + " << packedOffset;
      output << ", vl" << chunk << ");\n"
             << "  " << signedType << " codes" << chunk
             << " = __riscv_vreinterpret_v_" << unsignedSuffix << "_"
             << signedSuffix << "(__riscv_"
             << (highNibble ? "vsrl_vx_" : "vand_vx_") << unsignedSuffix
             << "(packed" << chunk << ", "
             << (highNibble ? "4" : "UINT8_C(15)") << ", vl" << chunk
             << "));\n";
      if (hasHighBits)
        output << "  const " << rvvMaskType(*maskRatio) << " high_mask" << chunk
               << " = __riscv_vlm_v_" << rvvMaskSuffix(*maskRatio)
               << "(high_bits + " << start / 8 << ", vl" << chunk << ");\n"
               << "  codes" << chunk << " = __riscv_vor_vx_" << signedSuffix
               << "_mu(high_mask" << chunk << ", codes" << chunk << ", codes"
               << chunk << ", INT8_C(16), vl" << chunk << ");\n";
      output << "  codes" << chunk << " = __riscv_vsub_vx_" << signedSuffix
             << "(codes" << chunk << ", zero_point, vl" << chunk << ");\n"
             << "  const " << signedType << " activation" << chunk
             << " = __riscv_vle8_v_" << signedSuffix
             << "((const int8_t *)(const void *)(activation_bytes + " << start
             << "), vl" << chunk << ");\n"
             << "  const " << widenedType << " products" << chunk
             << " = __riscv_vwmul_vv_" << widenedSuffix << "(codes" << chunk
             << ", activation" << chunk << ", vl" << chunk << ");\n";
    }
    output << "  " << widenedType << " products = products0;\n";
    for (unsigned chunk = 1; chunk < reduction->registerFactor; ++chunk)
      output << "  products = __riscv_vadd_vv_" << widenedSuffix
             << "(products, products" << chunk << ", vl0);\n";
  }
  output << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << widenedSuffix
         << "_i32m1(products, zero, "
         << (reduction->laneFactor == 32 ? "vl32" : "vl0") << "));\n"
         << "  return init + dot_scale * (float)integer_dot + additive_bias;\n"
         << "}\n\n";
  return true;
}

} // namespace

bool emitPackedI4LocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  return implementation.primitive == LocalPrimitiveKind::PackedI4I8 &&
         emitPackedDotLocalImplementation(output, implementation, false);
}

bool emitPackedI5LocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  return implementation.primitive == LocalPrimitiveKind::PackedI5I8 &&
         emitPackedDotLocalImplementation(output, implementation, true);
}

void emitNibbleCodebookLocalImplementations(llvm::raw_ostream &output,
                                        bool registerE8M2, bool registerE8M1) {
  if (registerE8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_nibble_codebook_i8_register_l32_e8m2(
    const uint8_t *packed_codes, const uint8_t *table_bytes,
    const uint8_t *activation_bytes, float dot_scale, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(packed_codes, vl16);
  const vint8m2_t table = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)table_bytes, vl16);
  const vuint8m1_t low_index =
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16);
  const vuint8m1_t high_index =
      __riscv_vsrl_vx_u8m1(packed, 4, vl16);
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t index =
      __riscv_vcreate_v_u8m1_u8m2(low_index, high_index);
  const vint8m2_t decoded =
      __riscv_vrgather_vv_i8m2(table, index, vl32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m4_t product =
      __riscv_vwmul_vv_i16m4(decoded, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m4_i32m1(product, zero, vl32));
  return init + dot_scale * (float)integer_dot;
}

)c";
  }

  if (registerE8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_nibble_codebook_i8_register_l32_e8m1(
    const uint8_t *packed_codes, const uint8_t *table_bytes,
    const uint8_t *activation_bytes, float dot_scale, float init) {
  const size_t vl16 = __riscv_vsetvl_e8mf2(16);
  const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(packed_codes, vl16);
  const vint8mf2_t table = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)table_bytes, vl16);
  const vuint8mf2_t low_index =
      __riscv_vand_vx_u8mf2(packed, UINT8_C(15), vl16);
  const vuint8mf2_t high_index =
      __riscv_vsrl_vx_u8mf2(packed, 4, vl16);
  const vint8mf2_t low =
      __riscv_vrgather_vv_i8mf2(table, low_index, vl16);
  const vint8mf2_t high =
      __riscv_vrgather_vv_i8mf2(table, high_index, vl16);
  const vint8mf2_t activation_low = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)activation_bytes, vl16);
  const vint8mf2_t activation_high = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)(activation_bytes + 16), vl16);
  const vint16m1_t product = __riscv_vadd_vv_i16m1(
      __riscv_vwmul_vv_i16m1(low, activation_low, vl16),
      __riscv_vwmul_vv_i16m1(high, activation_high, vl16), vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(product, zero, vl16));
  return init + dot_scale * (float)integer_dot;
}

)c";
  }
}

} // namespace weft::riscv_internal
