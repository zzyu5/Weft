#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitBase3TernaryLocalImplementations(llvm::raw_ostream &output,
                                         bool base3E8M2, bool base3E8M1) {
  if (base3E8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_base3_ternary_i8_register_l32_e8m2(
    const uint8_t *codes, const uint8_t *high_digits,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint8_t powers[16] = {
      1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t packed32 = __riscv_vle8_v_u8m2(codes, vl32);
  const vuint16m4_t digit0 = __riscv_vsrl_vx_u16m4(
      __riscv_vwmulu_vx_u16m4(packed32, 3, vl32), 8, vl32);
  const vint16m4_t activation0 = __riscv_vwcvt_x_x_v_i16m4(
      __riscv_vle8_v_i8m2(activation, vl32), vl32);
  vint16m4_t sum32 = __riscv_vmul_vv_i16m4(
      __riscv_vreinterpret_v_u16m4_i16m4(
          __riscv_vsub_vx_u16m4(digit0, 1, vl32)),
      activation0, vl32);
  uint8_t power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m4_t decoded = __riscv_vsrl_vx_u16m4(
        __riscv_vwmulu_vx_u16m4(
            __riscv_vmul_vx_u8m2(packed32, power, vl32), 3, vl32),
        8, vl32);
    const vint16m4_t values = __riscv_vwcvt_x_x_v_i16m4(
        __riscv_vle8_v_i8m2(activation + digit * 32, vl32), vl32);
    sum32 = __riscv_vmacc_vv_i16m4(
        sum32,
        __riscv_vreinterpret_v_u16m4_i16m4(
            __riscv_vsub_vx_u16m4(decoded, 1, vl32)),
        values, vl32);
    power = (uint8_t)(power * 3);
  }

  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed16 = __riscv_vle8_v_u8m1(codes + 32, vl16);
  const vuint16m2_t tail0 = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(packed16, 3, vl16), 8, vl16);
  const vint16m2_t tailActivation0 = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation + 160, vl16), vl16);
  vint16m2_t sum16 = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(tail0, 1, vl16)),
      tailActivation0, vl16);
  power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m2_t decoded = __riscv_vsrl_vx_u16m2(
        __riscv_vwmulu_vx_u16m2(
            __riscv_vmul_vx_u8m1(packed16, power, vl16), 3, vl16),
        8, vl16);
    const vint16m2_t values = __riscv_vwcvt_x_x_v_i16m2(
        __riscv_vle8_v_i8m1(activation + 160 + digit * 16, vl16), vl16);
    sum16 = __riscv_vmacc_vv_i16m2(
        sum16,
        __riscv_vreinterpret_v_u16m2_i16m2(
            __riscv_vsub_vx_u16m2(decoded, 1, vl16)),
        values, vl16);
    power = (uint8_t)(power * 3);
  }

  uint32_t packed_high = (uint32_t)high_digits[0] |
      ((uint32_t)high_digits[1] << 8) |
      ((uint32_t)high_digits[2] << 16) |
      ((uint32_t)high_digits[3] << 24);
  __asm__ __volatile__("" : "+r"(packed_high));
  const vuint8m1_t high = __riscv_vreinterpret_v_u32m1_u8m1(
      __riscv_vmv_v_x_u32m1(packed_high, 4));
  const vuint8m1_t high_power = __riscv_vle8_v_u8m1(powers, vl16);
  const vuint16m2_t high_decoded = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(
          __riscv_vmul_vv_u8m1(high, high_power, vl16), 3, vl16),
      8, vl16);
  const vint16m2_t high_activation = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation + 240, vl16), vl16);
  const vint16m2_t high_sum = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(high_decoded, 1, vl16)),
      high_activation, vl16);

  vint16m2_t combined = __riscv_vadd_vv_i16m2(
      __riscv_vget_v_i16m4_i16m2(sum32, 0),
      __riscv_vget_v_i16m4_i16m2(sum32, 1), vl16);
  combined = __riscv_vadd_vv_i16m2(combined, sum16, vl16);
  combined = __riscv_vadd_vv_i16m2(combined, high_sum, vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(combined, zero, vl16));
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }

  if (base3E8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_base3_ternary_i8_register_l32_e8m1(
    const uint8_t *codes, const uint8_t *high_digits,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint8_t powers[16] = {
      1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  const vuint8m1_t packed32 = __riscv_vle8_v_u8m1(codes, vl32);
  const vuint16m2_t digit0 = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(packed32, 3, vl32), 8, vl32);
  const vint16m2_t activation0 = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation, vl32), vl32);
  vint16m2_t sum32 = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(digit0, 1, vl32)),
      activation0, vl32);
  uint8_t power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m2_t decoded = __riscv_vsrl_vx_u16m2(
        __riscv_vwmulu_vx_u16m2(
            __riscv_vmul_vx_u8m1(packed32, power, vl32), 3, vl32),
        8, vl32);
    const vint16m2_t values = __riscv_vwcvt_x_x_v_i16m2(
        __riscv_vle8_v_i8m1(activation + digit * 32, vl32), vl32);
    sum32 = __riscv_vmacc_vv_i16m2(
        sum32,
        __riscv_vreinterpret_v_u16m2_i16m2(
            __riscv_vsub_vx_u16m2(decoded, 1, vl32)),
        values, vl32);
    power = (uint8_t)(power * 3);
  }

  const size_t vl16 = __riscv_vsetvl_e8mf2(16);
  const vuint8mf2_t packed16 = __riscv_vle8_v_u8mf2(codes + 32, vl16);
  const vuint16m1_t tail0 = __riscv_vsrl_vx_u16m1(
      __riscv_vwmulu_vx_u16m1(packed16, 3, vl16), 8, vl16);
  const vint16m1_t tailActivation0 = __riscv_vwcvt_x_x_v_i16m1(
      __riscv_vle8_v_i8mf2(activation + 160, vl16), vl16);
  vint16m1_t sum16 = __riscv_vmul_vv_i16m1(
      __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vsub_vx_u16m1(tail0, 1, vl16)),
      tailActivation0, vl16);
  power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m1_t decoded = __riscv_vsrl_vx_u16m1(
        __riscv_vwmulu_vx_u16m1(
            __riscv_vmul_vx_u8mf2(packed16, power, vl16), 3, vl16),
        8, vl16);
    const vint16m1_t values = __riscv_vwcvt_x_x_v_i16m1(
        __riscv_vle8_v_i8mf2(activation + 160 + digit * 16, vl16), vl16);
    sum16 = __riscv_vmacc_vv_i16m1(
        sum16,
        __riscv_vreinterpret_v_u16m1_i16m1(
            __riscv_vsub_vx_u16m1(decoded, 1, vl16)),
        values, vl16);
    power = (uint8_t)(power * 3);
  }

  uint32_t packed_high = (uint32_t)high_digits[0] |
      ((uint32_t)high_digits[1] << 8) |
      ((uint32_t)high_digits[2] << 16) |
      ((uint32_t)high_digits[3] << 24);
  __asm__ __volatile__("" : "+r"(packed_high));
  const vuint8mf2_t high = __riscv_vreinterpret_v_u32mf2_u8mf2(
      __riscv_vmv_v_x_u32mf2(packed_high, 4));
  const vuint8mf2_t high_power = __riscv_vle8_v_u8mf2(powers, vl16);
  const vuint16m1_t high_decoded = __riscv_vsrl_vx_u16m1(
      __riscv_vwmulu_vx_u16m1(
          __riscv_vmul_vv_u8mf2(high, high_power, vl16), 3, vl16),
      8, vl16);
  const vint16m1_t high_activation = __riscv_vwcvt_x_x_v_i16m1(
      __riscv_vle8_v_i8mf2(activation + 240, vl16), vl16);
  const vint16m1_t high_sum = __riscv_vmul_vv_i16m1(
      __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vsub_vx_u16m1(high_decoded, 1, vl16)),
      high_activation, vl16);

  vint16m1_t combined = __riscv_vadd_vv_i16m1(
      __riscv_vget_v_i16m2_i16m1(sum32, 0),
      __riscv_vget_v_i16m2_i16m1(sum32, 1), vl16);
  combined = __riscv_vadd_vv_i16m1(combined, sum16, vl16);
  combined = __riscv_vadd_vv_i16m1(combined, high_sum, vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(combined, zero, vl16));
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }

}

bool emitPackedI2TernaryLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (!isPackedI2TernaryLocalImplementationMapping(implementation) ||
      implementation.helperSymbol.empty())
    return false;
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape widenedShape = implementation.valueShapes[1];
  const std::string unsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string unsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string signedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string signedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string widenedType =
      rvvVectorType(RVVElementCategory::SignedInteger, widenedShape);
  const std::string widenedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, widenedShape);
  const std::string setVL = rvvSetVLIntrinsic(laneShape);
  if (!reduction || unsignedType.empty() || unsignedSuffix.empty() ||
      signedType.empty() || signedSuffix.empty() || widenedType.empty() ||
      widenedSuffix.empty() || setVL.empty())
    return false;

  const unsigned logicalChunk =
      reduction->laneFactor * reduction->registerFactor;
  const unsigned halfCount = reduction->sequentialFactor / 4;
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n"
         << "    const uint8_t *codes, const uint8_t *activation_bytes,\n"
         << "    float weight_scale, float activation_scale, float init) {\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  const size_t vl = " << setVL << "("
         << reduction->laneFactor << ");\n"
         << "  int32_t integer_sum = 0;\n"
         << "  for (size_t half = 0; half < " << halfCount << "; ++half) {\n"
         << "    for (size_t repetition = 0; repetition < "
         << reduction->registerFactor << "; ++repetition) {\n"
         << "      const " << unsignedType << " packed = __riscv_vle8_v_"
         << unsignedSuffix << "(codes + half * " << logicalChunk
         << " + repetition * " << reduction->laneFactor << ", vl);\n"
         << "      " << widenedType << " accumulator = __riscv_vmv_v_x_"
         << widenedSuffix << "(0, vl);\n"
         << "      for (size_t field = 0; field < 4; ++field) {\n"
         << "        " << unsignedType << " unpacked = __riscv_vsrl_vx_"
         << unsignedSuffix << "(packed, 2 * field, vl);\n"
         << "        if (field != 3)\n"
         << "          unpacked = __riscv_vand_vx_" << unsignedSuffix
         << "(unpacked, 3, vl);\n"
         << "        const " << signedType
         << " ternary = __riscv_vsub_vx_" << signedSuffix
         << "(__riscv_vreinterpret_v_" << unsignedSuffix << "_"
         << signedSuffix << "(unpacked), 1, vl);\n"
         << "        const " << signedType
         << " values = __riscv_vle8_v_" << signedSuffix
         << "(activation + (half * 4 + field) * " << logicalChunk
         << " + repetition * " << reduction->laneFactor << ", vl);\n"
         << "        accumulator = __riscv_vwmacc_vv_" << widenedSuffix
         << "(accumulator, ternary, values, vl);\n"
         << "      }\n"
         << "      const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "      integer_sum += __riscv_vmv_x_s_i32m1_i32(\n"
         << "          __riscv_vwredsum_vs_" << widenedSuffix
         << "_i32m1(accumulator, zero, vl));\n"
         << "    }\n"
         << "  }\n"
         << "  return init + (float)integer_sum * weight_scale * "
            "activation_scale;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
