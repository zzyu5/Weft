#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitBase3TernaryLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::Base3TernaryI8 ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVIntrinsic ||
      implementation.valueShapes.size() < 5 || !implementation.schedule ||
      implementation.schedule.decode.chunksPerStep != 5)
    return false;
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape halfShape = implementation.valueShapes[1];
  const RVVVectorShape widenedShape = implementation.valueShapes[2];
  const RVVVectorShape halfWidenedShape = implementation.valueShapes[3];
  const RVVVectorShape highWordShape = implementation.valueShapes[4];

  const std::string laneUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string widenedUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, widenedShape);
  const std::string widenedUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, widenedShape);
  const std::string widenedSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, widenedShape);
  const std::string widenedSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, widenedShape);
  const std::string halfUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, halfShape);
  const std::string halfUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, halfShape);
  const std::string halfSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, halfShape);
  const std::string halfSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, halfShape);
  const std::string halfWidenedUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, halfWidenedShape);
  const std::string halfWidenedUnsignedSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::UnsignedInteger, halfWidenedShape);
  const std::string halfWidenedSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, halfWidenedShape);
  const std::string halfWidenedSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger,
                             halfWidenedShape);
  const std::string highWordType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, highWordShape);
  const std::string highWordSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger,
                             highWordShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  const std::string halfSetVL = rvvSetVLIntrinsic(halfShape);
  if (laneUnsignedType.empty() || laneUnsignedSuffix.empty() ||
      laneSignedType.empty() || laneSignedSuffix.empty() ||
      widenedUnsignedType.empty() || widenedUnsignedSuffix.empty() ||
      widenedSignedType.empty() || widenedSignedSuffix.empty() ||
      halfUnsignedType.empty() || halfUnsignedSuffix.empty() ||
      halfSignedType.empty() || halfSignedSuffix.empty() ||
      halfWidenedUnsignedType.empty() || halfWidenedUnsignedSuffix.empty() ||
      halfWidenedSignedType.empty() || halfWidenedSignedSuffix.empty() ||
      highWordType.empty() || highWordSuffix.empty() || laneSetVL.empty() ||
      halfSetVL.empty())
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *codes, const uint8_t *high_digits,\n"
         << "    const uint8_t *activation_bytes, float weight_scale,\n"
         << "    float activation_scale, float init) {\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  const uint8_t powers[16] = {\n"
         << "      1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, "
            "27, 27, 27, 27};\n\n"
         << "  const size_t vl32 = " << laneSetVL << "(32);\n"
         << "  const " << laneUnsignedType
         << " packed32 = __riscv_vle8_v_" << laneUnsignedSuffix
         << "(codes, vl32);\n"
         << "  const " << widenedUnsignedType
         << " digit0 = __riscv_vsrl_vx_" << widenedUnsignedSuffix << "(\n"
         << "      __riscv_vwmulu_vx_" << widenedUnsignedSuffix
         << "(packed32, 3, vl32), 8, vl32);\n"
         << "  const " << widenedSignedType
         << " activation0 = __riscv_vwcvt_x_x_v_" << widenedSignedSuffix
         << "(\n      __riscv_vle8_v_" << laneSignedSuffix
         << "(activation, vl32), vl32);\n"
         << "  " << widenedSignedType << " sum32 = __riscv_vmul_vv_"
         << widenedSignedSuffix << "(\n"
         << "      __riscv_vreinterpret_v_" << widenedUnsignedSuffix << "_"
         << widenedSignedSuffix << "(\n"
         << "          __riscv_vsub_vx_" << widenedUnsignedSuffix
         << "(digit0, 1, vl32)),\n      activation0, vl32);\n"
         << "  uint8_t power = 3;\n"
         << "  for (size_t digit = 1; digit < "
         << implementation.schedule.decode.chunksPerStep << "; ++digit) {\n"
         << "    const " << widenedUnsignedType
         << " decoded = __riscv_vsrl_vx_" << widenedUnsignedSuffix << "(\n"
         << "        __riscv_vwmulu_vx_" << widenedUnsignedSuffix << "(\n"
         << "            __riscv_vmul_vx_" << laneUnsignedSuffix
         << "(packed32, power, vl32), 3, vl32),\n        8, vl32);\n"
         << "    const " << widenedSignedType
         << " values = __riscv_vwcvt_x_x_v_" << widenedSignedSuffix << "(\n"
         << "        __riscv_vle8_v_" << laneSignedSuffix
         << "(activation + digit * 32, vl32), vl32);\n"
         << "    sum32 = __riscv_vmacc_vv_" << widenedSignedSuffix
         << "(sum32,\n"
         << "        __riscv_vreinterpret_v_" << widenedUnsignedSuffix << "_"
         << widenedSignedSuffix << "(\n"
         << "            __riscv_vsub_vx_" << widenedUnsignedSuffix
         << "(decoded, 1, vl32)),\n        values, vl32);\n"
         << "    power = (uint8_t)(power * 3);\n"
         << "  }\n\n"
         << "  const size_t vl16 = " << halfSetVL << "(16);\n"
         << "  const " << halfUnsignedType
         << " packed16 = __riscv_vle8_v_" << halfUnsignedSuffix
         << "(codes + 32, vl16);\n"
         << "  const " << halfWidenedUnsignedType
         << " tail0 = __riscv_vsrl_vx_" << halfWidenedUnsignedSuffix << "(\n"
         << "      __riscv_vwmulu_vx_" << halfWidenedUnsignedSuffix
         << "(packed16, 3, vl16), 8, vl16);\n"
         << "  const " << halfWidenedSignedType
         << " tailActivation0 = __riscv_vwcvt_x_x_v_"
         << halfWidenedSignedSuffix << "(\n"
         << "      __riscv_vle8_v_" << halfSignedSuffix
         << "(activation + 160, vl16), vl16);\n"
         << "  " << halfWidenedSignedType
         << " sum16 = __riscv_vmul_vv_" << halfWidenedSignedSuffix << "(\n"
         << "      __riscv_vreinterpret_v_" << halfWidenedUnsignedSuffix << "_"
         << halfWidenedSignedSuffix << "(\n"
         << "          __riscv_vsub_vx_" << halfWidenedUnsignedSuffix
         << "(tail0, 1, vl16)),\n      tailActivation0, vl16);\n"
         << "  power = 3;\n"
         << "  for (size_t digit = 1; digit < "
         << implementation.schedule.decode.chunksPerStep << "; ++digit) {\n"
         << "    const " << halfWidenedUnsignedType
         << " decoded = __riscv_vsrl_vx_" << halfWidenedUnsignedSuffix << "(\n"
         << "        __riscv_vwmulu_vx_" << halfWidenedUnsignedSuffix << "(\n"
         << "            __riscv_vmul_vx_" << halfUnsignedSuffix
         << "(packed16, power, vl16), 3, vl16),\n        8, vl16);\n"
         << "    const " << halfWidenedSignedType
         << " values = __riscv_vwcvt_x_x_v_" << halfWidenedSignedSuffix
         << "(\n        __riscv_vle8_v_" << halfSignedSuffix
         << "(activation + 160 + digit * 16, vl16), vl16);\n"
         << "    sum16 = __riscv_vmacc_vv_" << halfWidenedSignedSuffix
         << "(sum16,\n"
         << "        __riscv_vreinterpret_v_" << halfWidenedUnsignedSuffix << "_"
         << halfWidenedSignedSuffix << "(\n"
         << "            __riscv_vsub_vx_" << halfWidenedUnsignedSuffix
         << "(decoded, 1, vl16)),\n        values, vl16);\n"
         << "    power = (uint8_t)(power * 3);\n"
         << "  }\n\n"
         << "  uint32_t packed_high = (uint32_t)high_digits[0] |\n"
         << "      ((uint32_t)high_digits[1] << 8) |\n"
         << "      ((uint32_t)high_digits[2] << 16) |\n"
         << "      ((uint32_t)high_digits[3] << 24);\n"
         << "  __asm__ __volatile__(\"\" : \"+r\"(packed_high));\n"
         << "  const " << halfUnsignedType
         << " high = __riscv_vreinterpret_v_" << highWordSuffix << "_"
         << halfUnsignedSuffix << "(\n"
         << "      __riscv_vmv_v_x_" << highWordSuffix
         << "(packed_high, 4));\n"
         << "  const " << halfUnsignedType
         << " high_power = __riscv_vle8_v_" << halfUnsignedSuffix
         << "(powers, vl16);\n"
         << "  const " << halfWidenedUnsignedType
         << " high_decoded = __riscv_vsrl_vx_" << halfWidenedUnsignedSuffix
         << "(\n      __riscv_vwmulu_vx_" << halfWidenedUnsignedSuffix
         << "(\n          __riscv_vmul_vv_" << halfUnsignedSuffix
         << "(high, high_power, vl16), 3, vl16),\n      8, vl16);\n"
         << "  const " << halfWidenedSignedType
         << " high_activation = __riscv_vwcvt_x_x_v_"
         << halfWidenedSignedSuffix << "(\n"
         << "      __riscv_vle8_v_" << halfSignedSuffix
         << "(activation + 240, vl16), vl16);\n"
         << "  const " << halfWidenedSignedType
         << " high_sum = __riscv_vmul_vv_" << halfWidenedSignedSuffix << "(\n"
         << "      __riscv_vreinterpret_v_" << halfWidenedUnsignedSuffix << "_"
         << halfWidenedSignedSuffix << "(\n"
         << "          __riscv_vsub_vx_" << halfWidenedUnsignedSuffix
         << "(high_decoded, 1, vl16)),\n      high_activation, vl16);\n\n"
         << "  " << halfWidenedSignedType
         << " combined = __riscv_vadd_vv_" << halfWidenedSignedSuffix << "(\n"
         << "      __riscv_vget_v_" << widenedSignedSuffix << "_"
         << halfWidenedSignedSuffix << "(sum32, 0),\n"
         << "      __riscv_vget_v_" << widenedSignedSuffix << "_"
         << halfWidenedSignedSuffix << "(sum32, 1), vl16);\n"
         << "  combined = __riscv_vadd_vv_" << halfWidenedSignedSuffix
         << "(combined, sum16, vl16);\n"
         << "  combined = __riscv_vadd_vv_" << halfWidenedSignedSuffix
         << "(combined, high_sum, vl16);\n"
         << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t integer_sum = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << halfWidenedSignedSuffix
         << "_i32m1(combined, zero, vl16));\n"
         << "  return init + (float)integer_sum * weight_scale * "
            "activation_scale;\n"
         << "}\n\n";
  return true;
}

bool emitPackedI2TernaryLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::PackedI2TernaryI8 ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVIntrinsic ||
      implementation.valueShapes.size() < 2 || !implementation.schedule ||
      implementation.schedule.decode.chunksPerStep == 0)
    return false;
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
  if (unsignedType.empty() || unsignedSuffix.empty() ||
      signedType.empty() || signedSuffix.empty() || widenedType.empty() ||
      widenedSuffix.empty() || setVL.empty())
    return false;

  const unsigned logicalChunk = implementation.schedule.iterationElements;
  const unsigned halfCount =
      implementation.schedule.sequentialIterations /
      implementation.schedule.decode.chunksPerStep;
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *codes, const uint8_t *activation_bytes,\n"
         << "    float weight_scale, float activation_scale, float init) {\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  const size_t vl = " << setVL << "("
         << implementation.schedule.laneFactor << ");\n"
         << "  int32_t integer_sum = 0;\n"
         << "  for (size_t half = 0; half < " << halfCount << "; ++half) {\n"
         << "    for (size_t repetition = 0; repetition < "
         << implementation.schedule.iterationRegisterFactor
         << "; ++repetition) {\n"
         << "      const " << unsignedType << " packed = __riscv_vle8_v_"
         << unsignedSuffix << "(codes + half * " << logicalChunk
         << " + repetition * " << implementation.schedule.laneFactor
         << ", vl);\n"
         << "      " << widenedType << " accumulator = __riscv_vmv_v_x_"
         << widenedSuffix << "(0, vl);\n"
         << "      for (size_t field = 0; field < "
         << implementation.schedule.decode.chunksPerStep << "; ++field) {\n"
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
         << " + repetition * " << implementation.schedule.laneFactor
         << ", vl);\n"
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
