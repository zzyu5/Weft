#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitIQ1LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::IQ1MI8 ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVRegister ||
      implementation.valueShapes.size() < 6)
    return false;
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape indexShape = implementation.valueShapes[1];
  const RVVVectorShape tableShape = implementation.valueShapes[2];
  const RVVVectorShape deltaWordShape = implementation.valueShapes[3];
  const RVVVectorShape productShape = implementation.valueShapes[4];
  const RVVVectorShape segmentProductShape = implementation.valueShapes[5];
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string indexType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, indexShape);
  const std::string indexSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, indexShape);
  const std::string tableType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string tableSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string deltaWordType =
      rvvVectorType(RVVElementCategory::SignedInteger, deltaWordShape);
  const std::string deltaWordSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, deltaWordShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string segmentProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, segmentProductShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  if (!reduction || laneSignedType.empty() || laneSignedSuffix.empty() ||
      laneUnsignedSuffix.empty() || indexType.empty() || indexSuffix.empty() ||
      tableType.empty() || tableSuffix.empty() || deltaWordType.empty() ||
      deltaWordSuffix.empty() || productType.empty() || productSuffix.empty() ||
      segmentProductSuffix.empty() || laneSetVL.empty())
    return false;

  const unsigned groupsPerVector = reduction->laneFactor / 32;
  const unsigned vectorCount = 4 * groupsPerVector;
  const unsigned segmentCount = 2 * groupsPerVector;
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *codes, const uint8_t *high_delta_bits,\n"
         << "    const uint8_t *scales, const uint8_t *activation_bytes,\n"
         << "    float activation_scale, float init) {\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  uint16_t scale_words[4];\n"
         << "  for (size_t word = 0; word < 4; ++word)\n"
         << "    scale_words[word] = (uint16_t)scales[2 * word] |\n"
         << "                        ((uint16_t)scales[2 * word + 1] << 8);\n"
         << "  const uint16_t scale_bits =\n"
         << "      (scale_words[0] >> 12) | ((scale_words[1] >> 8) & 0x00f0) |\n"
         << "      ((scale_words[2] >> 4) & 0x0f00) | (scale_words[3] & 0xf000);\n"
         << "  const size_t vl = " << laneSetVL << "("
         << reduction->laneFactor << ");\n"
         << "  int32_t grid_sum = 0;\n"
         << "  int32_t delta_sum = 0;\n"
         << "  for (size_t batch = 0; batch < "
         << reduction->sequentialFactor << "; ++batch) {\n"
         << "    const size_t first_group = batch * " << groupsPerVector
         << ";\n"
         << "    uint16_t byte_offsets[" << vectorCount << "];\n"
         << "    int64_t delta_words[" << vectorCount << "];\n"
         << "    for (size_t vector = 0; vector < " << vectorCount
         << "; ++vector) {\n"
         << "      const size_t group = first_group + vector / 4;\n"
         << "      const size_t within_group = vector % 4;\n"
         << "      const uint8_t high = high_delta_bits[group * 2 + "
            "within_group / 2];\n"
         << "      const uint16_t index = (uint16_t)codes[group * 4 + "
            "within_group] |\n"
         << "          (uint16_t)(((uint16_t)high << "
            "(8 - 4 * (within_group & 1))) & 0x700);\n"
         << "      byte_offsets[vector] = (uint16_t)(index * 8);\n"
         << "      delta_words[vector] =\n"
         << "          high & ((within_group & 1) ? UINT8_C(0x80) : "
            "UINT8_C(0x08))\n"
         << "              ? INT64_C(-1) : INT64_C(0x0101010101010101);\n"
         << "    }\n"
         << "    const " << indexType << " offsets = __riscv_vle16_v_"
         << indexSuffix << "(byte_offsets, " << vectorCount << ");\n"
         << "    const " << tableType
         << " packed_grid = __riscv_vluxei16_v_" << tableSuffix << "(\n"
         << "        (const uint64_t *)(const void *)__weft_iq1_m_grid, "
            "offsets, "
         << vectorCount << ");\n"
         << "    const " << laneSignedType << " grid = __riscv_vreinterpret_v_"
         << laneUnsignedSuffix << "_" << laneSignedSuffix << "(\n"
         << "        __riscv_vreinterpret_v_" << tableSuffix << "_"
         << laneUnsignedSuffix << "(packed_grid));\n"
         << "    const " << deltaWordType << " packed_delta = __riscv_vle64_v_"
         << deltaWordSuffix << "(delta_words, " << vectorCount << ");\n"
         << "    const " << laneSignedType << " delta = __riscv_vreinterpret_v_"
         << deltaWordSuffix << "_" << laneSignedSuffix << "(packed_delta);\n"
         << "    const " << laneSignedType << " q8 = __riscv_vle8_v_"
         << laneSignedSuffix << "(activation + first_group * 32, vl);\n"
         << "    const " << productType
         << " grid_product = __riscv_vwmul_vv_" << productSuffix
         << "(grid, q8, vl);\n"
         << "    const " << productType
         << " delta_product = __riscv_vwmul_vv_" << productSuffix
         << "(delta, q8, vl);\n"
         << "    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n";
  for (unsigned segment = 0; segment < segmentCount; ++segment) {
    output << "    const int32_t grid" << segment
           << " = __riscv_vmv_x_s_i32m1_i32(\n"
           << "        __riscv_vwredsum_vs_" << segmentProductSuffix
           << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
           << segmentProductSuffix << "(grid_product, " << segment
           << "), zero, 16));\n"
           << "    const int32_t delta" << segment
           << " = __riscv_vmv_x_s_i32m1_i32(\n"
           << "        __riscv_vwredsum_vs_" << segmentProductSuffix
           << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
           << segmentProductSuffix << "(delta_product, " << segment
           << "), zero, 16));\n";
  }
  for (unsigned segment = 0; segment < segmentCount; ++segment) {
    output << "    {\n"
           << "      const size_t group = first_group + " << segment / 2
           << ";\n"
           << "      const unsigned shift = 6 * (group & 1) + "
           << (segment % 2) * 3 << ";\n"
           << "      const int32_t scale = 1 + 2 * "
              "((scale_words[group / 2] >> shift) & 7);\n"
           << "      grid_sum += grid" << segment << " * scale;\n"
           << "      delta_sum += delta" << segment << " * scale;\n"
           << "    }\n";
  }
  output << "  }\n"
         << "  return init + (float)__weft_bitcast_u16_f16(scale_bits) *\n"
         << "                    activation_scale *\n"
         << "                    ((float)grid_sum + 0.125f * "
            "(float)delta_sum);\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
