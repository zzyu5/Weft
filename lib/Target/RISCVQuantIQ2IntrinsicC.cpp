#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitIQ2LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation) {
  if (implementation.leaf.kind != LocalLeafKind::IQ2SI8Register ||
      implementation.valueShapes.size() < 6)
    return false;
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape indexShape = implementation.valueShapes[1];
  const RVVVectorShape tableShape = implementation.valueShapes[2];
  const RVVVectorShape signShape = implementation.valueShapes[3];
  const RVVVectorShape productShape = implementation.valueShapes[4];
  const RVVVectorShape segmentProductShape = implementation.valueShapes[5];
  const std::string laneUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string indexType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, indexShape);
  const std::string indexSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, indexShape);
  const std::string tableType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string tableSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string signType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, signShape);
  const std::string signSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, signShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string segmentProductType =
      rvvVectorType(RVVElementCategory::SignedInteger, segmentProductShape);
  const std::string segmentProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, segmentProductShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(laneShape);
  if (!reduction || laneUnsignedType.empty() || laneUnsignedSuffix.empty() ||
      laneSignedType.empty() || laneSignedSuffix.empty() || indexType.empty() ||
      indexSuffix.empty() || tableType.empty() || tableSuffix.empty() ||
      signType.empty() || signSuffix.empty() || productType.empty() ||
      productSuffix.empty() || segmentProductType.empty() ||
      segmentProductSuffix.empty() || laneSetVL.empty() || !maskRatio)
    return false;

  const unsigned groupsPerVector = reduction->laneFactor / 32;
  const unsigned vectorCount = 4 * groupsPerVector;
  const unsigned segmentCount = 2 * groupsPerVector;
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *codes, const uint8_t *high_bits,\n"
         << "    const uint8_t *sign_bits, const uint8_t *scales,\n"
         << "    const uint8_t *activation_bytes, float weight_scale,\n"
         << "    float activation_scale, float init) {\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  const size_t vl = " << laneSetVL << "("
         << reduction->laneFactor << ");\n"
         << "  const " << laneUnsignedType << " lane = __riscv_vid_v_"
         << laneUnsignedSuffix << "(vl);\n"
         << "  const " << laneUnsignedType
         << " sign_source_index = __riscv_vsrl_vx_" << laneUnsignedSuffix
         << "(lane, 3, vl);\n"
         << "  const " << laneUnsignedType
         << " sign_bit = __riscv_vsll_vv_" << laneUnsignedSuffix << "(\n"
         << "      __riscv_vmv_v_x_" << laneUnsignedSuffix
         << "(1, vl),\n      __riscv_vand_vx_" << laneUnsignedSuffix
         << "(lane, 7, vl), vl);\n"
         << "  int32_t integer_sum = 0;\n"
         << "  for (size_t batch = 0; batch < "
         << reduction->sequentialFactor << "; ++batch) {\n"
         << "    const size_t first_group = batch * " << groupsPerVector
         << ";\n"
         << "    uint16_t byte_offsets[" << vectorCount << "];\n"
         << "    for (size_t vector = 0; vector < " << vectorCount
         << "; ++vector) {\n"
         << "      const size_t group = first_group + vector / 4;\n"
         << "      const size_t within_group = vector % 4;\n"
         << "      byte_offsets[vector] = (uint16_t)((codes[group * 4 + "
            "within_group] |\n"
         << "          (((uint16_t)high_bits[group] << "
            "(8 - 2 * within_group)) & 0x300)) * 8);\n"
         << "    }\n"
         << "    const " << indexType << " offsets = __riscv_vle16_v_"
         << indexSuffix << "(byte_offsets, " << vectorCount << ");\n"
         << "    const " << tableType
         << " packed_grid = __riscv_vluxei16_v_" << tableSuffix << "(\n"
         << "        (const uint64_t *)(const void *)__weft_iq2_s_grid, "
            "offsets, "
         << vectorCount << ");\n"
         << "    const " << laneSignedType
         << " grid = __riscv_vreinterpret_v_" << laneUnsignedSuffix << "_"
         << laneSignedSuffix << "(\n        __riscv_vreinterpret_v_"
         << tableSuffix << "_" << laneUnsignedSuffix << "(packed_grid));\n"
         << "    const " << signType
         << " packed_signs = __riscv_vle8_v_" << signSuffix
         << "(sign_bits + first_group * 4, " << vectorCount << ");\n"
         << "    const " << laneUnsignedType
         << " sign_source = __riscv_vlmul_ext_v_" << signSuffix << "_"
         << laneUnsignedSuffix << "(packed_signs);\n"
         << "    const " << laneUnsignedType
         << " expanded_signs = __riscv_vrgather_vv_" << laneUnsignedSuffix
         << "(sign_source, sign_source_index, vl);\n"
         << "    const " << rvvMaskType(*maskRatio)
         << " negative = __riscv_vmsne_vx_" << laneUnsignedSuffix << "_"
         << rvvMaskSuffix(*maskRatio) << "(\n"
         << "        __riscv_vand_vv_" << laneUnsignedSuffix
         << "(expanded_signs, sign_bit, vl), 0, vl);\n"
         << "    const " << laneSignedType << " q8 = __riscv_vle8_v_"
         << laneSignedSuffix << "(activation + first_group * 32, vl);\n"
         << "    const " << laneSignedType
         << " signed_q8 = __riscv_vrsub_vx_" << laneSignedSuffix
         << "_mu(negative, q8, q8, 0, vl);\n"
         << "    const " << productType << " product = __riscv_vwmul_vv_"
         << productSuffix << "(grid, signed_q8, vl);\n"
         << "    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n";
  for (unsigned segment = 0; segment < segmentCount; ++segment) {
    output << "    const int32_t partial" << segment
           << " = __riscv_vmv_x_s_i32m1_i32(\n"
           << "        __riscv_vwredsum_vs_" << segmentProductSuffix
           << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
           << segmentProductSuffix << "(product, " << segment
           << "), zero, 16));\n";
  }
  for (unsigned segment = 0; segment < segmentCount; ++segment) {
    output << "    integer_sum += partial" << segment << " * (1 + 2 * ";
    if (segment % 2 == 0)
      output << "(scales[first_group + " << segment / 2
             << "] & UINT8_C(15))";
    else
      output << "(scales[first_group + " << segment / 2 << "] >> 4)";
    output << ");\n";
  }
  output << "  }\n"
         << "  return init + 0.125f * (float)integer_sum * weight_scale * "
            "activation_scale;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
