#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitIQ3LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation) {
  if (!isIQ3SLocalImplementationMapping(implementation) ||
      implementation.helperSymbol.empty())
    return false;
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape codeShape = implementation.valueShapes[1];
  const RVVVectorShape indexShape = implementation.valueShapes[2];
  const RVVVectorShape tableShape = implementation.valueShapes[3];
  const RVVVectorShape signShape = implementation.valueShapes[4];
  const RVVVectorShape productShape = implementation.valueShapes[5];
  const RVVVectorShape halfProductShape = implementation.valueShapes[6];
  const std::string laneUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string codeType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, codeShape);
  const std::string codeSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, codeShape);
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
  const std::string halfProductType =
      rvvVectorType(RVVElementCategory::SignedInteger, halfProductShape);
  const std::string halfProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, halfProductShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(laneShape);
  if (!reduction || laneUnsignedType.empty() || laneUnsignedSuffix.empty() ||
      laneSignedType.empty() || laneSignedSuffix.empty() || codeType.empty() ||
      codeSuffix.empty() || indexType.empty() || indexSuffix.empty() ||
      tableType.empty() || tableSuffix.empty() || signType.empty() ||
      signSuffix.empty() || productType.empty() || productSuffix.empty() ||
      halfProductType.empty() || halfProductSuffix.empty() || !maskRatio)
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n"
         << "    const uint8_t *codes, const uint8_t *high_bits,\n"
         << "    const uint8_t *sign_bits, const uint8_t *scales,\n"
         << "    const uint8_t *activation_bytes, float weight_scale,\n"
         << "    float activation_scale, float init) {\n"
         << "  static const uint8_t sign_gather_indices[64] = {\n"
         << "      0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,\n"
         << "      2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,\n"
         << "      4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,\n"
         << "      6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};\n"
         << "  static const uint8_t sign_masks[64] = {\n"
         << "      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,\n"
         << "      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,\n"
         << "      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,\n"
         << "      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};\n"
         << "  static const uint16_t high_shifts[16] = {\n"
         << "      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};\n"
         << "  const int8_t *activation = "
            "(const int8_t *)(const void *)activation_bytes;\n"
         << "  const " << laneUnsignedType
         << " gather_indices = __riscv_vle8_v_" << laneUnsignedSuffix
         << "(sign_gather_indices, 64);\n"
         << "  const " << laneUnsignedType
         << " masks = __riscv_vle8_v_" << laneUnsignedSuffix
         << "(sign_masks, 64);\n"
         << "  const " << indexType << " shifts = __riscv_vle16_v_"
         << indexSuffix << "(high_shifts, 16);\n"
         << "  int32_t integer_sum = 0;\n"
         << "  for (size_t group_pair = 0; group_pair < "
         << reduction->sequentialFactor << "; ++group_pair) {\n"
         << "    const " << codeType << " low8 = __riscv_vle8_v_"
         << codeSuffix << "(codes + group_pair * 16, 16);\n"
         << "    const uint16_t high_word =\n"
         << "        (uint16_t)high_bits[group_pair * 2] |\n"
         << "        ((uint16_t)high_bits[group_pair * 2 + 1] << 8);\n"
         << "    " << indexType << " high = __riscv_vmv_v_x_"
         << indexSuffix << "(high_word, 16);\n"
         << "    high = __riscv_vand_vx_" << indexSuffix << "(\n"
         << "        __riscv_vsrl_vv_" << indexSuffix
         << "(high, shifts, 16), 1, 16);\n"
         << "    const " << indexType
         << " low = __riscv_vwcvtu_x_x_v_" << indexSuffix
         << "(low8, 16);\n"
         << "    const " << indexType << " offsets = __riscv_vor_vv_"
         << indexSuffix << "(\n        __riscv_vsll_vx_" << indexSuffix
         << "(low, 2, 16),\n        __riscv_vsll_vx_" << indexSuffix
         << "(high, 10, 16), 16);\n"
         << "    const " << tableType
         << " packed_grid = __riscv_vluxei16_v_" << tableSuffix << "(\n"
         << "        (const uint32_t *)(const void *)__weft_iq3_s_grid, "
            "offsets, 16);\n"
         << "    const " << laneUnsignedType
         << " grid = __riscv_vreinterpret_v_" << tableSuffix << "_"
         << laneUnsignedSuffix << "(packed_grid);\n"
         << "    const " << signType
         << " packed_signs = __riscv_vle8_v_" << signSuffix
         << "(sign_bits + group_pair * 8, 8);\n"
         << "    const " << laneUnsignedType
         << " sign_source = __riscv_vlmul_ext_v_" << signSuffix << "_"
         << laneUnsignedSuffix << "(packed_signs);\n"
         << "    const " << laneUnsignedType
         << " sign_bytes = __riscv_vrgather_vv_" << laneUnsignedSuffix
         << "(sign_source, gather_indices, 64);\n"
         << "    const " << rvvMaskType(*maskRatio)
         << " negative = __riscv_vmsne_vx_" << laneUnsignedSuffix << "_"
         << rvvMaskSuffix(*maskRatio) << "(\n"
         << "        __riscv_vand_vv_" << laneUnsignedSuffix
         << "(sign_bytes, masks, 64), 0, 64);\n"
         << "    const " << laneSignedType
         << " q8 = __riscv_vle8_v_" << laneSignedSuffix
         << "(activation + group_pair * 64, 64);\n"
         << "    const " << laneSignedType
         << " signed_q8 = __riscv_vrsub_vx_" << laneSignedSuffix
         << "_mu(negative, q8, q8, 0, 64);\n"
         << "    const " << productType
         << " product = __riscv_vwmulsu_vv_" << productSuffix
         << "(signed_q8, grid, 64);\n"
         << "    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "    const int32_t low_sum = __riscv_vmv_x_s_i32m1_i32(\n"
         << "        __riscv_vwredsum_vs_" << halfProductSuffix
         << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
         << halfProductSuffix << "(product, 0), zero, 32));\n"
         << "    const int32_t high_sum = __riscv_vmv_x_s_i32m1_i32(\n"
         << "        __riscv_vwredsum_vs_" << halfProductSuffix
         << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
         << halfProductSuffix << "(product, 1), zero, 32));\n"
         << "    const uint8_t packed_scale = scales[group_pair];\n"
         << "    integer_sum += low_sum * (1 + 2 * (packed_scale & 15)) +\n"
         << "                   high_sum * (1 + 2 * (packed_scale >> 4));\n"
         << "  }\n"
         << "  return init + (float)integer_sum * weight_scale * "
            "activation_scale;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
