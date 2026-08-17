#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitSignedCodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (!isSignedCodebookLocalImplementationMapping(implementation) ||
      implementation.helperSymbol.empty())
    return false;
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape codeShape = implementation.valueShapes[1];
  const RVVVectorShape indexShape = implementation.valueShapes[2];
  const RVVVectorShape tableShape = implementation.valueShapes[3];
  const RVVVectorShape productShape = implementation.valueShapes[4];
  const RVVVectorShape signSourceShape = implementation.valueShapes[5];
  const RVVVectorShape signWordShape{32, signSourceShape.lmulEighths};
  const unsigned codeCount = 32 / implementation.entryWidth;
  const unsigned offsetShift = implementation.entryWidth == 8 ? 3 : 2;

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
  const std::string laneUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string signSourceType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, signSourceShape);
  const std::string signSourceSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger,
                             signSourceShape);
  const std::string signWordSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger,
                             signWordShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(laneShape);
  if (codeType.empty() || codeSuffix.empty() || indexType.empty() ||
      indexSuffix.empty() || tableType.empty() || tableSuffix.empty() ||
      laneUnsignedType.empty() || laneUnsignedSuffix.empty() ||
      laneSignedType.empty() || laneSignedSuffix.empty() || productType.empty() ||
      productSuffix.empty() || signSourceType.empty() ||
      signSourceSuffix.empty() || signWordSuffix.empty() || laneSetVL.empty() ||
      !maskRatio)
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n"
         << "    const uint8_t *codes, uint32_t sign_metadata,\n"
         << "    const uint8_t *activation_bytes, const uint8_t *grid_table,\n"
         << "    const uint8_t *sign_table, float dot_scale, float init) {\n"
         << "  const size_t code_count = " << codeCount << ";\n"
         << "  const size_t vl32 = " << laneSetVL << "(32);\n"
         << "  const " << codeType << " code8 = __riscv_vle8_v_"
         << codeSuffix << "(codes, code_count);\n"
         << "  const " << indexType << " offsets = __riscv_vsll_vx_"
         << indexSuffix << "(\n"
         << "      __riscv_vwcvtu_x_x_v_" << indexSuffix
         << "(code8, code_count), " << offsetShift << ", code_count);\n"
         << "  const " << tableType << " packed_grid = __riscv_vluxei16_v_"
         << tableSuffix << "(\n      (const uint" << tableShape.sew
         << "_t *)(const void *)grid_table, offsets, code_count);\n"
         << "  const " << laneSignedType
         << " grid = __riscv_vreinterpret_v_" << laneUnsignedSuffix << "_"
         << laneSignedSuffix << "(\n"
         << "      __riscv_vreinterpret_v_" << tableSuffix << "_"
         << laneUnsignedSuffix << "(packed_grid));\n\n"
         << "  const uint32_t sign_word =\n"
         << "      (uint32_t)sign_table[(sign_metadata >> 0) & 127] |\n"
         << "      ((uint32_t)sign_table[(sign_metadata >> 7) & 127] << 8) |\n"
         << "      ((uint32_t)sign_table[(sign_metadata >> 14) & 127] << 16) |\n"
         << "      ((uint32_t)sign_table[(sign_metadata >> 21) & 127] << 24);\n"
         << "  const " << signSourceType
         << " sign_source = __riscv_vreinterpret_v_" << signWordSuffix << "_"
         << signSourceSuffix << "(\n      __riscv_vmv_v_x_" << signWordSuffix
         << "(sign_word, 1));\n"
         << "  const " << laneUnsignedType << " lane = __riscv_vid_v_"
         << laneUnsignedSuffix << "(vl32);\n"
         << "  const " << laneUnsignedType
         << " sign_index = __riscv_vsrl_vx_" << laneUnsignedSuffix
         << "(lane, 3, vl32);\n"
         << "  const " << laneUnsignedType
         << " sign_bit = __riscv_vsll_vv_" << laneUnsignedSuffix << "(\n"
         << "      __riscv_vmv_v_x_" << laneUnsignedSuffix
         << "(1, vl32),\n      __riscv_vand_vx_" << laneUnsignedSuffix
         << "(lane, 7, vl32), vl32);\n"
         << "  const " << laneUnsignedType << " signs = __riscv_vrgather_vv_"
         << laneUnsignedSuffix << "(\n      ";
  if (signSourceShape == laneShape)
    output << "sign_source";
  else
    output << "__riscv_vlmul_ext_v_" << signSourceSuffix << "_"
           << laneUnsignedSuffix << "(sign_source)";
  output << ", sign_index, vl32);\n"
         << "  const " << rvvMaskType(*maskRatio)
         << " negative = __riscv_vmsne_vx_" << laneUnsignedSuffix << "_"
         << rvvMaskSuffix(*maskRatio) << "(\n"
         << "      __riscv_vand_vv_" << laneUnsignedSuffix
         << "(signs, sign_bit, vl32), 0, vl32);\n"
         << "  const " << laneSignedType
         << " activation = __riscv_vle8_v_" << laneSignedSuffix
         << "(\n      (const int8_t *)(const void *)activation_bytes, vl32);\n"
         << "  const " << laneSignedType
         << " signed_activation = __riscv_vrsub_vx_" << laneSignedSuffix
         << "_mu(\n      negative, activation, activation, 0, vl32);\n"
         << "  const " << productType << " product = __riscv_vwmul_vv_"
         << productSuffix << "(\n      grid, signed_activation, vl32);\n"
         << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << productSuffix
         << "_i32m1(product, zero, vl32));\n"
         << "  return init + dot_scale * (float)integer_dot;\n"
         << "}\n\n";
  return true;
}

bool emitPackedU9U7CodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (!isPackedU9U7CodebookLocalImplementationMapping(implementation) ||
      implementation.helperSymbol.empty())
    return false;
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape wordShape = implementation.valueShapes[2];
  const RVVVectorShape tableShape = implementation.valueShapes[3];
  const RVVVectorShape productShape = implementation.valueShapes[4];
  const RVVVectorShape signSourceShape = implementation.valueShapes[5];
  const RVVVectorShape halfProductShape = implementation.valueShapes[6];
  const RVVVectorShape signWordShape{32, signSourceShape.lmulEighths};
  const std::string wordType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, wordShape);
  const std::string wordSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, wordShape);
  const std::string tableType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string tableSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, tableShape);
  const std::string laneUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string halfProductType =
      rvvVectorType(RVVElementCategory::SignedInteger, halfProductShape);
  const std::string halfProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, halfProductShape);
  const std::string signSourceType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, signSourceShape);
  const std::string signSourceSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger,
                             signSourceShape);
  const std::string signWordSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger,
                             signWordShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  const std::string halfSetVL = rvvSetVLIntrinsic(halfProductShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(laneShape);
  if (wordType.empty() || wordSuffix.empty() || tableType.empty() ||
      tableSuffix.empty() || laneUnsignedType.empty() ||
      laneUnsignedSuffix.empty() || laneSignedType.empty() ||
      laneSignedSuffix.empty() || productType.empty() || productSuffix.empty() ||
      halfProductType.empty() || halfProductSuffix.empty() ||
      signSourceType.empty() || signSourceSuffix.empty() ||
      signWordSuffix.empty() || laneSetVL.empty() || halfSetVL.empty() ||
      !maskRatio)
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n"
         << "    const uint8_t *packed_codes, uint8_t scale_byte,\n"
         << "    const uint8_t *activation_bytes, const uint8_t *grid_table,\n"
         << "    const uint8_t *sign_table, float dot_scale, float init) {\n"
         << "  const size_t code_count = 4;\n"
         << "  const size_t vl16 = " << halfSetVL << "(16);\n"
         << "  const size_t vl32 = " << laneSetVL << "(32);\n"
         << "  const " << wordType << " words = __riscv_vle16_v_"
         << wordSuffix << "(\n"
         << "      (const uint16_t *)(const void *)packed_codes, code_count);\n"
         << "  const " << wordType << " offsets = __riscv_vsll_vx_"
         << wordSuffix << "(\n      __riscv_vand_vx_" << wordSuffix
         << "(words, 511, code_count), 3, code_count);\n"
         << "  const " << tableType << " packed_grid = __riscv_vluxei16_v_"
         << tableSuffix
         << "(\n      (const uint64_t *)(const void *)grid_table, offsets, code_count);\n"
         << "  const " << laneSignedType
         << " grid = __riscv_vreinterpret_v_" << laneUnsignedSuffix << "_"
         << laneSignedSuffix << "(\n      __riscv_vreinterpret_v_"
         << tableSuffix << "_" << laneUnsignedSuffix << "(packed_grid));\n\n"
         << "  const uint16_t word0 =\n"
         << "      (uint16_t)packed_codes[0] | ((uint16_t)packed_codes[1] << 8);\n"
         << "  const uint16_t word1 =\n"
         << "      (uint16_t)packed_codes[2] | ((uint16_t)packed_codes[3] << 8);\n"
         << "  const uint16_t word2 =\n"
         << "      (uint16_t)packed_codes[4] | ((uint16_t)packed_codes[5] << 8);\n"
         << "  const uint16_t word3 =\n"
         << "      (uint16_t)packed_codes[6] | ((uint16_t)packed_codes[7] << 8);\n"
         << "  const uint32_t sign_word =\n"
         << "      (uint32_t)sign_table[word0 >> 9] |\n"
         << "      ((uint32_t)sign_table[word1 >> 9] << 8) |\n"
         << "      ((uint32_t)sign_table[word2 >> 9] << 16) |\n"
         << "      ((uint32_t)sign_table[word3 >> 9] << 24);\n"
         << "  const " << signSourceType
         << " sign_source = __riscv_vreinterpret_v_" << signWordSuffix << "_"
         << signSourceSuffix << "(\n      __riscv_vmv_v_x_" << signWordSuffix
         << "(sign_word, 1));\n"
         << "  const " << laneUnsignedType << " lane = __riscv_vid_v_"
         << laneUnsignedSuffix << "(vl32);\n"
         << "  const " << laneUnsignedType
         << " sign_index = __riscv_vsrl_vx_" << laneUnsignedSuffix
         << "(lane, 3, vl32);\n"
         << "  const " << laneUnsignedType
         << " sign_bit = __riscv_vsll_vv_" << laneUnsignedSuffix << "(\n"
         << "      __riscv_vmv_v_x_" << laneUnsignedSuffix
         << "(1, vl32),\n      __riscv_vand_vx_" << laneUnsignedSuffix
         << "(lane, 7, vl32), vl32);\n"
         << "  const " << laneUnsignedType << " signs = __riscv_vrgather_vv_"
         << laneUnsignedSuffix << "(\n      ";
  if (signSourceShape == laneShape)
    output << "sign_source";
  else
    output << "__riscv_vlmul_ext_v_" << signSourceSuffix << "_"
           << laneUnsignedSuffix << "(sign_source)";
  output << ", sign_index, vl32);\n"
         << "  const " << rvvMaskType(*maskRatio)
         << " negative = __riscv_vmsne_vx_" << laneUnsignedSuffix << "_"
         << rvvMaskSuffix(*maskRatio) << "(\n"
         << "      __riscv_vand_vv_" << laneUnsignedSuffix
         << "(signs, sign_bit, vl32), 0, vl32);\n"
         << "  const " << laneSignedType
         << " activation = __riscv_vle8_v_" << laneSignedSuffix
         << "(\n      (const int8_t *)(const void *)activation_bytes, vl32);\n"
         << "  const " << laneSignedType
         << " signed_activation = __riscv_vrsub_vx_" << laneSignedSuffix
         << "_mu(\n      negative, activation, activation, 0, vl32);\n"
         << "  const " << productType << " product = __riscv_vwmul_vv_"
         << productSuffix << "(grid, signed_activation, vl32);\n"
         << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t low_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << halfProductSuffix
         << "_i32m1(\n          __riscv_vget_v_" << productSuffix << "_"
         << halfProductSuffix << "(product, 0), zero, vl16));\n"
         << "  const int32_t high_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << halfProductSuffix
         << "_i32m1(\n          __riscv_vget_v_" << productSuffix << "_"
         << halfProductSuffix << "(product, 1), zero, vl16));\n"
         << "  const int32_t low_scale = 1 + 2 * (int32_t)(scale_byte & 15);\n"
         << "  const int32_t high_scale = 1 + 2 * (int32_t)(scale_byte >> 4);\n"
         << "  return init + dot_scale * (float)(low_scale * low_dot +\n"
         << "                                    high_scale * high_dot);\n"
         << "}\n\n";
  return true;
}

bool emitPackedU11GridDeltaLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (!isPackedU11GridDeltaLocalImplementationMapping(implementation) ||
      implementation.helperSymbol.empty())
    return false;
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape codeShape = implementation.valueShapes[1];
  const RVVVectorShape indexShape = implementation.valueShapes[2];
  const RVVVectorShape tableShape = implementation.valueShapes[3];
  const RVVVectorShape productShape = implementation.valueShapes[4];
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
  const std::string laneUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, laneShape);
  const std::string laneSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  if (codeType.empty() || codeSuffix.empty() || indexType.empty() ||
      indexSuffix.empty() || tableType.empty() || tableSuffix.empty() ||
      laneUnsignedSuffix.empty() || laneSignedType.empty() ||
      laneSignedSuffix.empty() || productType.empty() || productSuffix.empty() ||
      laneSetVL.empty())
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << implementation.helperSymbol << "(\n"
         << "    const uint8_t *codes, uint16_t metadata,\n"
         << "    const uint8_t *activation_bytes, const uint8_t *grid_table,\n"
         << "    int32_t activation_sum, float dot_scale, float init) {\n"
         << "  const size_t code_count = 4;\n"
         << "  const size_t vl32 = " << laneSetVL << "(32);\n"
         << "  const " << codeType << " code8 = __riscv_vle8_v_"
         << codeSuffix << "(codes, code_count);\n"
         << "  const " << indexType << " low = __riscv_vwcvtu_x_x_v_"
         << indexSuffix << "(code8, code_count);\n"
         << "  const " << indexType << " field = __riscv_vid_v_"
         << indexSuffix << "(code_count);\n"
         << "  const " << indexType << " shift = __riscv_vmul_vx_"
         << indexSuffix << "(field, 3, code_count);\n"
         << "  const " << indexType << " high = __riscv_vand_vx_"
         << indexSuffix << "(\n      __riscv_vsrl_vv_" << indexSuffix
         << "(\n          __riscv_vmv_v_x_" << indexSuffix
         << "(metadata, code_count), shift, code_count),\n      7, code_count);\n"
         << "  const " << indexType << " index = __riscv_vor_vv_"
         << indexSuffix << "(\n      low, __riscv_vsll_vx_" << indexSuffix
         << "(high, 8, code_count), code_count);\n"
         << "  const " << indexType << " offsets = __riscv_vsll_vx_"
         << indexSuffix << "(index, 3, code_count);\n"
         << "  const " << tableType << " packed_grid = __riscv_vluxei16_v_"
         << tableSuffix
         << "(\n      (const uint64_t *)(const void *)grid_table, offsets, code_count);\n"
         << "  const " << laneSignedType
         << " grid = __riscv_vreinterpret_v_" << laneUnsignedSuffix << "_"
         << laneSignedSuffix << "(\n      __riscv_vreinterpret_v_"
         << tableSuffix << "_" << laneUnsignedSuffix << "(packed_grid));\n"
         << "  const " << laneSignedType
         << " activation = __riscv_vle8_v_" << laneSignedSuffix
         << "(\n      (const int8_t *)(const void *)activation_bytes, vl32);\n"
         << "  const " << productType << " product = __riscv_vwmul_vv_"
         << productSuffix << "(grid, activation, vl32);\n"
         << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t grid_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << productSuffix
         << "_i32m1(product, zero, vl32));\n"
         << "  const int32_t local_scale = "
            "1 + 2 * (int32_t)((metadata >> 12) & 7);\n"
         << "  const float delta = "
            "(metadata & 0x8000u) ? -0.125f : 0.125f;\n"
         << "  return init + dot_scale * (float)local_scale *\n"
         << "                    ((float)grid_dot + delta * "
            "(float)activation_sum);\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
