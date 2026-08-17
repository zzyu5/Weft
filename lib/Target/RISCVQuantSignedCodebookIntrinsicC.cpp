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

void emitPackedU9U7CodebookLocalImplementations(llvm::raw_ostream &output,
                                            bool registerE8M2, bool registerE8M1) {
  if (registerE8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_u9_u7_codebook_i8_register_l32_e8m2(
    const uint8_t *packed_codes, uint8_t scale_byte,
    const uint8_t *activation_bytes, const uint8_t *grid_table,
    const uint8_t *sign_table, float dot_scale, float init) {
  const size_t code_count = 4;
  const size_t vl16 = __riscv_vsetvl_e16m2(16);
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint16mf2_t words = __riscv_vle16_v_u16mf2(
      (const uint16_t *)(const void *)packed_codes, code_count);
  const vuint16mf2_t offsets = __riscv_vsll_vx_u16mf2(
      __riscv_vand_vx_u16mf2(words, 511, code_count), 3, code_count);
  const vuint64m2_t packed_grid = __riscv_vluxei16_v_u64m2(
      (const uint64_t *)(const void *)grid_table, offsets, code_count);
  const vint8m2_t grid = __riscv_vreinterpret_v_u8m2_i8m2(
      __riscv_vreinterpret_v_u64m2_u8m2(packed_grid));

  const uint16_t word0 =
      (uint16_t)packed_codes[0] | ((uint16_t)packed_codes[1] << 8);
  const uint16_t word1 =
      (uint16_t)packed_codes[2] | ((uint16_t)packed_codes[3] << 8);
  const uint16_t word2 =
      (uint16_t)packed_codes[4] | ((uint16_t)packed_codes[5] << 8);
  const uint16_t word3 =
      (uint16_t)packed_codes[6] | ((uint16_t)packed_codes[7] << 8);
  const uint32_t sign_word =
      (uint32_t)sign_table[word0 >> 9] |
      ((uint32_t)sign_table[word1 >> 9] << 8) |
      ((uint32_t)sign_table[word2 >> 9] << 16) |
      ((uint32_t)sign_table[word3 >> 9] << 24);
  const vuint8m1_t sign_source = __riscv_vreinterpret_v_u32m1_u8m1(
      __riscv_vmv_v_x_u32m1(sign_word, 1));
  const vuint8m2_t lane = __riscv_vid_v_u8m2(vl32);
  const vuint8m2_t sign_index = __riscv_vsrl_vx_u8m2(lane, 3, vl32);
  const vuint8m2_t sign_bit = __riscv_vsll_vv_u8m2(
      __riscv_vmv_v_x_u8m2(1, vl32),
      __riscv_vand_vx_u8m2(lane, 7, vl32), vl32);
  const vuint8m2_t signs = __riscv_vrgather_vv_u8m2(
      __riscv_vlmul_ext_v_u8m1_u8m2(sign_source), sign_index, vl32);
  const vbool4_t negative = __riscv_vmsne_vx_u8m2_b4(
      __riscv_vand_vv_u8m2(signs, sign_bit, vl32), 0, vl32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint8m2_t signed_activation = __riscv_vrsub_vx_i8m2_mu(
      negative, activation, activation, 0, vl32);
  const vint16m4_t product = __riscv_vwmul_vv_i16m4(
      grid, signed_activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t low_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(
          __riscv_vget_v_i16m4_i16m2(product, 0), zero, vl16));
  const int32_t high_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(
          __riscv_vget_v_i16m4_i16m2(product, 1), zero, vl16));
  const int32_t low_scale = 1 + 2 * (int32_t)(scale_byte & 15);
  const int32_t high_scale = 1 + 2 * (int32_t)(scale_byte >> 4);
  return init + dot_scale * (float)(low_scale * low_dot +
                                    high_scale * high_dot);
}

)c";
  }

  if (registerE8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_u9_u7_codebook_i8_register_l32_e8m1(
    const uint8_t *packed_codes, uint8_t scale_byte,
    const uint8_t *activation_bytes, const uint8_t *grid_table,
    const uint8_t *sign_table, float dot_scale, float init) {
  const size_t code_count = 4;
  const size_t vl16 = __riscv_vsetvl_e16m1(16);
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  const vuint16mf4_t words = __riscv_vle16_v_u16mf4(
      (const uint16_t *)(const void *)packed_codes, code_count);
  const vuint16mf4_t offsets = __riscv_vsll_vx_u16mf4(
      __riscv_vand_vx_u16mf4(words, 511, code_count), 3, code_count);
  const vuint64m1_t packed_grid = __riscv_vluxei16_v_u64m1(
      (const uint64_t *)(const void *)grid_table, offsets, code_count);
  const vint8m1_t grid = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vreinterpret_v_u64m1_u8m1(packed_grid));

  const uint16_t word0 =
      (uint16_t)packed_codes[0] | ((uint16_t)packed_codes[1] << 8);
  const uint16_t word1 =
      (uint16_t)packed_codes[2] | ((uint16_t)packed_codes[3] << 8);
  const uint16_t word2 =
      (uint16_t)packed_codes[4] | ((uint16_t)packed_codes[5] << 8);
  const uint16_t word3 =
      (uint16_t)packed_codes[6] | ((uint16_t)packed_codes[7] << 8);
  const uint32_t sign_word =
      (uint32_t)sign_table[word0 >> 9] |
      ((uint32_t)sign_table[word1 >> 9] << 8) |
      ((uint32_t)sign_table[word2 >> 9] << 16) |
      ((uint32_t)sign_table[word3 >> 9] << 24);
  const vuint8m1_t sign_source = __riscv_vreinterpret_v_u32m1_u8m1(
      __riscv_vmv_v_x_u32m1(sign_word, 1));
  const vuint8m1_t lane = __riscv_vid_v_u8m1(vl32);
  const vuint8m1_t sign_index = __riscv_vsrl_vx_u8m1(lane, 3, vl32);
  const vuint8m1_t sign_bit = __riscv_vsll_vv_u8m1(
      __riscv_vmv_v_x_u8m1(1, vl32),
      __riscv_vand_vx_u8m1(lane, 7, vl32), vl32);
  const vuint8m1_t signs =
      __riscv_vrgather_vv_u8m1(sign_source, sign_index, vl32);
  const vbool8_t negative = __riscv_vmsne_vx_u8m1_b8(
      __riscv_vand_vv_u8m1(signs, sign_bit, vl32), 0, vl32);
  const vint8m1_t activation = __riscv_vle8_v_i8m1(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint8m1_t signed_activation = __riscv_vrsub_vx_i8m1_mu(
      negative, activation, activation, 0, vl32);
  const vint16m2_t product = __riscv_vwmul_vv_i16m2(
      grid, signed_activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t low_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(
          __riscv_vget_v_i16m2_i16m1(product, 0), zero, vl16));
  const int32_t high_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(
          __riscv_vget_v_i16m2_i16m1(product, 1), zero, vl16));
  const int32_t low_scale = 1 + 2 * (int32_t)(scale_byte & 15);
  const int32_t high_scale = 1 + 2 * (int32_t)(scale_byte >> 4);
  return init + dot_scale * (float)(low_scale * low_dot +
                                    high_scale * high_dot);
}

)c";
  }
}

void emitPackedU11GridDeltaLocalImplementations(llvm::raw_ostream &output,
                                            bool registerE8M2, bool registerE8M1) {
  if (registerE8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_u11_grid_delta_i8_register_l32_e8m2(
    const uint8_t *codes, uint16_t metadata,
    const uint8_t *activation_bytes, const uint8_t *grid_table,
    int32_t activation_sum, float dot_scale, float init) {
  const size_t code_count = 4;
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8mf4_t code8 = __riscv_vle8_v_u8mf4(codes, code_count);
  const vuint16mf2_t low =
      __riscv_vwcvtu_x_x_v_u16mf2(code8, code_count);
  const vuint16mf2_t field = __riscv_vid_v_u16mf2(code_count);
  const vuint16mf2_t shift =
      __riscv_vmul_vx_u16mf2(field, 3, code_count);
  const vuint16mf2_t high = __riscv_vand_vx_u16mf2(
      __riscv_vsrl_vv_u16mf2(
          __riscv_vmv_v_x_u16mf2(metadata, code_count), shift, code_count),
      7, code_count);
  const vuint16mf2_t index = __riscv_vor_vv_u16mf2(
      low, __riscv_vsll_vx_u16mf2(high, 8, code_count), code_count);
  const vuint16mf2_t offsets =
      __riscv_vsll_vx_u16mf2(index, 3, code_count);
  const vuint64m2_t packed_grid = __riscv_vluxei16_v_u64m2(
      (const uint64_t *)(const void *)grid_table, offsets, code_count);
  const vint8m2_t grid = __riscv_vreinterpret_v_u8m2_i8m2(
      __riscv_vreinterpret_v_u64m2_u8m2(packed_grid));
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m4_t product =
      __riscv_vwmul_vv_i16m4(grid, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t grid_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m4_i32m1(product, zero, vl32));
  const int32_t local_scale = 1 + 2 * (int32_t)((metadata >> 12) & 7);
  const float delta = (metadata & 0x8000u) ? -0.125f : 0.125f;
  return init + dot_scale * (float)local_scale *
                    ((float)grid_dot + delta * (float)activation_sum);
}

)c";
  }

  if (registerE8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_u11_grid_delta_i8_register_l32_e8m1(
    const uint8_t *codes, uint16_t metadata,
    const uint8_t *activation_bytes, const uint8_t *grid_table,
    int32_t activation_sum, float dot_scale, float init) {
  const size_t code_count = 4;
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  const vuint8mf8_t code8 = __riscv_vle8_v_u8mf8(codes, code_count);
  const vuint16mf4_t low =
      __riscv_vwcvtu_x_x_v_u16mf4(code8, code_count);
  const vuint16mf4_t field = __riscv_vid_v_u16mf4(code_count);
  const vuint16mf4_t shift =
      __riscv_vmul_vx_u16mf4(field, 3, code_count);
  const vuint16mf4_t high = __riscv_vand_vx_u16mf4(
      __riscv_vsrl_vv_u16mf4(
          __riscv_vmv_v_x_u16mf4(metadata, code_count), shift, code_count),
      7, code_count);
  const vuint16mf4_t index = __riscv_vor_vv_u16mf4(
      low, __riscv_vsll_vx_u16mf4(high, 8, code_count), code_count);
  const vuint16mf4_t offsets =
      __riscv_vsll_vx_u16mf4(index, 3, code_count);
  const vuint64m1_t packed_grid = __riscv_vluxei16_v_u64m1(
      (const uint64_t *)(const void *)grid_table, offsets, code_count);
  const vint8m1_t grid = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vreinterpret_v_u64m1_u8m1(packed_grid));
  const vint8m1_t activation = __riscv_vle8_v_i8m1(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m2_t product =
      __riscv_vwmul_vv_i16m2(grid, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t grid_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(product, zero, vl32));
  const int32_t local_scale = 1 + 2 * (int32_t)((metadata >> 12) & 7);
  const float delta = (metadata & 0x8000u) ? -0.125f : 0.125f;
  return init + dot_scale * (float)local_scale *
                    ((float)grid_dot + delta * (float)activation_sum);
}

)c";
  }
}

} // namespace weft::riscv_internal
