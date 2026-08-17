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
      findUniqueAxisMapping(implementation.mapping, LogicalAxisRole::Reduction);
  const LocalPrimitiveKind expectedPrimitive =
      hasHighBits ? LocalPrimitiveKind::PackedI5I8
                  : LocalPrimitiveKind::PackedI4I8;
  if (implementation.primitive != expectedPrimitive ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVIntrinsic ||
      implementation.operation.projection == LocalOperationProjection::None ||
      implementation.valueShapes.size() < 3 || !reduction)
    return false;

  const RVVVectorShape shape = implementation.mapping.laneShape;
  const RVVVectorShape decodeShape = implementation.valueShapes[1];
  const RVVVectorShape widened = implementation.valueShapes[2];
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
         << localImplementationSymbol(implementation) << "(\n";
  if (hasHighBits)
    output << "    const uint8_t *low_bits, const uint8_t *high_bits,\n";
  else
    output << "    const uint8_t *packed_codes,\n";
  output << "    const uint8_t *activation_bytes, int32_t zero_point,\n"
         << "    float dot_scale, float additive_bias, float init) {\n";

  const bool laneProjection =
      implementation.operation.projection ==
          LocalOperationProjection::PackedDotLaneSlide ||
      implementation.operation.projection ==
          LocalOperationProjection::PackedDotLaneCreate;
  if (laneProjection) {
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
    if (implementation.operation.projection ==
        LocalOperationProjection::PackedDotLaneSlide)
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
  } else if (implementation.operation.projection ==
             LocalOperationProjection::PackedDotRegisterChunks) {
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
  } else {
    return false;
  }
  output << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << widenedSuffix
         << "_i32m1(products, zero, "
         << (laneProjection ? "vl32" : "vl0") << "));\n"
         << "  return init + dot_scale * (float)integer_dot + additive_bias;\n"
         << "}\n\n";
  return true;
}

} // namespace

bool emitPackedI4LocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  return emitPackedDotLocalImplementation(output, implementation, false);
}

bool emitPackedI5LocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  return emitPackedDotLocalImplementation(output, implementation, true);
}

bool emitNibbleCodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::NibbleCodebookI8 ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVIntrinsic ||
      implementation.valueShapes.size() < 4)
    return false;
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape packedShape = implementation.valueShapes[1];
  const RVVVectorShape tableShape = implementation.valueShapes[2];
  const RVVVectorShape productShape = implementation.valueShapes[3];
  const bool combined =
      implementation.operation.projection ==
      LocalOperationProjection::NibbleCodebookCombined;
  if (!combined && implementation.operation.projection !=
                       LocalOperationProjection::NibbleCodebookSplit)
    return false;
  const std::string packedUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, packedShape);
  const std::string packedUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, packedShape);
  const std::string tableType =
      rvvVectorType(RVVElementCategory::SignedInteger, tableShape);
  const std::string tableSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, tableShape);
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
  const std::string packedSetVL = rvvSetVLIntrinsic(packedShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  if (packedUnsignedType.empty() || packedUnsignedSuffix.empty() ||
      tableType.empty() || tableSuffix.empty() || laneUnsignedType.empty() ||
      laneUnsignedSuffix.empty() || laneSignedType.empty() ||
      laneSignedSuffix.empty() || productType.empty() ||
      productSuffix.empty() || packedSetVL.empty() || laneSetVL.empty())
    return false;

  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *packed_codes, const uint8_t *table_bytes,\n"
         << "    const uint8_t *activation_bytes, float dot_scale, float init) {\n"
         << "  const size_t vl16 = " << packedSetVL << "(16);\n"
         << "  const " << packedUnsignedType
         << " packed = __riscv_vle8_v_" << packedUnsignedSuffix
         << "(packed_codes, vl16);\n"
         << "  const " << tableType << " table = __riscv_vle8_v_"
         << tableSuffix
         << "((const int8_t *)(const void *)table_bytes, vl16);\n"
         << "  const " << packedUnsignedType
         << " low_index = __riscv_vand_vx_" << packedUnsignedSuffix
         << "(packed, UINT8_C(15), vl16);\n"
         << "  const " << packedUnsignedType
         << " high_index = __riscv_vsrl_vx_" << packedUnsignedSuffix
         << "(packed, 4, vl16);\n";
  if (combined) {
    output << "  const size_t vl32 = " << laneSetVL << "(32);\n"
           << "  const " << laneUnsignedType
           << " index = __riscv_vcreate_v_" << packedUnsignedSuffix << "_"
           << laneUnsignedSuffix << "(low_index, high_index);\n"
           << "  const " << laneSignedType
           << " decoded = __riscv_vrgather_vv_" << laneSignedSuffix
           << "(table, index, vl32);\n"
           << "  const " << laneSignedType
           << " activation = __riscv_vle8_v_" << laneSignedSuffix
           << "((const int8_t *)(const void *)activation_bytes, vl32);\n"
           << "  const " << productType
           << " product = __riscv_vwmul_vv_" << productSuffix
           << "(decoded, activation, vl32);\n";
  } else {
    output << "  const " << tableType
           << " low = __riscv_vrgather_vv_" << tableSuffix
           << "(table, low_index, vl16);\n"
           << "  const " << tableType
           << " high = __riscv_vrgather_vv_" << tableSuffix
           << "(table, high_index, vl16);\n"
           << "  const " << tableType
           << " activation_low = __riscv_vle8_v_" << tableSuffix
           << "((const int8_t *)(const void *)activation_bytes, vl16);\n"
           << "  const " << tableType
           << " activation_high = __riscv_vle8_v_" << tableSuffix
           << "((const int8_t *)(const void *)(activation_bytes + 16), vl16);\n"
           << "  const " << productType
           << " product = __riscv_vadd_vv_" << productSuffix << "(\n"
           << "      __riscv_vwmul_vv_" << productSuffix
           << "(low, activation_low, vl16),\n"
           << "      __riscv_vwmul_vv_" << productSuffix
           << "(high, activation_high, vl16), vl16);\n";
  }
  output << "  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n"
         << "  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(\n"
         << "      __riscv_vwredsum_vs_" << productSuffix
         << "_i32m1(product, zero, " << (combined ? "vl32" : "vl16")
         << "));\n"
         << "  return init + dot_scale * (float)integer_dot;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
