#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

bool emitPackedI3GroupedLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::PackedI3GroupedI8 ||
      implementation.operation.kind !=
          LocalHardwareOperationKind::RVVRegister ||
      implementation.valueShapes.size() < 3)
    return false;
  const PhysicalAxisDecomposition *reduction =
      findAxisMapping(implementation.mapping, kCoreAxisK);
  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape productShape = implementation.valueShapes[1];
  const RVVVectorShape segmentProductShape = implementation.valueShapes[2];
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
  const std::string segmentProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, segmentProductShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  std::optional<unsigned> maskRatio = rvvMaskRatio(laneShape);
  if (!reduction || laneUnsignedType.empty() || laneUnsignedSuffix.empty() ||
      laneSignedType.empty() || laneSignedSuffix.empty() || productType.empty() ||
      productSuffix.empty() || segmentProductSuffix.empty() ||
      laneSetVL.empty() || !maskRatio)
    return false;

  const unsigned groupsPerVector = reduction->laneFactor / 16;
  const unsigned groupsPerIteration =
      groupsPerVector * reduction->registerFactor;
  output << R"c(static inline __attribute__((always_inline, unused)) int32_t
__weft_packed_i3_group_scale(const uint8_t *scales, size_t group) {
  const size_t quarter = group / 4;
  const size_t lane = group % 4;
  const uint8_t base = scales[(quarter % 2) * 4 + lane];
  const uint8_t low = quarter >= 2 ? base >> 4 : base & UINT8_C(15);
  const uint8_t high =
      (scales[8 + lane] >> (uint8_t)(quarter * 2)) & UINT8_C(3);
  return (int32_t)(low | (uint8_t)(high << 4)) - 32;
}

)c";
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *low_bits, const uint8_t *high_bits,\n"
         << "    const uint8_t *scales, const uint8_t *activation_bytes,\n"
         << "    float weight_scale, float activation_scale, float init) {\n"
         << "  const size_t vl = " << laneSetVL << "("
         << reduction->laneFactor << ");\n"
         << "  int32_t integer_sum = 0;\n"
         << "#pragma GCC unroll " << reduction->sequentialFactor << "\n"
         << "  for (size_t batch = 0; batch < "
         << reduction->sequentialFactor << "; ++batch) {\n"
         << "    const size_t first_iteration_group = batch * "
         << groupsPerIteration << ";\n"
         << "    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n";
  for (unsigned repeat = 0; repeat < reduction->registerFactor; ++repeat) {
    output << "    {\n"
           << "      const size_t first_group = first_iteration_group + "
           << repeat * groupsPerVector << ";\n"
           << "      const size_t half = first_group / 8;\n"
           << "      const size_t within_half = first_group % 8;\n"
           << "      const size_t field = within_half / 2;\n"
           << "      const size_t lane_group = within_half % 2;\n"
           << "      const uint8_t low_shift = (uint8_t)(field * 2);\n"
           << "      const uint8_t high_mask = (uint8_t)(1u << (half * 4 + "
              "field));\n"
           << "      const " << laneUnsignedType
           << " packed = __riscv_vle8_v_" << laneUnsignedSuffix
           << "(low_bits + half * 32 + lane_group * 16, vl);\n"
           << "      const " << laneUnsignedType
           << " high = __riscv_vle8_v_" << laneUnsignedSuffix
           << "(high_bits + lane_group * 16, vl);\n"
           << "      const " << laneUnsignedType
           << " low = __riscv_vand_vx_" << laneUnsignedSuffix
           << "(\n          __riscv_vsrl_vx_" << laneUnsignedSuffix
           << "(packed, low_shift, vl), UINT8_C(3), vl);\n"
           << "      " << laneSignedType << " code = __riscv_vreinterpret_v_"
           << laneUnsignedSuffix << "_" << laneSignedSuffix << "(low);\n"
           << "      const " << rvvMaskType(*maskRatio)
           << " missing = __riscv_vmseq_vx_" << laneUnsignedSuffix << "_"
           << rvvMaskSuffix(*maskRatio) << "(\n          __riscv_vand_vx_"
           << laneUnsignedSuffix << "(high, high_mask, vl), 0, vl);\n"
           << "      code = __riscv_vsub_vx_" << laneSignedSuffix
           << "_mu(missing, code, code, 4, vl);\n"
           << "      const " << laneSignedType
           << " activation = __riscv_vle8_v_" << laneSignedSuffix
           << "(\n          (const int8_t *)(const void *)(activation_bytes + "
              "first_group * 16), vl);\n"
           << "      const " << productType
           << " product = __riscv_vwmul_vv_" << productSuffix
           << "(code, activation, vl);\n";
    for (unsigned group = 0; group < groupsPerVector; ++group) {
      output << "      const int32_t partial" << group
             << " = __riscv_vmv_x_s_i32m1_i32(\n"
             << "          __riscv_vwredsum_vs_" << segmentProductSuffix
             << "_i32m1(\n              ";
      if (productShape == segmentProductShape)
        output << "product";
      else
        output << "__riscv_vget_v_" << productSuffix << "_"
               << segmentProductSuffix << "(product, " << group << ")";
      output << ", zero, 16));\n"
             << "      integer_sum += __weft_packed_i3_group_scale(\n"
             << "          scales, first_group + " << group << ") * partial"
             << group << ";\n";
    }
    output << "    }\n";
  }
  output << "  }\n"
         << "  return init + weight_scale * activation_scale * "
            "(float)integer_sum;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
