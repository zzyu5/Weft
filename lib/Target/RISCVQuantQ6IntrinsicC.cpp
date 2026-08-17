#include "RISCVQuantIntrinsicC.h"

#include "RISCVRVVSpelling.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {
namespace {

void emitQ6KRVVAssembly(llvm::raw_ostream &output,
                        const LocalImplementation &implementation) {
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << R"c((
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *group_scale_bytes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *group_scales =
      (const int8_t *)(const void *)group_scale_bytes;
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const float combined_scale = weight_scale * activation_scale;
  const uint8_t *low = low_bits;
  const uint8_t *high = high_bits;
  const int8_t *scale = group_scales;
  const int8_t *q8 = activation;
  int low_high;
  float temporary;
  float result = init;
  for (size_t half = 0; half < )c"
         << implementation.schedule.decode.chunksPerStep << R"c(; ++half) {
    __asm__ volatile(
        "addi %[low_high], %[low], 32\n\t"
        "ld t0, 0(%[scale])\n\t"
        "addi %[scale], %[scale], 8\n\t"
        "slli t6, t0, 1 * 8\n\t"
        "lb zero, 0(%[low])\n\t"
        "slli t5, t0, 2 * 8\n\t"
        "slli t4, t0, 3 * 8\n\t"
        "lb zero, 0(%[low_high])\n\t"
        "slli t3, t0, 4 * 8\n\t"
        "slli t2, t0, 5 * 8\n\t"
        "lb zero, 0(%[high])\n\t"
        "lb zero, 31(%[low_high])\n\t"
        "slli t1, t0, 6 * 8\n\t"
        "srai a7, t0, 56\n\t"
        "vsetvli zero, %[vl32], e8, m2\n\t"
        "vle8.v v8, (%[low])\n\t"
        "srai t6, t6, 56\n\t"
        "srai t5, t5, 56\n\t"
        "srai t4, t4, 56\n\t"
        "srai t3, t3, 56\n\t"
        "vle8.v v10, (%[low_high])\n\t"
        "addi %[low], %[low], 64\n\t"
        "slli t0, t0, 7 * 8\n\t"
        "srai t2, t2, 56\n\t"
        "srai t1, t1, 56\n\t"
        "srai t0, t0, 56\n\t"
        "vle8.v v4, (%[high])\n\t"
        "vsrl.vi v12, v8, 4\n\t"
        "vsrl.vi v14, v10, 4\n\t"
        "lb zero, 0(%[q8])\n\t"
        "vand.vi v8, v8, 0xF\n\t"
        "vand.vi v10, v10, 0xF\n\t"
        "lb zero, 32(%[q8])\n\t"
        "vsll.vi v0, v4, 4\n\t"
        "vsll.vi v2, v4, 2\n\t"
        "lb zero, 64(%[q8])\n\t"
        "vsrl.vi v6, v4, 2\n\t"
        "lb zero, 96(%[q8])\n\t"
        "vand.vx v0, v0, %[mask]\n\t"
        "vand.vx v2, v2, %[mask]\n\t"
        "vand.vx v4, v4, %[mask]\n\t"
        "vand.vx v6, v6, %[mask]\n\t"
        "vor.vv v8, v8, v0\n\t"
        "vor.vv v10, v10, v2\n\t"
        "vor.vv v12, v12, v4\n\t"
        "vor.vv v14, v14, v6\n\t"
        "lb zero, 127(%[q8])\n\t"
        "vsetvli zero, %[vl128], e8, m8\n\t"
        "vle8.v v0, (%[q8])\n\t"
        "vsub.vx v8, v8, %[vl32]\n\t"
        "vsetvli zero, %[vl64], e8, m4\n\t"
        "vwmul.vv v16, v0, v8\n\t"
        "vwmul.vv v24, v4, v12\n\t"
        "vsetivli zero, 16, e16, m2\n\t"
        "vmv.v.x v0, zero\n\t"
        "vwredsum.vs v10, v16, v0\n\t"
        "vwredsum.vs v9, v18, v0\n\t"
        "vwredsum.vs v8, v20, v0\n\t"
        "vwredsum.vs v7, v22, v0\n\t"
        "vwredsum.vs v11, v24, v0\n\t"
        "vwredsum.vs v12, v26, v0\n\t"
        "vwredsum.vs v13, v28, v0\n\t"
        "vwredsum.vs v14, v30, v0\n\t"
        "vsetivli zero, 4, e32, m1\n\t"
        "vmul.vx v0, v10, t0\n\t"
        "vmul.vx v1, v9, t1\n\t"
        "vmacc.vx v0, t2, v8\n\t"
        "vmacc.vx v1, t3, v7\n\t"
        "vmacc.vx v0, t4, v11\n\t"
        "vmacc.vx v1, t5, v12\n\t"
        "vmacc.vx v0, t6, v13\n\t"
        "vmacc.vx v1, a7, v14\n\t"
        "vadd.vv v0, v0, v1\n\t"
        "vfcvt.f.x.v v0, v0\n\t"
        "vfmv.f.s %[temporary], v0\n\t"
        "fmadd.s %[result], %[combined_scale], %[temporary], %[result]"
        : [low] "+&r"(low), [low_high] "=&r"(low_high),
          [scale] "+&r"(scale), [result] "+&f"(result),
          [temporary] "=&f"(temporary)
        : [high] "r"(high), [q8] "r"(q8), [vl32] "r"(32),
          [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30),
          [combined_scale] "f"(combined_scale)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
          "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7");
    high += 32;
    q8 += 128;
  }
  return result;
}

)c";
}

} // namespace

bool emitQ6LocalImplementation(llvm::raw_ostream &output,
                               const LocalImplementation &implementation) {
  if (implementation.primitive != LocalPrimitiveKind::Q6KI8 ||
      (implementation.operation.kind !=
           LocalHardwareOperationKind::RVVInlineAsm &&
       implementation.operation.kind !=
           LocalHardwareOperationKind::RVVIntrinsic) ||
      implementation.valueShapes.size() < 4 || !implementation.schedule)
    return false;
  if (implementation.operation.kind ==
      LocalHardwareOperationKind::RVVInlineAsm) {
    emitQ6KRVVAssembly(output, implementation);
    return true;
  }
  if (implementation.operation.kind !=
      LocalHardwareOperationKind::RVVIntrinsic)
    return false;

  const RVVVectorShape laneShape = implementation.valueShapes[0];
  const RVVVectorShape chunkShape = implementation.valueShapes[1];
  const RVVVectorShape productShape = implementation.valueShapes[2];
  const RVVVectorShape segmentProductShape = implementation.valueShapes[3];
  const std::string laneType =
      rvvVectorType(RVVElementCategory::SignedInteger, laneShape);
  const std::string laneSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, laneShape);
  const std::string chunkUnsignedType =
      rvvVectorType(RVVElementCategory::UnsignedInteger, chunkShape);
  const std::string chunkUnsignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::UnsignedInteger, chunkShape);
  const std::string chunkSignedType =
      rvvVectorType(RVVElementCategory::SignedInteger, chunkShape);
  const std::string chunkSignedSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, chunkShape);
  const std::string productType =
      rvvVectorType(RVVElementCategory::SignedInteger, productShape);
  const std::string productSuffix =
      rvvIntrinsicTypeSuffix(RVVElementCategory::SignedInteger, productShape);
  const std::string segmentProductSuffix = rvvIntrinsicTypeSuffix(
      RVVElementCategory::SignedInteger, segmentProductShape);
  const std::string laneSetVL = rvvSetVLIntrinsic(laneShape);
  const std::string chunkSetVL = rvvSetVLIntrinsic(chunkShape);
  if (laneType.empty() || laneSuffix.empty() ||
      chunkUnsignedType.empty() || chunkUnsignedSuffix.empty() ||
      chunkSignedType.empty() || chunkSignedSuffix.empty() ||
      productType.empty() || productSuffix.empty() ||
      segmentProductSuffix.empty() || laneSetVL.empty() || chunkSetVL.empty())
    return false;

  const unsigned chunksPerVector =
      implementation.schedule.decode.chunksPerStep;
  const unsigned segmentCount =
      implementation.schedule.decode.reductionSegments;
  output << "static inline __attribute__((always_inline, unused)) float\n"
         << localImplementationSymbol(implementation) << "(\n"
         << "    const uint8_t *low_bits, const uint8_t *high_bits,\n"
         << "    const uint8_t *group_scale_bytes, const uint8_t "
            "*activation_bytes,\n"
         << "    float weight_scale, float activation_scale, float init) {\n"
         << "  const int8_t *group_scales =\n"
         << "      (const int8_t *)(const void *)group_scale_bytes;\n"
         << "  const int8_t *activation =\n"
         << "      (const int8_t *)(const void *)activation_bytes;\n"
         << "  const size_t chunk_vl = " << chunkSetVL << "(32);\n"
         << "  const size_t vl = " << laneSetVL << "("
         << implementation.schedule.laneFactor << ");\n"
         << "  int32_t integer_sum = 0;\n"
         << "  for (size_t batch = 0; batch < "
         << implementation.schedule.sequentialIterations << "; ++batch) {\n";
  for (unsigned chunk = 0; chunk < chunksPerVector; ++chunk) {
    output << "    const size_t logical_chunk" << chunk << " = batch * "
           << chunksPerVector << " + " << chunk << ";\n"
           << "    const size_t half" << chunk << " = logical_chunk" << chunk
           << " / 4;\n"
           << "    const size_t quarter" << chunk << " = logical_chunk" << chunk
           << " % 4;\n"
           << "    const uint8_t *low" << chunk << " = low_bits + half" << chunk
           << " * 64 + (quarter" << chunk << " & 1) * 32;\n"
           << "    const uint8_t *high" << chunk << " = high_bits + half" << chunk
           << " * 32;\n"
           << "    const " << chunkUnsignedType << " packed" << chunk
           << " = __riscv_vle8_v_" << chunkUnsignedSuffix << "(low" << chunk
           << ", chunk_vl);\n"
           << "    const " << chunkUnsignedType << " high_bits" << chunk
           << " = __riscv_vle8_v_" << chunkUnsignedSuffix << "(high" << chunk
           << ", chunk_vl);\n"
           << "    const " << chunkUnsignedType << " low_code" << chunk
           << " = quarter" << chunk << " < 2\n"
           << "        ? __riscv_vand_vx_" << chunkUnsignedSuffix << "(packed"
           << chunk << ", UINT8_C(15), chunk_vl)\n"
           << "        : __riscv_vsrl_vx_" << chunkUnsignedSuffix << "(packed"
           << chunk << ", 4, chunk_vl);\n"
           << "    const " << chunkUnsignedType << " high_code" << chunk
           << " = __riscv_vand_vx_" << chunkUnsignedSuffix << "(\n"
           << "        __riscv_vsrl_vx_" << chunkUnsignedSuffix << "(high_bits"
           << chunk << ", 2 * quarter" << chunk
           << ", chunk_vl), UINT8_C(3), chunk_vl);\n"
           << "    const " << chunkSignedType << " codes" << chunk
           << " = __riscv_vsub_vx_" << chunkSignedSuffix << "(\n"
           << "        __riscv_vreinterpret_v_" << chunkUnsignedSuffix << "_"
           << chunkSignedSuffix << "(__riscv_vor_vv_" << chunkUnsignedSuffix
           << "(\n            low_code" << chunk << ", __riscv_vsll_vx_"
           << chunkUnsignedSuffix << "(high_code" << chunk
           << ", 4, chunk_vl), chunk_vl)),\n"
           << "        32, chunk_vl);\n"
           << "    const " << chunkSignedType << " activation" << chunk
           << " = __riscv_vle8_v_" << chunkSignedSuffix
           << "(activation + logical_chunk" << chunk << " * 32, chunk_vl);\n";
  }
  if (chunksPerVector == 1) {
    output << "    const " << laneType << " codes = codes0;\n"
           << "    const " << laneType << " q8 = activation0;\n";
  } else {
    output << "    const " << laneType << " codes = __riscv_vcreate_v_"
           << chunkSignedSuffix << "_" << laneSuffix
           << "(codes0, codes1);\n"
           << "    const " << laneType << " q8 = __riscv_vcreate_v_"
           << chunkSignedSuffix << "_" << laneSuffix
           << "(activation0, activation1);\n";
  }
  output << "    const " << productType
         << " product = __riscv_vwmul_vv_" << productSuffix
         << "(codes, q8, vl);\n"
         << "    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);\n";
  for (unsigned segment = 0; segment < segmentCount; ++segment) {
    output << "    const int32_t partial" << segment
           << " = __riscv_vmv_x_s_i32m1_i32(\n"
           << "        __riscv_vwredsum_vs_" << segmentProductSuffix
           << "_i32m1(\n            __riscv_vget_v_" << productSuffix << "_"
           << segmentProductSuffix << "(product, " << segment
           << "), zero, 16));\n"
           << "    integer_sum += partial" << segment
           << " * group_scales[batch * " << segmentCount << " + " << segment
           << "];\n";
  }
  output << "  }\n"
         << "  return init + (float)integer_sum * weight_scale * "
            "activation_scale;\n"
         << "}\n\n";
  return true;
}

} // namespace weft::riscv_internal
