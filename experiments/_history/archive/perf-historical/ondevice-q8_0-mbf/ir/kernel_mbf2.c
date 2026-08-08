#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
  size_t v10 = __riscv_vsetvl_e8m2(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v11;
  v11 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v12 = v1 / 32;
  size_t v13 = v12 % 2;
  size_t v14 = v12 - v13;
  for (size_t v15 = 0; v15 < v14; v15 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v16 = v15 * 34;
    const uint8_t* v17 = v4 + v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v18 = v15 * 34;
    const uint8_t* v19 = v6 + v18;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v21 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v22;
    v22 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v23 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v24 = v17 + 2;
    const uint8_t* v25 = v24 + 0;
    const int8_t* v26 = (const int8_t*) v25;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v27 = __riscv_vle8_v_i8m2(v26, v23);
    const uint8_t* v28 = v19 + 2;
    const uint8_t* v29 = v28 + 0;
    const int8_t* v30 = (const int8_t*) v29;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v31 = __riscv_vle8_v_i8m2(v30, v23);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v32 = __riscv_vwmul_vv_i16m4(v27, v31, v23);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v33 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v34 = __riscv_vwredsum_vs_i16m4_i32m1(v32, v33, v23);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v35 = __riscv_vmv_x_s_i32m1_i32(v34);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v35;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v36 = v15 + 1;
    size_t v37 = v36 * 34;
    const uint8_t* v38 = v4 + v37;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v39 = v15 + 1;
    size_t v40 = v39 * 34;
    const uint8_t* v41 = v6 + v40;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v38);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v43 = (float)*(const _Float16 *)(v41);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v44;
    v44 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v45 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v46 = v38 + 2;
    const uint8_t* v47 = v46 + 0;
    const int8_t* v48 = (const int8_t*) v47;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v49 = __riscv_vle8_v_i8m2(v48, v45);
    const uint8_t* v50 = v41 + 2;
    const uint8_t* v51 = v50 + 0;
    const int8_t* v52 = (const int8_t*) v51;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v53 = __riscv_vle8_v_i8m2(v52, v45);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v54 = __riscv_vwmul_vv_i16m4(v49, v53, v45);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v55 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v56 = __riscv_vwredsum_vs_i16m4_i32m1(v54, v55, v45);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v57 = __riscv_vmv_x_s_i32m1_i32(v56);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v44 = v57;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v58 = v22;
    float v59 = v11;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v59 + (float) v58 * (v20 * v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v60 = v44;
    float v61 = v11;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v61 + (float) v60 * (v42 * v43);
  }
  for (size_t v62 = v14; v62 < v12; v62 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v63 = v62 * 34;
    const uint8_t* v64 = v4 + v63;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v65 = v62 * 34;
    const uint8_t* v66 = v6 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v67 = (float)*(const _Float16 *)(v64);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v68 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v69;
    v69 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v70 = __riscv_vsetvl_e8m2(32);
    for (size_t v71 = 0; v71 < 32; v71 += v70) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v72 = 32 - v71;
      size_t v73 = __riscv_vsetvl_e8m2(v72);
      const uint8_t* v74 = v64 + 2;
      const uint8_t* v75 = v74 + v71;
      const int8_t* v76 = (const int8_t*) v75;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v77 = __riscv_vle8_v_i8m2(v76, v73);
      const uint8_t* v78 = v66 + 2;
      const uint8_t* v79 = v78 + v71;
      const int8_t* v80 = (const int8_t*) v79;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v81 = __riscv_vle8_v_i8m2(v80, v73);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v82 = __riscv_vwmul_vv_i16m4(v77, v81, v73);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v83 = v69;
      vint32m1_t v84 = __riscv_vmv_v_x_i32m1(v83, 1);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v85 = __riscv_vwredsum_vs_i16m4_i32m1(v82, v84, v73);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v86 = __riscv_vmv_x_s_i32m1_i32(v85);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v69 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v87 = v69;
    float v88 = v11;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v88 + (float) v87 * (v67 * v68);
  }
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v89 = v11;
  v2[0] = v89;
  return;
}


