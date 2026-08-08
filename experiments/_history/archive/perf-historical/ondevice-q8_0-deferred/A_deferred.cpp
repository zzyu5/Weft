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
  size_t v13 = v12 % 4;
  size_t v14 = v12 - v13;
  for (size_t v15 = 0; v15 < v14; v15 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=deferred_sumi_pack
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v16 = __riscv_vmv_v_x_i32m1(0, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v17 = v15 * 34;
    const uint8_t* v18 = v4 + v17;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v19 = v15 * 34;
    const uint8_t* v20 = v6 + v19;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v21 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v22 = v18 + 2;
    const int8_t* v23 = (const int8_t*) v22;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v24 = __riscv_vle8_v_i8m2(v23, v21);
    const uint8_t* v25 = v20 + 2;
    const int8_t* v26 = (const int8_t*) v25;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v27 = __riscv_vle8_v_i8m2(v26, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v28 = __riscv_vwmul_vv_i16m4(v24, v27, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v29 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v30 = __riscv_vwredsum_vs_i16m4_i32m1(v28, v29, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v31 = __riscv_vmv_x_s_i32m1_i32(v30);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslide1down_vx_i32m1
    vint32m1_t v32 = __riscv_vslide1down_vx_i32m1(v16, v31, 4);
    size_t v33 = v15 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v34 = v33 * 34;
    const uint8_t* v35 = v4 + v34;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v36 = v33 * 34;
    const uint8_t* v37 = v6 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v38 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v39 = v35 + 2;
    const int8_t* v40 = (const int8_t*) v39;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v41 = __riscv_vle8_v_i8m2(v40, v38);
    const uint8_t* v42 = v37 + 2;
    const int8_t* v43 = (const int8_t*) v42;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v44 = __riscv_vle8_v_i8m2(v43, v38);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v45 = __riscv_vwmul_vv_i16m4(v41, v44, v38);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v46 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v47 = __riscv_vwredsum_vs_i16m4_i32m1(v45, v46, v38);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v48 = __riscv_vmv_x_s_i32m1_i32(v47);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslide1down_vx_i32m1
    vint32m1_t v49 = __riscv_vslide1down_vx_i32m1(v32, v48, 4);
    size_t v50 = v15 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v51 = v50 * 34;
    const uint8_t* v52 = v4 + v51;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v53 = v50 * 34;
    const uint8_t* v54 = v6 + v53;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v55 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v56 = v52 + 2;
    const int8_t* v57 = (const int8_t*) v56;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v58 = __riscv_vle8_v_i8m2(v57, v55);
    const uint8_t* v59 = v54 + 2;
    const int8_t* v60 = (const int8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v61 = __riscv_vle8_v_i8m2(v60, v55);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v62 = __riscv_vwmul_vv_i16m4(v58, v61, v55);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v63 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v64 = __riscv_vwredsum_vs_i16m4_i32m1(v62, v63, v55);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslide1down_vx_i32m1
    vint32m1_t v66 = __riscv_vslide1down_vx_i32m1(v49, v65, 4);
    size_t v67 = v15 + 3;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v68 = v67 * 34;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v70 = v67 * 34;
    const uint8_t* v71 = v6 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v72 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v73 = v69 + 2;
    const int8_t* v74 = (const int8_t*) v73;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v75 = __riscv_vle8_v_i8m2(v74, v72);
    const uint8_t* v76 = v71 + 2;
    const int8_t* v77 = (const int8_t*) v76;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v78 = __riscv_vle8_v_i8m2(v77, v72);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v79 = __riscv_vwmul_vv_i16m4(v75, v78, v72);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v80 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v81 = __riscv_vwredsum_vs_i16m4_i32m1(v79, v80, v72);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v82 = __riscv_vmv_x_s_i32m1_i32(v81);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslide1down_vx_i32m1
    vint32m1_t v83 = __riscv_vslide1down_vx_i32m1(v66, v82, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=deferred_scale_gather
    size_t v84 = v15 * 34;
    const uint8_t* v85 = v4 + v84;
    const _Float16* v86 = (const _Float16*) v85;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_f16mf2
    vfloat16mf2_t v87 = __riscv_vlse16_v_f16mf2(v86, 34, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m1
    vfloat32m1_t v88 = __riscv_vfwcvt_f_f_v_f32m1(v87, 4);
    size_t v89 = v15 * 34;
    const uint8_t* v90 = v6 + v89;
    const _Float16* v91 = (const _Float16*) v90;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_f16mf2
    vfloat16mf2_t v92 = __riscv_vlse16_v_f16mf2(v91, 34, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m1
    vfloat32m1_t v93 = __riscv_vfwcvt_f_f_v_f32m1(v92, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=deferred_ordered_fold
    float v94 = v11;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
    vfloat32m1_t v95 = __riscv_vfmv_v_f_f32m1(v94, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v96 = __riscv_vfcvt_f_x_v_f32m1(v83, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m1
    vfloat32m1_t v97 = __riscv_vfmul_vv_f32m1(v96, v88, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m1
    vfloat32m1_t v98 = __riscv_vfmul_vv_f32m1(v97, v93, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfredosum_vs_f32m1_f32m1
    vfloat32m1_t v99 = __riscv_vfredosum_vs_f32m1_f32m1(v98, v95, 4);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f32m1_f32
    float v100 = __riscv_vfmv_f_s_f32m1_f32(v99);
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v100;
  }
  for (size_t v101 = v14; v101 < v12; v101 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v102 = v101 * 34;
    const uint8_t* v103 = v4 + v102;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v104 = v101 * 34;
    const uint8_t* v105 = v6 + v104;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v106 = (float)*(const _Float16 *)(v103);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v107 = (float)*(const _Float16 *)(v105);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v108;
    v108 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v109 = __riscv_vsetvl_e8m2(32);
    for (size_t v110 = 0; v110 < 32; v110 += v109) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v111 = 32 - v110;
      size_t v112 = __riscv_vsetvl_e8m2(v111);
      const uint8_t* v113 = v103 + 2;
      const uint8_t* v114 = v113 + v110;
      const int8_t* v115 = (const int8_t*) v114;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v116 = __riscv_vle8_v_i8m2(v115, v112);
      const uint8_t* v117 = v105 + 2;
      const uint8_t* v118 = v117 + v110;
      const int8_t* v119 = (const int8_t*) v118;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v120 = __riscv_vle8_v_i8m2(v119, v112);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v121 = __riscv_vwmul_vv_i16m4(v116, v120, v112);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v122 = v108;
      vint32m1_t v123 = __riscv_vmv_v_x_i32m1(v122, 1);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v124 = __riscv_vwredsum_vs_i16m4_i32m1(v121, v123, v112);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v125 = __riscv_vmv_x_s_i32m1_i32(v124);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v108 = v125;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v126 = v108;
    float v127 = v11;
    float v128 = (float) v126;
    float v129 = v128 * v106;
    float v130 = v129 * v107;
    float v131 = v127 + v130;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v131;
  }
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v132 = v11;
  v2[0] = v132;
  return;
}


