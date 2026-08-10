#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  uint32_t v7[4];
  // tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v8[8];
  // tcrv_emitc.local_variable=sums source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat32m2_t v9;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v10 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v9 = v10;
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v11;
  v11 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v12 = 0; v12 < v6; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v13 = v12 * 176;
    const uint8_t* v14 = v3 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v15 = v12 * 292;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_bit_dance
    const uint8_t* v17 = v14 + 4;
    const uint32_t* v18 = (const uint32_t*) v17;
    const uint32_t v19 = v18[0];
    uint32_t v20 = (uint32_t) v19;
    const uint32_t v21 = v18[1];
    uint32_t v22 = (uint32_t) v21;
    const uint32_t v23 = v18[2];
    uint32_t v24 = (uint32_t) v23;
    uint32_t v25 = v22 >> 6;
    uint32_t v26 = v25 & 0x03030303;
    uint32_t v27 = v26 << 4;
    uint32_t v28 = v24 >> 4;
    uint32_t v29 = v28 & 0x0f0f0f0f;
    uint32_t v30 = v29 | v27;
    uint32_t v31 = v22 & 0x3f3f3f3f;
    uint32_t v32 = v20 >> 6;
    uint32_t v33 = v32 & 0x03030303;
    uint32_t v34 = v33 << 4;
    uint32_t v35 = v24 & 0x0f0f0f0f;
    uint32_t v36 = v35 | v34;
    uint32_t v37 = v20 & 0x3f3f3f3f;
    v7[0] = v37;
    v7[1] = v36;
    v7[2] = v31;
    v7[3] = v30;
    uint32_t* v38 = &v7[0];
    const uint8_t* v39 = (const uint8_t*) v38;
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v40;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v41 = __riscv_vmv_v_x_i32m2(0, 8);
    v40 = v41;
    const uint8_t* v42 = v16 + 4;
    const int8_t* v43 = (const int8_t*) v42;
    const uint8_t* v44 = v14 + 48;
    const uint8_t* v45 = (const uint8_t*) v44;
    const uint8_t* v46 = v14 + 16;
    const uint8_t* v47 = (const uint8_t*) v46;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v48 = 0; v48 < 8; v48 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const uint8_t v49 = v39[v48];
      int v50 = (int) v49;
      size_t v51 = v48 * 32;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=register_resident_qs_group
      size_t v52 = v48 >> 1;
      size_t v53 = v52 * 32;
      const uint8_t* v54 = v45 + v53;
      size_t v55 = v48 & 1;
      size_t v56 = v55 * 4;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v57 = __riscv_vsetvl_e8mf2(8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=register_resident_decode
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v58 = __riscv_vle8_v_u8mf2(v54, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v59 = __riscv_vsrl_vx_u8mf2(v58, v56, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v60 = __riscv_vand_vx_u8mf2(v59, 0x0F, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v61 = __riscv_vle8_v_u8mf2(v47, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v62 = __riscv_vsrl_vx_u8mf2(v61, v48, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v63 = __riscv_vand_vx_u8mf2(v62, 0x01, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v64 = __riscv_vsll_vx_u8mf2(v63, 0x04, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8mf2
      vuint8mf2_t v65 = __riscv_vadd_vv_u8mf2(v60, v64, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v66 = __riscv_vreinterpret_v_u8mf2_i8mf2(v65);
      const int8_t* v67 = v43 + v51;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v68 = __riscv_vle8_v_i8mf2(v67, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v69 = __riscv_vwmul_vv_i16m1(v68, v66, v57);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v70 = v40;
      vint32m2_t v71 = __riscv_vwmacc_vx_i32m2(v70, v50, v69, v57);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v40 = v71;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v72 = __riscv_vsetvl_e8mf2(8);
      size_t v73 = v51 + 8;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=register_resident_decode
      const uint8_t* v74 = v54 + 8;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v75 = __riscv_vle8_v_u8mf2(v74, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v76 = __riscv_vsrl_vx_u8mf2(v75, v56, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v77 = __riscv_vand_vx_u8mf2(v76, 0x0F, v72);
      const uint8_t* v78 = v47 + 8;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v79 = __riscv_vle8_v_u8mf2(v78, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v80 = __riscv_vsrl_vx_u8mf2(v79, v48, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v81 = __riscv_vand_vx_u8mf2(v80, 0x01, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v82 = __riscv_vsll_vx_u8mf2(v81, 0x04, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8mf2
      vuint8mf2_t v83 = __riscv_vadd_vv_u8mf2(v77, v82, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v84 = __riscv_vreinterpret_v_u8mf2_i8mf2(v83);
      const int8_t* v85 = v43 + v73;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v86 = __riscv_vle8_v_i8mf2(v85, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v87 = __riscv_vwmul_vv_i16m1(v86, v84, v72);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v88 = v40;
      vint32m2_t v89 = __riscv_vwmacc_vx_i32m2(v88, v50, v87, v72);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v40 = v89;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v90 = __riscv_vsetvl_e8mf2(8);
      size_t v91 = v51 + 16;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=register_resident_decode
      const uint8_t* v92 = v54 + 16;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v93 = __riscv_vle8_v_u8mf2(v92, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v94 = __riscv_vsrl_vx_u8mf2(v93, v56, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v95 = __riscv_vand_vx_u8mf2(v94, 0x0F, v90);
      const uint8_t* v96 = v47 + 16;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v97 = __riscv_vle8_v_u8mf2(v96, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v98 = __riscv_vsrl_vx_u8mf2(v97, v48, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v99 = __riscv_vand_vx_u8mf2(v98, 0x01, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v100 = __riscv_vsll_vx_u8mf2(v99, 0x04, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8mf2
      vuint8mf2_t v101 = __riscv_vadd_vv_u8mf2(v95, v100, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v102 = __riscv_vreinterpret_v_u8mf2_i8mf2(v101);
      const int8_t* v103 = v43 + v91;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v104 = __riscv_vle8_v_i8mf2(v103, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v105 = __riscv_vwmul_vv_i16m1(v104, v102, v90);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v106 = v40;
      vint32m2_t v107 = __riscv_vwmacc_vx_i32m2(v106, v50, v105, v90);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v40 = v107;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v108 = __riscv_vsetvl_e8mf2(8);
      size_t v109 = v51 + 24;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=register_resident_decode
      const uint8_t* v110 = v54 + 24;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v111 = __riscv_vle8_v_u8mf2(v110, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v112 = __riscv_vsrl_vx_u8mf2(v111, v56, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v113 = __riscv_vand_vx_u8mf2(v112, 0x0F, v108);
      const uint8_t* v114 = v47 + 24;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v115 = __riscv_vle8_v_u8mf2(v114, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v116 = __riscv_vsrl_vx_u8mf2(v115, v48, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v117 = __riscv_vand_vx_u8mf2(v116, 0x01, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v118 = __riscv_vsll_vx_u8mf2(v117, 0x04, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8mf2
      vuint8mf2_t v119 = __riscv_vadd_vv_u8mf2(v113, v118, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v119);
      const int8_t* v121 = v43 + v109;
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v122 = __riscv_vle8_v_i8mf2(v121, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v123 = __riscv_vwmul_vv_i16m1(v122, v120, v108);
      // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v124 = v40;
      vint32m2_t v125 = __riscv_vwmacc_vx_i32m2(v124, v50, v123, v108);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v40 = v125;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v126 = (const float*) v16;
    const float v127 = v126[0];
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v128 = v16 + 260;
    const int16_t* v129 = (const int16_t*) v128;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v130;
    v130 = 0;
    const int16_t v131 = v129[0];
    int v132 = (int) v131;
    const uint8_t v133 = v39[8];
    int v134 = (int) v133;
    int v135 = v132 * v134;
    int v136 = v130;
    int v137 = v136 + v135;
    v130 = v137;
    const int16_t v138 = v129[1];
    int v139 = (int) v138;
    const uint8_t v140 = v39[8];
    int v141 = (int) v140;
    int v142 = v139 * v141;
    int v143 = v130;
    int v144 = v143 + v142;
    v130 = v144;
    const int16_t v145 = v129[2];
    int v146 = (int) v145;
    const uint8_t v147 = v39[9];
    int v148 = (int) v147;
    int v149 = v146 * v148;
    int v150 = v130;
    int v151 = v150 + v149;
    v130 = v151;
    const int16_t v152 = v129[3];
    int v153 = (int) v152;
    const uint8_t v154 = v39[9];
    int v155 = (int) v154;
    int v156 = v153 * v155;
    int v157 = v130;
    int v158 = v157 + v156;
    v130 = v158;
    const int16_t v159 = v129[4];
    int v160 = (int) v159;
    const uint8_t v161 = v39[10];
    int v162 = (int) v161;
    int v163 = v160 * v162;
    int v164 = v130;
    int v165 = v164 + v163;
    v130 = v165;
    const int16_t v166 = v129[5];
    int v167 = (int) v166;
    const uint8_t v168 = v39[10];
    int v169 = (int) v168;
    int v170 = v167 * v169;
    int v171 = v130;
    int v172 = v171 + v170;
    v130 = v172;
    const int16_t v173 = v129[6];
    int v174 = (int) v173;
    const uint8_t v175 = v39[11];
    int v176 = (int) v175;
    int v177 = v174 * v176;
    int v178 = v130;
    int v179 = v178 + v177;
    v130 = v179;
    const int16_t v180 = v129[7];
    int v181 = (int) v180;
    const uint8_t v182 = v39[11];
    int v183 = (int) v182;
    int v184 = v181 * v183;
    int v185 = v130;
    int v186 = v185 + v184;
    v130 = v186;
    const int16_t v187 = v129[8];
    int v188 = (int) v187;
    const uint8_t v189 = v39[12];
    int v190 = (int) v189;
    int v191 = v188 * v190;
    int v192 = v130;
    int v193 = v192 + v191;
    v130 = v193;
    const int16_t v194 = v129[9];
    int v195 = (int) v194;
    const uint8_t v196 = v39[12];
    int v197 = (int) v196;
    int v198 = v195 * v197;
    int v199 = v130;
    int v200 = v199 + v198;
    v130 = v200;
    const int16_t v201 = v129[10];
    int v202 = (int) v201;
    const uint8_t v203 = v39[13];
    int v204 = (int) v203;
    int v205 = v202 * v204;
    int v206 = v130;
    int v207 = v206 + v205;
    v130 = v207;
    const int16_t v208 = v129[11];
    int v209 = (int) v208;
    const uint8_t v210 = v39[13];
    int v211 = (int) v210;
    int v212 = v209 * v211;
    int v213 = v130;
    int v214 = v213 + v212;
    v130 = v214;
    const int16_t v215 = v129[12];
    int v216 = (int) v215;
    const uint8_t v217 = v39[14];
    int v218 = (int) v217;
    int v219 = v216 * v218;
    int v220 = v130;
    int v221 = v220 + v219;
    v130 = v221;
    const int16_t v222 = v129[13];
    int v223 = (int) v222;
    const uint8_t v224 = v39[14];
    int v225 = (int) v224;
    int v226 = v223 * v225;
    int v227 = v130;
    int v228 = v227 + v226;
    v130 = v228;
    const int16_t v229 = v129[14];
    int v230 = (int) v229;
    const uint8_t v231 = v39[15];
    int v232 = (int) v231;
    int v233 = v230 * v232;
    int v234 = v130;
    int v235 = v234 + v233;
    v130 = v235;
    const int16_t v236 = v129[15];
    int v237 = (int) v236;
    const uint8_t v238 = v39[15];
    int v239 = (int) v238;
    int v240 = v237 * v239;
    int v241 = v130;
    int v242 = v241 + v240;
    v130 = v242;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v243 = (float)*(const _Float16 *)(v14);
    float v244 = v243 * v127;
    vint32m2_t v245 = v40;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v246 = __riscv_vfcvt_f_x_v_f32m2(v245, 8);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v247 = __riscv_vfmul_vf_f32m2(v246, v244, 8);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v248 = v9;
    vfloat32m2_t v249 = __riscv_vfadd_vv_f32m2(v248, v247, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v249;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v250 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v251 = (float)*(const _Float16 *)(v250);
    float v252 = v251 * v127;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract
    int v253 = v130;
    float v254 = v11;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v254 - v252 * (float) v253;
  }
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v255 = &v8[0];
  vfloat32m2_t v256 = v9;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v255, v256, 8);
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v257 = v11;
  float v258 = v8[0];
  float v259 = v257 + v258;
  float v260 = v8[1];
  float v261 = v259 + v260;
  float v262 = v8[2];
  float v263 = v261 + v262;
  float v264 = v8[3];
  float v265 = v263 + v264;
  float v266 = v8[4];
  float v267 = v265 + v266;
  float v268 = v8[5];
  float v269 = v267 + v268;
  float v270 = v8[6];
  float v271 = v269 + v270;
  float v272 = v8[7];
  float v273 = v271 + v272;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v273;
  return;
}


