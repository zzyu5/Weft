module {
  tcrv.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_q8_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %zero_seed = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_q8_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "robust", multi_block_factor = 2 : i64} {
        ^bb0(%block_index: index, %acc: f32):
          %dd = tcrv_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          %wv = tcrv_rvv.load %vx, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          %av = tcrv_rvv.load %vy, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          %prod = tcrv_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
          %red = tcrv_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %sumi = tcrv_rvv.typed_vector_lane0_to_scalar_extract %red, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          %bterm = tcrv_rvv.block_computed_scale_dequant %sumi, %dd {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %bterm {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
