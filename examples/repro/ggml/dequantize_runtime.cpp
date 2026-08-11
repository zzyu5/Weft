#include "ggml-quants.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kN = 1024;
constexpr std::size_t kK = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

using dequantize_fn = void (*)(const void *, float *, std::int64_t);
using quantize_fn = void (*)(const float *, void *, std::int64_t);

template <typename Block,
          void (*Function)(const Block *, float *, std::int64_t)>
void dequantize_adapter(const void *input, float *output, std::int64_t count) {
  Function(static_cast<const Block *>(input), output, count);
}

struct dequantize_case {
  const char *name;
  ggml_type type;
  dequantize_fn dequantize;
  quantize_fn quantize;
};

const dequantize_case cases[] = {
    {"q1_0", GGML_TYPE_Q1_0,
     dequantize_adapter<block_q1_0, dequantize_row_q1_0>, quantize_row_q1_0},
    {"q4_0", GGML_TYPE_Q4_0,
     dequantize_adapter<block_q4_0, dequantize_row_q4_0>, quantize_row_q4_0},
    {"q4_1", GGML_TYPE_Q4_1,
     dequantize_adapter<block_q4_1, dequantize_row_q4_1>, quantize_row_q4_1},
    {"q5_0", GGML_TYPE_Q5_0,
     dequantize_adapter<block_q5_0, dequantize_row_q5_0>, quantize_row_q5_0},
    {"q5_1", GGML_TYPE_Q5_1,
     dequantize_adapter<block_q5_1, dequantize_row_q5_1>, quantize_row_q5_1},
    {"q8_0", GGML_TYPE_Q8_0,
     dequantize_adapter<block_q8_0, dequantize_row_q8_0>, quantize_row_q8_0},
    {"mxfp4", GGML_TYPE_MXFP4,
     dequantize_adapter<block_mxfp4, dequantize_row_mxfp4>, quantize_row_mxfp4},
    {"nvfp4", GGML_TYPE_NVFP4,
     dequantize_adapter<block_nvfp4, dequantize_row_nvfp4>, quantize_row_nvfp4},
    {"q2_K", GGML_TYPE_Q2_K,
     dequantize_adapter<block_q2_K, dequantize_row_q2_K>, quantize_row_q2_K},
    {"q3_K", GGML_TYPE_Q3_K,
     dequantize_adapter<block_q3_K, dequantize_row_q3_K>, quantize_row_q3_K},
    {"q4_K", GGML_TYPE_Q4_K,
     dequantize_adapter<block_q4_K, dequantize_row_q4_K>, quantize_row_q4_K},
    {"q5_K", GGML_TYPE_Q5_K,
     dequantize_adapter<block_q5_K, dequantize_row_q5_K>, quantize_row_q5_K},
    {"q6_K", GGML_TYPE_Q6_K,
     dequantize_adapter<block_q6_K, dequantize_row_q6_K>, quantize_row_q6_K},
    {"tq1_0", GGML_TYPE_TQ1_0,
     dequantize_adapter<block_tq1_0, dequantize_row_tq1_0>, quantize_row_tq1_0},
    {"tq2_0", GGML_TYPE_TQ2_0,
     dequantize_adapter<block_tq2_0, dequantize_row_tq2_0>, quantize_row_tq2_0},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS,
     dequantize_adapter<block_iq2_xxs, dequantize_row_iq2_xxs>, nullptr},
    {"iq2_xs", GGML_TYPE_IQ2_XS,
     dequantize_adapter<block_iq2_xs, dequantize_row_iq2_xs>, nullptr},
    {"iq2_s", GGML_TYPE_IQ2_S,
     dequantize_adapter<block_iq2_s, dequantize_row_iq2_s>, nullptr},
    {"iq3_xxs", GGML_TYPE_IQ3_XXS,
     dequantize_adapter<block_iq3_xxs, dequantize_row_iq3_xxs>, nullptr},
    {"iq3_s", GGML_TYPE_IQ3_S,
     dequantize_adapter<block_iq3_s, dequantize_row_iq3_s>, nullptr},
    {"iq1_s", GGML_TYPE_IQ1_S,
     dequantize_adapter<block_iq1_s, dequantize_row_iq1_s>, nullptr},
    {"iq1_m", GGML_TYPE_IQ1_M,
     dequantize_adapter<block_iq1_m, dequantize_row_iq1_m>, nullptr},
    {"iq4_nl", GGML_TYPE_IQ4_NL,
     dequantize_adapter<block_iq4_nl, dequantize_row_iq4_nl>,
     quantize_row_iq4_nl},
    {"iq4_xs", GGML_TYPE_IQ4_XS,
     dequantize_adapter<block_iq4_xs, dequantize_row_iq4_xs>,
     quantize_row_iq4_xs},
};

volatile std::uint64_t flush_sink = 0;

const dequantize_case *find_case(const char *name) {
  for (const dequantize_case &candidate : cases) {
    if (std::strcmp(name, candidate.name) == 0) {
      return &candidate;
    }
  }
  return nullptr;
}

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0) {
    return 0;
  }
  return static_cast<std::size_t>(value);
}

void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t i = 0; i < buffer.size(); i += 64) {
    buffer[i] = static_cast<std::uint8_t>(buffer[i] + 1U);
    flush_sink += buffer[i];
  }
}

void run_dequantize(const dequantize_case &selected,
                    const std::uint8_t *input, float *output,
                    std::size_t input_row_bytes) {
  for (std::size_t row = 0; row < kN; ++row) {
    selected.dequantize(input + row * input_row_bytes, output + row * kK, kK);
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <dequantize-format> <repetitions>\n", argv[0]);
    return 2;
  }
  const dequantize_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML dequantizer: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  std::vector<float> source(kK);
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0F;
  }
  const std::size_t input_row_bytes = ggml_row_size(selected->type, kK);
  std::vector<std::uint8_t> input_row(input_row_bytes, 0);
  if (selected->quantize != nullptr) {
    selected->quantize(source.data(), input_row.data(), kK);
  }
  std::vector<std::uint8_t> input(kN * input_row_bytes);
  for (std::size_t row = 0; row < kN; ++row) {
    std::memcpy(input.data() + row * input_row_bytes, input_row.data(),
                input_row_bytes);
  }
  std::vector<float> output(kN * kK);
  std::vector<std::uint8_t> flush_buffer(kFlushBytes, 1);

  run_dequantize(*selected, input.data(), output.data(), input_row_bytes);
  for (float value : output) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite output from dequantize_row_%s\n", argv[1]);
      return 1;
    }
  }

  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush_buffer);
    const auto begin = std::chrono::steady_clock::now();
    run_dequantize(*selected, input.data(), output.data(), input_row_bytes);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double cold_median_us = median(samples);
  const double elements = static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=dequantize\n");
  std::printf("kernel=dequantize_row_%s\n", selected->name);
  std::printf("implementation=scalar\n");
  std::printf("model_shape=DeepSeek-R1-Distill-Llama-8B.attn_k\n");
  std::printf("N=%zu\nK=%zu\n", kN, kK);
  std::printf("cold_protocol=64MiB-evict-then-full-tensor\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_melements_s=%.6f\n", elements / cold_median_us);
  std::printf("output_sample=%.9g\n", output[0]);
  return 0;
}
