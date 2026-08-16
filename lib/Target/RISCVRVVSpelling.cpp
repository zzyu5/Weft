#include "RISCVRVVSpelling.h"

namespace weft::riscv_internal {

std::string rvvShapeSuffix(const RVVVectorShape &shape) {
  if (!shape)
    return {};
  std::string lmul;
  if (shape.lmulEighths == 1)
    lmul = "mf8";
  else if (shape.lmulEighths == 2)
    lmul = "mf4";
  else if (shape.lmulEighths == 4)
    lmul = "mf2";
  else if (std::optional<unsigned> integer = rvvIntegerLMUL(shape))
    lmul = "m" + std::to_string(*integer);
  else
    return {};
  return std::to_string(shape.sew) + lmul;
}

std::string rvvIntrinsicTypeSuffix(RVVElementCategory category,
                                   const RVVVectorShape &shape) {
  std::string shapeSuffix = rvvShapeSuffix(shape);
  if (shapeSuffix.empty())
    return {};
  switch (category) {
  case RVVElementCategory::Floating:
    return "f" + shapeSuffix;
  case RVVElementCategory::SignedInteger:
    return "i" + shapeSuffix;
  case RVVElementCategory::UnsignedInteger:
    return "u" + shapeSuffix;
  }
  return {};
}

std::string rvvVectorType(RVVElementCategory category,
                          const RVVVectorShape &shape) {
  std::string shapeSuffix = rvvShapeSuffix(shape);
  if (shapeSuffix.empty())
    return {};
  switch (category) {
  case RVVElementCategory::Floating:
    return "vfloat" + shapeSuffix + "_t";
  case RVVElementCategory::SignedInteger:
    return "vint" + shapeSuffix + "_t";
  case RVVElementCategory::UnsignedInteger:
    return "vuint" + shapeSuffix + "_t";
  }
  return {};
}

std::string rvvMaskSuffix(unsigned maskRatio) {
  switch (maskRatio) {
  case 1:
  case 2:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
    return "b" + std::to_string(maskRatio);
  }
  return {};
}

std::string rvvMaskType(unsigned maskRatio) {
  std::string suffix = rvvMaskSuffix(maskRatio);
  return suffix.empty() ? std::string{} : "vbool" + suffix.substr(1) + "_t";
}

std::string rvvSetVLIntrinsic(const RVVVectorShape &shape) {
  std::string suffix = rvvShapeSuffix(shape);
  return suffix.empty() ? std::string{} : "__riscv_vsetvl_e" + suffix;
}

std::string rvvSetVLMaxIntrinsic(const RVVVectorShape &shape) {
  std::string suffix = rvvShapeSuffix(shape);
  return suffix.empty() ? std::string{} : "__riscv_vsetvlmax_e" + suffix;
}

} // namespace weft::riscv_internal
