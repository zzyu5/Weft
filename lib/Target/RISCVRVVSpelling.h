#ifndef WEFT_LIB_TARGET_RISCVRVVSPELLING_H
#define WEFT_LIB_TARGET_RISCVRVVSPELLING_H

#include "RISCVPhysicalPlanning.h"

#include <string>

namespace weft::riscv_internal {

enum class RVVElementCategory {
  Floating,
  SignedInteger,
  UnsignedInteger,
};

std::string rvvShapeSuffix(const RVVVectorShape &shape);
std::string rvvIntrinsicTypeSuffix(RVVElementCategory category,
                                   const RVVVectorShape &shape);
std::string rvvVectorType(RVVElementCategory category,
                          const RVVVectorShape &shape);
std::string rvvMaskSuffix(unsigned maskRatio);
std::string rvvMaskType(unsigned maskRatio);
std::string rvvSetVLIntrinsic(const RVVVectorShape &shape);
std::string rvvSetVLMaxIntrinsic(const RVVVectorShape &shape);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVRVVSPELLING_H
