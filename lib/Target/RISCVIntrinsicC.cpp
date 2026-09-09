#include "Emission/Emitter.h"

using namespace weft;
using namespace weft::riscv_emission;


mlir::LogicalResult weft::emitSelectedRISCVIntrinsicC(mlir::ModuleOp module,
                                                      std::string &result,
                                                      std::vector<RISCVKernelABI> &kernels) {
  std::string body;
  llvm::raw_string_ostream output(body);
  output << "#include <stddef.h>\n"
         << "#include <stdint.h>\n"
         << "#include <stdbool.h>\n"
         << "#include <math.h>\n"
         << "#include <string.h>\n"
         << "#include <riscv_vector.h>\n\n"
         << "static inline __attribute__((unused)) int16_t weft_load_i16_le(const uint8_t *p) {\n"
         << "  int16_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint16_t weft_load_u16_le(const uint8_t *p) {\n"
         << "  uint16_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint32_t weft_load_u32_le(const uint8_t *p) {\n"
         << "  uint32_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) _Float16 weft_load_f16_le(const uint8_t *p) {\n"
         << "  _Float16 value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_load_f32_le(const uint8_t *p) {\n"
         << "  float value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_i16_le(uint8_t *p, int16_t value) {\n"
         << "  uint16_t bits = (uint16_t)value; p[0] = (uint8_t)bits; p[1] = (uint8_t)(bits >> 8);\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f16_le(uint8_t *p, _Float16 value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f32_le(uint8_t *p, float value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "\n";
  bool found = false;
  kernels.clear();
  for (riscv::ArtifactPackOp artifact :
       module.getOps<riscv::ArtifactPackOp>()) {
    Emitter emitter(module, artifact, output);
    if (mlir::failed(emitter.emitArtifact()))
      return mlir::failure();
  }
  for (riscv::KernelOp kernel : module.getOps<riscv::KernelOp>()) {
    Emitter emitter(module, kernel, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
    auto abi = emitter.kernelABI();
    if (mlir::failed(abi))
      return mlir::failure();
    kernels.push_back(std::move(*abi));
    found = true;
  }
  if (!found)
    return module.emitError("intrinsic-C emission requires a verified RISC-V physical kernel");
  output.flush();
  result = std::move(body);
  return mlir::success();
}
