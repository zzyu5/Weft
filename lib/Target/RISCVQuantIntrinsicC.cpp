#include "RISCVIntrinsicC.h"
#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace weft::riscv_internal {

#include "RISCVQuantCodebooks.inc"

namespace {

void emitI64Table(llvm::raw_ostream &output, llvm::StringRef name,
                  const int64_t *values, size_t count) {
  output << "static const int64_t " << name << "[" << count << "] = {\n";
  for (size_t index = 0; index < count; ++index) {
    if (index % 4 == 0)
      output << "  ";
    output << values[index];
    if (index + 1 != count)
      output << ", ";
    output << (index % 4 == 3 ? "\n" : "");
  }
  if (count % 4 != 0)
    output << "\n";
  output << "};\n\n";
}

void emitI32Table(llvm::raw_ostream &output, llvm::StringRef name,
                  const int32_t *values, size_t count) {
  output << "static const int32_t " << name << "[" << count << "] = {\n";
  for (size_t index = 0; index < count; ++index) {
    if (index % 8 == 0)
      output << "  ";
    output << values[index];
    if (index + 1 != count)
      output << ", ";
    output << (index % 8 == 7 ? "\n" : "");
  }
  if (count % 8 != 0)
    output << "\n";
  output << "};\n\n";
}

} // namespace

bool emitQuantLocalImplementations(llvm::raw_ostream &output,
                                   const SelectedLocalImplementations &selected,
                                   std::string &unsupportedSymbol) {
  const LocalImplementation *iq2Implementation = nullptr;
  const LocalImplementation *iq3Implementation = nullptr;
  const LocalImplementation *iq1Implementation = nullptr;
  const LocalImplementation *q6Implementation = nullptr;
  const LocalImplementation *packedI3Implementation = nullptr;
  bool supported = true;
  selected.forEach([&](const LocalImplementation &implementation) {
    if (!supported)
      return;
    const std::string symbol = localImplementationSymbol(implementation);
    switch (implementation.primitive) {
    case LocalPrimitiveKind::PackedI4I8:
      supported = emitPackedI4LocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI5I8:
      supported = emitPackedI5LocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI3GroupedI8:
      packedI3Implementation = &implementation;
      break;
    case LocalPrimitiveKind::Base3TernaryI8:
      supported = emitBase3TernaryLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedI2TernaryI8:
      supported =
          emitPackedI2TernaryLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::SignedCodebook8I8:
    case LocalPrimitiveKind::SignedCodebook4I8:
      supported =
          emitSignedCodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedU9U7CodebookI8:
      supported =
          emitPackedU9U7CodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::PackedU11GridDeltaI8:
      supported =
          emitPackedU11GridDeltaLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::NibbleCodebookI8:
      supported =
          emitNibbleCodebookLocalImplementation(output, implementation);
      break;
    case LocalPrimitiveKind::IQ2SI8:
      iq2Implementation = &implementation;
      break;
    case LocalPrimitiveKind::IQ3SI8:
      iq3Implementation = &implementation;
      break;
    case LocalPrimitiveKind::IQ1MI8:
      iq1Implementation = &implementation;
      break;
    case LocalPrimitiveKind::Q6KI8:
      q6Implementation = &implementation;
      break;
    default:
      return;
    }
    if (!supported)
      unsupportedSymbol = symbol.empty() ? "<unnamed local implementation>"
                                         : symbol;
  });
  if (!supported)
    return false;
  if (!iq2Implementation && !iq3Implementation && !iq1Implementation &&
      !q6Implementation && !packedI3Implementation)
    return true;
  if (iq1Implementation)
    emitI64Table(output, "__weft_iq1_m_grid", __weft_iq1_m_grid, 2048);
  if (iq2Implementation)
    emitI64Table(output, "__weft_iq2_s_grid", __weft_iq2_s_grid, 1024);
  if (iq3Implementation)
    emitI32Table(output, "__weft_iq3_s_grid", __weft_iq3_s_grid, 512);

  if (iq2Implementation &&
      !emitIQ2LocalImplementation(output, *iq2Implementation)) {
    unsupportedSymbol = localImplementationSymbol(*iq2Implementation);
    return false;
  }
  if (iq3Implementation &&
      !emitIQ3LocalImplementation(output, *iq3Implementation)) {
    unsupportedSymbol = localImplementationSymbol(*iq3Implementation);
    return false;
  }
  if (iq1Implementation &&
      !emitIQ1LocalImplementation(output, *iq1Implementation)) {
    unsupportedSymbol = localImplementationSymbol(*iq1Implementation);
    return false;
  }
  if (q6Implementation &&
      !emitQ6LocalImplementation(output, *q6Implementation)) {
    unsupportedSymbol = localImplementationSymbol(*q6Implementation);
    return false;
  }
  if (packedI3Implementation && !emitPackedI3GroupedLocalImplementation(
                                    output, *packedI3Implementation)) {
    unsupportedSymbol = localImplementationSymbol(*packedI3Implementation);
    return false;
  }
  return true;
}

} // namespace weft::riscv_internal
