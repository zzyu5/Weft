#ifndef WEFT_LIB_TARGET_RISCVINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVINTRINSICC_H

#include "RISCVPhysicalPlanning.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <set>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

std::string rvvShapeSuffix(const RVVVectorShape &shape);

enum class IntrinsicCLeaf {
  None,
  RVVF32M2Math,
  RVVSymmetricI4I8N16K32,
  RVVAffineI4I8N16K32,
  RVVSymmetricI4I8M4N16K32,
  RVVAffineI4I8M4N16K32,
  IME1SymmetricI4I8N16K32,
  IME1AffineI4I8N16K32,
  IME1SymmetricI4I8M4N16K32,
  IME1AffineI4I8M4N16K32,
  GroupedAffineI4I8VLEN128,
  GroupedAffineI4I8VLEN256,
  GroupedAffineI4I8Scalable,
  E2M1E8M0I8VLEN128,
  E2M1E8M0I8VLEN256,
  E2M1E8M0I8Scalable,
  PackedI4I8VLEN128,
  PackedI4I8VLEN256,
  PackedI5I8VLEN128,
  PackedI5I8VLEN256,
  Base3TernaryI8VLEN128,
  Base3TernaryI8VLEN256,
  PackedI2TernaryI8VLEN128,
  PackedI2TernaryI8VLEN256,
  SignedCodebook8I8VLEN128,
  SignedCodebook8I8VLEN256,
  SignedCodebook4I8VLEN128,
  SignedCodebook4I8VLEN256,
  PackedU9U7CodebookI8VLEN128,
  PackedU9U7CodebookI8VLEN256,
  PackedU11GridDeltaI8VLEN128,
  PackedU11GridDeltaI8VLEN256,
  IQ2SI8FixedLanes32,
  IQ2SI8FixedLanes64,
  IQ2SI8Scalable,
  IQ3SI8FixedLanes64,
  IQ3SI8Scalable,
  IQ1MI8FixedLanes32,
  IQ1MI8FixedLanes64,
  IQ1MI8Scalable,
  Q6KI8FixedLanes32,
  Q6KI8FixedLanes64,
  Q6KI8Scalable,
};

class SelectedIntrinsicCLeaves {
public:
  void add(IntrinsicCLeaf leaf) {
    if (leaf != IntrinsicCLeaf::None)
      leaves.insert(leaf);
  }
  bool contains(IntrinsicCLeaf leaf) const {
    return leaves.find(leaf) != leaves.end();
  }
  bool empty() const { return leaves.empty(); }

private:
  std::set<IntrinsicCLeaf> leaves;
};

llvm::StringRef intrinsicCLeafName(IntrinsicCLeaf leaf);
void emitRVVIntrinsicCLeaves(llvm::raw_ostream &output,
                             bool usesRVVSymmetricI4I8,
                             bool usesRVVAffineI4I8,
                             bool usesRVVSymmetricI4I8M4,
                             bool usesRVVAffineI4I8M4,
                             bool usesGroupedI4I8VLEN128,
                             bool usesGroupedI4I8VLEN256,
                             bool usesGroupedI4I8Scalable,
                             bool usesE2M1VLEN128, bool usesE2M1VLEN256,
                             bool usesE2M1Scalable);
void emitQuantIntrinsicCLeaves(llvm::raw_ostream &output,
                               const SelectedIntrinsicCLeaves &leaves);
void emitIMEIntrinsicCLeaves(llvm::raw_ostream &output,
                             bool usesIME1SymmetricI4I8,
                             bool usesIME1AffineI4I8,
                             bool usesIME1SymmetricI4I8M4,
                             bool usesIME1AffineI4I8M4);
void emitIntrinsicCPrelude(llvm::raw_ostream &output,
                           const SelectedIntrinsicCLeaves &leaves);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVINTRINSICC_H
