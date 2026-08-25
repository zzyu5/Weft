#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace weft;

namespace {

struct LinearIndex {
  llvm::DenseMap<mlir::Value, int64_t> terms;
  int64_t constant = 0;
  bool valid = true;
};

std::optional<int64_t> constantIndex(mlir::Value value) {
  llvm::APInt constant;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) &&
      constant.isSignedIntN(64))
    return constant.getSExtValue();
  return std::nullopt;
}

void decomposeIndex(mlir::Value value, int64_t coefficient,
                    LinearIndex &result) {
  if (!result.valid)
    return;
  if (auto constant = constantIndex(value)) {
    result.constant += coefficient * *constant;
    return;
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    decomposeIndex(add.getLhs(), coefficient, result);
    decomposeIndex(add.getRhs(), coefficient, result);
    return;
  }
  if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
    decomposeIndex(sub.getLhs(), coefficient, result);
    decomposeIndex(sub.getRhs(), -coefficient, result);
    return;
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto lhs = constantIndex(multiply.getLhs())) {
      decomposeIndex(multiply.getRhs(), coefficient * *lhs, result);
      return;
    }
    if (auto rhs = constantIndex(multiply.getRhs())) {
      decomposeIndex(multiply.getLhs(), coefficient * *rhs, result);
      return;
    }
  }
  result.terms[value] += coefficient;
  if (result.terms[value] == 0)
    result.terms.erase(value);
}

bool sameTerms(const LinearIndex &lhs, const LinearIndex &rhs) {
  if (!lhs.valid || !rhs.valid || lhs.terms.size() != rhs.terms.size())
    return false;
  for (const auto &[value, coefficient] : lhs.terms) {
    auto found = rhs.terms.find(value);
    if (found == rhs.terms.end() || found->second != coefficient)
      return false;
  }
  return true;
}

bool isLayerExtract(riscv::ExtractOp extract, riscv::FieldOp &field,
                    riscv::PhysicalPointOp &point) {
  field = extract.getInput().getDefiningOp<riscv::FieldOp>();
  point = extract.getIndices().size() == 1
              ? extract.getIndices().front().getDefiningOp<
                    riscv::PhysicalPointOp>()
              : riscv::PhysicalPointOp();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  return field && point && result && input &&
         extract.getSelectors().size() == 1 &&
         mlir::cast<mlir::StringAttr>(extract.getSelectors()[0]).getValue() ==
             "domain" &&
         extract.getAccess().getForm() == "indexed" &&
         extract.getAccess().getMapping() == "grouped_layered" &&
         extract.getAccess().getGroupSize() ==
             extract.getAccess().getLayerSize() * 2 &&
         result.getShape().size() == 1 &&
         result.getShape()[0] == extract.getAccess().getLayerSize() &&
         result.getLayout().getCarrier() == "rvv" &&
         result.getLayout().getValidity() == "full";
}

class ShareRISCVLayeredWindowsPass
    : public mlir::PassWrapper<ShareRISCVLayeredWindowsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-share-layered-windows";
  }
  llvm::StringRef getDescription() const override {
    return "Represent adjacent packed layers as one shared RVV storage window";
  }

  void runOnOperation() override {
    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    llvm::DenseSet<mlir::Operation *> consumed;
    mlir::IRRewriter rewriter(&getContext());

    for (riscv::ExtractOp first : extracts) {
      if (consumed.contains(first))
        continue;
      riscv::FieldOp firstField;
      riscv::PhysicalPointOp firstPoint;
      if (!isLayerExtract(first, firstField, firstPoint))
        continue;
      const int64_t layer = first.getAccess().getLayerSize();

      LinearIndex firstIndex;
      decomposeIndex(firstPoint.getBase(), 1, firstIndex);
      if (!firstIndex.valid)
        continue;

      riscv::ExtractOp second;
      riscv::FieldOp secondField;
      for (riscv::ExtractOp candidate : extracts) {
        if (candidate == first || consumed.contains(candidate) ||
            candidate->getBlock() != first->getBlock() ||
            first->isBeforeInBlock(candidate) == false)
          continue;
        riscv::PhysicalPointOp candidatePoint;
        riscv::FieldOp candidateField;
        if (!isLayerExtract(candidate, candidateField, candidatePoint) ||
            candidate.getResult().getType() != first.getResult().getType() ||
            candidate.getAccess() != first.getAccess() ||
            candidateField.getName() != firstField.getName() ||
            candidateField.getOwner() != firstField.getOwner() ||
            candidatePoint.getParent() != firstPoint.getParent())
          continue;
        LinearIndex candidateIndex;
        decomposeIndex(candidatePoint.getBase(), 1, candidateIndex);
        if (!sameTerms(firstIndex, candidateIndex) ||
            candidateIndex.constant - firstIndex.constant != layer)
          continue;
        second = candidate;
        secondField = candidateField;
        break;
      }
      if (!second)
        continue;

      auto firstType = mlir::cast<riscv::ValueType>(first.getResult().getType());
      const int64_t groups = firstType.getLayout().getRegisterGroups();
      rewriter.setInsertionPoint(first);
      auto shared = rewriter.create<riscv::RVVLayeredWindowOp>(
          first.getLoc(),
          mlir::TypeRange{first.getResult().getType(), second.getResult().getType()},
          firstField.getResult(), firstPoint.getResult(), first.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "layered-window",
                               "rvv.layered-window", "rvv.layered-window", 0,
                               groups * 2, groups));
      if (auto origin = first->getAttr("source_origin"))
        shared->setAttr("source_origin", origin);
      shared->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.extract"));
      first.getResult().replaceAllUsesWith(shared.getFirst());
      second.getResult().replaceAllUsesWith(shared.getSecond());
      consumed.insert(first);
      consumed.insert(second);
      rewriter.eraseOp(first);
      rewriter.eraseOp(second);
      if (secondField.getResult().use_empty())
        rewriter.eraseOp(secondField);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createShareRISCVLayeredWindowsPass() {
  return std::make_unique<ShareRISCVLayeredWindowsPass>();
}
