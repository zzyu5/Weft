#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
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

mlir::Value stripRepresentationConversions(
    mlir::Value value,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    auto input = mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
    auto result = mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
    llvm::StringRef effect = conversion.getConversion().getEffect();
    if (!input || !result || input.getShape() != result.getShape() ||
        input.getAxisIds() != result.getAxisIds() ||
        (effect != "pure" && effect != "read"))
      break;
    if (conversions)
      conversions->push_back(conversion);
    value = conversion.getInput();
  }
  return value;
}

bool isReadOnlyOrPure(mlir::Operation *operation) {
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
    llvm::StringRef effect = conversion.getConversion().getEffect();
    return effect == "pure" || effect == "read";
  }
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &instance) {
    return mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

bool canFoldReadConversions(
    riscv::ExtractOp first, riscv::ExtractOp second,
    llvm::ArrayRef<riscv::ConvertLayoutOp> firstConversions,
  llvm::ArrayRef<riscv::ConvertLayoutOp> secondConversions) {
  llvm::SmallVector<mlir::Operation *> reads;
  auto collectReads = [&](llvm::ArrayRef<riscv::ConvertLayoutOp> conversions) {
    for (riscv::ConvertLayoutOp conversion : conversions) {
      if (conversion.getConversion().getEffect() != "read")
        continue;
      if (conversion.getConversion().getKind() != "local_load" ||
          conversion->getBlock() != first->getBlock())
        return false;
      reads.push_back(conversion);
    }
    return true;
  };
  if (!collectReads(firstConversions) || !collectReads(secondConversions))
    return false;
  if (reads.empty())
    return true;

  mlir::Operation *begin = reads.front();
  for (mlir::Operation *read : reads)
    if (read->isBeforeInBlock(begin))
      begin = read;
  if (second->isBeforeInBlock(begin))
    return false;
  for (mlir::Operation *operation = begin; operation;
       operation = operation->getNextNode()) {
    if (!isReadOnlyOrPure(operation))
      return false;
    if (operation == second.getOperation())
      return true;
  }
  return false;
}

bool isLayerExtract(
    riscv::ExtractOp extract, riscv::FieldOp &field,
    riscv::PhysicalPointOp &point,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  field = stripRepresentationConversions(extract.getInput(), conversions)
              .getDefiningOp<riscv::FieldOp>();
  point = extract.getIndices().size() == 1
              ? extract.getIndices().front().getDefiningOp<
                    riscv::PhysicalPointOp>()
              : riscv::PhysicalPointOp();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  int64_t domainSelectors = 0;
  for (mlir::Attribute selector : extract.getSelectors())
    domainSelectors +=
        mlir::cast<mlir::StringAttr>(selector).getValue() == "domain";
  if (!field || !point || !result || !input)
    return false;
  auto axis = llvm::find(result.getAxisIds().asArrayRef(),
                         point.getResult().getType().getDomain().getAxisId());
  const bool hasLayerAxis =
      result && axis != result.getAxisIds().asArrayRef().end() &&
      result.getShape()[static_cast<size_t>(
          axis - result.getAxisIds().asArrayRef().begin())] ==
          extract.getAccess().getLayerSize();
  return domainSelectors == 1 &&
         extract.getAccess().getForm() == "indexed" &&
         extract.getAccess().getMapping() == "grouped_layered" &&
         extract.getAccess().getGroupSize() ==
             extract.getAccess().getLayerSize() * 2 &&
         result.getShape().size() == input.getShape().size() && hasLayerAxis &&
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
      llvm::SmallVector<riscv::ConvertLayoutOp> firstConversions;
      if (!isLayerExtract(first, firstField, firstPoint, &firstConversions))
        continue;
      const int64_t layer = first.getAccess().getLayerSize();

      LinearIndex firstIndex;
      decomposeIndex(firstPoint.getBase(), 1, firstIndex);
      if (!firstIndex.valid)
        continue;

      riscv::ExtractOp second;
      riscv::FieldOp secondField;
      llvm::SmallVector<riscv::ConvertLayoutOp> secondConversions;
      for (riscv::ExtractOp candidate : extracts) {
        if (candidate == first || consumed.contains(candidate) ||
            candidate->getBlock() != first->getBlock() ||
            first->isBeforeInBlock(candidate) == false)
          continue;
        riscv::PhysicalPointOp candidatePoint;
        riscv::FieldOp candidateField;
        llvm::SmallVector<riscv::ConvertLayoutOp> candidateConversions;
        if (!isLayerExtract(candidate, candidateField, candidatePoint,
                            &candidateConversions) ||
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
        secondConversions = std::move(candidateConversions);
        break;
      }
      if (!second)
        continue;
      if (!canFoldReadConversions(first, second, firstConversions,
                                  secondConversions))
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
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(firstConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(secondConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      if (secondField.getResult().use_empty())
        rewriter.eraseOp(secondField);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createShareRISCVLayeredWindowsPass() {
  return std::make_unique<ShareRISCVLayeredWindowsPass>();
}
