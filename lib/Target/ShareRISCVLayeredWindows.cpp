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

#include <algorithm>
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
  return riscv_internal::constantInt(value);
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
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), conversions);
  field = riscv_internal::sourceField(source);
  point = extract.getIndices().size() == 1
              ? riscv_internal::stripRepresentationConversions(
                    extract.getIndices().front())
                    .getDefiningOp<riscv::PhysicalPointOp>()
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
  llvm::StringRef validity = result.getLayout().getValidity();
  return domainSelectors == 1 &&
         extract.getAccess().getForm() == "indexed" &&
         extract.getAccess().getMapping() == "grouped_layered" &&
         extract.getAccess().getGroupSize() ==
             extract.getAccess().getLayerSize() * 2 &&
         result.getShape().size() == input.getShape().size() && hasLayerAxis &&
         result.getLayout().getCarrier() == "rvv" &&
         (validity == "full" || validity == "tail");
}

bool sameFieldEdge(riscv::FieldOp lhs, riscv::FieldOp rhs) {
  if (!lhs || !rhs || lhs.getOwner() != rhs.getOwner() ||
      lhs.getResult().getType() != rhs.getResult().getType() ||
      lhs.getAccess() != rhs.getAccess())
    return false;
  riscv_internal::FieldFacts left = riscv_internal::fieldFacts(lhs);
  riscv_internal::FieldFacts right = riscv_internal::fieldFacts(rhs);
  return left.mapping == right.mapping &&
         left.scalarPerRecord == right.scalarPerRecord &&
         left.logicalRank == right.logicalRank &&
         left.group == right.group && left.layer == right.layer &&
         left.joinFields == right.joinFields &&
         left.joinLowBits == right.joinLowBits &&
         left.joinRole == right.joinRole && left.bitOffset == right.bitOffset &&
         left.storageBits == right.storageBits &&
         left.alignment == right.alignment && left.order == right.order;
}

bool isLayeredStreamField(riscv::FieldOp field) {
  auto value = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  auto element = value
                     ? mlir::dyn_cast<mlir::IntegerType>(value.getElementType())
                     : mlir::IntegerType();
  riscv::AccessAttr access = field.getAccess();
  const int64_t group = access ? access.getGroupSize() : 0;
  const int64_t layer = access ? access.getLayerSize() : 0;
  const int64_t layers = layer > 0 ? group / layer : 0;
  if (!value || !element || element.isSigned() || !access ||
      value.getLayout().getCarrier() != "rvv" ||
      access.getForm() != "indexed" ||
      access.getMapping() != "grouped_layered" ||
      group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers > 8 || access.getBitOffset() % 8)
    return false;
  bool hasStreamAxis = false;
  for (size_t index = 0; index < value.getShape().size(); ++index) {
    const int64_t extent = value.getShape()[index];
    const int64_t time = value.getLayout().getTimeFactors()[index];
    const int64_t lanes = value.getLayout().getLaneFactors()[index];
    const bool streamAxis = extent > 0 && extent % group == 0 && lanes > 1 &&
                            layer % lanes == 0 && time > 1 &&
                            time * lanes == extent;
    if (streamAxis) {
      if (hasStreamAxis)
        return false;
      hasStreamAxis = true;
    } else if (time != 1 || lanes != 1) {
      return false;
    }
  }
  return hasStreamAxis;
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
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<riscv::FieldOp> fields;
    getOperation().walk(
        [&](riscv::FieldOp field) { fields.push_back(field); });
    for (riscv::FieldOp field : fields) {
      if (!isLayeredStreamField(field))
        continue;
      auto type = mlir::cast<riscv::ValueType>(field.getResult().getType());
      const int64_t resultGroups = type.getLayout().getRegisterGroups();
      const int64_t temporaryGroups =
          std::max<int64_t>(1,
                            (type.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = type.getLayout().getValidity() == "tail";
      rewriter.setInsertionPointAfter(field);
      auto stream = rewriter.create<riscv::RVVLayeredStreamOp>(
          field.getLoc(), type, field.getResult(), field.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "layered-stream",
                               "rvv.layered-stream", "rvv.layered-stream", 0,
                               resultGroups, temporaryGroups, 0, "none",
                               tail ? "agnostic" : "exact"));
      if (auto origin = field->getAttr("source_origin"))
        stream->setAttr("source_origin", origin);
      stream->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.field"));
      field.getResult().replaceAllUsesExcept(stream.getResult(),
                                             stream.getOperation());
    }

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    llvm::DenseSet<mlir::Operation *> consumed;

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
            !sameFieldEdge(candidateField, firstField) ||
            candidatePoint.getParent() != firstPoint.getParent() ||
            candidatePoint.getActive() != firstPoint.getActive() ||
            candidatePoint.getPartition() != firstPoint.getPartition() ||
            candidatePoint.getResult().getType() !=
                firstPoint.getResult().getType())
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
      const bool tail = firstType.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(first);
      auto shared = rewriter.create<riscv::RVVLayeredWindowOp>(
          first.getLoc(),
          mlir::TypeRange{first.getResult().getType(), second.getResult().getType()},
          firstField.getResult(), firstPoint.getResult(), first.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "layered-window",
                               "rvv.layered-window", "rvv.layered-window", 0,
                               groups * 2, groups, 0, "none",
                               tail ? "agnostic" : "exact"));
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
