#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>

using namespace weft;

namespace {

struct Mapping {
  std::string kind;
  int64_t laneAxis = 0;
  int64_t physicalLanes = 1;
  int64_t streamParts = 1;
  int64_t registerParts = 1;
  int64_t vectorParts = 1;
  int64_t sew = 0;
  int64_t lmulEighths = 0;
  llvm::SmallVector<int64_t, 4> registerAxes;
  llvm::SmallVector<int64_t, 4> registerExtents;

  bool isScalar() const { return kind == "scalar" || kind == "sequential"; }
  bool isTuple() const { return kind == "register-scalar-tuple"; }
  bool isVector() const { return llvm::StringRef(kind).starts_with("rvv"); }
  int64_t partCount() const {
    if (isVector())
      return vectorParts;
    if (isTuple())
      return registerParts;
    return 1;
  }
};

Mapping mappingFor(mlir::DictionaryAttr value) {
  Mapping result;
  result.kind = riscv_internal::string(value, "physical_kind").value_or("").str();
  result.laneAxis = riscv_internal::integer(value, "lane_axis").value_or(0);
  result.physicalLanes =
      riscv_internal::integer(value, "physical_lanes").value_or(1);
  result.streamParts =
      riscv_internal::integer(value, "stream_parts").value_or(1);
  result.registerParts =
      riscv_internal::integer(value, "register_parts").value_or(1);
  result.vectorParts =
      riscv_internal::integer(value, "vector_parts").value_or(1);
  result.sew = riscv_internal::integer(value, "physical_sew").value_or(0);
  result.lmulEighths =
      riscv_internal::integer(value, "lmul_eighths").value_or(0);
  if (auto axes = value.getAs<mlir::DenseI64ArrayAttr>("register_axes"))
    result.registerAxes.append(axes.asArrayRef().begin(), axes.asArrayRef().end());
  if (auto extents = value.getAs<mlir::DenseI64ArrayAttr>("register_extents"))
    result.registerExtents.append(extents.asArrayRef().begin(),
                                  extents.asArrayRef().end());
  return result;
}

std::optional<llvm::SmallVector<int64_t, 4>>
registerCoordinates(const Mapping &mapping, int64_t registerPart) {
  if (mapping.registerAxes.size() != mapping.registerExtents.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> coordinates(mapping.registerAxes.size(), 0);
  for (int64_t index = static_cast<int64_t>(mapping.registerAxes.size()) - 1;
       index >= 0; --index) {
    const int64_t extent = mapping.registerExtents[index];
    if (extent <= 0)
      return std::nullopt;
    coordinates[index] = registerPart % extent;
    registerPart /= extent;
  }
  if (registerPart != 0)
    return std::nullopt;
  return coordinates;
}

bool isMappedValue(const Mapping &mapping) {
  return mapping.isScalar() || mapping.isTuple() || mapping.isVector();
}

bool carriesPointwiseMapping(llvm::StringRef name) {
  return name == "weft_kernel.binary" || name == "weft_kernel.unary" ||
         name == "weft_kernel.compare" || name == "weft_kernel.cast" ||
         name == "weft_kernel.widen" || name == "weft_kernel.narrow";
}

llvm::SmallVector<int64_t, 2> mappedOperands(llvm::StringRef name,
                                             size_t operandCount) {
  if (name == "weft_kernel.lookup")
    return operandCount == 2 ? llvm::SmallVector<int64_t, 2>{1}
                             : llvm::SmallVector<int64_t, 2>{};
  if (!carriesPointwiseMapping(name))
    return {};
  llvm::SmallVector<int64_t, 2> result;
  for (size_t index = 0; index < operandCount; ++index)
    result.push_back(static_cast<int64_t>(index));
  return result;
}

mlir::FailureOr<mlir::DictionaryAttr>
buildUseConversion(mlir::Builder &builder, llvm::StringRef sourceId,
                   mlir::DictionaryAttr sourceValue, llvm::StringRef targetId,
                   mlir::DictionaryAttr targetValue, int64_t operandIndex) {
  Mapping source = mappingFor(sourceValue);
  Mapping target = mappingFor(targetValue);
  if (!isMappedValue(source) || !isMappedValue(target))
    return mlir::failure();

  const int64_t targetParts = target.partCount();
  if (targetParts <= 0)
    return mlir::failure();
  llvm::SmallVector<int64_t, 16> sourceParts;
  llvm::SmallVector<int64_t, 16> laneOffsets;
  sourceParts.reserve(targetParts);
  laneOffsets.reserve(targetParts);
  llvm::StringRef relation = "identity";

  if (source.isScalar()) {
    relation = target.isScalar() ? "identity" : "scalar-broadcast";
    sourceParts.assign(targetParts, 0);
    laneOffsets.assign(targetParts, -1);
  } else {
    bool laneToRegister = source.laneAxis > 0 &&
                          source.laneAxis != target.laneAxis &&
                          llvm::is_contained(target.registerAxes,
                                             source.laneAxis);
    if (source.laneAxis > 0 && source.laneAxis != target.laneAxis &&
        !laneToRegister)
      return mlir::failure();
    relation = laneToRegister ? "lane-to-register" : "project";
    for (int64_t targetPart = 0; targetPart < targetParts; ++targetPart) {
      const int64_t targetStream =
          target.isVector() ? targetPart % target.streamParts : 0;
      const int64_t targetRegister =
          target.isVector() ? targetPart / target.streamParts : targetPart;
      auto targetCoordinates = registerCoordinates(target, targetRegister);
      if (!targetCoordinates)
        return mlir::failure();

      int64_t sourceRegister = 0;
      for (auto [axis, extent] :
           llvm::zip(source.registerAxes, source.registerExtents)) {
        auto found = llvm::find(target.registerAxes, axis);
        if (found == target.registerAxes.end() || extent <= 0)
          return mlir::failure();
        const int64_t coordinate =
            (*targetCoordinates)[found - target.registerAxes.begin()];
        if (coordinate < 0 || coordinate >= extent)
          return mlir::failure();
        sourceRegister = sourceRegister * extent + coordinate;
      }

      int64_t sourceStream = 0;
      int64_t laneOffset = -1;
      if (laneToRegister) {
        auto lanePosition =
            llvm::find(target.registerAxes, source.laneAxis);
        if (lanePosition == target.registerAxes.end() ||
            source.physicalLanes <= 0 || source.streamParts <= 0)
          return mlir::failure();
        const int64_t logicalLane =
            (*targetCoordinates)[lanePosition - target.registerAxes.begin()];
        sourceStream = logicalLane / source.physicalLanes;
        laneOffset = logicalLane % source.physicalLanes;
      } else if (source.laneAxis > 0) {
        if (source.laneAxis != target.laneAxis ||
            targetStream >= source.streamParts)
          return mlir::failure();
        sourceStream = source.streamParts == 1 ? 0 : targetStream;
      }
      const int64_t sourcePart =
          sourceRegister * source.streamParts + sourceStream;
      if (sourcePart < 0 || sourcePart >= source.partCount())
        return mlir::failure();
      sourceParts.push_back(sourcePart);
      laneOffsets.push_back(laneOffset);
    }
    bool exactIdentity = source.kind == target.kind &&
                         source.laneAxis == target.laneAxis &&
                         source.physicalLanes == target.physicalLanes &&
                         source.streamParts == target.streamParts &&
                         source.registerAxes == target.registerAxes &&
                         source.registerExtents == target.registerExtents &&
                         source.sew == target.sew &&
                         source.lmulEighths == target.lmulEighths;
    if (exactIdentity)
      for (int64_t part = 0; part < targetParts; ++part)
        exactIdentity &= sourceParts[part] == part && laneOffsets[part] < 0;
    if (exactIdentity)
      relation = "identity";
  }

  return riscv_internal::dictionary(
      builder,
      {{"operand", builder.getI64IntegerAttr(operandIndex)},
       {"source", builder.getStringAttr(sourceId)},
       {"target", builder.getStringAttr(targetId)},
       {"relation", builder.getStringAttr(relation)},
       {"source_lane_axis", builder.getI64IntegerAttr(source.laneAxis)},
       {"target_lane_axis", builder.getI64IntegerAttr(target.laneAxis)},
       {"source_sew", builder.getI64IntegerAttr(source.sew)},
       {"target_sew", builder.getI64IntegerAttr(target.sew)},
       {"source_lmul_eighths",
        builder.getI64IntegerAttr(source.lmulEighths)},
       {"target_lmul_eighths",
        builder.getI64IntegerAttr(target.lmulEighths)},
       {"source_parts", builder.getDenseI64ArrayAttr(sourceParts)},
       {"lane_offsets", builder.getDenseI64ArrayAttr(laneOffsets)},
       {"decision_owner", builder.getStringAttr("value-use-edge")}});
}

class ResolveRISCVLayoutConversionsPass final
    : public mlir::PassWrapper<ResolveRISCVLayoutConversionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ResolveRISCVLayoutConversionsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-resolve-layout-conversions";
  }
  llvm::StringRef getDescription() const final {
    return "materialize typed value-use physical conversions";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;
      if (problem.getStage() != "representations") {
        problem.emitError(
            "layout conversion pass requires assigned representations");
        signalPassFailure();
        return;
      }
      llvm::StringMap<mlir::DictionaryAttr> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        values[*riscv_internal::string(value, "id")] = value;
      }

      bool legal = true;
      std::string reason;
      llvm::SmallVector<mlir::Attribute> operations;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name =
            riscv_internal::string(operation, "name").value_or("");
        auto operands = operation.getAs<mlir::ArrayAttr>("operands");
        llvm::SmallVector<int64_t, 2> mapped =
            mappedOperands(name, operands ? operands.size() : 0);
        if (mapped.empty()) {
          operations.push_back(operation);
          continue;
        }
        auto results = operation.getAs<mlir::ArrayAttr>("results");
        if (!operands || !results || results.size() != 1) {
          operations.push_back(operation);
          continue;
        }
        llvm::StringRef targetId =
            mlir::cast<mlir::StringAttr>(results[0]).getValue();
        auto target = values.find(targetId);
        if (target == values.end()) {
          legal = false;
          reason = "pointwise result has no assigned physical mapping";
          break;
        }
        llvm::SmallVector<mlir::Attribute> conversions;
        for (int64_t index : mapped) {
          mlir::Attribute operandAttribute = operands[index];
          llvm::StringRef sourceId =
              mlir::cast<mlir::StringAttr>(operandAttribute).getValue();
          auto source = values.find(sourceId);
          if (source == values.end()) {
            legal = false;
            reason = "pointwise operand has no assigned physical mapping";
            break;
          }
          auto conversion = buildUseConversion(
              builder, sourceId, source->second, targetId, target->second,
              index);
          if (mlir::failed(conversion)) {
            legal = false;
            reason = "pointwise value-use edge has no typed physical conversion";
            break;
          }
          conversions.push_back(*conversion);
        }
        if (!legal)
          break;
        operation = riscv_internal::set(
            operation, "use_conversions", builder.getArrayAttr(conversions));
        operations.push_back(operation);
      }
      if (!legal) {
        problem.setResourcesAttr(riscv_internal::dictionary(
            builder, {{"invalid_reason", builder.getStringAttr(reason)}}));
        problem.setStageAttr(builder.getStringAttr("invalid"));
        continue;
      }
      problem.setOperationsAttr(builder.getArrayAttr(operations));
      problem.setStageAttr(builder.getStringAttr("conversions"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createResolveRISCVLayoutConversionsPass() {
  return std::make_unique<ResolveRISCVLayoutConversionsPass>();
}
