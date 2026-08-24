#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringExtras.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace weft;

namespace {

std::string valueKind(mlir::Type type) {
  if (mlir::isa<kernel::ViewType>(type))
    return "view";
  if (mlir::isa<kernel::SliceType>(type))
    return "slice";
  if (mlir::isa<kernel::EncodingType>(type))
    return "encoded_value";
  if (mlir::isa<kernel::ValueType>(type))
    return "local_value";
  if (mlir::isa<kernel::DomainType, kernel::PointType>(type))
    return "control";
  if (type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
    return "scalar";
  return "unknown";
}

kernel::EncodingType encodingOf(mlir::Type type) {
  if (auto encoding = mlir::dyn_cast<kernel::EncodingType>(type))
    return encoding;
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return mlir::dyn_cast<kernel::EncodingType>(view.getEncoding());
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return mlir::dyn_cast<kernel::EncodingType>(slice.getEncoding());
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return mlir::dyn_cast<kernel::EncodingType>(value.getElementType());
  return {};
}

class KernelFactCollector {
public:
  KernelFactCollector(mlir::Builder &builder, kernel::KernelOp kernel)
      : builder(builder), kernel(kernel) {}

  mlir::LogicalResult collect() {
    llvm::SmallVector<std::string> levelPath;
    llvm::SmallVector<std::string> controlPath;
    visitBlock(kernel.getBody().front(), "kernel", levelPath, controlPath);
    formHandoffClasses();
    return failed ? mlir::failure() : mlir::success();
  }

  mlir::ArrayAttr valueAttributes() const {
    return builder.getArrayAttr(values);
  }
  mlir::ArrayAttr operationAttributes() const {
    return builder.getArrayAttr(operations);
  }

private:
  std::string addValue(mlir::Value value, llvm::StringRef source) {
    auto found = ids.find(value);
    if (found != ids.end())
      return found->second;
    std::string id = "v" + std::to_string(nextValue++);
    ids.try_emplace(value, id);
    valueOrdinals.try_emplace(value, handoffParent.size());
    handoffParent.push_back(handoffParent.size());
    mlir::Type type = value.getType();
    kernel::EncodingType encoding = encodingOf(type);
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> fields{
        {"id", builder.getStringAttr(id)},
        {"source", builder.getStringAttr(source)},
        {"type", mlir::TypeAttr::get(type)},
        {"type_spelling", builder.getStringAttr(riscv_internal::printType(type))},
        {"kind", builder.getStringAttr(valueKind(type))},
        {"shape", riscv_internal::integers(builder,
                                            riscv_internal::logicalShape(type))},
        {"axes", riscv_internal::integers(builder,
                                           riscv_internal::logicalAxes(type))},
        {"logical_sew", builder.getI64IntegerAttr(
                            riscv_internal::logicalBitWidth(type))}};
    if (encoding) {
      fields.push_back(
          {"encoding_family", builder.getStringAttr(encoding.getFamily())});
      fields.push_back(
          {"encoding_kind", builder.getStringAttr(encoding.getKind())});
      fields.push_back({"layout_identity",
                        builder.getStringAttr(encoding.getLayoutIdentity())});
    }
    kernel::DomainType domain;
    if (auto direct = mlir::dyn_cast<kernel::DomainType>(type))
      domain = direct;
    else if (auto point = mlir::dyn_cast<kernel::PointType>(type))
      domain = point.getDomain();
    if (domain) {
      fields.push_back(
          {"domain_axis", builder.getI64IntegerAttr(domain.getAxisId())});
      fields.push_back(
          {"domain_relation", builder.getStringAttr(domain.getRelation())});
      fields.push_back(
          {"domain_extent", builder.getStringAttr(domain.getExtent())});
      fields.push_back(
          {"domain_partition", builder.getStringAttr(domain.getPartition())});
      fields.push_back({"domain_multiplicity",
                        builder.getStringAttr(domain.getMultiplicity())});
      fields.push_back({"domain_tail", builder.getStringAttr(domain.getTail())});
    }
    values.push_back(riscv_internal::dictionary(builder, fields));
    return id;
  }

  void visitBlock(mlir::Block &block, llvm::StringRef source,
                  llvm::SmallVectorImpl<std::string> &levelPath,
                  llvm::SmallVectorImpl<std::string> &controlPath) {
    for (auto [index, argument] : llvm::enumerate(block.getArguments()))
      addValue(argument, source.str() + ".arg" + std::to_string(index));
    for (mlir::Operation &operation : block)
      visitOperation(operation, levelPath, controlPath);
  }

  void visitOperation(mlir::Operation &operation,
                      llvm::SmallVectorImpl<std::string> &levelPath,
                      llvm::SmallVectorImpl<std::string> &controlPath) {
    std::string operationId = "op" + std::to_string(nextOperation++);
    llvm::SmallVector<std::string> resultIds;
    for (auto [index, result] : llvm::enumerate(operation.getResults()))
      resultIds.push_back(addValue(result, operationId + ".result" +
                                               std::to_string(index)));
    llvm::SmallVector<std::string> operandIds;
    for (mlir::Value operand : operation.getOperands()) {
      auto found = ids.find(operand);
      if (found == ids.end()) {
        operation.emitError("planning fact construction found an unbound SSA operand");
        failed = true;
        operandIds.push_back("unbound");
      } else {
        operandIds.push_back(found->second);
      }
    }
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> fields{
        {"id", builder.getStringAttr(operationId)},
        {"name", builder.getStringAttr(operation.getName().getStringRef())},
        {"operands", riscv_internal::strings(builder, operandIds)},
        {"results", riscv_internal::strings(builder, resultIds)},
        {"source_attributes", operation.getAttrDictionary()},
        {"level_path", riscv_internal::strings(builder, levelPath)},
        {"control_path", riscv_internal::strings(builder, controlPath)},
        {"location",
         builder.getStringAttr(riscv_internal::printAttribute(operation.getLoc()))}};
    if (auto engine = operation.getAttrOfType<mlir::StringAttr>("engine"))
      fields.push_back({"engine", engine});
    else
      fields.push_back({"engine", builder.getStringAttr("")});
    const bool opensLevel = mlir::isa<kernel::LevelOp>(operation);
    const bool opensControl = mlir::isa<kernel::ForOp, kernel::WhileOp,
                                        kernel::IfOp>(operation);
    if (opensLevel)
      levelPath.push_back(operationId);
    if (opensControl)
      controlPath.push_back(operationId);
    for (auto [regionIndex, region] : llvm::enumerate(operation.getRegions()))
      for (auto [blockIndex, nested] : llvm::enumerate(region))
        visitBlock(nested,
                   operationId + ".region" + std::to_string(regionIndex) +
                       ".block" + std::to_string(blockIndex),
                   levelPath, controlPath);
    if (opensControl)
      controlPath.pop_back();
    if (opensLevel)
      levelPath.pop_back();
    operations.push_back(riscv_internal::dictionary(builder, fields));
  }

  unsigned findHandoff(unsigned value) {
    if (handoffParent[value] != value)
      handoffParent[value] = findHandoff(handoffParent[value]);
    return handoffParent[value];
  }

  void uniteHandoff(mlir::Value lhs, mlir::Value rhs) {
    auto lhsIt = valueOrdinals.find(lhs);
    auto rhsIt = valueOrdinals.find(rhs);
    if (lhsIt == valueOrdinals.end() || rhsIt == valueOrdinals.end())
      return;
    unsigned lhsRoot = findHandoff(lhsIt->second);
    unsigned rhsRoot = findHandoff(rhsIt->second);
    if (lhsRoot != rhsRoot)
      handoffParent[rhsRoot] = lhsRoot;
  }

  void formHandoffClasses() {
    kernel.walk([&](kernel::LevelOp level) {
      mlir::Block &body = level.getBody().front();
      auto handoff = mlir::cast<kernel::HandoffOp>(body.getTerminator());
      for (auto [index, carried] : llvm::enumerate(level.getCarried())) {
        mlir::Value argument = body.getArgument(1 + index);
        uniteHandoff(carried, argument);
        uniteHandoff(argument, handoff.getValues()[index]);
        uniteHandoff(handoff.getValues()[index], level.getResult(index));
      }
      auto stateYield = mlir::cast<kernel::BirthsYieldOp>(
          level.getStateBirths().front().getTerminator());
      auto stagedYield = mlir::cast<kernel::BirthsYieldOp>(
          level.getStagedBirths().front().getTerminator());
      size_t cursor = 1 + level.getCarried().size();
      for (mlir::Value birth : stateYield.getValues())
        uniteHandoff(birth, body.getArgument(cursor++));
      for (mlir::Value birth : stagedYield.getValues())
        uniteHandoff(birth, body.getArgument(cursor++));
    });
    kernel.walk([&](kernel::ForOp loop) {
      mlir::Block &body = loop.getBody().front();
      auto yield = mlir::cast<kernel::YieldOp>(body.getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInitArgs())) {
        uniteHandoff(initial, body.getArgument(index + 1));
        uniteHandoff(body.getArgument(index + 1), yield.getValues()[index]);
        uniteHandoff(yield.getValues()[index], loop.getResult(index));
      }
    });
    kernel.walk([&](kernel::IfOp branch) {
      auto thenYield = mlir::cast<kernel::YieldOp>(
          branch.getThenRegion().front().getTerminator());
      auto elseYield = mlir::cast<kernel::YieldOp>(
          branch.getElseRegion().front().getTerminator());
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        uniteHandoff(thenYield.getValues()[index], result);
        uniteHandoff(elseYield.getValues()[index], result);
      }
    });
    kernel.walk([&](kernel::WhileOp loop) {
      mlir::Block &condition = loop.getConditionRegion().front();
      mlir::Block &body = loop.getBodyRegion().front();
      auto conditionTerminator =
          mlir::cast<kernel::ConditionOp>(condition.getTerminator());
      auto yield = mlir::cast<kernel::YieldOp>(body.getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInitArgs())) {
        uniteHandoff(initial, condition.getArgument(index));
        uniteHandoff(condition.getArgument(index),
                     conditionTerminator.getValues()[index]);
        uniteHandoff(conditionTerminator.getValues()[index],
                     body.getArgument(index));
        uniteHandoff(body.getArgument(index), yield.getValues()[index]);
        uniteHandoff(yield.getValues()[index], loop.getResult(index));
      }
    });

    for (auto [index, attribute] : llvm::enumerate(values)) {
      auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
      values[index] = riscv_internal::set(
          value, "handoff_class",
          builder.getStringAttr("h" + std::to_string(findHandoff(index))));
    }
  }

  mlir::Builder &builder;
  kernel::KernelOp kernel;
  llvm::DenseMap<mlir::Value, std::string> ids;
  llvm::DenseMap<mlir::Value, unsigned> valueOrdinals;
  llvm::SmallVector<unsigned> handoffParent;
  llvm::SmallVector<mlir::Attribute> values;
  llvm::SmallVector<mlir::Attribute> operations;
  unsigned nextValue = 0;
  unsigned nextOperation = 0;
  bool failed = false;
};

struct AutoDimension {
  std::string name;
  llvm::SmallVector<int64_t> choices;
};

mlir::FailureOr<llvm::SmallVector<AutoDimension>>
collectAutoDimensions(kernel::KernelOp kernel,
                      const RISCVCompilerOptions &options) {
  llvm::SmallVector<AutoDimension> dimensions;
  llvm::SmallSet<std::string, 8> seen;
  mlir::WalkResult result = kernel.walk([&](kernel::DomainOp operation) {
    kernel::DomainType domain = operation.getResult().getType();
    llvm::StringRef partition = domain.getPartition();
    if (!partition.starts_with("auto:"))
      return mlir::WalkResult::advance();
    llvm::StringRef spelling = partition.drop_front(5);
    if (!seen.insert(spelling.str()).second)
      return mlir::WalkResult::advance();
    AutoDimension dimension;
    dimension.name = spelling.str();
    llvm::SmallVector<llvm::StringRef> choices;
    spelling.split(choices, '|', -1, false);
    for (llvm::StringRef choice : choices) {
      int64_t value = 0;
      if (!choice.getAsInteger(10, value) && value > 0) {
        dimension.choices.push_back(value);
        continue;
      }
      auto binding = options.metaBindings.find(choice);
      if (binding == options.metaBindings.end() || binding->second.empty()) {
        operation.emitError()
            << "auto parameter '" << choice
            << "' requires positive --meta choices in this invocation";
        return mlir::WalkResult::interrupt();
      }
      dimension.choices.append(binding->second.begin(), binding->second.end());
    }
    dimensions.push_back(std::move(dimension));
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted())
    return mlir::failure();
  return dimensions;
}

struct ScheduleSite {
  std::string level;
  bool unroll = false;
  bool pipeline = false;
};

std::pair<bool, bool>
scheduleCapabilities(mlir::DictionaryAttr operation) {
  llvm::StringRef name =
      riscv_internal::string(operation, "name").value_or("");
  if (name == "weft_kernel.for")
    return {false, true};
  if (name == "weft_kernel.mac_pairs" || name == "weft_kernel.mac_groups" ||
      name == "weft_kernel.outer_contract")
    return {true, true};
  return {false, false};
}

llvm::SmallVector<ScheduleSite>
collectLeafScheduleLevels(mlir::ArrayAttr operations) {
  llvm::StringMap<ScheduleSite> localLevels;
  llvm::SmallSet<std::string, 8> enclosingLevels;
  for (mlir::Attribute attribute : operations) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    auto [unroll, pipeline] = scheduleCapabilities(operation);
    if (!unroll && !pipeline)
      continue;
    auto path = operation.getAs<mlir::ArrayAttr>("level_path");
    if (!path || path.empty())
      continue;
    for (size_t index = 0; index + 1 < path.size(); ++index)
      enclosingLevels.insert(
          mlir::cast<mlir::StringAttr>(path[index]).getValue().str());
    llvm::StringRef level =
        mlir::cast<mlir::StringAttr>(path[path.size() - 1]).getValue();
    ScheduleSite &site = localLevels[level];
    site.level = level.str();
    site.unroll |= unroll;
    site.pipeline |= pipeline;
  }
  llvm::SmallVector<ScheduleSite> result;
  for (const auto &entry : localLevels)
    if (!enclosingLevels.contains(entry.getKey().str()))
      result.push_back(entry.getValue());
  llvm::sort(result, [](const ScheduleSite &lhs, const ScheduleSite &rhs) {
    return lhs.level < rhs.level;
  });
  return result;
}

struct CandidateBindings {
  llvm::StringMap<int64_t> source;
  llvm::StringMap<int64_t> schedule;
};

void expandScheduleDimension(
    llvm::SmallVectorImpl<CandidateBindings> &bindings, llvm::StringRef level,
    llvm::StringRef parameter, llvm::ArrayRef<int64_t> choices) {
  llvm::SmallVector<CandidateBindings> expanded;
  const std::string key = (level + "." + parameter).str();
  for (const CandidateBindings &binding : bindings)
    for (int64_t choice : choices) {
      CandidateBindings next = binding;
      next.schedule[key] = choice;
      expanded.push_back(std::move(next));
    }
  bindings.assign(std::make_move_iterator(expanded.begin()),
                  std::make_move_iterator(expanded.end()));
}

mlir::ArrayAttr instantiateCandidates(
    mlir::Builder &builder, kernel::KernelOp kernel,
    llvm::ArrayRef<AutoDimension> dimensions,
    llvm::ArrayRef<ScheduleSite> scheduleLevels,
    const RISCVCompilerOptions &options) {
  llvm::SmallVector<CandidateBindings> bindings(1);
  for (const AutoDimension &dimension : dimensions) {
    llvm::SmallVector<CandidateBindings> expanded;
    for (const CandidateBindings &binding : bindings)
      for (int64_t choice : dimension.choices) {
        CandidateBindings next = binding;
        next.source[dimension.name] = choice;
        expanded.push_back(std::move(next));
      }
    bindings = std::move(expanded);
  }
  for (const ScheduleSite &site : scheduleLevels) {
    if (site.unroll)
      expandScheduleDimension(bindings, site.level, "unroll",
                              options.unrollChoices);
    else
      for (CandidateBindings &binding : bindings)
        binding.schedule[site.level + ".unroll"] = 1;
    if (site.pipeline)
      expandScheduleDimension(bindings, site.level, "pipeline_depth",
                              options.pipelineDepthChoices);
    else
      for (CandidateBindings &binding : bindings)
        binding.schedule[site.level + ".pipeline_depth"] = 1;
  }
  llvm::SmallVector<mlir::Attribute> candidates;
  for (auto [index, binding] : llvm::enumerate(bindings)) {
    llvm::SmallVector<mlir::NamedAttribute> concrete;
    for (const auto &entry : binding.source)
      concrete.push_back(builder.getNamedAttr(
          entry.getKey(), builder.getI64IntegerAttr(entry.getValue())));
    llvm::SmallVector<mlir::NamedAttribute> schedule;
    for (const ScheduleSite &site : scheduleLevels) {
      llvm::SmallVector<mlir::NamedAttribute> parameters;
      for (llvm::StringRef parameter : {"unroll", "pipeline_depth"}) {
        const std::string key = site.level + "." + parameter.str();
        parameters.push_back(builder.getNamedAttr(
            parameter, builder.getI64IntegerAttr(binding.schedule.lookup(key))));
      }
      schedule.push_back(builder.getNamedAttr(
          site.level, builder.getDictionaryAttr(parameters)));
    }
    candidates.push_back(riscv_internal::dictionary(
        builder,
        {{"id", builder.getStringAttr("candidate" + std::to_string(index))},
         {"specialization", builder.getStringAttr(kernel.getSymName())},
         {"auto_bindings", builder.getDictionaryAttr(concrete)},
         {"schedule_bindings", builder.getDictionaryAttr(schedule)}}));
  }
  return builder.getArrayAttr(candidates);
}

class ConstructRISCVProblemsPass final
    : public mlir::PassWrapper<ConstructRISCVProblemsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstructRISCVProblemsPass)

  explicit ConstructRISCVProblemsPass(RISCVCompilerOptions options)
      : options(std::make_shared<const RISCVCompilerOptions>(
            std::move(options))) {}
  ConstructRISCVProblemsPass(const ConstructRISCVProblemsPass &other)
      : mlir::PassWrapper<ConstructRISCVProblemsPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        options(other.options) {}

  llvm::StringRef getArgument() const final {
    return "weft-riscv-construct-problems";
  }
  llvm::StringRef getDescription() const final {
    return "construct immutable logical facts and finite program candidates";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (!options->target.supportsFixedRVV()) {
      module.emitError("RISC-V physicalization requires RVV and a fixed VLEN");
      signalPassFailure();
      return;
    }
    bool preplanned = false;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<riscv::ProblemOp, riscv::AssignmentOp>(operation))
        preplanned = true;
    });
    if (preplanned) {
      module.emitError("physical planning operations are compiler-owned and cannot be input");
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    unsigned problemIndex = 0;
    for (kernel::KernelOp kernel : module.getOps<kernel::KernelOp>()) {
      auto dimensions = collectAutoDimensions(kernel, *options);
      if (mlir::failed(dimensions)) {
        signalPassFailure();
        return;
      }
      KernelFactCollector collector(builder, kernel);
      if (mlir::failed(collector.collect())) {
        signalPassFailure();
        return;
      }
      llvm::SmallVector<ScheduleSite> scheduleLevels =
          collectLeafScheduleLevels(collector.operationAttributes());
      mlir::ArrayAttr candidates = instantiateCandidates(
          builder, kernel, *dimensions, scheduleLevels, *options);
      for (mlir::Attribute attribute : candidates) {
        auto candidate = mlir::cast<mlir::DictionaryAttr>(attribute);
        mlir::OperationState state(kernel.getLoc(),
                                   riscv::ProblemOp::getOperationName());
        state.addAttribute("sym_name", builder.getStringAttr(
                                           "__weft_candidate_" +
                                           std::to_string(problemIndex++)));
        state.addAttribute("kernel",
                           mlir::FlatSymbolRefAttr::get(module.getContext(),
                                                        kernel.getSymName()));
        state.addAttribute("target",
                           riscv_internal::targetFacts(builder, options->target));
        state.addAttribute("candidate", candidate);
        state.addAttribute("values", collector.valueAttributes());
        state.addAttribute("operations", collector.operationAttributes());
        state.addAttribute("resources", builder.getDictionaryAttr({}));
        state.addAttribute("stage", builder.getStringAttr("facts"));
        builder.create(state);
      }
    }
    if (problemIndex == 0) {
      module.emitError("RISC-V planning requires at least one Weft kernel");
      signalPassFailure();
    }
  }

private:
  std::shared_ptr<const RISCVCompilerOptions> options;
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createConstructRISCVProblemsPass(RISCVCompilerOptions options) {
  return std::make_unique<ConstructRISCVProblemsPass>(std::move(options));
}
