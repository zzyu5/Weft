#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

bool isCanonicalKernelOperation(llvm::StringRef name) {
  return name == "weft_kernel.root_domain" ||
         name == "weft_kernel.symbol" || name == "weft_kernel.domain" ||
         name == "weft_kernel.level" ||
         name == "weft_kernel.births_yield" ||
         name == "weft_kernel.handoff" || name == "weft_kernel.return" ||
         name == "weft_kernel.constant" || name == "weft_kernel.iota" ||
         name == "weft_kernel.new" ||
         name == "weft_kernel.materialize" ||
         name == "weft_kernel.admit" || name == "weft_kernel.commit" ||
         name == "weft_kernel.slice" || name == "weft_kernel.field" ||
         name == "weft_kernel.extract" || name == "weft_kernel.update" ||
         name == "weft_kernel.unary" || name == "weft_kernel.binary" ||
         name == "weft_kernel.compare" || name == "weft_kernel.cast" ||
         name == "weft_kernel.narrow" ||
         name == "weft_kernel.mac_groups" || name == "weft_kernel.widen" ||
         name == "weft_kernel.reduce" || name == "weft_kernel.fold2" ||
         name == "weft_kernel.dot" || name == "weft_kernel.contract" ||
         name == "weft_kernel.outer_contract" ||
         name == "weft_kernel.lookup" || name == "arith.constant" ||
         name == "arith.ceildivui" || name == "scf.for" ||
         name == "scf.if" || name == "scf.while" ||
         name == "scf.yield" || name == "scf.condition";
}

std::optional<std::string> indexExpression(mlir::Value value) {
  if (auto symbol = value.getDefiningOp<kernel::SymbolOp>()) {
    if (symbol.getKind() == "source_auto")
      return "auto:" + symbol.getName().str();
    return symbol.getName().str();
  }
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return std::to_string(integer.getInt());
  }
  if (auto ceil = value.getDefiningOp<mlir::arith::CeilDivUIOp>()) {
    auto lhs = indexExpression(ceil.getLhs());
    auto rhs = indexExpression(ceil.getRhs());
    if (lhs && rhs)
      return "ceildiv(" + *lhs + "," + *rhs + ")";
  }
  return std::nullopt;
}

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
    kernel.walk([&](kernel::DomainOp domain) {
      domains.try_emplace(domain.getResult().getType().getDomainId(), domain);
    });
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
      auto domainOperation = domains.find(domain.getDomainId());
      std::string extent = "1";
      std::string partition = "1";
      std::string multiplicity = "1";
      if (domainOperation == domains.end() && domain.getDomainId() != 0) {
        kernel.emitError()
            << "non-root domain identity " << domain.getDomainId()
            << " has no canonical domain operation";
        failed = true;
      } else if (domainOperation != domains.end()) {
        auto extentValue = indexExpression(domainOperation->second.getExtent());
        auto partitionValue =
            indexExpression(domainOperation->second.getPartition());
        auto multiplicityValue =
            indexExpression(domainOperation->second.getMultiplicity());
        if (!extentValue || !partitionValue || !multiplicityValue) {
          domainOperation->second.emitError(
              "domain index expressions must be canonical shape/auto/constant/ceildiv values");
          failed = true;
        } else {
          extent = std::move(*extentValue);
          partition = std::move(*partitionValue);
          multiplicity = std::move(*multiplicityValue);
        }
      }
      fields.push_back({"domain_extent", builder.getStringAttr(extent)});
      fields.push_back({"domain_partition", builder.getStringAttr(partition)});
      fields.push_back(
          {"domain_multiplicity", builder.getStringAttr(multiplicity)});
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
    if (!isCanonicalKernelOperation(operation.getName().getStringRef())) {
      operation.emitError(
          "operation is outside the canonical Kernel IR to RISC-V contract");
      failed = true;
    }
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
    const bool opensLevel = mlir::isa<kernel::LevelOp>(operation);
    const bool opensControl = mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp,
                                        mlir::scf::IfOp>(operation);
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
    kernel.walk([&](mlir::scf::ForOp loop) {
      mlir::Block &body = *loop.getBody();
      auto yield = mlir::cast<mlir::scf::YieldOp>(body.getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInitArgs())) {
        uniteHandoff(initial, body.getArgument(index + 1));
        uniteHandoff(body.getArgument(index + 1), yield.getResults()[index]);
        uniteHandoff(yield.getResults()[index], loop.getResult(index));
      }
    });
    kernel.walk([&](mlir::scf::IfOp branch) {
      auto thenYield = mlir::cast<mlir::scf::YieldOp>(
          branch.getThenRegion().front().getTerminator());
      auto elseYield = mlir::cast<mlir::scf::YieldOp>(
          branch.getElseRegion().front().getTerminator());
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        uniteHandoff(thenYield.getResults()[index], result);
        uniteHandoff(elseYield.getResults()[index], result);
      }
    });
    kernel.walk([&](mlir::scf::WhileOp loop) {
      mlir::Block &condition = loop.getBefore().front();
      mlir::Block &body = loop.getAfter().front();
      auto conditionTerminator =
          mlir::cast<mlir::scf::ConditionOp>(condition.getTerminator());
      auto yield = mlir::cast<mlir::scf::YieldOp>(body.getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInits())) {
        uniteHandoff(initial, condition.getArgument(index));
        uniteHandoff(condition.getArgument(index),
                     conditionTerminator.getArgs()[index]);
        uniteHandoff(conditionTerminator.getArgs()[index],
                     body.getArgument(index));
        uniteHandoff(body.getArgument(index), yield.getResults()[index]);
        uniteHandoff(yield.getResults()[index], loop.getResult(index));
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
  llvm::DenseMap<int64_t, kernel::DomainOp> domains;
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
  mlir::WalkResult result = kernel.walk([&](kernel::SymbolOp operation) {
    if (operation.getKind() != "source_auto")
      return mlir::WalkResult::advance();
    llvm::StringRef name = operation.getName();
    if (!seen.insert(name.str()).second)
      return mlir::WalkResult::advance();
    AutoDimension dimension;
    dimension.name = name.str();
    auto declaredChoices = operation.getChoices();
    dimension.choices.append(declaredChoices.begin(), declaredChoices.end());
    if (dimension.choices.empty()) {
      auto binding = options.metaBindings.find(name);
      if (binding == options.metaBindings.end() || binding->second.empty()) {
        operation.emitError()
            << "auto parameter '" << name
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
  if (name == "scf.for")
    return {false, true};
  if (name == "weft_kernel.mac_groups" ||
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
