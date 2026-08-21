#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>

using namespace weft;

namespace {

using Domains = llvm::StringMap<llvm::SmallVector<std::string>>;
using Choice = llvm::StringMap<std::string>;

llvm::SmallVector<std::string> stringsFrom(mlir::ArrayAttr values) {
  llvm::SmallVector<std::string> result;
  if (!values)
    return result;
  for (mlir::Attribute value : values) {
    if (auto text = mlir::dyn_cast<mlir::StringAttr>(value))
      result.push_back(text.getValue().str());
    else if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value))
      result.push_back(std::to_string(integer.getInt()));
  }
  return result;
}

llvm::SmallVector<std::string> stringsFrom(mlir::DenseI64ArrayAttr values) {
  llvm::SmallVector<std::string> result;
  if (!values)
    return result;
  for (int64_t value : values.asArrayRef())
    result.push_back(std::to_string(value));
  return result;
}

Domains readDomains(mlir::DictionaryAttr source) {
  Domains result;
  for (mlir::NamedAttribute named : source) {
    if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(named.getValue()))
      result[named.getName()] = stringsFrom(array);
    else if (auto integers =
                 mlir::dyn_cast<mlir::DenseI64ArrayAttr>(named.getValue()))
      result[named.getName()] = stringsFrom(integers);
  }
  return result;
}

bool retain(llvm::SmallVectorImpl<std::string> &domain,
            llvm::function_ref<bool(llvm::StringRef)> predicate) {
  size_t oldSize = domain.size();
  llvm::erase_if(domain,
                 [&](const std::string &value) { return !predicate(value); });
  return oldSize != domain.size();
}

bool arc(Domains &domains, llvm::StringRef lhsName, llvm::StringRef rhsName,
         llvm::function_ref<bool(llvm::StringRef, llvm::StringRef)> compatible,
         bool &changed) {
  auto lhs = domains.find(lhsName);
  auto rhs = domains.find(rhsName);
  if (lhs == domains.end() || rhs == domains.end())
    return false;
  changed |= retain(lhs->second, [&](llvm::StringRef lhsValue) {
    return llvm::any_of(rhs->second, [&](llvm::StringRef rhsValue) {
      return compatible(lhsValue, rhsValue);
    });
  });
  changed |= retain(rhs->second, [&](llvm::StringRef rhsValue) {
    return llvm::any_of(lhs->second, [&](llvm::StringRef lhsValue) {
      return compatible(lhsValue, rhsValue);
    });
  });
  return lhs->second.empty() || rhs->second.empty();
}

bool propagate(Domains &domains) {
  bool changed = true;
  while (changed) {
    changed = false;
    if (arc(domains, "nibble_unpack", "byte_interleave",
            [](llvm::StringRef unpack, llvm::StringRef layout) {
              if (unpack == "none")
                return true;
              if (unpack == "and-shift")
                return layout.contains("pair-major");
              if (unpack == "indexed-gather")
                return layout.contains("row-major");
              return false;
            },
            changed))
      return false;
    if (arc(domains, "partial_layout", "horizontal_reduce",
            [](llvm::StringRef partial, llvm::StringRef reduction) {
              if (partial == "not-applicable")
                return reduction == "not-applicable";
              return (partial == "group-major" && reduction == "streamed") ||
                     (partial == "lane-major" && reduction == "end-of-sub");
            },
            changed))
      return false;
    if (arc(domains, "partial_layout", "byte_interleave",
            [](llvm::StringRef partial, llvm::StringRef layout) {
              if (partial == "not-applicable")
                return true;
              if (layout == "not-applicable")
                return true;
              return (partial == "group-major" &&
                      layout.contains("pair-major")) ||
                     (partial == "lane-major" &&
                      layout.contains("row-major"));
            },
            changed))
      return false;
    if (arc(domains, "byte_interleave", "load_stride_alignment",
            [](llvm::StringRef layout, llvm::StringRef load) {
              return (layout.contains("pair-major") && load == "unit-stride") ||
                     (layout.contains("row-major") && load == "strided") ||
                     (layout == "not-applicable" && load == "unit-stride");
            },
            changed))
      return false;
    if (arc(domains, "byte_interleave", "scale_broadcast",
            [](llvm::StringRef layout, llvm::StringRef broadcast) {
              return (layout.contains("pair-major") &&
                      broadcast == "lane-aligned") ||
                     (layout.contains("row-major") &&
                      broadcast == "scalar-splat") ||
                     layout == "not-applicable";
            },
            changed))
      return false;
    if (arc(domains, "pipeline_depth", "prefetch_distance",
            [](llvm::StringRef depth, llvm::StringRef distance) {
              int64_t d = 0, p = 0;
              depth.getAsInteger(10, d);
              distance.getAsInteger(10, p);
              return d == 1 || p >= 1;
            },
            changed))
      return false;
    if (arc(domains, "pipeline_depth", "unroll",
            [](llvm::StringRef depth, llvm::StringRef unroll) {
              int64_t d = 0, u = 0;
              depth.getAsInteger(10, d);
              unroll.getAsInteger(10, u);
              return d == 1 || u >= 2;
            },
            changed))
      return false;
    if (arc(domains, "lane_axis", "lane_cohort_pair",
            [](llvm::StringRef axis, llvm::StringRef pair) {
              return pair.starts_with(axis.str() + ":");
            },
            changed))
      return false;
    if (arc(domains, "cohort", "lane_cohort_pair",
            [](llvm::StringRef cohort, llvm::StringRef pair) {
              return pair.ends_with(":" + cohort.str());
            },
            changed))
      return false;
    llvm::SmallVector<std::string> subbyteInstructions;
    for (const auto &domain : domains)
      if (domain.getKey().starts_with("instruction_subbyte."))
        subbyteInstructions.push_back(domain.getKey().str());
    for (const std::string &instruction : subbyteInstructions)
      if (arc(domains, instruction, "nibble_unpack",
              [](llvm::StringRef hardware, llvm::StringRef unpack) {
                if (hardware == "rvv.vwmaccsu" ||
                    hardware.starts_with("matrix."))
                  return unpack != "none";
                return true;
              },
              changed))
        return false;
    for (const auto &domain : domains)
      if (domain.getValue().empty())
        return false;
  }
  return true;
}

std::optional<int64_t> chosenInteger(const Choice &choice,
                                     llvm::StringRef name) {
  auto found = choice.find(name);
  if (found == choice.end())
    return std::nullopt;
  int64_t result = 0;
  if (llvm::StringRef(found->second).getAsInteger(10, result))
    return std::nullopt;
  return result;
}

std::string lmulSpelling(int64_t eighths) {
  if (eighths < 8)
    return "mf" + std::to_string(8 / eighths);
  return "m" + std::to_string(eighths / 8);
}

std::optional<int64_t> chooseLMUL(int64_t lanes, unsigned sew,
                                  int64_t vlenBits,
                                  mlir::DenseI64ArrayAttr legal,
                                  int64_t multiplier = 1) {
  if (lanes <= 0 || sew == 0 || vlenBits <= 0 || !legal)
    return std::nullopt;
  int64_t required =
      (lanes * static_cast<int64_t>(sew) * 8 + vlenBits - 1) / vlenBits;
  required *= std::max<int64_t>(1, multiplier);
  for (int64_t candidate : legal.asArrayRef())
    if (candidate >= required)
      return candidate;
  return std::nullopt;
}

bool containsAxis(mlir::DictionaryAttr value, int64_t axis) {
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
  return axes && llvm::is_contained(axes.asArrayRef(), axis);
}

struct ResourceResult {
  int64_t peak = 0;
  int64_t stackBytes = 0;
  int64_t spillGroups = 0;
  llvm::SmallVector<std::string> peakLiveClasses;
};

struct Solution {
  Choice choice;
  mlir::DictionaryAttr candidate;
  llvm::StringMap<int64_t> lmulByValue;
  llvm::StringMap<int64_t> registersByValue;
  ResourceResult resources;
  int64_t laneAxis = 0;
  int64_t cohort = 0;
  int64_t groups = 0;
  int64_t lanesPerGroup = 0;
  int64_t physicalLanes = 0;
  int64_t lmulMultiplier = 1;
  int64_t cost = std::numeric_limits<int64_t>::max();
};

int64_t laneExtent(mlir::DictionaryAttr value, int64_t laneAxis);

std::optional<ResourceResult>
calculateResources(riscv::ProblemOp problem, const Choice &choice,
                   int64_t laneAxis, int64_t groups, int64_t physicalLanes,
                   llvm::StringMap<int64_t> &lmulByValue,
                   llvm::StringMap<int64_t> &registersByValue) {
  if (laneAxis < 0)
    return ResourceResult{};
  int64_t vlenBits =
      *riscv_internal::integer(problem.getTarget(), "vlen_bits");
  auto legal = problem.getTarget().getAs<mlir::DenseI64ArrayAttr>(
      "legal_lmul_eighths");
  auto supportedSEW =
      problem.getTarget().getAs<mlir::DenseI64ArrayAttr>("supported_sew");
  const int64_t lmulMultiplier =
      chosenInteger(choice, "lmul_multiplier").value_or(1);
  llvm::SmallVector<mlir::DictionaryAttr> values;
  for (mlir::Attribute attribute : problem.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    std::string id = riscv_internal::string(value, "id")->str();
    values.push_back(value);
    unsigned sew = *riscv_internal::integer(value, "logical_sew");
    if (!containsAxis(value, laneAxis) || sew == 0)
      continue;
    sew = std::max(8u, sew);
    if (!supportedSEW ||
        !llvm::is_contained(supportedSEW.asArrayRef(),
                            static_cast<int64_t>(sew)))
      return std::nullopt;
    const int64_t valueLanes =
        std::min<int64_t>(physicalLanes, laneExtent(value, laneAxis));
    auto lmul =
        chooseLMUL(valueLanes, sew, vlenBits, legal, lmulMultiplier);
    if (!lmul)
      return std::nullopt;
    int64_t registers = ((*lmul + 7) / 8) * groups;
    lmulByValue[id] = *lmul;
    registersByValue[id] = registers;
  }

  int64_t localPeak = 0;
  llvm::SmallVector<std::string> peakLiveClasses;
  for (int64_t ordinal = 0,
               operationCount = problem.getOperations().size();
       ordinal < operationCount; ++ordinal) {
    llvm::StringMap<int64_t> liveClasses;
    for (mlir::DictionaryAttr value : values) {
      int64_t begin =
          riscv_internal::integer(value, "live_start").value_or(0);
      int64_t end = riscv_internal::integer(value, "live_end").value_or(begin);
      if (ordinal < begin || ordinal > end)
        continue;
      llvm::StringRef id = *riscv_internal::string(value, "id");
      int64_t registers = registersByValue.lookup(id);
      if (!registers)
        continue;
      llvm::StringRef handoff =
          riscv_internal::string(value, "handoff_class").value_or(id);
      liveClasses[handoff] = std::max(liveClasses.lookup(handoff), registers);
    }
    int64_t registers = 0;
    for (const auto &entry : liveClasses)
      registers += entry.getValue();
    if (registers > localPeak) {
      localPeak = registers;
      peakLiveClasses.clear();
      for (const auto &entry : liveClasses)
        peakLiveClasses.push_back(entry.getKey().str() + ":" +
                                  std::to_string(entry.getValue()));
      llvm::sort(peakLiveClasses);
    }
  }
  int64_t pipeline = *chosenInteger(choice, "pipeline_depth");
  int64_t unroll = *chosenInteger(choice, "unroll");
  auto operandLMUL =
      chooseLMUL(physicalLanes, 8, vlenBits, legal, lmulMultiplier);
  if (!operandLMUL)
    return std::nullopt;
  int64_t operandGroups = ((*operandLMUL + 7) / 8) * groups;
  int64_t bufferGroups = (pipeline - 1) * 2 * operandGroups;
  int64_t unrollGroups =
      (choice.lookup("partial_layout") == "group-major" ? unroll - 1 : 0) *
      operandGroups;
  int64_t fragmentGroups = 0;
  auto fragments =
      problem.getTarget().getAs<mlir::ArrayAttr>("matrix_fragments");
  for (const auto &selected : choice) {
    llvm::StringRef instruction = selected.getValue();
    if (!instruction.consume_front("matrix."))
      continue;
    for (mlir::Attribute attribute : fragments) {
      auto fragment = mlir::cast<mlir::DictionaryAttr>(attribute);
      if (*riscv_internal::string(fragment, "identity") == instruction)
        fragmentGroups +=
            *riscv_internal::integer(fragment, "fixed_resource_groups");
    }
  }
  ResourceResult result;
  result.peak =
      localPeak + bufferGroups + unrollGroups + fragmentGroups + 2;
  if (bufferGroups)
    peakLiveClasses.push_back("pipeline-buffers:" +
                              std::to_string(bufferGroups));
  if (unrollGroups)
    peakLiveClasses.push_back("unroll-temporaries:" +
                              std::to_string(unrollGroups));
  if (fragmentGroups)
    peakLiveClasses.push_back("matrix-fragments:" +
                              std::to_string(fragmentGroups));
  peakLiveClasses.push_back("reserved:2");
  llvm::sort(peakLiveClasses);
  result.peakLiveClasses = std::move(peakLiveClasses);
  int64_t budget =
      *riscv_internal::integer(problem.getTarget(), "vector_registers");
  if (result.peak > budget) {
    result.spillGroups = result.peak - budget;
    result.stackBytes = result.spillGroups * (vlenBits / 8);
  }
  return result;
}

std::optional<Solution> materializeSolution(riscv::ProblemOp problem,
                                            const Choice &choice,
                                            mlir::DictionaryAttr candidate) {
  Solution solution;
  solution.choice = choice;
  solution.candidate = candidate;
  auto laneAxis = chosenInteger(choice, "lane_axis");
  auto cohort = chosenInteger(choice, "cohort");
  auto groups = chosenInteger(choice, "accumulator_groups");
  if (!laneAxis || !cohort || !groups)
    return std::nullopt;
  solution.laneAxis = *laneAxis;
  solution.cohort = *cohort;
  solution.groups = *groups;
  solution.lmulMultiplier =
      chosenInteger(choice, "lmul_multiplier").value_or(1);
  if (solution.cohort <= 0 || solution.groups <= 0 ||
      solution.cohort % solution.groups)
    return std::nullopt;
  solution.lanesPerGroup = solution.cohort / solution.groups;
  solution.physicalLanes = 1;
  if (solution.laneAxis >= 0) {
    int64_t vlenBits =
        *riscv_internal::integer(problem.getTarget(), "vlen_bits");
    auto legalLMUL = problem.getTarget().getAs<mlir::DenseI64ArrayAttr>(
        "legal_lmul_eighths");
    unsigned maximumSEW = 0;
    for (mlir::Attribute attribute : problem.getValues()) {
      auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
      if (!containsAxis(value, solution.laneAxis))
        continue;
      maximumSEW = std::max<unsigned>(
          maximumSEW,
          std::max<int64_t>(8, *riscv_internal::integer(value, "logical_sew")));
    }
    if (!maximumSEW || !legalLMUL || legalLMUL.empty())
      return std::nullopt;
    const int64_t maximumLMUL = *llvm::max_element(legalLMUL.asArrayRef());
    const int64_t laneCapacity =
        vlenBits * maximumLMUL / (static_cast<int64_t>(maximumSEW) * 8);
    if (laneCapacity <= 0)
      return std::nullopt;
    solution.physicalLanes = std::min(solution.lanesPerGroup, laneCapacity);
  }
  auto resources = calculateResources(
      problem, choice, solution.laneAxis, solution.groups,
      solution.physicalLanes, solution.lmulByValue,
      solution.registersByValue);
  if (!resources)
    return std::nullopt;
  solution.resources = *resources;
  solution.cost = 0;
  int64_t budget =
      *riscv_internal::integer(problem.getTarget(), "vector_registers");
  int64_t stackBudget =
      *riscv_internal::integer(problem.getTarget(), "max_private_stack_bytes");
  llvm::StringRef spill = choice.lookup("spill");
  if (solution.resources.peak > budget && spill != "stack")
    return std::nullopt;
  if (solution.resources.stackBytes > stackBudget)
    return std::nullopt;
  if (solution.laneAxis < 0) {
    solution.cost = 0;
    return solution;
  }
  if (solution.resources.peak <= budget && spill == "stack")
    solution.cost += 5000;
  solution.cost += solution.groups * 40;
  solution.cost -= solution.physicalLanes;
  solution.cost += (solution.lmulMultiplier - 1) * 2;
  if (choice.lookup("co_reduce_schedule") == "shared")
    solution.cost += solution.physicalLanes >= 64 ? 4 : -2;
  solution.cost += choice.lookup("nibble_unpack") == "indexed-gather" ? 35 : 0;
  solution.cost += choice.lookup("partial_layout") == "lane-major" ? 20 : 0;
  solution.cost += solution.resources.spillGroups * 10000;
  solution.cost -= *chosenInteger(choice, "unroll") * 3;
  solution.cost -= *chosenInteger(choice, "pipeline_depth") * 5;
  int64_t prefetch = *chosenInteger(choice, "prefetch_distance");
  solution.cost += prefetch == 1 ? -2 : prefetch * 2;
  return solution;
}

const llvm::StringMap<unsigned> variablePriority{
    {"nibble_unpack", 1},         {"byte_interleave", 2},
    {"partial_layout", 3},
    {"horizontal_reduce", 4},     {"lane_axis", 5},
    {"lane_cohort_pair", 6},      {"cohort", 7},
    {"accumulator_groups", 8},      {"lmul_multiplier", 9},
    {"co_reduce_schedule", 10},     {"pipeline_depth", 11},
    {"unroll", 12},                 {"prefetch_distance", 13},
    {"load_stride_alignment", 14},  {"scale_broadcast", 15},
    {"spill", 16}};

std::optional<int64_t> resolvePartition(llvm::StringRef partition,
                                        mlir::DictionaryAttr candidate) {
  int64_t fixed = 0;
  if (!partition.getAsInteger(10, fixed) && fixed > 0)
    return fixed;
  if (!partition.consume_front("auto:"))
    return std::nullopt;
  auto bindings = candidate.getAs<mlir::DictionaryAttr>("auto_bindings");
  if (!bindings)
    return std::nullopt;
  if (auto value = bindings.getAs<mlir::IntegerAttr>(partition))
    return value.getInt();
  return std::nullopt;
}

bool restrictLaneCohorts(riscv::ProblemOp problem,
                         mlir::DictionaryAttr candidate, Domains &domains) {
  llvm::StringSet<> pairs;
  llvm::StringSet<> authorizedAxes;
  for (const std::string &axis : domains.lookup("lane_axis"))
    authorizedAxes.insert(axis);
  for (mlir::Attribute attribute : problem.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    auto axis = riscv_internal::integer(value, "domain_axis");
    auto partition = riscv_internal::string(value, "domain_partition");
    if (!axis || !partition ||
        !authorizedAxes.contains(std::to_string(*axis)))
      continue;
    auto cohort = resolvePartition(*partition, candidate);
    if (!cohort)
      return false;
    pairs.insert(std::to_string(*axis) + ":" + std::to_string(*cohort));
  }
  if (pairs.empty())
    pairs.insert("-1:1");
  llvm::SmallVector<std::string> pairDomain;
  for (const auto &pair : pairs)
    pairDomain.push_back(pair.getKey().str());
  llvm::sort(pairDomain);
  domains["lane_cohort_pair"] = std::move(pairDomain);
  return true;
}

std::optional<std::string> selectVariable(const Domains &domains) {
  std::optional<std::string> selected;
  unsigned selectedPriority = std::numeric_limits<unsigned>::max();
  for (const auto &entry : domains) {
    if (entry.getValue().size() <= 1)
      continue;
    unsigned priority = variablePriority.lookup(entry.getKey());
    if (entry.getKey().starts_with("instruction.") ||
        entry.getKey().starts_with("instruction_subbyte."))
      priority = 0;
    else if (!variablePriority.contains(entry.getKey()))
      priority = 100;
    if (!selected || priority < selectedPriority ||
        (priority == selectedPriority && entry.getKey() < *selected)) {
      selected = entry.getKey().str();
      selectedPriority = priority;
    }
  }
  return selected;
}

Choice singletonChoice(const Domains &domains) {
  Choice result;
  for (const auto &entry : domains)
    if (entry.getValue().size() == 1)
      result[entry.getKey()] = entry.getValue().front();
  return result;
}

void search(riscv::ProblemOp problem, Domains domains,
            mlir::DictionaryAttr candidate, std::optional<Solution> &best) {
  if (!propagate(domains))
    return;
  std::optional<std::string> variable = selectVariable(domains);
  if (!variable) {
    Choice choice = singletonChoice(domains);
    auto solution = materializeSolution(problem, choice, candidate);
    if (solution && (!best || solution->cost < best->cost))
      best = std::move(solution);
    return;
  }
  llvm::SmallVector<std::string> choices = domains[*variable];
  for (const std::string &value : choices) {
    Domains branch = domains;
    branch[*variable] = {value};
    search(problem, std::move(branch), candidate, best);
  }
}

int64_t laneExtent(mlir::DictionaryAttr value, int64_t laneAxis) {
  auto shape = value.getAs<mlir::DenseI64ArrayAttr>("shape");
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
  if (!shape || !axes)
    return 1;
  for (auto [extent, axis] : llvm::zip(shape.asArrayRef(), axes.asArrayRef()))
    if (axis == laneAxis)
      return extent > 0 ? extent : 1;
  return 1;
}

std::string physicalKind(mlir::DictionaryAttr value, const Solution &solution) {
  llvm::StringRef kind = *riscv_internal::string(value, "kind");
  if (kind == "view" || kind == "slice")
    return "memory";
  if (kind == "encoded_value")
    return "encoded-record";
  if (kind == "control")
    return "control";
  if (kind == "scalar")
    return "scalar";
  if (*riscv_internal::integer(value, "logical_sew") == 0)
    return "encoded-record";
  if (containsAxis(value, solution.laneAxis)) {
    auto shape = value.getAs<mlir::DenseI64ArrayAttr>("shape");
    return (shape && shape.size() > 1) ||
                   laneExtent(value, solution.laneAxis) > solution.physicalLanes
               ? "rvv-stream"
               : "rvv-lane";
  }
  return "sequential";
}

std::string selectedMemoryIdentity(mlir::DictionaryAttr value,
                                   const Solution &solution, int64_t vlenBits) {
  llvm::StringRef kind =
      riscv_internal::string(value, "encoding_kind").value_or("");
  if (kind == "derived_family")
    return solution.choice.lookup("byte_interleave") == "not-applicable"
               ? "not-applicable"
               : solution.choice.lookup("byte_interleave") + ".vlen" +
                     std::to_string(vlenBits);
  return riscv_internal::string(value, "layout_identity").value_or("").str();
}

mlir::ArrayAttr assignedValues(mlir::Builder &builder, riscv::ProblemOp problem,
                               const Solution &solution) {
  int64_t vlenBits =
      *riscv_internal::integer(problem.getTarget(), "vlen_bits");
  const int64_t registerBudget =
      *riscv_internal::integer(problem.getTarget(), "vector_registers");
  llvm::StringMap<int64_t> useCounts;
  llvm::StringMap<std::string> producers;
  for (mlir::Attribute attribute : problem.getOperations()) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    llvm::StringRef name = *riscv_internal::string(operation, "name");
    if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
      for (mlir::Attribute operand : operands)
        ++useCounts[mlir::cast<mlir::StringAttr>(operand).getValue()];
    if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
      for (mlir::Attribute result : results)
        producers[mlir::cast<mlir::StringAttr>(result).getValue()] = name.str();
  }
  llvm::SmallVector<mlir::Attribute> result;
  for (mlir::Attribute attribute : problem.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    std::string id = riscv_internal::string(value, "id")->str();
    unsigned sew = *riscv_internal::integer(value, "logical_sew");
    std::string kind = physicalKind(value, solution);
    llvm::StringRef logicalEncodingKind =
        riscv_internal::string(value, "encoding_kind").value_or("");
    std::string physicalEncodingKind =
        logicalEncodingKind == "derived_family"
            ? "derived_instance"
            : logicalEncodingKind.empty() ? "not-applicable"
                                          : logicalEncodingKind.str();
    int64_t lmul = solution.lmulByValue.lookup(id);
    int64_t registers = solution.registersByValue.lookup(id);
    int64_t logicalLanes = laneExtent(value, solution.laneAxis);
    int64_t valuePhysicalLanes =
        containsAxis(value, solution.laneAxis)
            ? std::min(logicalLanes, solution.physicalLanes)
            : 1;
    int64_t streamParts =
        containsAxis(value, solution.laneAxis)
            ? std::max<int64_t>(
                  1, (logicalLanes + valuePhysicalLanes - 1) /
                         valuePhysicalLanes)
            : 1;
    const bool shareAdmittedValue =
        producers.lookup(id) == "weft_kernel.admit" && useCounts.lookup(id) > 1 &&
        registers * streamParts <= std::max<int64_t>(1, registerBudget / 4);
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> fields{
        {"id", value.get("id")},
        {"source", value.get("source")},
        {"logical_type", value.get("type_spelling")},
        {"physical_kind", builder.getStringAttr(kind)},
        {"physical_encoding_kind",
         builder.getStringAttr(physicalEncodingKind)},
        {"logical_sew", builder.getI64IntegerAttr(sew)},
        {"physical_sew",
         builder.getI64IntegerAttr(sew == 0 ? 0 : std::max(8u, sew))},
        {"lane_axis",
         builder.getI64IntegerAttr(containsAxis(value, solution.laneAxis)
                                       ? solution.laneAxis
                                       : 0)},
        {"physical_lanes",
         builder.getI64IntegerAttr(lmul ? valuePhysicalLanes : 1)},
        {"stream_parts", builder.getI64IntegerAttr(streamParts)},
        {"lmul_eighths", builder.getI64IntegerAttr(lmul)},
        {"lmul", builder.getStringAttr(lmul ? lmulSpelling(lmul) : "none")},
        {"vl", builder.getStringAttr(
                   lmul ? "min(" + std::to_string(valuePhysicalLanes) +
                              ",remaining-axis" +
                              std::to_string(solution.laneAxis) + ")"
                        : "1")},
        {"register_groups", builder.getI64IntegerAttr(registers)},
        {"storage", builder.getStringAttr(
                        kind == "memory" ? "pinned-memory"
                        : kind == "control" ? "control"
                        : kind == "scalar" ? "scalar-register"
                        : kind == "encoded-record" ? "local-record"
                        : kind == "sequential" ? "scalar-or-rematerialized"
                                               : "vector-register")},
        {"materialization",
         builder.getStringAttr(
             producers.lookup(id) == "weft_kernel.admit"
                 ? (shareAdmittedValue ? "shared-register" : "reload-per-use")
                 : "not-applicable")}};
    std::string identity = selectedMemoryIdentity(value, solution, vlenBits);
    if (!identity.empty())
      fields.push_back({"layout_identity", builder.getStringAttr(identity)});
    if (auto family = value.get("encoding_family"))
      fields.push_back({"encoding_family", family});
    result.push_back(riscv_internal::dictionary(builder, fields));
  }
  return builder.getArrayAttr(result);
}

std::string realizationFor(mlir::DictionaryAttr operation,
                           const llvm::StringMap<mlir::DictionaryAttr> &values,
                           const llvm::StringMap<mlir::DictionaryAttr> &operations,
                           mlir::DictionaryAttr resourceModel,
                           const Solution &solution) {
  llvm::StringRef name = *riscv_internal::string(operation, "name");
  if (name == "weft_kernel.for" || name == "weft_kernel.if" ||
      name == "weft_kernel.while")
    return "ordered-scalar-control";
  if (name == "weft_kernel.level") {
    auto operands = operation.getAs<mlir::ArrayAttr>("operands");
    auto domainId = operands && !operands.empty()
                        ? mlir::dyn_cast<mlir::StringAttr>(operands[0])
                        : mlir::StringAttr();
    auto domain = domainId ? values.find(domainId.getValue()) : values.end();
    if (domain == values.end())
      return "invalid-level-domain";
    int64_t axis =
        riscv_internal::integer(domain->second, "domain_axis").value_or(0);
    llvm::StringRef tail =
        riscv_internal::string(domain->second, "domain_tail").value_or("");
    std::string realization = axis == solution.laneAxis
                                  ? "wide-lane-level.vsetvl-" +
                                        std::to_string(solution.physicalLanes) +
                                        "." + tail.str()
                                  : "ordered-sequential-level." + tail.str();
    llvm::StringRef operationId =
        riscv_internal::string(operation, "id").value_or("");
    llvm::StringRef pipelineLevel =
        riscv_internal::string(resourceModel, "pipeline_level")
            .value_or("not-applicable");
    if (operationId == pipelineLevel)
      realization += ".pipeline-depth" +
                     solution.choice.lookup("pipeline_depth") + ".unroll" +
                     solution.choice.lookup("unroll") + ".prefetch" +
                     solution.choice.lookup("prefetch_distance");
    return realization;
  }
  if (name == "weft_kernel.admit")
    return "transfer." + solution.choice.lookup("load_stride_alignment") +
           ".with-level-validity";
  if (name == "weft_kernel.commit")
    return "transfer.store." +
           solution.choice.lookup("load_stride_alignment") +
           ".with-level-validity";
  if (name == "weft_kernel.mac_pairs" ||
      name == "weft_kernel.mac_groups" || name == "weft_kernel.dot" ||
      name == "weft_kernel.contract" ||
      name == "weft_kernel.outer_contract" ||
      name == "weft_kernel.lookup") {
    llvm::StringRef instructionKey =
        riscv_internal::string(operation, "instruction_domain_key")
            .value_or("");
    std::string instruction =
        solution.choice.lookup(instructionKey);
    auto operands = operation.getAs<mlir::ArrayAttr>("operands");
    if (instruction == "rvv.vwmaccsu" && operands && operands.size() >= 2) {
      auto lhs = values.find(mlir::cast<mlir::StringAttr>(operands[0]).getValue());
      auto rhs = values.find(mlir::cast<mlir::StringAttr>(operands[1]).getValue());
      if (lhs != values.end() && rhs != values.end())
        instruction += "(lhs=" +
                       riscv_internal::string(lhs->second, "type_spelling")
                           .value_or("unknown")
                           .str() +
                       ",rhs=" +
                       riscv_internal::string(rhs->second, "type_spelling")
                           .value_or("unknown")
                           .str() +
                       ";typed-order-preserved)";
    }
    return instruction;
  }
  if (name == "weft_kernel.widen") {
    auto results = operation.getAs<mlir::ArrayAttr>("results");
    if (results && results.size() == 1) {
      llvm::StringRef resultId =
          mlir::cast<mlir::StringAttr>(results[0]).getValue();
      mlir::DictionaryAttr soleConsumer;
      int64_t consumerCount = 0;
      for (const auto &entry : operations) {
        auto operands = entry.getValue().getAs<mlir::ArrayAttr>("operands");
        if (!operands)
          continue;
        for (mlir::Attribute operand : operands)
          if (mlir::cast<mlir::StringAttr>(operand).getValue() == resultId) {
            ++consumerCount;
            soleConsumer = entry.getValue();
          }
      }
      if (consumerCount == 1 && soleConsumer &&
          riscv_internal::string(soleConsumer, "name").value_or("") ==
              "weft_kernel.reduce")
        return "rvv.widen-deferred-to-reduction";
    }
    return "rvv.widen-preserve-lanes";
  }
  if (name == "weft_kernel.reduce") {
    auto operands = operation.getAs<mlir::ArrayAttr>("operands");
    if (operands && operands.size() == 1) {
      auto value = values.find(
          mlir::cast<mlir::StringAttr>(operands[0]).getValue());
      if (value != values.end()) {
        llvm::StringRef source =
            riscv_internal::string(value->second, "source").value_or("");
        llvm::StringRef producerId = source.split('.').first;
        auto producer = operations.find(producerId);
        if (producer != operations.end() &&
            riscv_internal::string(producer->second, "name").value_or("") ==
                "weft_kernel.widen") {
          auto producerOperands =
              producer->second.getAs<mlir::ArrayAttr>("operands");
          auto producerResults =
              producer->second.getAs<mlir::ArrayAttr>("results");
          if (producerOperands && producerOperands.size() == 1 && producerResults &&
              producerResults.size() == 1) {
            auto sourceValue = values.find(
                mlir::cast<mlir::StringAttr>(producerOperands[0]).getValue());
            auto targetValue = values.find(
                mlir::cast<mlir::StringAttr>(producerResults[0]).getValue());
            if (sourceValue != values.end() && targetValue != values.end() &&
                riscv_internal::integer(sourceValue->second, "logical_sew")
                        .value_or(0) *
                        2 ==
                    riscv_internal::integer(targetValue->second, "logical_sew")
                        .value_or(0))
              return "rvv.widen-reduce." +
                     riscv_internal::string(operation.getAs<mlir::DictionaryAttr>(
                                                "source_attributes"),
                                            "kind")
                         .value_or("unknown")
                         .str();
          }
        }
      }
    }
    return "rvv.reduce." + solution.choice.lookup("horizontal_reduce");
  }
  if (name == "weft_kernel.fold2")
    return "ordered-pair-fold";
  if (name == "weft_kernel.field") {
    llvm::StringRef packing =
        riscv_internal::string(operation, "field_packing").value_or("");
    if (packing.starts_with("nibble:"))
      return "encoded-field." + solution.choice.lookup("nibble_unpack") + "." +
             packing.str();
    if (packing.starts_with("packed:"))
      return "encoded-field.packed-extract." + packing.str();
    return "encoded-field-extract";
  }
  if (name == "weft_kernel.binary" || name == "weft_kernel.unary" ||
      name == "weft_kernel.cast")
    return "mapped-pointwise";
  if (name == "weft_kernel.materialize")
    return "stage-once";
  if (name == "weft_kernel.pack")
    return "primitive-local-pack";
  return "structural";
}

std::string validityFor(mlir::DictionaryAttr operation) {
  auto path = operation.getAs<mlir::ArrayAttr>("level_path");
  if (!path || path.empty())
    return "always";
  std::string result = "intersection(";
  for (auto [index, level] : llvm::enumerate(path)) {
    if (index)
      result += ",";
    result += mlir::cast<mlir::StringAttr>(level).getValue().str() + ".active";
  }
  result += ")";
  return result;
}

mlir::DictionaryAttr levelMapping(
    mlir::Builder &builder, mlir::DictionaryAttr operation,
    const llvm::StringMap<mlir::DictionaryAttr> &values,
    const Solution &solution) {
  auto operands = operation.getAs<mlir::ArrayAttr>("operands");
  if (!operands || operands.empty())
    return {};
  auto domainId = mlir::dyn_cast<mlir::StringAttr>(operands[0]);
  auto domain = domainId ? values.find(domainId.getValue()) : values.end();
  if (domain == values.end())
    return {};
  int64_t axis =
      riscv_internal::integer(domain->second, "domain_axis").value_or(0);
  llvm::StringRef relation =
      riscv_internal::string(domain->second, "domain_relation").value_or("");
  llvm::StringRef extent =
      riscv_internal::string(domain->second, "domain_extent").value_or("");
  llvm::StringRef partition =
      riscv_internal::string(domain->second, "domain_partition").value_or("");
  llvm::StringRef multiplicity =
      riscv_internal::string(domain->second, "domain_multiplicity").value_or("");
  llvm::StringRef tail =
      riscv_internal::string(domain->second, "domain_tail").value_or("");
  auto physicalPartition = resolvePartition(partition, solution.candidate);
  bool lane = axis == solution.laneAxis && physicalPartition &&
              *physicalPartition > 1;
  const int64_t levelPhysicalLanes =
      lane ? std::min(*physicalPartition, solution.physicalLanes) : 1;
  return riscv_internal::dictionary(
      builder,
      {{"axis", builder.getI64IntegerAttr(axis)},
       {"relation", builder.getStringAttr(relation)},
       {"extent", builder.getStringAttr(extent)},
       {"partition", builder.getStringAttr(partition)},
       {"multiplicity", builder.getStringAttr(multiplicity)},
       {"logical_tail", builder.getStringAttr(tail)},
       {"physical_iteration",
        builder.getStringAttr(
            lane ? (*physicalPartition > levelPhysicalLanes ? "rvv-stream"
                                                            : "rvv-lane")
                 : "ordered-sequential")},
       {"physical_lanes",
        builder.getI64IntegerAttr(levelPhysicalLanes)},
       {"active_extent",
        builder.getStringAttr(
            lane ? "min(" + std::to_string(levelPhysicalLanes) +
                       ",remaining(" + extent.str() + "))"
                 : "clamp(" + extent.str() + "-point*" +
                       partition.str() + ",0," + partition.str() + ")")},
       {"tail_policy",
        builder.getStringAttr(
            lane ? "strip logical partition at physical_lanes; set vl to each active strip"
                 : "execute explicit multiplicity; guard final partition; child intersects parent validity")}});
}

mlir::ArrayAttr assignedOperations(mlir::Builder &builder,
                                   riscv::ProblemOp problem,
                                   const Solution &solution) {
  llvm::StringMap<mlir::DictionaryAttr> values;
  for (mlir::Attribute attribute : problem.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    values[*riscv_internal::string(value, "id")] = value;
  }
  llvm::StringMap<mlir::DictionaryAttr> operations;
  for (mlir::Attribute attribute : problem.getOperations()) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    operations[*riscv_internal::string(operation, "id")] = operation;
  }
  llvm::StringMap<std::string> coReducePartner;
  llvm::StringSet<> coReduceLeader;
  auto operationAttributes = problem.getOperations();
  for (size_t first = 0; first < operationAttributes.size(); ++first) {
    auto lhs = mlir::cast<mlir::DictionaryAttr>(operationAttributes[first]);
    if (riscv_internal::string(lhs, "name").value_or("") !=
        "weft_kernel.reduce")
      continue;
    auto lhsAttrs = lhs.getAs<mlir::DictionaryAttr>("source_attributes");
    auto lhsOperands = lhs.getAs<mlir::ArrayAttr>("operands");
    llvm::StringRef lhsKind =
        riscv_internal::string(lhsAttrs, "kind").value_or("");
    if (!lhsOperands || lhsOperands.size() != 1 ||
        (lhsKind != "max" && lhsKind != "min"))
      continue;
    for (size_t second = first + 1; second < operationAttributes.size(); ++second) {
      auto rhs = mlir::cast<mlir::DictionaryAttr>(operationAttributes[second]);
      if (riscv_internal::string(rhs, "name").value_or("") !=
          "weft_kernel.reduce")
        continue;
      auto rhsAttrs = rhs.getAs<mlir::DictionaryAttr>("source_attributes");
      auto rhsOperands = rhs.getAs<mlir::ArrayAttr>("operands");
      llvm::StringRef rhsKind =
          riscv_internal::string(rhsAttrs, "kind").value_or("");
      if (!rhsOperands || rhsOperands.size() != 1 || lhsKind == rhsKind ||
          (rhsKind != "max" && rhsKind != "min") ||
          lhsOperands[0] != rhsOperands[0] ||
          lhs.get("level_path") != rhs.get("level_path"))
        continue;
      llvm::StringRef lhsId = *riscv_internal::string(lhs, "id");
      llvm::StringRef rhsId = *riscv_internal::string(rhs, "id");
      coReducePartner[lhsId] = rhsId.str();
      coReducePartner[rhsId] = lhsId.str();
      coReduceLeader.insert(lhsId);
      break;
    }
  }
  llvm::SmallVector<mlir::Attribute> result;
  for (mlir::Attribute attribute : problem.getOperations()) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    llvm::StringRef operationId = *riscv_internal::string(operation, "id");
    std::string realization = realizationFor(
        operation, values, operations, problem.getResourceModel(), solution);
    if (solution.choice.lookup("co_reduce_schedule") == "shared" &&
        coReducePartner.contains(operationId))
      realization = coReduceLeader.contains(operationId)
                        ? "rvv.co-reduce.max-min.leader"
                        : "rvv.co-reduce.max-min.follower";
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> fields{
        {"id", operation.get("id")},
        {"source_op", operation.get("name")},
        {"source_attributes", operation.get("source_attributes")},
        {"realization", builder.getStringAttr(realization)},
        {"operands", operation.get("operands")},
        {"results", operation.get("results")},
        {"level_path", operation.get("level_path")},
        {"validity", builder.getStringAttr(validityFor(operation))}};
    if (auto partner = coReducePartner.find(operationId);
        partner != coReducePartner.end())
      fields.push_back(
          {"co_reduce_partner", builder.getStringAttr(partner->second)});
    if (*riscv_internal::string(operation, "name") == "weft_kernel.level")
      fields.push_back(
          {"level_mapping", levelMapping(builder, operation, values, solution)});
    result.push_back(riscv_internal::dictionary(builder, fields));
  }
  return builder.getArrayAttr(result);
}

mlir::DictionaryAttr cDecisions(mlir::Builder &builder,
                                riscv::ProblemOp problem,
                                const Solution &solution) {
  int64_t vlen = *riscv_internal::integer(problem.getTarget(), "vlen_bits");
  int64_t budget =
      *riscv_internal::integer(problem.getTarget(), "vector_registers");
  const bool hasReduction = llvm::any_of(
      problem.getOperations(), [](mlir::Attribute attribute) {
        return riscv_internal::string(
                   mlir::cast<mlir::DictionaryAttr>(attribute), "name")
                   .value_or("") == "weft_kernel.reduce";
      });
  if (solution.laneAxis < 0)
    return riscv_internal::dictionary(
        builder,
        {{"sew", builder.getStringAttr("scalar logical widths")},
         {"lmul", builder.getStringAttr("not-applicable")},
         {"vl", builder.getStringAttr("not-applicable")},
         {"tail", builder.getStringAttr("explicit ordered scalar control")},
         {"accumulator_grouping", builder.getStringAttr("not-applicable")},
         {"register_budget", builder.getStringAttr("0 vector groups")},
         {"partial_layout", builder.getStringAttr("not-applicable")},
         {"horizontal_reduce", builder.getStringAttr("not-applicable")},
         {"vlen_specialization",
          builder.getStringAttr("not-applicable; no explicit wide value")}});
  return riscv_internal::dictionary(
      builder,
      {{"sew", builder.getStringAttr("per-value; logical width preserved, sub-byte arithmetic widens to e8")},
       {"lmul", builder.getStringAttr(
                    "per-value; derived from SEW x lanes / VLEN with legal multiplier " +
                    std::to_string(solution.lmulMultiplier) + ", see values")},
       {"vl", builder.getStringAttr("min(" +
                                    std::to_string(solution.physicalLanes) +
                                    ", remaining-axis" +
                                    std::to_string(solution.laneAxis) + ")")},
       {"tail", builder.getStringAttr(
                    "per-Level level_mapping defines active extent, masks and parent-validity intersection")},
       {"accumulator_grouping",
        builder.getStringAttr(std::to_string(solution.groups) + " groups x " +
                              std::to_string(solution.lanesPerGroup) +
                              " logical lanes; " +
                              std::to_string(solution.physicalLanes) +
                              " lanes per RVV strip")},
       {"register_budget",
        builder.getStringAttr(std::to_string(solution.resources.peak) + "/" +
                              std::to_string(budget) + " vector groups")},
       {"partial_layout",
        builder.getStringAttr(solution.choice.lookup("partial_layout"))},
       {"horizontal_reduce",
        builder.getStringAttr(
            hasReduction ? "per-operation; see selected reduce realization"
                         : solution.choice.lookup("horizontal_reduce"))},
       {"vlen_specialization",
        builder.getStringAttr("fixed-vlen" + std::to_string(vlen) +
                              "; LMUL and vl re-solved per target")}});
}

mlir::DictionaryAttr dDecisions(mlir::Builder &builder,
                                riscv::ProblemOp problem,
                                const Solution &solution) {
  int64_t prefetch = *chosenInteger(solution.choice, "prefetch_distance");
  int64_t pipeline = *chosenInteger(solution.choice, "pipeline_depth");
  int64_t unroll = *chosenInteger(solution.choice, "unroll");
  int64_t vlen = *riscv_internal::integer(problem.getTarget(), "vlen_bits");
  std::string layout = solution.choice.lookup("byte_interleave");
  bool hasInterleave = layout != "not-applicable";
  mlir::DictionaryAttr layoutSource;
  if (auto sources = problem.getResourceModel().getAs<mlir::ArrayAttr>(
          "derived_layout_sources")) {
    for (mlir::Attribute attribute : sources) {
      auto source = mlir::cast<mlir::DictionaryAttr>(attribute);
      llvm::StringRef family = *riscv_internal::string(source, "family");
      std::string prefix = family.str() + ".";
      if (llvm::StringRef(layout).starts_with(prefix)) {
        layoutSource = source;
        break;
      }
    }
  }
  if (hasInterleave && !layoutSource)
    return {};
  int64_t rows =
      layoutSource ? *riscv_internal::integer(layoutSource, "rows") : 0;
  int64_t storageBits = layoutSource
                            ? *riscv_internal::integer(
                                  layoutSource, "base_record_storage_bits")
                            : 0;
  if (hasInterleave)
    layout += ".vlen" + std::to_string(vlen);
  llvm::StringMap<mlir::DictionaryAttr> values;
  for (mlir::Attribute attribute : problem.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    values[*riscv_internal::string(value, "id")] = value;
  }
  llvm::SmallVector<mlir::Attribute> instructions;
  for (mlir::Attribute attribute : problem.getOperations()) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    llvm::StringRef name = *riscv_internal::string(operation, "name");
    if (name != "weft_kernel.mac_pairs" &&
        name != "weft_kernel.mac_groups" && name != "weft_kernel.dot" &&
        name != "weft_kernel.contract" &&
        name != "weft_kernel.outer_contract")
      continue;
    llvm::StringRef instructionKey =
        riscv_internal::string(operation, "instruction_domain_key")
            .value_or("");
    auto operands = operation.getAs<mlir::ArrayAttr>("operands");
    mlir::Attribute lhsType = builder.getStringAttr("not-applicable");
    mlir::Attribute rhsType = builder.getStringAttr("not-applicable");
    if (operands && operands.size() >= 2) {
      auto lhs = values.find(mlir::cast<mlir::StringAttr>(operands[0]).getValue());
      auto rhs = values.find(mlir::cast<mlir::StringAttr>(operands[1]).getValue());
      if (lhs != values.end())
        lhsType = lhs->second.get("type_spelling");
      if (rhs != values.end())
        rhsType = rhs->second.get("type_spelling");
    }
    instructions.push_back(riscv_internal::dictionary(
        builder,
        {{"operation", operation.get("id")},
         {"engine", operation.get("engine")},
         {"instruction",
          builder.getStringAttr(solution.choice.lookup(instructionKey))},
         {"lhs_type", lhsType},
         {"rhs_type", rhsType},
         {"operand_binding",
          builder.getStringAttr("lhs,rhs; signedness follows typed operands")}}));
  }
  return riscv_internal::dictionary(
      builder,
      {{"nibble_unpack",
        builder.getStringAttr(solution.choice.lookup("nibble_unpack"))},
       {"mac_instruction",
        builder.getArrayAttr(instructions)},
       {"scale_broadcast",
        builder.getStringAttr(solution.choice.lookup("scale_broadcast"))},
       {"byte_interleave",
        riscv_internal::dictionary(
            builder,
            {{"builder", builder.getStringAttr(hasInterleave ? "interleave" : "none")},
             {"layout_identity", builder.getStringAttr(layout)},
             {"bit_order",
              hasInterleave ? layoutSource.get("bit_order")
                            : builder.getStringAttr("not-applicable")},
             {"byte_order",
              hasInterleave ? layoutSource.get("byte_order")
                            : builder.getStringAttr("not-applicable")},
             {"base_record_storage_bits",
              builder.getI64IntegerAttr(storageBits)},
             {"rows", builder.getI64IntegerAttr(rows)},
             {"instance_size_bytes",
              builder.getI64IntegerAttr(rows * storageBits / 8)},
             {"alignment",
              hasInterleave ? layoutSource.get("alignment")
                            : builder.getI64IntegerAttr(1)},
             {"target_compatibility",
              builder.getStringAttr("rvv-fixed-vlen")}})},
       {"load_stride_alignment",
        builder.getStringAttr(solution.choice.lookup("load_stride_alignment") +
                              "; base alignment=1")},
       {"prefetch_distance",
        riscv_internal::dictionary(
            builder,
            {{"level", problem.getResourceModel().get("pipeline_level")},
             {"distance", builder.getI64IntegerAttr(prefetch)},
             {"objects",
              problem.getResourceModel().get("prefetch_sources")}})},
       {"pipeline_unroll",
        riscv_internal::dictionary(
            builder,
            {{"level", problem.getResourceModel().get("pipeline_level")},
             {"depth", builder.getI64IntegerAttr(pipeline)},
             {"unroll", builder.getI64IntegerAttr(unroll)}})}});
}

mlir::DictionaryAttr resources(mlir::Builder &builder,
                               riscv::ProblemOp problem,
                               const Solution &solution) {
  int64_t budget =
      *riscv_internal::integer(problem.getTarget(), "vector_registers");
  return riscv_internal::dictionary(
      builder,
      {{"vector_register_budget", builder.getI64IntegerAttr(budget)},
       {"peak_vector_groups",
        builder.getI64IntegerAttr(solution.resources.peak)},
       {"peak_live_handoff_classes",
        riscv_internal::strings(builder, solution.resources.peakLiveClasses)},
       {"spill", builder.getStringAttr(
                     solution.resources.spillGroups ? "stack" : "none")},
       {"spilled_vector_groups",
        builder.getI64IntegerAttr(solution.resources.spillGroups)},
       {"stack_bytes", builder.getI64IntegerAttr(solution.resources.stackBytes)},
       {"cost", builder.getI64IntegerAttr(solution.cost)}});
}

class SolveRISCVProblemsPass final
    : public mlir::PassWrapper<SolveRISCVProblemsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SolveRISCVProblemsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-solve-problems";
  }
  llvm::StringRef getDescription() const final {
    return "solve finite physical domains with propagation and resource backtracking";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    for (riscv::ProblemOp problem :
         llvm::make_early_inc_range(module.getOps<riscv::ProblemOp>())) {
      if (problem.getStage() != "resources") {
        problem.emitError("solver requires a resources-stage problem");
        signalPassFailure();
        return;
      }
      Domains base = readDomains(problem.getDecisionDomains());
      for (llvm::StringRef required :
           {"lane_axis", "cohort", "accumulator_groups", "nibble_unpack",
            "scale_broadcast", "byte_interleave",
            "load_stride_alignment", "partial_layout", "horizontal_reduce",
            "prefetch_distance", "pipeline_depth", "unroll", "spill"}) {
        if (!base.contains(required) || base[required].empty()) {
          problem.emitError() << "physical domain '" << required
                              << "' is empty before solving";
          signalPassFailure();
          return;
        }
      }
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        if (name != "weft_kernel.mac_pairs" &&
            name != "weft_kernel.mac_groups" && name != "weft_kernel.dot" &&
            name != "weft_kernel.contract" &&
            name != "weft_kernel.outer_contract")
          continue;
        std::string key =
            riscv_internal::string(operation, "instruction_domain_key")
                .value_or("")
                .str();
        if (!base.contains(key) || base[key].empty()) {
          problem.emitError() << "instruction domain '" << key
                              << "' is empty before solving";
          signalPassFailure();
          return;
        }
      }
      std::optional<Solution> best;
      for (mlir::Attribute candidateAttribute : problem.getCandidates()) {
        auto candidate = mlir::cast<mlir::DictionaryAttr>(candidateAttribute);
        Domains candidateDomains = base;
        if (!restrictLaneCohorts(problem, candidate, candidateDomains))
          continue;
        search(problem, std::move(candidateDomains), candidate, best);
      }
      if (!best) {
        problem.emitError(
            "no resource-legal RISC-V physical assignment; no fallback exists");
        signalPassFailure();
        return;
      }

      builder.setInsertionPoint(problem);
      mlir::OperationState state(problem.getLoc(),
                                 riscv::AssignmentOp::getOperationName());
      state.addAttribute(
          "sym_name",
          builder.getStringAttr(problem.getSymName().str() + ".assignment"));
      state.addAttribute("kernel", problem.getKernelAttr());
      state.addAttribute("target", problem.getTargetAttr());
      state.addAttribute("candidate", best->candidate);
      state.addAttribute("values", assignedValues(builder, problem, *best));
      state.addAttribute("operations",
                         assignedOperations(builder, problem, *best));
      state.addAttribute("c_decisions", cDecisions(builder, problem, *best));
      state.addAttribute("d_decisions", dDecisions(builder, problem, *best));
      state.addAttribute("resources", resources(builder, problem, *best));
      state.addAttribute("status", builder.getStringAttr("complete"));
      builder.create(state);
      problem.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSolveRISCVProblemsPass() {
  return std::make_unique<SolveRISCVProblemsPass>();
}
