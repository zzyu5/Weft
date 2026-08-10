#include "Weft/Compiler/Selection.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

namespace {

using namespace weft;
using namespace weft::execution;
using namespace weft::kernel;

struct Candidate {
  llvm::StringRef provider;
  llvm::StringRef realization;
  llvm::StringRef strategy;
  int64_t sew = 0;
  llvm::StringRef lmul = "none";
  int64_t unroll = 1;
};

llvm::StringRef selectedRecordName(mlir::Operation *operation);

mlir::Type bareType(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return masked.getValueType();
  return type;
}

mlir::Type elementType(mlir::Type type) {
  type = bareType(type);
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  return type;
}

bool isTrueScalarPredicate(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  return value.getType().isInteger(1) && integer && !integer.getValue().isZero();
}

bool isF32RegionValue(mlir::Type type) {
  type = bareType(type);
  auto region = mlir::dyn_cast<RegionType>(type);
  if (!region)
    return false;
  mlir::Type element = region.getElementType();
  if (auto pointer = mlir::dyn_cast<PtrType>(element))
    element = pointer.getElementType();
  return element.isF32();
}

bool isUnitStrideRegionPointer(mlir::Value value, mlir::BlockArgument coordinate) {
  auto pointer = value.getDefiningOp<PtrAddOp>();
  if (!pointer || pointer.getOffset() != coordinate)
    return false;
  mlir::Type base = bareType(pointer.getBase().getType());
  return mlir::isa<PtrType>(base);
}

bool isRVVElementwiseVLA(VLAOp vla, const RISCVTargetProfile &target) {
  if (!target.hasRVV)
    return false;
  mlir::Block &body = vla.getBody().front();
  auto coordinate = body.getArgument(0);
  bool hasMemory = false;
  for (mlir::Operation &operation : body.without_terminator()) {
    if (mlir::isa<ConstantOp, InvalidOp>(operation))
      continue;
    if (auto pointer = mlir::dyn_cast<PtrAddOp>(operation)) {
      mlir::Type result = bareType(pointer.getResult().getType());
      if (mlir::isa<RegionType>(result) &&
          !isUnitStrideRegionPointer(pointer.getResult(), coordinate))
        return false;
      if (mlir::isa<RegionType>(result) && !isF32RegionValue(result))
        return false;
      continue;
    }
    if (auto binary = mlir::dyn_cast<BinaryOp>(operation)) {
      if (!llvm::is_contained({"add", "sub", "mul", "div"}, binary.getKind()))
        return false;
      if (mlir::isa<RegionType>(bareType(binary.getResult().getType())) &&
          !isF32RegionValue(binary.getResult().getType()))
        return false;
      continue;
    }
    if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
      if (!isUnitStrideRegionPointer(load.getPointer(), coordinate) ||
          !isF32RegionValue(load.getResult().getType()) ||
          !isTrueScalarPredicate(load.getWhere()) ||
          !mlir::isa<mlir::NoneType>(load.getOther().getType()))
        return false;
      hasMemory = true;
      continue;
    }
    if (auto store = mlir::dyn_cast<StoreOp>(operation)) {
      if (!isUnitStrideRegionPointer(store.getPointer(), coordinate) ||
          !isF32RegionValue(store.getValue().getType()) ||
          !isTrueScalarPredicate(store.getWhere()))
        return false;
      hasMemory = true;
      continue;
    }
    if (auto reduce = mlir::dyn_cast<ReduceOp>(operation)) {
      if (reduce.getAxis() != -1 || reduce.getKind() != "add" ||
          reduce.getOrder() != "relaxed" ||
          !isF32RegionValue(reduce.getInput().getType()) ||
          !reduce.getResult().getType().isF32() ||
          !reduce.getIdentity().getType().isF32() ||
          !isTrueScalarPredicate(reduce.getWhere()))
        return false;
      if (!llvm::all_of(reduce.getResult().getUsers(),
                        [](mlir::Operation *user) { return mlir::isa<YieldOp>(user); }))
        return false;
      continue;
    }
    return false;
  }
  auto yield = mlir::cast<YieldOp>(body.getTerminator());
  if (yield.getNumOperands() != vla.getNumResults())
    return false;
  for (mlir::Value yielded : yield.getOperands())
    if (!yielded.getDefiningOp<ReduceOp>())
      return false;
  return hasMemory;
}

llvm::DenseSet<mlir::Operation *>
findRVVVLAs(KernelOp kernel, const RISCVTargetProfile &target) {
  llvm::DenseSet<mlir::Operation *> result;
  kernel.walk([&](VLAOp vla) {
    if (isRVVElementwiseVLA(vla, target))
      result.insert(vla.getOperation());
  });
  return result;
}

bool belongsToRVVVLA(mlir::Operation *operation,
                     const llvm::DenseSet<mlir::Operation *> &rvvVLAs) {
  if (auto vla = operation->getParentOfType<VLAOp>())
    return rvvVLAs.contains(vla.getOperation());
  return false;
}

bool isRVVF16F32Contract(ContractOp contract,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || contract.getOrder() != "relaxed" ||
      contract.getMath() != "native" || !contract.getAccDtype().isF32() ||
      !contract.getOutDtype().isF32() || !contract.getOutputOrder().empty() ||
      contract.getLhsAxes().size() != 1 || contract.getLhsAxes().front() != 1 ||
      contract.getRhsAxes().size() != 1 || contract.getRhsAxes().front() != 0 ||
      !isTrueScalarPredicate(contract.getWhereLhs()) ||
      !isTrueScalarPredicate(contract.getWhereRhs()))
    return false;
  auto lhs = mlir::dyn_cast<BlockType>(bareType(contract.getLhs().getType()));
  auto rhs = mlir::dyn_cast<BlockType>(bareType(contract.getRhs().getType()));
  auto init = mlir::dyn_cast<BlockType>(bareType(contract.getInit().getType()));
  auto result = mlir::dyn_cast<BlockType>(bareType(contract.getResult().getType()));
  return lhs && rhs && init && result && lhs.getShape().size() == 2 &&
         rhs.getShape().size() == 2 && init.getShape().size() == 2 &&
         result.getShape().size() == 2 && lhs.getElementType().isF16() &&
         rhs.getElementType().isF16() && init.getElementType().isF32() &&
         result.getElementType().isF32();
}

llvm::SmallVector<Candidate>
buildCandidates(mlir::Operation *canonical,
                const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                const RISCVTargetProfile &target) {
  llvm::SmallVector<Candidate> candidates;
  llvm::StringRef name = selectedRecordName(canonical);
  if (name == "weft_execution.axis_plan")
    candidates.push_back({"scalar", "scalar", {}, 0, "none", 1});
  else if (name == "weft_execution.memory_plan")
    candidates.push_back({"scalar", {}, "scalar_direct"});
  else if (name == "weft_execution.reduce_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.scan_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.summary_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.contract_plan")
    candidates.push_back({"scalar", {}, "scalar_nested"});
  else if (name == "weft_execution.math_plan")
    candidates.push_back({"scalar", {}, "scalar_libm"});
  else if (name == "weft_execution.primitive_plan")
    candidates.push_back({"scalar", {}, "scalar_direct"});

  if (auto vla = mlir::dyn_cast<VLAOp>(canonical);
      vla && rvvVLAs.contains(vla.getOperation()))
    candidates.push_back({"rvv", "rvv", {}, 32, "m1", 1});
  if (mlir::isa<LoadOp, StoreOp>(canonical) &&
      belongsToRVVVLA(canonical, rvvVLAs))
    candidates.push_back({"rvv", {}, "rvv_unit_stride"});
  if (auto reduce = mlir::dyn_cast<ReduceOp>(canonical);
      reduce && belongsToRVVVLA(canonical, rvvVLAs) &&
      reduce.getOrder() == "relaxed")
    candidates.push_back({"rvv", {}, "rvv_tree"});
  if (auto contract = mlir::dyn_cast<ContractOp>(canonical);
      contract && isRVVF16F32Contract(contract, target))
    candidates.push_back({"rvv", {}, "rvv_f16_f32_contract"});
  return candidates;
}

Candidate chooseCandidate(mlir::Operation *canonical,
                          const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                          const RISCVTargetProfile &target) {
  llvm::SmallVector<Candidate> candidates =
      buildCandidates(canonical, rvvVLAs, target);
  assert(!candidates.empty() && "planned anchor requires a legal candidate");
  return candidates.back();
}

llvm::StringRef selectedRecordName(mlir::Operation *operation) {
  if (mlir::isa<MetaValueOp>(operation))
    return "weft_execution.meta_binding";
  if (mlir::isa<VLAOp, BlockAxisOp>(operation))
    return "weft_execution.axis_plan";
  if (mlir::isa<LoadOp, StoreOp, PrefetchOp, AtomicAddOp, FenceOp>(operation))
    return "weft_execution.memory_plan";
  if (mlir::isa<ReduceOp>(operation))
    return "weft_execution.reduce_plan";
  if (mlir::isa<ScanOp>(operation))
    return "weft_execution.scan_plan";
  if (mlir::isa<SummaryFoldOp>(operation))
    return "weft_execution.summary_plan";
  if (mlir::isa<ContractOp, weft::extension::BlockScaledContractOp>(operation))
    return "weft_execution.contract_plan";
  if (auto unary = mlir::dyn_cast<UnaryOp>(operation);
      unary && unary.getKind() != "neg")
    return "weft_execution.math_plan";
  if (mlir::isa<PermuteOp, LookupOp, DecodeOp, WidenOp, NarrowOp>(operation))
    return "weft_execution.primitive_plan";
  return {};
}

bool isMechanicallySupported(mlir::Operation *operation) {
  return mlir::isa<
      KernelOp, ReturnOp, YieldOp, ConditionOp, ConstantOp, IfOp, ForOp,
      WhileOp, FullOp, ExpandDimsOp, BroadcastToOp, ReshapeOp, TransposeOp,
      PtrAddOp, UnaryOp, BinaryOp, CompareOp, CastOp, BitcastOp, SelectOp,
      TupleOp, TupleGetOp, SpecialValueOp, InvalidOp, ValidOp, FillOp>(operation);
}

mlir::Operation *createOperation(mlir::OpBuilder &builder,
                                 llvm::StringRef name,
                                 mlir::Location location,
                                 mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::OperationState state(location, name);
  state.addAttributes(attrs);
  return builder.create(state);
}

mlir::NamedAttribute attr(mlir::OpBuilder &builder, llvm::StringRef name,
                          mlir::Attribute value) {
  return builder.getNamedAttr(name, value);
}

mlir::Operation *createSelectedRecord(mlir::OpBuilder &builder,
                                      mlir::Operation *canonical,
                                      int64_t anchor,
                                      const SelectionOptions &options,
                                      const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                                      llvm::StringSet<> &consumedMeta) {
  llvm::StringRef name = selectedRecordName(canonical);
  if (name.empty())
    return nullptr;
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(attr(builder, "anchor", builder.getI64IntegerAttr(anchor)));

  if (auto meta = mlir::dyn_cast<MetaValueOp>(canonical)) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(meta.getInput());
    auto kernel = canonical->getParentOfType<KernelOp>();
    if (!argument || !kernel)
      return nullptr;
    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(
        kernel.getArgNames()[argument.getArgNumber()]);
    if (!nameAttr)
      return nullptr;
    auto binding = options.metaBindings.find(nameAttr.getValue());
    if (binding == options.metaBindings.end()) {
      canonical->emitError() << "missing --meta binding for "
                             << nameAttr.getValue();
      return nullptr;
    }
    consumedMeta.insert(nameAttr.getValue());
    attrs.push_back(attr(builder, "value",
                         builder.getI64IntegerAttr(binding->second)));
  } else {
    Candidate selected = chooseCandidate(canonical, rvvVLAs, options.target);
    attrs.push_back(attr(builder, "provider", builder.getStringAttr(selected.provider)));
    if (name == "weft_execution.axis_plan") {
      attrs.push_back(attr(builder, "realization",
                           builder.getStringAttr(selected.realization)));
      attrs.push_back(attr(builder, "sew", builder.getI64IntegerAttr(selected.sew)));
      attrs.push_back(attr(builder, "lmul", builder.getStringAttr(selected.lmul)));
      attrs.push_back(attr(builder, "unroll", builder.getI64IntegerAttr(selected.unroll)));
    } else {
      attrs.push_back(attr(builder, "strategy", builder.getStringAttr(selected.strategy)));
    }
  }
  if (name == "weft_execution.contract_plan") {
    attrs.push_back(attr(builder, "micro_m", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_n", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_k", builder.getI64IntegerAttr(1)));
  }
  return createOperation(builder, name, canonical->getLoc(), attrs);
}

mlir::FailureOr<PlanOp> createPlan(mlir::ModuleOp module, KernelOp kernel,
                                   const SelectionOptions &options) {
  mlir::OpBuilder builder(module.getContext());
  std::string symbol = (kernel.getSymName() + "__selected").str();
  if (module.lookupSymbol(symbol)) {
    kernel.emitError("selected plan symbol already exists");
    return mlir::failure();
  }
  builder.setInsertionPointAfter(kernel);
  mlir::OperationState state(kernel.getLoc(), PlanOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symbol));
  state.addAttribute("kernel_ref", mlir::FlatSymbolRefAttr::get(
                                       module.getContext(), kernel.getSymName()));
  state.addAttribute("target_triple",
                     builder.getStringAttr(options.target.triple));
  state.addAttribute("march", builder.getStringAttr(options.target.march));
  state.addAttribute("abi", builder.getStringAttr(options.target.abi));
  state.addAttribute("xlen", builder.getI64IntegerAttr(options.target.xlen));
  state.addAttribute("little_endian",
                     builder.getBoolAttr(options.target.littleEndian));
  state.addAttribute("has_rvv", builder.getBoolAttr(options.target.hasRVV));
  state.addAttribute("vlen_bits",
                     builder.getI64IntegerAttr(options.target.vlenBits));
  state.addAttribute(
      "vector_registers",
      builder.getI64IntegerAttr(options.target.vectorRegisters));
  state.addRegion();
  auto plan = mlir::cast<PlanOp>(builder.create(state));
  plan.getBody().emplaceBlock();

  llvm::SmallVector<mlir::Operation *> planned;
  bool unsupported = false;
  kernel.walk([&](mlir::Operation *operation) {
    if (!selectedRecordName(operation).empty()) {
      planned.push_back(operation);
      return;
    }
    if (isMechanicallySupported(operation))
      return;
    operation->emitError("has no registered realization provider or mechanical lowering");
    unsupported = true;
  });
  if (unsupported) {
    plan.erase();
    return mlir::failure();
  }
  builder.setInsertionPointToStart(&plan.getBody().front());
  llvm::StringSet<> consumedMeta;
  llvm::DenseSet<mlir::Operation *> rvvVLAs =
      findRVVVLAs(kernel, options.target);
  for (auto [anchor, operation] : llvm::enumerate(planned)) {
    operation->setAttr(kCanonicalAnchorAttr,
                       builder.getI64IntegerAttr(anchor));
    if (!createSelectedRecord(builder, operation, anchor, options, rvvVLAs,
                              consumedMeta)) {
      plan.erase();
      return mlir::failure();
    }
  }
  for (const auto &binding : options.metaBindings) {
    if (!consumedMeta.contains(binding.getKey())) {
      kernel.emitError() << "--meta names no constexpr argument: "
                         << binding.getKey();
      plan.erase();
      return mlir::failure();
    }
  }
  createOperation(builder, EndOp::getOperationName(), kernel.getLoc(), {});
  return plan;
}

} // namespace

mlir::LogicalResult
weft::selectExecution(mlir::ModuleOp module,
                      const SelectionOptions &options) {
  module.getContext()->getOrLoadDialect<WEFTExecutionDialect>();
  llvm::SmallVector<KernelOp> kernels;
  for (KernelOp kernel : module.getOps<KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("module contains no canonical Weft kernel");
  for (KernelOp kernel : kernels)
    if (mlir::failed(createPlan(module, kernel, options)))
      return mlir::failure();
  return mlir::success();
}
