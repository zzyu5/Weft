#include "Weft/Compiler/Selection.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

namespace {

using namespace weft;
using namespace weft::execution;
using namespace weft::kernel;

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
  } else if (name == "weft_execution.axis_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "realization", builder.getStringAttr("scalar")));
    attrs.push_back(attr(builder, "sew", builder.getI64IntegerAttr(0)));
    attrs.push_back(attr(builder, "lmul", builder.getStringAttr("none")));
    attrs.push_back(attr(builder, "unroll", builder.getI64IntegerAttr(1)));
  } else if (name == "weft_execution.memory_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_direct")));
  } else if (name == "weft_execution.reduce_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_linear")));
  } else if (name == "weft_execution.scan_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_linear")));
  } else if (name == "weft_execution.summary_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_linear")));
  } else if (name == "weft_execution.contract_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_nested")));
    attrs.push_back(attr(builder, "micro_m", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_n", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_k", builder.getI64IntegerAttr(1)));
  } else if (name == "weft_execution.math_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_libm")));
  } else if (name == "weft_execution.primitive_plan") {
    attrs.push_back(attr(builder, "provider", builder.getStringAttr("scalar")));
    attrs.push_back(
        attr(builder, "strategy", builder.getStringAttr("scalar_direct")));
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
  for (auto [anchor, operation] : llvm::enumerate(planned)) {
    operation->setAttr(kCanonicalAnchorAttr,
                       builder.getI64IntegerAttr(anchor));
    if (!createSelectedRecord(builder, operation, anchor, options,
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
