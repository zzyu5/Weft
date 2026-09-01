#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <limits>
#include <optional>
#include <string>

using namespace weft;

namespace {

bool isTerminalRISCVOperation(mlir::Operation *operation) {
  return mlir::isa<
      riscv::EncodingDeclOp, riscv::DerivedEncodingOp, riscv::ArtifactPackOp,
      riscv::ArtifactSizeYieldOp, riscv::ArtifactReturnOp,
      riscv::StorageLoadOp, riscv::StorageStoreOp, riscv::KernelOp,
      riscv::ReturnOp, riscv::RootDomainOp, riscv::RootPointOp,
      riscv::SymbolOp, riscv::DomainOp, riscv::PhysicalPointOp,
      riscv::ConstantOp, riscv::IotaOp, riscv::NewOp,
      riscv::StagedViewOp,
      riscv::MemoryViewOp, riscv::SliceOp, riscv::LoadOp, riscv::StoreOp,
      riscv::FieldOp,
      riscv::ExtractOp, riscv::UpdateOp, riscv::UnaryOp, riscv::BinaryOp,
      riscv::CompareOp, riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
      riscv::ReduceOp, riscv::Fold2Op, riscv::LookupOp,
      riscv::RVVIndexedEntryLoadOp,
      riscv::ConvertLayoutOp,
      riscv::LocalAllocOp, riscv::LocalCapacityGuardOp,
      riscv::LocalBindOp, riscv::LocalLoadOp,
      riscv::LocalStoreOp, riscv::RVVLocalMaterializeOp,
      riscv::EncodedLocalPackOp,
      riscv::SpillOp, riscv::ReloadOp,
      riscv::IMEPackOp, riscv::IMEFragmentMMAOp, riscv::IMEUnpackOp,
      riscv::RegisterMaterializeOp,
      riscv::RVVBitplaneMergeOp,
      riscv::PackedPlaneMergeOp,
      riscv::RVVBitmaskDecodeOp, riscv::RVVSignedBitmaskReduceOp,
      riscv::RVVBitmaskWindowLoadOp,
      riscv::RVVGroupedMacReduceOp,
      riscv::RVVGroupedMacLoadOp,
      riscv::RVVGroupedMacStepOp, riscv::RVVEncodedDotLoadOp,
      riscv::RVVEncodedDotStepOp, riscv::RVVWidenDotOp,
      riscv::RVVWidenMultiplyOp, riscv::RVVWidenScalarMultiplyOp,
      riscv::RVVRegularRepeatIndexOp,
      riscv::RVVRegularRepeatGatherOp,
      riscv::RVVRegularRepeatScalarLoadOp,
      riscv::RVVStorageWindowOp, riscv::RVVUnitEntryWindowLoadOp,
      riscv::RVVLayeredRecordLoadOp,
      riscv::RVVLayeredStorageLoadOp,
      riscv::RVVLayeredStorageDecodeOp, riscv::RVVReplicaStorageLoadOp,
      riscv::RVVIssueSliceOp,
      riscv::RVVWidenAccumulateOp,
      riscv::RVVFinalizeWidenDotOp,
      riscv::RVVPartialSetOp, riscv::RVVPartialCaptureOp,
      riscv::RVVPartialRepackOp,
      riscv::RVVPartialMergeOp, riscv::RVVPartialReduceOp,
      riscv::RVVPartialScaleCombineOp, riscv::RVVPartialWidenScaleOp,
      riscv::RVVPartialCombineOp,
      riscv::RVVPartialFinalizeOp, riscv::RVVAssembleReplicasOp,
      riscv::RVVWidenReduceOp,
      riscv::RVVPartitionedWidenReduceStoreOp,
      riscv::RVVLayeredWindowOp, riscv::RVVLayeredStreamOp,
      riscv::RVVProjectedLayeredStreamOp,
      riscv::RVVStreamReduceOp, riscv::RVVStreamDotOp,
      riscv::RVVStreamContractOp, riscv::RVVSplatOp,
      riscv::RVVAxisBroadcastOp,
      riscv::ProjectReductionOperandOp, riscv::RVVContractStepOp,
      riscv::RVVEncodedContractStepOp>(operation);
}

bool requiresLeaf(mlir::Operation *operation) {
  return mlir::isa<
      riscv::StorageLoadOp, riscv::StorageStoreOp, riscv::IotaOp,
      riscv::LoadOp, riscv::StoreOp, riscv::FieldOp, riscv::ExtractOp,
      riscv::UpdateOp, riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
      riscv::CastOp, riscv::NarrowOp, riscv::WidenOp, riscv::ReduceOp,
      riscv::Fold2Op, riscv::LookupOp, riscv::RVVIndexedEntryLoadOp,
      riscv::ConvertLayoutOp,
      riscv::LocalCapacityGuardOp, riscv::LocalLoadOp, riscv::LocalStoreOp,
      riscv::RVVLocalMaterializeOp, riscv::EncodedLocalPackOp,
      riscv::SpillOp, riscv::ReloadOp,
      riscv::IMEPackOp,
      riscv::IMEFragmentMMAOp, riscv::IMEUnpackOp,
      riscv::RVVBitplaneMergeOp, riscv::PackedPlaneMergeOp,
      riscv::RVVBitmaskDecodeOp, riscv::RVVSignedBitmaskReduceOp,
      riscv::RVVBitmaskWindowLoadOp,
      riscv::RVVGroupedMacReduceOp, riscv::RVVGroupedMacLoadOp,
      riscv::RVVGroupedMacStepOp,
      riscv::RVVEncodedDotLoadOp, riscv::RVVEncodedDotStepOp,
      riscv::RVVWidenDotOp, riscv::RVVWidenMultiplyOp,
      riscv::RVVWidenScalarMultiplyOp,
      riscv::RVVRegularRepeatIndexOp, riscv::RVVRegularRepeatGatherOp,
      riscv::RVVRegularRepeatScalarLoadOp,
      riscv::RVVStorageWindowOp, riscv::RVVUnitEntryWindowLoadOp,
      riscv::RVVLayeredRecordLoadOp,
      riscv::RVVLayeredStorageLoadOp, riscv::RVVLayeredStorageDecodeOp,
      riscv::RVVReplicaStorageLoadOp, riscv::RVVIssueSliceOp,
      riscv::RVVWidenAccumulateOp,
      riscv::RVVFinalizeWidenDotOp,
      riscv::RVVPartialSetOp, riscv::RVVPartialCaptureOp,
      riscv::RVVPartialRepackOp,
      riscv::RVVPartialMergeOp, riscv::RVVPartialReduceOp,
      riscv::RVVPartialScaleCombineOp, riscv::RVVPartialWidenScaleOp,
      riscv::RVVPartialCombineOp,
      riscv::RVVPartialFinalizeOp, riscv::RVVAssembleReplicasOp,
      riscv::RVVWidenReduceOp,
      riscv::RVVPartitionedWidenReduceStoreOp,
      riscv::RVVLayeredWindowOp, riscv::RVVLayeredStreamOp,
      riscv::RVVProjectedLayeredStreamOp,
      riscv::RVVStreamReduceOp, riscv::RVVStreamDotOp,
      riscv::RVVStreamContractOp, riscv::RVVSplatOp,
      riscv::RVVAxisBroadcastOp,
      riscv::ProjectReductionOperandOp,
      riscv::RVVContractStepOp, riscv::RVVEncodedContractStepOp>(operation);
}

int64_t resourceGroups(mlir::Type type) {
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::WindowType>(type))
    return window.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::LayeredWindowType>(type))
    return window.getResourceGroups();
  if (auto partials = mlir::dyn_cast<riscv::PartialSetType>(type))
    return partials.getResourceGroups();
  if (auto layout = riscv_internal::layoutOf(type))
    return layout.getRegisterGroups();
  return 0;
}

bool checkedAdd(int64_t &total, int64_t value) {
  if (value < 0 || total < 0 ||
      total > std::numeric_limits<int64_t>::max() - value)
    return false;
  total += value;
  return true;
}

riscv::EncodingDeclOp findEncoding(mlir::ModuleOp module,
                                   llvm::StringRef family) {
  riscv::EncodingDeclOp result;
  module.walk([&](riscv::EncodingDeclOp declaration) {
    if (!result && declaration.getSymName() == family)
      result = declaration;
  });
  return result;
}

riscv::DerivedEncodingOp findDerivedEncoding(
    mlir::ModuleOp module, kernel::EncodingType encoding) {
  riscv::DerivedEncodingOp result;
  module.walk([&](riscv::DerivedEncodingOp derived) {
    if (!result && derived.getResultFamily() == encoding.getFamily() &&
        derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
        derived.getParameterValues() == encoding.getParameters().asArrayRef())
      result = derived;
  });
  return result;
}

std::optional<int64_t> denseStorageBits(kernel::EncodingType encoding) {
  if (encoding.getKind() != "dense")
    return std::nullopt;
  llvm::StringRef family = encoding.getFamily();
  if (family == "f16" || family == "bf16")
    return 16;
  if (family == "f32")
    return 32;
  if (family == "f64")
    return 64;
  if (family.consume_front("i") || family.consume_front("u")) {
    int64_t width = 0;
    if (!family.getAsInteger(10, width) && width > 0)
      return width;
  }
  return std::nullopt;
}

mlir::LogicalResult verifyDescriptorFacts(mlir::Operation *owner,
                                          riscv::MemDescType descriptor,
                                          mlir::ModuleOp module) {
  auto encoding =
      mlir::dyn_cast<kernel::EncodingType>(descriptor.getEncoding());
  if (!encoding)
    return owner->emitError(
        "memory descriptor lost its canonical Encoding identity");
  if (descriptor.getLayoutIdentity() != encoding.getLayoutIdentity())
    return owner->emitError(
        "memory descriptor layout identity disagrees with its Encoding");
  if (encoding.getKind() == "dense") {
    auto expectedBits = denseStorageBits(encoding);
    if (!expectedBits || descriptor.getStorageBits() != *expectedBits ||
        descriptor.getElements() != 1 || descriptor.getInterleaveRows() != 0)
      return owner->emitError(
          "dense descriptor has non-dense record or interleave geometry");
    return mlir::success();
  }
  llvm::StringRef baseFamily = encoding.getFamily();
  int64_t interleaveRows = 0;
  if (encoding.getKind() == "derived_instance") {
    auto derived = findDerivedEncoding(module, encoding);
    if (!derived)
      return owner->emitError(
          "derived descriptor has no unique physical Encoding declaration");
    baseFamily = derived.getSourceFamily();
    interleaveRows = derived.getInterleaveRows();
  } else if (encoding.getKind() != "base") {
    return owner->emitError(
        "runtime memory descriptor cannot use an abstract Encoding family");
  }
  auto declaration = findEncoding(module, baseFamily);
  if (!declaration || descriptor.getStorageBits() != declaration.getStorageBits() ||
      descriptor.getElements() != declaration.getElements() ||
      descriptor.getAlignment() != declaration.getAlignment() ||
      descriptor.getInterleaveRows() != interleaveRows)
    return owner->emitError(
        "memory descriptor storage facts disagree with its physical Encoding declarations");
  return mlir::success();
}

bool requiresIntegerWidening(mlir::Operation *operation) {
  if (mlir::isa<riscv::RVVGroupedMacReduceOp,
                riscv::RVVGroupedMacStepOp,
                riscv::RVVEncodedDotStepOp,
                riscv::RVVWidenDotOp,
                riscv::RVVPartialSetOp,
                riscv::RVVPartialCaptureOp, riscv::RVVPartialRepackOp,
                riscv::RVVPartialMergeOp,
                riscv::RVVPartialReduceOp,
                riscv::RVVPartialScaleCombineOp,
                riscv::RVVPartialWidenScaleOp,
                riscv::RVVPartialCombineOp,
                riscv::RVVPartialFinalizeOp,
                riscv::RVVWidenAccumulateOp,
                riscv::RVVFinalizeWidenDotOp,
                riscv::RVVWidenReduceOp,
                riscv::RVVPartitionedWidenReduceStoreOp,
                riscv::RVVEncodedContractStepOp>(operation))
    return true;
  auto widen = mlir::dyn_cast<riscv::WidenOp>(operation);
  if (!widen)
    return false;
  return mlir::isa<mlir::IntegerType>(
      riscv_internal::logicalElement(widen.getInput().getType()));
}

bool requiresFloatWidening(mlir::Operation *operation) {
  auto widen = mlir::dyn_cast<riscv::WidenOp>(operation);
  return widen && mlir::isa<mlir::FloatType>(
                      riscv_internal::logicalElement(widen.getInput().getType()));
}

class VerifyFinalRISCVPass
    : public mlir::PassWrapper<VerifyFinalRISCVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-verify-final";
  }
  llvm::StringRef getDescription() const override {
    return "Verify the complete target-aware RISC-V Physical IR contract";
  }

  void runOnOperation() override {
    bool failed = false;
    mlir::ModuleOp module = getOperation();
    llvm::DenseSet<std::pair<int64_t, int64_t>> partialBirths;
    getOperation().walk([&](mlir::Operation *operation) {
      auto verifyPartialBirth = [&](int64_t owner, int64_t birth) {
        if (!partialBirths.insert({owner, birth}).second) {
          operation->emitError()
              << "partial owner " << owner << " has duplicate birth " << birth;
          failed = true;
        }
      };
      if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
        verifyPartialBirth(partial.getOwnerDomainId(), partial.getBirthId());
      if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
        verifyPartialBirth(capture.getOwnerDomainId(), capture.getBirthId());
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "weft_kernel") {
        operation->emitError(
            "Canonical Kernel IR operation survived target conversion");
        failed = true;
      }
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "weft_riscv" &&
          !isTerminalRISCVOperation(operation)) {
        auto diagnostic = operation->emitError()
            << operation->getName()
            << " is not in the final RISC-V terminal allowlist";
        for (mlir::Value result : operation->getResults())
          for (mlir::OpOperand &use : result.getUses())
            diagnostic << " [used by " << use.getOwner()->getName() << "]";
        failed = true;
      }
      for (mlir::Value result : operation->getResults())
        if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
          if (value.getLayout().getCarrier() == "unassigned") {
            operation->emitError("final shaped result has no physical layout");
            failed = true;
          }
          if (value.getLayout().getCarrier() == "ime") {
            operation->emitError(
                "IME representation must be carried by a typed fragment, not a generic value");
            failed = true;
          }
        }
      auto checkDescriptor = [&](mlir::Type type) {
        auto descriptor = mlir::dyn_cast<riscv::MemDescType>(type);
        if (descriptor && mlir::failed(verifyDescriptorFacts(
                              operation, descriptor, module)))
          failed = true;
      };
      for (mlir::Type type : operation->getResultTypes())
        checkDescriptor(type);
      for (mlir::Region &region : operation->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            if (auto value =
                    mlir::dyn_cast<riscv::ValueType>(argument.getType()))
              if (value.getLayout().getCarrier() == "unassigned") {
                operation->emitError(
                    "final shaped block argument has no physical layout");
                failed = true;
              }
            else
              checkDescriptor(argument.getType());
      if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf")) {
        if (leaf.getEngine() == "unselected" ||
            leaf.getInstruction().starts_with("pending-")) {
          operation->emitError()
              << "final target operation " << operation->getName()
              << " has no closed local leaf; leaf=" << leaf
              << ", operands=" << operation->getOperandTypes()
              << ", results=" << operation->getResultTypes();
          failed = true;
        }
        if (mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp,
                      riscv::CompareOp, riscv::CastOp, riscv::NarrowOp,
                      riscv::WidenOp, riscv::ReduceOp,
                      riscv::Fold2Op>(operation)) {
          std::string expected =
              riscv_internal::terminalInstruction(operation);
          if (expected.empty() || leaf.getInstruction() != expected ||
              leaf.getSpelling() != expected) {
            operation->emitError()
                << "final leaf does not match the typed operation; expected "
                << (expected.empty() ? llvm::StringRef("<unsupported>")
                                     : llvm::StringRef(expected))
                << ", got " << leaf.getInstruction();
            failed = true;
          }
        }
      } else if (requiresLeaf(operation)) {
        operation->emitError(
            "final target-local operation has no exact leaf contract");
        failed = true;
      }
      if (operation->hasAttr("implementation")) {
        operation->emitError(
            "structural implementation anchor survived terminal leaf lowering");
        failed = true;
      }
      if (auto dot = mlir::dyn_cast<riscv::RVVWidenDotOp>(operation)) {
        llvm::StringRef kind = dot.getPartialTopology().getKind();
        if (kind != "sequential_fused" &&
            kind != "sequential_per_stream") {
          dot.emitError()
              << "selected partial topology was not materialized; topology="
              << dot.getPartialTopology();
          failed = true;
        }
      }
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
          conversion &&
          conversion.getConversion().getKind() == "time_to_lane") {
        mlir::Operation *definition = conversion.getInput().getDefiningOp();
        bool rematerializable =
            mlir::isa_and_nonnull<riscv::FieldOp, riscv::SliceOp>(definition);
        if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition))
          rematerializable =
              static_cast<bool>(riscv_internal::sourceField(extract.getInput()));
        if (!rematerializable) {
          conversion.emitError(
              "final time-to-lane conversion was not rematerialized to an encoded field edge; input_def=")
              << (definition ? definition->getName().getStringRef()
                             : llvm::StringRef("<block-argument>"))
              << ", input_type=" << conversion.getInput().getType()
              << ", result_type=" << conversion.getResult().getType();
          failed = true;
        }
      }
      if (auto access = operation->getAttrOfType<riscv::AccessAttr>("access"))
        if (access.getForm() == "unassigned") {
          operation->emitError("final memory edge has no selected form");
          failed = true;
        }
      if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation)) {
        auto access = field.getAccess();
        auto leaf = field.getLeaf();
        if (!access || access.getMapping() == "opaque" ||
            leaf.getEngine() != "transfer" ||
            leaf.getFamily() != "encoded-field" ||
            leaf.getInstruction() !=
                ("rvv.encoded." + access.getMapping()).str()) {
          field.emitError(
              "final encoded field has no closed storage mapping and transfer leaf");
          failed = true;
        }
      }
      if (auto load = mlir::dyn_cast<riscv::LoadOp>(operation)) {
        auto access = load.getAccess();
        auto leaf = load.getLeaf();
        if (!access || leaf.getEngine() != "transfer" ||
            leaf.getFamily() != "load" ||
            leaf.getInstruction() != ("rvv.load." + access.getForm()).str() ||
            leaf.getSpelling() != leaf.getInstruction()) {
          load.emitError(
              "final load has no exact access-form transfer leaf");
          failed = true;
        }
      }
      if (auto store = mlir::dyn_cast<riscv::StoreOp>(operation)) {
        auto access = store.getAccess();
        auto leaf = store.getLeaf();
        auto value = mlir::dyn_cast<riscv::ValueType>(store.getValue().getType());
        auto encoding = mlir::dyn_cast<kernel::EncodingType>(
            store.getRegion().getType().getEncoding());
        const bool scalarValue =
            !value || value.getLayout().getCarrier() == "scalar";
        const bool scalarStore =
            access && scalarValue &&
            (access.getForm() == "unit" || access.getForm() == "strided") &&
            access.getMapping() == "dense" && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "store" &&
            leaf.getInstruction() == "scalar.store" &&
            leaf.getSpelling() == leaf.getInstruction() && encoding &&
            encoding.getKind() == "dense" &&
            store.getRegion().getType().getElements() == 1 &&
            store.getRegion().getType().getInterleaveRows() == 0;
        const bool vectorStore =
            access && !scalarValue && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "store" &&
            leaf.getInstruction() == ("rvv.store." + access.getForm()).str() &&
            leaf.getSpelling() == leaf.getInstruction();
        if (!scalarStore && !vectorStore) {
          store.emitError(
              "final store has no exact access-form transfer leaf");
          failed = true;
        }
      }
      if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation)) {
        auto access = extract.getAccess();
        auto leaf = extract.getLeaf();
        const bool local = access && access.getForm() == "local";
        const bool localGather =
            access && access.getForm() == "indexed" &&
            leaf.getEngine() == "rvv" &&
            leaf.getFamily() == "local-gather" &&
            leaf.getInstruction() == "rvv.local-gather";
        const bool localExtract =
            local && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "local-extract" &&
            leaf.getInstruction() == "local.extract";
        const bool registerExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "rvv" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() == "rvv.extract.vrgather";
        const bool laneReplicaExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "rvv" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() == "rvv.extract.lane-to-replica";
        const bool scalarReplicaExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "scalar" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() == "scalar.replica-gather";
        const bool ordinaryExtract =
            access && !local && !localGather && !registerExtract &&
            !laneReplicaExtract && !scalarReplicaExtract &&
            leaf.getEngine() == "transfer" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() ==
                ("rvv.extract." + access.getForm()).str();
        if (!access ||
            (!localExtract && !localGather && !registerExtract &&
             !laneReplicaExtract &&
             !scalarReplicaExtract && !ordinaryExtract)) {
          extract.emitError(
              "final extract has no closed physical access and transfer leaf");
          failed = true;
        }
      }
      if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation)) {
        auto layout = update.getResult().getType().getLayout();
        auto leaf = update.getLeaf();
        if (layout.getCarrier() != "local" || leaf.getEngine() != "scalar" ||
            leaf.getFamily() != "state-update" ||
            leaf.getInstruction() != "local.update" ||
            leaf.getSpelling() != leaf.getInstruction()) {
          update.emitError(
              "final update has no implemented local-state realization");
          failed = true;
        }
      }
      if (auto schedule =
              operation->getAttrOfType<riscv::ScheduleAttr>("schedule"))
        if (schedule.getPipelineDepth() > 1) {
          operation->emitError(
              "unexpanded pipeline schedule survived final physical lowering");
          failed = true;
        }
      if (operation->hasAttr("weft.riscv.schedule")) {
        operation->emitError(
            "typed physical schedule survived without structural expansion");
        failed = true;
      }
      if (auto state = mlir::dyn_cast<riscv::NewOp>(operation))
        if (auto value = mlir::dyn_cast<riscv::ValueType>(
                state.getResult().getType());
            value && value.getLayout().getCarrier() == "local") {
          state.emitError(
              "local state initialization survived without explicit local storage and control");
          failed = true;
        }
      if (auto store = mlir::dyn_cast<riscv::StoreOp>(operation))
        if (auto value =
                mlir::dyn_cast<riscv::ValueType>(store.getValue().getType());
            value && value.getLayout().getCarrier() == "local") {
          store.emitError(
              "local value commit survived without explicit element transfer control");
          failed = true;
        }
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
        auto level = loop->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
        auto direction = loop->getAttrOfType<mlir::StringAttr>(
            "weft.riscv.direction");
        llvm::StringRef ordered =
            level ? level.getDirection()
                  : direction ? direction.getValue() : llvm::StringRef();
        if (ordered != "ascending" && ordered != "descending") {
          loop.emitError("final ordered loop has no physical direction");
          failed = true;
        }
        if (level) {
          riscv::PhysicalPointOp point;
          for (mlir::Operation &nested : loop.getBody()->without_terminator())
            if (!point)
              point = mlir::dyn_cast<riscv::PhysicalPointOp>(nested);
          if (!point) {
            loop.emitError(
                "final physical Level has no explicit point for its logical domain");
            failed = true;
          } else {
            auto domain = point.getResult().getType().getDomain();
            if (domain.getDomainId() != level.getDomainId() ||
                domain.getAxisId() != level.getAxisId() ||
                domain.getRelation() != level.getRelation()) {
              loop.emitError(
                  "final Level identity disagrees with its physical point domain");
              failed = true;
            }
          }
          llvm::DenseSet<int64_t> stateBirths;
          llvm::DenseSet<int64_t> stagedBirths;
          for (mlir::Operation &nested : loop.getBody()->without_terminator()) {
            if (auto state = mlir::dyn_cast<riscv::NewOp>(nested);
                state && state.getOwnerDomainId() == level.getDomainId())
              stateBirths.insert(state.getBirthId());
            if (auto state = mlir::dyn_cast<riscv::LocalBindOp>(nested);
                state && state.getStorage().getType().getPurpose() == "handoff" &&
                state.getStorage().getType().getOwnerDomainId() ==
                    level.getDomainId())
              stateBirths.insert(state.getStorage().getType().getBirthId());
            if (auto staged =
                    mlir::dyn_cast<riscv::RegisterMaterializeOp>(nested);
                staged && staged.getRealization() != "physical-share" &&
                staged.getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(staged.getBirthId());
            if (auto staged = mlir::dyn_cast<riscv::StagedViewOp>(nested);
                staged && staged.getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(staged.getBirthId());
            if (auto staged =
                    mlir::dyn_cast<riscv::RVVLocalMaterializeOp>(nested);
                staged &&
                staged.getDestination()
                        .getDefiningOp<riscv::LocalBindOp>()
                        .getStorage()
                        .getType()
                        .getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(
                  staged.getDestination()
                      .getDefiningOp<riscv::LocalBindOp>()
                      .getStorage()
                      .getType()
                      .getBirthId());
            if (auto staged =
                    mlir::dyn_cast<riscv::EncodedLocalPackOp>(nested);
                staged) {
              auto storage = staged.getStorage().getType();
              if (storage.getOwnerDomainId() == level.getDomainId())
                stagedBirths.insert(storage.getBirthId());
            }
          }
          const int64_t actualState = stateBirths.size();
          const int64_t actualStaged = stagedBirths.size();
          if (level.getStateBirthCount() != actualState ||
              level.getStagedBirthCount() != actualStaged ||
              level.getHandoffCount() !=
                  static_cast<int64_t>(loop.getNumResults())) {
            loop.emitError()
                << "final physical Level lost a birth or handoff identity; expected state="
                << level.getStateBirthCount()
                << ", staged=" << level.getStagedBirthCount()
                << ", handoff=" << level.getHandoffCount()
                << ", observed state=" << actualState
                << ", staged=" << actualStaged
                << ", handoff=" << loop.getNumResults();
            failed = true;
          }
        }
      }
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
        auto level =
            branch->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
        if (!level)
          return;
        if (level.getHandoffCount() !=
            static_cast<int64_t>(branch.getNumResults())) {
          branch.emitError()
              << "pipelined physical Level lost its handoff identity; expected "
              << level.getHandoffCount() << ", observed "
              << branch.getNumResults();
          failed = true;
        }
        bool foundPoint = false;
        branch.getThenRegion().walk([&](riscv::PhysicalPointOp point) {
          auto domain = point.getResult().getType().getDomain();
          if (domain.getDomainId() == level.getDomainId() &&
              domain.getAxisId() == level.getAxisId() &&
              domain.getRelation() == level.getRelation())
            foundPoint = true;
        });
        if (!foundPoint) {
          branch.emitError(
              "pipelined physical Level has no point for its logical domain");
          failed = true;
        }
      }
    });
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>()) {
      for (mlir::BlockArgument argument : kernel.getBody().front().getArguments()) {
        if (!mlir::isa<riscv::MemDescType>(argument.getType()))
          continue;
        int64_t views = 0;
        for (mlir::OpOperand &use : argument.getUses()) {
          auto view = mlir::dyn_cast<riscv::MemoryViewOp>(use.getOwner());
          if (!view || use.getOperandNumber() != 0) {
            kernel.emitError(
                "kernel ABI descriptor bypasses its explicit runtime memory view");
            failed = true;
          } else {
            ++views;
          }
        }
        if (views != 1) {
          kernel.emitError(
              "kernel ABI descriptor requires exactly one explicit runtime memory view");
          failed = true;
        }
      }
      int64_t totalRegisterPeak = kernel.getVectorRegisterPeak();
      if (!checkedAdd(totalRegisterPeak, kernel.getFragmentRegisterPeak()) ||
          totalRegisterPeak > kernel.getTarget().getVectorRegisters()) {
        kernel.emitError("final physical resources exceed the target profile");
        failed = true;
      }
      if (kernel.getLocalStorageBytes() >
          kernel.getTarget().getMaxPrivateStackBytes()) {
        kernel.emitError(
            "final physical local storage exceeds the target profile");
        failed = true;
      }
      kernel.walk([&](mlir::Operation *operation) {
        auto checkType = [&](mlir::Type type) {
          auto value = mlir::dyn_cast<riscv::ValueType>(type);
          if (!value)
            return;
          auto layout = value.getLayout();
          auto shape = value.getShape().asArrayRef();
          auto time = layout.getTimeFactors().asArrayRef();
          auto lane = layout.getLaneFactors().asArrayRef();
          auto replica = layout.getReplicaFactors().asArrayRef();
          auto fragment = layout.getFragmentFactors().asArrayRef();
          auto local = layout.getLocalFactors().asArrayRef();
          for (size_t index = 0; index < shape.size(); ++index) {
            if (shape[index] < 0)
              continue;
            int64_t factors[] = {time[index], lane[index], replica[index],
                                 fragment[index], local[index]};
            auto represented = riscv_internal::staticProduct(factors);
            if (!represented || *represented != shape[index]) {
              operation->emitError()
                  << "physical layout factors are invalid or do not represent axis "
                  << value.getAxisIds()[index] << " with extent " << shape[index];
              failed = true;
            }
          }
          if (layout.getCarrier() != "rvv")
            return;
          if (!kernel.getTarget().getHasRVV() ||
              kernel.getTarget().getVlenBits() <= 0 ||
              kernel.getTarget().getVectorRegisters() <= 0 ||
              !llvm::is_contained(kernel.getTarget().getSupportedSEW().asArrayRef(),
                                  layout.getSew()) ||
              !llvm::is_contained(
                  kernel.getTarget().getLegalLMULEighths().asArrayRef(),
                                  layout.getLmulEighths()) ||
              (value.getElementType().isF16() &&
               !kernel.getTarget().getHasVectorF16())) {
            operation->emitError(
                "RVV layout uses a SEW/LMUL outside the target profile");
            failed = true;
            return;
          }
          int64_t vlmax = kernel.getTarget().getVlenBits() *
                          layout.getLmulEighths() /
                          (8 * layout.getSew());
          auto lanes = riscv_internal::staticProduct(lane);
          auto replicas = riscv_internal::staticProduct(replica);
          const int64_t groupsPerValue =
              (layout.getLmulEighths() + 7) / 8;
          const bool groupsOverflow =
              !replicas ||
              *replicas > std::numeric_limits<int64_t>::max() / groupsPerValue;
          const int64_t expectedGroups =
              groupsOverflow ? -1 : groupsPerValue * *replicas;
          if (!lanes || layout.getVl() > vlmax || *lanes > layout.getVl() ||
              layout.getRegisterGroups() != expectedGroups) {
            operation->emitError(
                "RVV layout lane/vl/register assignment disagrees with the target vector shape: layout=")
                << layout << ", vlmax=" << vlmax << ", lanes="
                << (lanes ? *lanes : -1) << ", expected-groups="
                << expectedGroups;
            failed = true;
          }
        };
        for (mlir::Type type : operation->getResultTypes())
          checkType(type);
        for (mlir::Region &region : operation->getRegions())
          for (mlir::Block &block : region)
            for (mlir::BlockArgument argument : block.getArguments())
              checkType(argument.getType());
        for (mlir::NamedAttribute named : operation->getAttrs()) {
          auto access = mlir::dyn_cast<riscv::AccessAttr>(named.getValue());
          if (!access)
            continue;
          if ((access.getForm() == "indexed" &&
               !kernel.getTarget().getHasIndexedMemory()) ||
              (access.getForm() == "segment" &&
               !kernel.getTarget().getHasSegmentMemory())) {
            operation->emitError()
                << "memory form " << access.getForm()
                << " is unsupported by the final target profile";
            failed = true;
          }
        }
        if (auto laneForm =
                operation->getAttrOfType<mlir::StringAttr>("lane_memory_form")) {
          if ((laneForm.getValue() == "indexed" &&
               !kernel.getTarget().getHasIndexedMemory()) ||
              (laneForm.getValue() == "segment" &&
               !kernel.getTarget().getHasSegmentMemory())) {
            operation->emitError()
                << "lane memory form " << laneForm.getValue()
                << " is unsupported by the final target profile";
            failed = true;
          }
        }
        auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
        if (!leaf)
          return;
        int64_t operandGroups = 0;
        int64_t resultGroups = 0;
        bool resourceOverflow = false;
        for (mlir::Value operand : operation->getOperands())
          resourceOverflow |=
              !checkedAdd(operandGroups, resourceGroups(operand.getType()));
        for (mlir::Value result : operation->getResults())
          resourceOverflow |=
              !checkedAdd(resultGroups, resourceGroups(result.getType()));
        if (resourceOverflow || leaf.getOperandGroups() != operandGroups ||
            leaf.getResultGroups() != resultGroups) {
          operation->emitError()
              << "leaf resource contract is stale: expected operands="
              << operandGroups << ", results=" << resultGroups
              << ", got operands=" << leaf.getOperandGroups()
              << ", results=" << leaf.getResultGroups();
          failed = true;
        }
        if (leaf.getEngine() == "rvv" && !kernel.getTarget().getHasRVV()) {
          operation->emitError("RVV leaf is illegal for a target without RVV");
          failed = true;
        }
        if (requiresIntegerWidening(operation) &&
            !kernel.getTarget().getHasWideningInteger()) {
          operation->emitError(
              "selected leaf requires integer widening unsupported by target");
          failed = true;
        }
        if (requiresFloatWidening(operation) &&
            !kernel.getTarget().getHasWideningFloat()) {
          operation->emitError(
              "selected leaf requires floating widening unsupported by target");
          failed = true;
        }
        const bool arithmeticResult = mlir::isa<
            riscv::IotaOp, riscv::UpdateOp, riscv::UnaryOp, riscv::BinaryOp,
            riscv::CompareOp, riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
            riscv::RVVSplatOp, riscv::RVVBitmaskDecodeOp,
            riscv::RVVSignedBitmaskReduceOp,
            riscv::RVVBitmaskWindowLoadOp,
            riscv::RVVGroupedMacReduceOp,
            riscv::RVVGroupedMacStepOp,
            riscv::RVVEncodedDotStepOp, riscv::RVVLayeredRecordLoadOp,
            riscv::RVVLayeredWindowOp,
            riscv::RVVLayeredStreamOp, riscv::RVVProjectedLayeredStreamOp,
            riscv::RVVContractStepOp,
            riscv::RVVEncodedContractStepOp>(operation);
        for (mlir::Value result : operation->getResults()) {
          if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
            llvm::StringRef carrier = value.getLayout().getCarrier();
            if (arithmeticResult &&
                ((leaf.getEngine() == "rvv" && carrier != "rvv") ||
                 (leaf.getEngine() == "scalar" && carrier != "scalar"))) {
              operation->emitError()
                  << "leaf engine " << leaf.getEngine()
                  << " and result physical carrier " << carrier
                  << " disagree for " << operation->getName();
              failed = true;
            }
          }
          if (mlir::isa<riscv::FragmentType>(result.getType()) &&
              leaf.getEngine() != "ime") {
            operation->emitError(
                "fragment result requires an IME target leaf");
            failed = true;
          }
        }
      });
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createVerifyFinalRISCVPass() {
  return std::make_unique<VerifyFinalRISCVPass>();
}
