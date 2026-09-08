#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Verification.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft;
using namespace weft::riscv;
using namespace weft::riscv::detail;

namespace {
llvm::StringRef signedness(RISCVFragmentSignedness value) {
  return value == RISCVFragmentSignedness::Signed ? "signed" : "unsigned";
}

FragmentPackingAttr packingAttr(mlir::MLIRContext *context,
                                const RISCVFragmentPacking &packing) {
  return FragmentPackingAttr::get(
      context, packing.schema,
      mlir::DenseI64ArrayAttr::get(context, packing.axisOrder),
      packing.rowsPerTile, packing.columnsPerTile, packing.tileOrder,
      packing.elementOrder, packing.storageBits, packing.alignment);
}

FragmentCapabilityAttr fragmentCapability(mlir::Operation *operation,
                                          llvm::StringRef instruction) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel)
    return {};
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto capability = mlir::cast<FragmentCapabilityAttr>(attribute);
    if (capability.getInstruction() == instruction)
      return capability;
  }
  return {};
}

} // namespace

FragmentCapabilityAttr weft::riscv::fragmentCapabilityAttr(
    mlir::MLIRContext *context, const RISCVFragmentCapability &capability) {
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Attribute> clobbers;
  for (const std::string &clobber : capability.clobbers)
    clobbers.push_back(builder.getStringAttr(clobber));
  return FragmentCapabilityAttr::get(
      context, capability.name, signedness(capability.lhsSignedness),
      signedness(capability.rhsSignedness), capability.lhsElementBits,
      capability.rhsElementBits, capability.accumulatorElementBits,
      capability.mFactor, capability.nFactor, capability.kFactor,
      capability.lhsResourceGroups, capability.rhsResourceGroups,
      capability.accumulatorResourceGroups,
      packingAttr(context, capability.lhsPacking),
      packingAttr(context, capability.rhsPacking),
      packingAttr(context, capability.accumulatorPacking),
      capability.mmaGroups, capability.mmaChunks, builder.getArrayAttr(clobbers),
      capability.volatileAsm, capability.memoryClobber);
}

bool weft::riscv::fragmentAccumulatorTypeMatches(
    FragmentCapabilityAttr capability, mlir::Type element) {
  const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
  return contract && elementBitWidth(element) == capability.getAccumulatorBits() &&
         integerSignednessMatches(element, signedness(contract->accumulatorSignedness));
}

int64_t weft::riscv::fragmentMMAAdditionalGroups(FragmentCapabilityAttr capability) {
  llvm::SmallVector<unsigned> registers;
  for (mlir::Attribute attribute : capability.getClobbers()) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(attribute).getValue();
    unsigned number = 0;
    if (name.consume_front("v") && !name.getAsInteger(10, number) &&
        number < 32 && !llvm::is_contained(registers, number))
      registers.push_back(number);
  }
  // Input fragment groups are already live before the instruction. Reserve
  // every remaining explicitly clobbered vector register at its issue point.
  return std::max<int64_t>(0, static_cast<int64_t>(registers.size()) -
                                capability.getLhsResourceGroups() -
                                capability.getRhsResourceGroups());
}

mlir::LogicalResult FragmentPackingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef schema, mlir::DenseI64ArrayAttr axisOrder,
    int64_t rowsPerTile, int64_t columnsPerTile,
    llvm::StringRef tileOrder, llvm::StringRef elementOrder,
    int64_t storageBits, int64_t alignment) {
  if (schema.empty() || rowsPerTile <= 0 || columnsPerTile <= 0 ||
      !axisOrder || axisOrder.size() != 2 || axisOrder[0] == axisOrder[1] ||
      llvm::any_of(axisOrder.asArrayRef(),
                   [](int64_t axis) { return axis < 0 || axis > 1; }) ||
      storageBits <= 0 || storageBits % 8 || alignment <= 0)
    return emitError()
           << "fragment packing requires one rank-two byte-addressable tiled storage mapping";
  auto validOrder = [](llvm::StringRef order) {
    return order == "row_major" || order == "column_major";
  };
  if (!validOrder(tileOrder) || !validOrder(elementOrder))
    return emitError()
           << "fragment packing requires explicit tile and in-tile orders";
  return mlir::success();
}

mlir::LogicalResult FragmentCapabilityAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef instruction, llvm::StringRef lhsSignedness,
    llvm::StringRef rhsSignedness, int64_t lhsBits, int64_t rhsBits,
    int64_t accumulatorBits, int64_t mFactor, int64_t nFactor,
    int64_t kFactor, int64_t lhsResourceGroups, int64_t rhsResourceGroups,
    int64_t accumulatorResourceGroups, FragmentPackingAttr lhsPacking,
    FragmentPackingAttr rhsPacking, FragmentPackingAttr accumulatorPacking,
    int64_t mmaGroups, int64_t mmaChunks, mlir::ArrayAttr clobbers,
    bool volatileAsm, bool memoryClobber) {
  const auto *contract = findRISCVFragmentCapability(instruction);
  if (!contract)
    return emitError() << "unsupported fragment instruction " << instruction;
  if (!lhsPacking || !rhsPacking || !accumulatorPacking || !clobbers)
    return emitError() << "fragment capability requires packing and clobbers";
  auto *context = lhsPacking.getContext();
  if (lhsSignedness != signedness(contract->lhsSignedness) ||
      rhsSignedness != signedness(contract->rhsSignedness) ||
      lhsBits != contract->lhsElementBits || rhsBits != contract->rhsElementBits ||
      accumulatorBits != contract->accumulatorElementBits ||
      mFactor != contract->mFactor || nFactor != contract->nFactor ||
      kFactor != contract->kFactor ||
      lhsResourceGroups != contract->lhsResourceGroups ||
      rhsResourceGroups != contract->rhsResourceGroups ||
      accumulatorResourceGroups != contract->accumulatorResourceGroups ||
      lhsPacking != packingAttr(context, contract->lhsPacking) ||
      rhsPacking != packingAttr(context, contract->rhsPacking) ||
      accumulatorPacking != packingAttr(context, contract->accumulatorPacking) ||
      mmaGroups != contract->mmaGroups || mmaChunks != contract->mmaChunks ||
      volatileAsm != contract->volatileAsm || memoryClobber != contract->memoryClobber ||
      clobbers.size() != contract->clobbers.size())
    return emitError() << "fragment capability disagrees with its registered ISA contract";
  for (auto [attribute, name] : llvm::zip(clobbers, contract->clobbers)) {
    auto clobber = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!clobber || clobber.getValue() != name)
      return emitError() << "fragment clobbers disagree with the registered ISA contract";
  }
  if (instruction.empty() || lhsBits <= 0 || rhsBits <= 0 ||
      accumulatorBits <= 0 || mFactor <= 0 || nFactor <= 0 || kFactor <= 0 ||
      (lhsSignedness != "signed" && lhsSignedness != "unsigned") ||
      (rhsSignedness != "signed" && rhsSignedness != "unsigned") ||
      lhsResourceGroups <= 0 || rhsResourceGroups <= 0 ||
      accumulatorResourceGroups <= 0 || !lhsPacking || !rhsPacking ||
      !accumulatorPacking ||
      mmaGroups <= 0 || mmaChunks <= 0 || !volatileAsm || !memoryClobber ||
      clobbers.empty() ||
      llvm::any_of(clobbers, [](mlir::Attribute attribute) {
        auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
        return !value || value.getValue().empty();
      }))
    return emitError() << "fragment capability fields must be positive and complete";
  if (lhsPacking.getRowsPerTile() * lhsPacking.getColumnsPerTile() *
              lhsPacking.getStorageBits() / 8 <=
          0 ||
      rhsPacking.getRowsPerTile() * rhsPacking.getColumnsPerTile() *
              rhsPacking.getStorageBits() / 8 <=
          0 ||
      accumulatorPacking.getRowsPerTile() *
              accumulatorPacking.getColumnsPerTile() *
              accumulatorPacking.getStorageBits() / 8 <=
          0)
    return emitError() << "fragment packing tile geometry is incomplete";
  if (mFactor % lhsPacking.getRowsPerTile() ||
      kFactor % lhsPacking.getColumnsPerTile() ||
      nFactor % rhsPacking.getRowsPerTile() ||
      kFactor % rhsPacking.getColumnsPerTile() ||
      mFactor % accumulatorPacking.getRowsPerTile() ||
      nFactor % accumulatorPacking.getColumnsPerTile() ||
      lhsPacking.getStorageBits() < lhsBits ||
      rhsPacking.getStorageBits() < rhsBits ||
      accumulatorPacking.getStorageBits() < accumulatorBits)
    return emitError()
           << "fragment packing geometry does not cover the capability operands";
  return mlir::success();
}

mlir::LogicalResult IMEFragmentMMAOp::verify() {
  FragmentType lhs = getLhs().getType();
  FragmentType rhs = getRhs().getType();
  FragmentType result = getResult().getType();
  auto capability = fragmentCapability(*this, getLeaf().getInstruction());
  const auto *contract = findRISCVFragmentCapability(getLeaf().getInstruction());
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-mma",
                 capability.getInstruction(), "none", "exact") ||
      lhs.getFamily() != rhs.getFamily() || result.getFamily() != lhs.getFamily() ||
      lhs.getFamily() != capability.getInstruction() ||
      lhs.getRole() != "lhs" || rhs.getRole() != "rhs" ||
      result.getRole() != "accumulator" ||
      lhs.getPacking() != capability.getLhsPacking() ||
      rhs.getPacking() != capability.getRhsPacking() ||
      result.getPacking() != capability.getAccumulatorPacking() ||
      elementBitWidth(lhs.getElementType()) != capability.getLhsBits() ||
      elementBitWidth(rhs.getElementType()) != capability.getRhsBits() ||
      !integerSignednessMatches(lhs.getElementType(),
                                capability.getLhsSignedness()) ||
      !integerSignednessMatches(rhs.getElementType(),
                                capability.getRhsSignedness()) ||
      !fragmentAccumulatorTypeMatches(capability, result.getElementType()) ||
      getLeaf().getFragmentGroups() != fragmentMMAAdditionalGroups(capability) ||
      getGroups() != capability.getMmaGroups() ||
      getChunks() != capability.getMmaChunks() ||
      getAsmClobbers() != capability.getClobbers() ||
      getVolatileAsm() != capability.getVolatileAsm() ||
      getMemoryClobber() != capability.getMemoryClobber() ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}) ||
      lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      getLeaf().getLocalBytes() != contract->mmaLocalBytes)
    return emitOpError("IME fragment MMA requires an IME leaf contract");
  if (lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      lhs.getShape()[0] != capability.getMFactor() ||
      lhs.getShape()[1] != capability.getKFactor() ||
      rhs.getShape()[0] != capability.getKFactor() ||
      rhs.getShape()[1] != capability.getNFactor() ||
      result.getShape()[0] != capability.getMFactor() ||
      result.getShape()[1] != capability.getNFactor() ||
      lhs.getAxisIds()[1] != rhs.getAxisIds()[0] ||
      lhs.getAxisIds()[0] != result.getAxisIds()[0] ||
      rhs.getAxisIds()[1] != result.getAxisIds()[1])
    return emitOpError(
        "IME fragment MMA axes and shapes do not match the selected capability");
  if (lhs.getResourceGroups() != capability.getLhsResourceGroups() ||
      rhs.getResourceGroups() != capability.getRhsResourceGroups() ||
      result.getResourceGroups() != capability.getAccumulatorResourceGroups())
    return emitOpError(
        "IME fragment resources do not match the target capability");
  return mlir::success();
}

mlir::LogicalResult IMEPackOp::verify() {
  FragmentType result = getResult().getType();
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto capability = fragmentCapability(*this, result.getFamily());
  const auto *contract = findRISCVFragmentCapability(result.getFamily());
  if (!input || !capability || !contract ||
      (getRole() != "lhs" && getRole() != "rhs") ||
      !exactLeaf(getLeaf(), "ime", "fragment-pack",
                 getRole() == "lhs" ? "ime.pack.lhs" : "ime.pack.rhs",
                 "none", "exact") ||
      getPacking() != (getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      result.getRole() != getRole() || result.getPacking() != getPacking() ||
      result.getResourceGroups() !=
          (getRole() == "lhs" ? capability.getLhsResourceGroups()
                              : capability.getRhsResourceGroups()) ||
      getPacking().getStorageBits() != 8 ||
      result.getLayout().getCarrier() != "ime" ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}))
    return emitOpError("IME pack requires one typed fragment operand contract");
  const bool lhs = getRole() == "lhs";
  auto packing = getPacking();
  const int64_t expectedBits = lhs ? capability.getLhsBits()
                                   : capability.getRhsBits();
  const llvm::StringRef expectedSignedness =
      lhs ? capability.getLhsSignedness() : capability.getRhsSignedness();
  if (elementBitWidth(result.getElementType()) != expectedBits ||
      elementBitWidth(input.getElementType()) != expectedBits ||
      !integerSignednessMatches(input.getElementType(), expectedSignedness) ||
      input.getAxisIds() != result.getAxisIds())
    return emitOpError("IME pack input and fragment logical domains must agree");
  auto layout = input.getLayout();
  int64_t laneAxes = 0;
  if (layout.getCarrier() != "rvv" || layout.getSew() != 8 ||
      input.getShape().size() != layout.getTimeFactors().size() ||
      input.getShape().size() != layout.getLaneFactors().size() ||
      input.getShape().size() != layout.getReplicaFactors().size() ||
      input.getShape().size() != layout.getFragmentFactors().size() ||
      input.getShape().size() != layout.getLocalFactors().size())
    return emitOpError("IME pack input requires one complete eight-bit RVV mapping");
  for (auto [extent, time, lane, replica, fragment, local] :
       llvm::zip(result.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef())) {
    if (extent <= 0 || time <= 0 || lane <= 0 || replica <= 0 ||
        fragment <= 0 || local <= 0 ||
        time * lane * replica * fragment * local != extent)
      return emitOpError("IME pack input time/lane/register factors are incomplete");
    laneAxes += lane > 1;
  }
  if (laneAxes != 1)
    return emitOpError("IME pack input requires exactly one SIMD lane axis");
  if ((lhs && (result.getShape().size() != 2 ||
               result.getShape()[0] != capability.getMFactor() ||
               result.getShape()[1] != capability.getKFactor())) ||
      (!lhs && (result.getShape().size() != 2 ||
                result.getShape()[0] != capability.getKFactor() ||
                result.getShape()[1] != capability.getNFactor())))
    return emitOpError("IME pack logical shape disagrees with the target fragment");
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  const int64_t rows = result.getShape()[axisOrder[0]];
  const int64_t columns = result.getShape()[axisOrder[1]];
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() < expectedBits)
    return emitOpError(
        "IME operand shape and element width do not fit the selected tiled packing");
  if (getLeaf().getLocalBytes() !=
          (lhs ? contract->lhsPackLocalBytes : contract->rhsPackLocalBytes) ||
      getLeaf().getTemporaryGroups() != contract->packTemporaryGroups)
    return emitOpError(
        "IME pack leaf must account for every primitive-private byte");
  if (getAccess().getMapping() == "opaque" || getAccess().getForm() == "opaque")
    return emitOpError("IME operand has no selected physical access form");
  return mlir::success();
}

mlir::LogicalResult IMEUnpackOp::verify() {
  auto capability = fragmentCapability(*this, getInput().getType().getFamily());
  const auto *contract = findRISCVFragmentCapability(getInput().getType().getFamily());
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-unpack", "ime.unpack.rvv",
                 "none", "exact") ||
      getConversion().getKind() != "fragment_to_rvv" ||
      getConversion().getEffect() != "handoff" ||
      getInput().getType().getRole() != "accumulator" ||
      getInput().getType().getPacking() !=
          capability.getAccumulatorPacking() ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      getInput().getType().getAxisIds() != getResult().getType().getAxisIds() ||
      getInput().getType().getElementType() !=
          getResult().getType().getElementType() ||
      !fragmentAccumulatorTypeMatches(capability, getResult().getType().getElementType()) ||
      getLeaf().getLocalBytes() != 0 ||
      getLeaf().getTemporaryGroups() != contract->unpackTemporaryGroups ||
      getConversion().getTemporaryGroups() != contract->unpackTemporaryGroups)
    return emitOpError("IME unpack must preserve a fragment logical domain into RVV");
  auto fragment = getInput().getType();
  auto layout = getResult().getType().getLayout();
  for (auto [extent, time, lane, replica, fragmentFactor, local] :
       llvm::zip(fragment.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef()))
    if (time * lane * replica * fragmentFactor * local != extent)
      return emitOpError(
          "IME unpack result layout does not cover the complete fragment domain");
  return mlir::success();
}
