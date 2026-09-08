#include "Emitter.h"
#include "../IME/FragmentEmission.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileIMEPack(riscv::IMEPackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto inputType =
      mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  auto fragment = operation.getResult().getType();
  if (!inputType)
    return fail(operation, "IME pack input has no typed physical value");
  auto integer = mlir::dyn_cast<mlir::IntegerType>(inputType.getElementType());
  if (!integer || integer.getWidth() > 8)
    return fail(operation,
                "IME pack currently requires an at-most-eight-bit integer value");
  if (input.kind == Binding::Kind::Field || input.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (input.kind != Binding::Kind::Vector ||
      static_cast<int64_t>(input.parts.size()) !=
          vectorPartCount(operation.getInput()))
    return fail(operation,
                "IME pack input does not materialize its complete selected RVV value");

  auto layout = inputType.getLayout();
  auto shape = fragment.getShape().asArrayRef();
  auto time = layout.getTimeFactors().asArrayRef();
  auto lanes = layout.getLaneFactors().asArrayRef();
  auto replicas = layout.getReplicaFactors().asArrayRef();
  if (shape.size() != time.size() || shape.size() != lanes.size() ||
      shape.size() != replicas.size())
    return fail(operation, "IME pack input layout rank is incomplete");
  int64_t laneDimension = -1;
  int64_t elements = 1;
  llvm::SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
       dimension >= 0; --dimension) {
    if (shape[dimension] <= 0 || time[dimension] <= 0 ||
        lanes[dimension] <= 0 || replicas[dimension] <= 0 ||
        time[dimension] * lanes[dimension] * replicas[dimension] !=
            shape[dimension])
      return fail(operation,
                  "IME pack requires a closed time/lane/register decomposition");
    if (lanes[dimension] > 1) {
      if (laneDimension >= 0)
        return fail(operation,
                    "IME pack currently accepts one SIMD lane axis");
      laneDimension = dimension;
    }
    strides[dimension] = elements;
    elements *= shape[dimension];
  }
  if (laneDimension < 0 || layout.getSew() != 8)
    return fail(operation,
                "IME pack requires one selected eight-bit RVV lane mapping");
  auto scalarType = scalarCType(inputType.getElementType());
  if (!scalarType)
    return fail(operation, "IME pack input has no intrinsic-C scalar type");

  const std::string logical = fresh("ime_pack_logical");
  line("_Alignas(" + std::to_string(operation.getPacking().getAlignment()) + ") " + *scalarType + " " + logical + "[" +
       std::to_string(elements) + "] = {0};");
  const int64_t streams = streamPartCount(operation.getInput());
  const int64_t registers = registerPartCount(operation.getInput());
  for (int64_t reg = 0; reg < registers; ++reg) {
    int64_t registerCursor = reg;
    llvm::SmallVector<int64_t> registerCoordinates(shape.size(), 0);
    for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
         dimension >= 0; --dimension) {
      registerCoordinates[dimension] = registerCursor % replicas[dimension];
      registerCursor /= replicas[dimension];
    }
    if (registerCursor)
      return fail(operation, "IME pack register coordinates are incomplete");
    for (int64_t stream = 0; stream < streams; ++stream) {
      int64_t streamCursor = stream;
      llvm::SmallVector<int64_t> timeCoordinates(shape.size(), 0);
      for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
           dimension >= 0; --dimension) {
        timeCoordinates[dimension] = streamCursor % time[dimension];
        streamCursor /= time[dimension];
      }
      if (streamCursor)
        return fail(operation, "IME pack time coordinates are incomplete");
      int64_t linearBase = 0;
      for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        int64_t coordinate =
            (timeCoordinates[dimension] * replicas[dimension] +
             registerCoordinates[dimension]) *
            lanes[dimension];
        linearBase += coordinate * strides[dimension];
      }
      const size_t part = static_cast<size_t>(reg * streams + stream);
      line("__riscv_vsse8_v_" + vectorSuffix(operation.getInput()) + "(" +
           logical + " + " + std::to_string(linearBase) + ", (ptrdiff_t)" +
           std::to_string(strides[laneDimension] *
                          std::max<int64_t>(1, integer.getWidth() / 8)) +
           ", " + input.parts[part] + ", " +
           partVL(operation.getInput(), part) + ");");
    }
  }

  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();

  if (operation.getRole() != "lhs" && operation.getRole() != "rhs")
    return fail(operation, "IME pack role must be lhs or rhs");
  auto packing = operation.getPacking();
  const bool lhs = operation.getRole() == "lhs";
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  if (shape.size() != 2 || axisOrder.size() != 2)
    return fail(operation, "IME pack requires one rank-two physical axis order");
  const int64_t rows = shape[axisOrder[0]];
  const int64_t columns = shape[axisOrder[1]];
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() != 8 || integer.getWidth() > 8)
    return fail(operation,
                "selected IME fragment has no closed tiled packing geometry");
  const int64_t tileBytes = packing.getRowsPerTile() *
                            packing.getColumnsPerTile() *
                            packing.getStorageBits() / 8;
  const int64_t tileRows = rows / packing.getRowsPerTile();
  const int64_t tileColumns = columns / packing.getColumnsPerTile();
  const int64_t storageBytes = rows * columns * packing.getStorageBits() / 8;
  const std::string name = fresh(lhs ? "ime_lhs_fragment" : "ime_rhs_fragment");
  line("_Alignas(" + std::to_string(packing.getAlignment()) + ") uint8_t " +
       name + "[" + std::to_string(storageBytes) + "];");
  const uint64_t mask = integer.getWidth() == 64
                            ? ~uint64_t(0)
                            : ((uint64_t(1) << integer.getWidth()) - 1);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column) {
      const int64_t tileRow = row / packing.getRowsPerTile();
      const int64_t tileColumn = column / packing.getColumnsPerTile();
      const int64_t tile = packing.getTileOrder() == "row_major"
                               ? tileRow * tileColumns + tileColumn
                               : tileColumn * tileRows + tileRow;
      const int64_t innerRow = row % packing.getRowsPerTile();
      const int64_t innerColumn = column % packing.getColumnsPerTile();
      const int64_t inner = packing.getElementOrder() == "row_major"
                                ? innerRow * packing.getColumnsPerTile() +
                                      innerColumn
                                : innerColumn * packing.getRowsPerTile() +
                                      innerRow;
      const int64_t index = tile * tileBytes + inner;
      llvm::SmallVector<int64_t, 2> logicalCoordinates(2, 0);
      logicalCoordinates[axisOrder[0]] = row;
      logicalCoordinates[axisOrder[1]] = column;
      const int64_t logicalIndex =
          logicalCoordinates[0] * shape[1] + logicalCoordinates[1];
      line(name + "[" + std::to_string(index) + "] = (uint8_t)(" + logical +
           "[" + std::to_string(logicalIndex) + "] & " +
           std::to_string(mask) + ");");
    }
  result.parts.push_back(name);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentMMA(
    riscv::IMEFragmentMMAOp operation) {
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  if (lhs.kind != Binding::Kind::Fragment ||
      rhs.kind != Binding::Kind::Fragment || lhs.parts.size() != 1 ||
      rhs.parts.size() != 1 ||
      lhs.fragmentFamily != operation.getResult().getType().getFamily() ||
      rhs.fragmentFamily != operation.getResult().getType().getFamily() ||
      !operation.getVolatileAsm() || !operation.getMemoryClobber() ||
      operation.getAsmClobbers().empty())
    return fail(operation, "IME MMA operands do not match the selected fragment leaf");
  auto freshName = [&](llvm::StringRef prefix) { return fresh(prefix); };
  auto emitLine = [&](llvm::StringRef text) { line(text.str()); };
  auto output = emitFragmentMMA(operation, lhs.parts.front(), rhs.parts.front(),
                                {freshName, emitLine});
  if (mlir::failed(output))
    return mlir::failure();

  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = operation.getResult().getType().getFamily().str();
  result.parts.push_back(*output);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEUnpack(riscv::IMEUnpackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  if (input.kind != Binding::Kind::Fragment || input.parts.size() != 1 ||
      input.fragmentFamily != operation.getInput().getType().getFamily() ||
      operation.getLeaf().getEngine() != "ime" ||
      operation.getLeaf().getFamily() != "fragment-unpack" ||
      operation.getLeaf().getInstruction() != "ime.unpack.rvv" ||
      operation.getConversion().getKind() != "fragment_to_rvv" ||
      operation.getConversion().getEffect() != "handoff")
    return fail(operation, "IME unpack input has no typed fragment binding");
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t lanes = physicalLanes(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    const std::string name = fresh("ime_unpack");
    line(vectorType(operation.getResult()) + " " + name +
         " = __riscv_vle32_v_" + suffix + "(" + input.parts.front() +
         " + " + std::to_string(part * lanes) + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(name);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
