#include "Emitter.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileFieldRead(riscv::FieldReadOp operation) {
  auto projection = fieldProjections.find(operation.getInput());
  if (projection == fieldProjections.end())
    return fail(operation, "field read input is not its selected storage projection");
  Binding input;
  input.kind = Binding::Kind::Field;
  input.field = projection->second;
  llvm::SmallVector<mlir::Value> captured;
  riscv::fieldReadProjection(operation.getInput(), &captured);
  auto bindIndex = [&](mlir::Value index) -> std::optional<mlir::Value> {
    auto found = llvm::find(captured, index);
    if (found == captured.end())
      return std::nullopt;
    return operation.getIndices()[static_cast<size_t>(found - captured.begin())];
  };
  if (input.field.index) {
    auto index = bindIndex(*input.field.index);
    if (!index)
      return fail(operation, "field read has no explicit primary address operand");
    input.field.index = *index;
  }
  for (mlir::Value &relative : input.field.relativeIndices) {
    auto index = bindIndex(relative);
    if (!index)
      return fail(operation, "field read has no explicit relative address operand");
    relative = *index;
  }
  input.field.useAccess = operation.getAccess();
  auto loaded = materializeNumeric(operation.getResult(), std::move(input));
  if (mlir::failed(loaded))
    return mlir::failure();
  bindings[operation.getResult()] = std::move(*loaded);
  return mlir::success();
}

std::optional<std::string> Emitter::denseAddress(
    const SliceInfo &slice,
    llvm::ArrayRef<std::pair<int64_t, std::string>> offsets) const {
  auto memoryIt = bindings.find(slice.base);
  if (memoryIt == bindings.end() || memoryIt->second.kind != Binding::Kind::Memory)
    return std::nullopt;
  const MemoryInfo &memory = memoryIt->second.memory;
  llvm::StringMap<std::string> offsetByAxis;
  for (auto [axis, expression] : offsets)
    offsetByAxis[std::to_string(axis)] = expression;
  std::string linear = "0";
  size_t domainCursor = 0;
  for (size_t dimension = 0; dimension < memory.axes.size(); ++dimension) {
    std::string coordinate = "0";
    if (dimension < slice.selectors.size() &&
        slice.selectors[dimension] != "all") {
      const std::string &selector = slice.selectors[dimension];
      if (domainCursor >= slice.indices.size())
        return std::nullopt;
      const Binding &index = bindings.lookup(slice.indices[domainCursor++]);
      if (selector == "domain")
        coordinate = index.point.base;
      else if (selector == "group_index")
        coordinate = "(" + index.point.base + " / " +
                     std::to_string(index.point.physicalExtent) + ")";
      else if (selector == "index")
        coordinate = index.scalar;
      else
        return std::nullopt;
    }
    for (auto [axis, value] : slice.localOffsets) {
      if (axis != memory.axes[dimension])
        continue;
      const Binding &offset = bindings.lookup(value);
      std::string expression;
      if (offset.kind == Binding::Kind::Scalar)
        expression = offset.scalar;
      else if (offset.kind == Binding::Kind::Point)
        expression = offset.point.base;
      else
        return std::nullopt;
      coordinate = "(" + coordinate + " + " + expression + ")";
    }
    for (auto [axis, offset] : slice.staticOffsets)
      if (axis == memory.axes[dimension] && offset != 0)
        coordinate = "(" + coordinate + " + " + std::to_string(offset) + ")";
    auto extra = offsetByAxis.find(std::to_string(memory.axes[dimension]));
    if (extra != offsetByAxis.end())
      coordinate = "(" + coordinate + " + " + extra->second + ")";
    if (dimension < memory.origins.size() && memory.origins[dimension] != "0")
      coordinate = "(" + coordinate + " - " + memory.origins[dimension] + ")";
    linear = "(" + linear + " + (" + coordinate + ") * (" +
             memory.strides[dimension] + "))";
  }
  return memory.name + " + " + linear;
}

mlir::FailureOr<Binding>
Emitter::loadDenseBlock(mlir::Value result, const Binding &memoryBlock) {
  Binding loaded;
  loaded.kind = Binding::Kind::Vector;
  auto slice = memoryBlock.slice;
  const Binding &base = bindings.lookup(slice.base);
  if (base.kind != Binding::Kind::Memory) {
    result.getDefiningOp()->emitError("selected dense load has no memory-backed slice");
    return mlir::failure();
  }
  const MemoryInfo &memory = base.memory;
  mlir::Type resultElement = riscv_internal::logicalElement(result.getType());
  if (!memory.elementType || memory.elementType != resultElement) {
    result.getDefiningOp()->emitError(
        "dense load element type does not match its selected local value");
    return mlir::failure();
  }
  auto elementType = scalarCType(resultElement);
  unsigned sew = riscv_internal::logicalBitWidth(result.getType());
  if (!elementType || !sew)
    return mlir::failure();
  const int64_t laneAxis = laneAxisFor(result);
  llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(result);
  size_t laneDimension = llvm::find(memory.axes, laneAxis) - memory.axes.begin();
  if (laneDimension >= memory.strides.size()) {
    result.getDefiningOp()->emitError("selected dense load has no physical lane axis");
    return mlir::failure();
  }
  const std::string suffix = vectorSuffix(result);
  const std::string type = vectorType(result);
  mlir::Operation *transfer = result.getDefiningOp();
  while (transfer) {
    if (auto fresh = mlir::dyn_cast<riscv::NewOp>(transfer);
        fresh && fresh.getInitialized()) {
      transfer = fresh.getInitial().getDefiningOp();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(transfer)) {
      transfer = conversion.getInput().getDefiningOp();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(transfer)) {
      transfer = materialize.getInput().getDefiningOp();
      continue;
    }
    break;
  }
  riscv::AccessAttr edge = accessOf(transfer);
  llvm::StringRef memoryForm = edge ? edge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided") {
    result.getDefiningOp()->emitError(
        "dense load has no pass-selected unit or strided memory form");
    return mlir::failure();
  }
  const int64_t streams = streamPartCount(result);
  const int64_t totalParts = vectorPartCount(result);
  for (int64_t part = 0; part < totalParts; ++part) {
    auto coordinates = registerCoordinates(result, part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size()) {
      result.getDefiningOp()->emitError(
          "selected dense load has no register-coordinate mapping");
      return mlir::failure();
    }
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string active = "1";
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      offsets.push_back({axis, std::to_string(coordinate)});
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
    }
    if (streams > 1)
      offsets.push_back({laneAxis, partOffset(result, part)});
    auto address = denseAddress(slice, offsets);
    if (!address) {
      result.getDefiningOp()->emitError("selected dense load has no address relation");
      return mlir::failure();
    }
    std::string expression;
    if (memoryForm == "unit")
      expression = "__riscv_vle" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + partVL(result, part) + ")";
    else
      expression = "__riscv_vlse" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + memory.strides[laneDimension] +
                   " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                   partVL(result, part) + ")";
    std::string name = fresh("load");
    if (!registerAxes.empty()) {
      std::string zero = mlir::isa<mlir::FloatType>(resultElement)
                             ? "__riscv_vfmv_v_f_" + suffix + "(0.0, " +
                                   partVL(result, part) + ")"
                             : "__riscv_vmv_v_x_" + suffix + "(0, " +
                                   partVL(result, part) + ")";
      line(type + " " + name + " = " + zero + ";");
      line("if (" + active + ") " + name + " = " + expression + ";");
    } else {
      line(type + " " + name + " = " + expression + ";");
    }
    loaded.parts.push_back(std::move(name));
  }
  return loaded;
}

mlir::FailureOr<Binding> Emitter::materializeNumeric(mlir::Value value,
                                                     Binding binding) {
  if (binding.kind == Binding::Kind::Slice) {
    auto layout = layoutOf(value);
    llvm::StringRef carrier = layout ? layout.getCarrier() : llvm::StringRef("scalar");
    if (carrier == "scalar") {
      const int64_t parts = registerPartCount(value);
      if (parts > 1) {
        const Binding &base = bindings.lookup(binding.slice.base);
        mlir::Type element = riscv_internal::logicalElement(value.getType());
        auto type = scalarCType(element);
        auto axes = registerAxesFor(value);
        if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
            base.memory.elementType != element || !type) {
          value.getDefiningOp()->emitError(
              "register-replicated dense load has no matching scalar memory type");
          return mlir::failure();
        }
        Binding tuple;
        tuple.kind = Binding::Kind::ScalarTuple;
        for (int64_t part = 0; part < parts; ++part) {
          auto coordinates = registerCoordinates(value, part);
          if (!coordinates || coordinates->size() != axes.size()) {
            value.getDefiningOp()->emitError(
                "register-replicated dense load has no typed coordinate mapping");
            return mlir::failure();
          }
          llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
          for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
            offsets.push_back({axis, std::to_string(coordinate)});
          auto address = denseAddress(binding.slice, offsets);
          if (!address)
            return mlir::failure();
          tuple.parts.push_back(materializeScalarRead(
              *type, "*(const " + *type + " *)(" + *address + ")"));
        }
        bindings[value] = tuple;
        return tuple;
      }
      auto valueType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      auto layout = valueType ? valueType.getLayout() : riscv::LayoutAttr();
      const bool scalarElement = value.getType().isIntOrFloat() ||
                                 value.getType().isIndex();
      if ((!valueType && !scalarElement) ||
          (valueType &&
           (!layout || product(layout.getTimeFactors()) != 1 ||
            product(layout.getLaneFactors()) != 1 ||
            product(layout.getReplicaFactors()) != 1 ||
            product(layout.getFragmentFactors()) != 1 ||
            product(layout.getLocalFactors()) != 1))) {
        value.getDefiningOp()->emitError(
            "sequential dense load requires one selected physical element");
        return mlir::failure();
      }
      const Binding &base = bindings.lookup(binding.slice.base);
      mlir::Type element = riscv_internal::logicalElement(value.getType());
      if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
          base.memory.elementType != element) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no matching memory element");
        return mlir::failure();
      }
      auto address = denseAddress(binding.slice, {});
      auto type = scalarCType(element);
      if (!address || !type) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no address or C type");
        return mlir::failure();
      }
      Binding loaded = scalar(materializeScalarRead(
          *type, "*(const " + *type + " *)(" + *address + ")"));
      bindings[value] = loaded;
      return loaded;
    }
    if (carrier != "rvv") {
      value.getDefiningOp()->emitError(
          "dense slice has no selected scalar or vector load realization");
      return mlir::failure();
    }
    mlir::FailureOr<Binding> loaded = loadDenseBlock(value, binding);
    if (mlir::failed(loaded))
      return mlir::failure();
    // A selected physical SSA load is evaluated once.  Any reload or
    // rematerialization must already be an explicit physical operation; the
    // terminal emitter must not silently duplicate the producer per consumer.
    bindings[value] = *loaded;
    return loaded;
  }
  if (binding.kind == Binding::Kind::Field) {
    auto selectedKind = selectedBindingKind(value);
    if (selectedKind && *selectedKind == Binding::Kind::ScalarTuple) {
      Binding owner = bindings.lookup(binding.field.owner);
      if (owner.kind == Binding::Kind::Slice) {
        mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
        if (mlir::failed(record))
          return mlir::failure();
        owner = std::move(*record);
      }
      auto field = fieldFor(binding);
      auto valueType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      llvm::SmallVector<int64_t, 4> registerAxes = registerAxesFor(value);
      const int64_t registerParts = registerPartCount(value);
      llvm::SmallVector<std::string, 4> registerStrides;
      llvm::SmallVector<int64_t, 4> indexAxes;
      llvm::SmallVector<int64_t, 4> directFieldAxes;
      llvm::SmallVector<std::string, 4> tupleLogicalIndices;
      for (int64_t registerAxis : registerAxes) {
        std::string stride;
        for (const auto &[axis, byteStride] : owner.recordByteStrides)
          if (axis == registerAxis) {
            stride = byteStride;
            break;
          }
        registerStrides.push_back(std::move(stride));
      }
      std::string logicalIndex = "0";
      if (binding.field.index) {
        const Binding &index = bindings.lookup(*binding.field.index);
        if (index.kind == Binding::Kind::Scalar) {
          logicalIndex = index.scalar;
        } else if (index.kind == Binding::Kind::Point &&
                   owner.recordElements > 0) {
          const int64_t elements = owner.recordElements;
          logicalIndex =
              binding.field.selector == "group_index"
                  ? "((" + index.point.base + " % " +
                        std::to_string(elements) + ") / " +
                        std::to_string(index.point.physicalExtent) + ")"
                  : "(" + index.point.base + " % " +
                        std::to_string(elements) + ")";
        } else if (index.kind == Binding::Kind::ScalarTuple) {
          auto indexType = mlir::dyn_cast<riscv::ValueType>(
              binding.field.index->getType());
          if (!indexType ||
              static_cast<int64_t>(index.parts.size()) !=
                  registerPartCount(*binding.field.index))
            return value.getDefiningOp()->emitError(
                       "selected scalar tuple field index has no closed register mapping"),
                   mlir::failure();
          indexAxes.assign(indexType.getAxisIds().asArrayRef().begin(),
                           indexType.getAxisIds().asArrayRef().end());
          tupleLogicalIndices.reserve(registerParts);
          for (int64_t part = 0; part < registerParts; ++part) {
            auto projected = projectRegisterPart(*binding.field.index, value, part);
            if (!projected || *projected >= index.parts.size())
              return value.getDefiningOp()->emitError(
                         "selected scalar tuple field index cannot project across result register axes"),
                     mlir::failure();
            tupleLogicalIndices.push_back(index.parts[*projected]);
          }
        } else {
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field has no usable logical index"),
                 mlir::failure();
        }
      }
      for (mlir::Value relativeValue : binding.field.relativeIndices) {
        const Binding &relative = bindings.lookup(relativeValue);
        if (relative.kind != Binding::Kind::Scalar)
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field has a non-scalar relative index"),
                 mlir::failure();
        if (tupleLogicalIndices.empty()) {
          logicalIndex =
              "((" + logicalIndex + ") + (" + relative.scalar + "))";
        } else {
          for (std::string &index : tupleLogicalIndices)
            index = "((" + index + ") + (" + relative.scalar + "))";
        }
      }
      unsigned logicalWidth =
          field ? riscv_internal::logicalBitWidth(field->type) : 0;
      const bool joined = field && field->access.getMapping() == "joined";
      if (!binding.field.index && valueType && binding.field.logicalRank > 0 &&
          binding.field.logicalRank <=
              static_cast<int64_t>(valueType.getAxisIds().size())) {
        const size_t first = valueType.getAxisIds().size() -
                             static_cast<size_t>(binding.field.logicalRank);
        directFieldAxes.append(valueType.getAxisIds().asArrayRef().begin() + first,
                               valueType.getAxisIds().asArrayRef().end());
      }
      if (owner.kind != Binding::Kind::Record || !field || registerAxes.empty() ||
          registerParts <= 0 || registerAxes.size() != registerStrides.size() ||
          field->bitOffset % 8)
        return value.getDefiningOp()->emitError(
                   "selected scalar tuple has no closed encoded-field mapping"),
               mlir::failure();
      for (auto [axis, stride] : llvm::zip(registerAxes, registerStrides))
        if (stride.empty() && !llvm::is_contained(indexAxes, axis) &&
            !llvm::is_contained(directFieldAxes, axis))
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple axis is represented by neither a record stride nor its typed gather index"),
                 mlir::failure();
      Binding tuple;
      tuple.kind = Binding::Kind::ScalarTuple;
      for (int64_t part = 0; part < registerParts; ++part) {
        auto coordinates = registerCoordinates(value, part);
        if (!coordinates || coordinates->size() != registerAxes.size())
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple has no register-coordinate mapping"),
                 mlir::failure();
        std::string record = "(" + owner.recordPointer;
        for (auto [coordinate, stride] :
             llvm::zip(*coordinates, registerStrides))
          if (!stride.empty())
            record += " + " + std::to_string(coordinate) + " * " + stride;
        record += ")";
        std::string partLogicalIndex =
            tupleLogicalIndices.empty()
                ? logicalIndex
                : tupleLogicalIndices[static_cast<size_t>(part)];
        // The selected scalar-replica layout already fixes the coordinates of
        // a whole encoded field value.  Project its trailing logical field
        // axes to the field's linear element index; this is a mechanical
        // spelling of the typed layout, not a terminal layout choice.
        if (tupleLogicalIndices.empty() && !directFieldAxes.empty()) {
          if (!valueType || binding.field.logicalRank < 0 ||
              binding.field.logicalRank >
                  static_cast<int64_t>(binding.field.shape.size()))
            return value.getDefiningOp()->emitError(
                       "selected scalar tuple field has no typed logical rank"),
                   mlir::failure();
          const size_t logicalRank =
              static_cast<size_t>(binding.field.logicalRank);
          const size_t valueOuterRank = valueType.getAxisIds().size() - logicalRank;
          const size_t shapeOuterRank = binding.field.shape.size() - logicalRank;
          int64_t linear = 0;
          for (size_t dimension = 0; dimension < logicalRank; ++dimension) {
            const int64_t axis = valueType.getAxisIds()[valueOuterRank + dimension];
            const int64_t extent = binding.field.shape[shapeOuterRank + dimension];
            auto registerAxis = llvm::find(registerAxes, axis);
            int64_t coordinate = 0;
            if (registerAxis != registerAxes.end())
              coordinate = (*coordinates)[static_cast<size_t>(
                  std::distance(registerAxes.begin(), registerAxis))];
            auto valueAxis = llvm::find(valueType.getAxisIds().asArrayRef(), axis);
            const size_t valuePosition = static_cast<size_t>(std::distance(
                valueType.getAxisIds().asArrayRef().begin(), valueAxis));
            if (extent <= 0 || valueAxis == valueType.getAxisIds().asArrayRef().end() ||
                coordinate < 0 || coordinate >= extent ||
                (registerAxis == registerAxes.end() &&
                 valueType.getLayout().getReplicaFactors()[valuePosition] != 1))
              return value.getDefiningOp()->emitError(
                         "selected scalar tuple field has no closed logical-axis projection"),
                     mlir::failure();
            linear = linear * extent + coordinate;
          }
          partLogicalIndex = "((" + logicalIndex + ") + " +
                             std::to_string(linear) + ")";
        }
        auto fragment =
            singleStorageFragment(*field, partLogicalIndex, logicalWidth);
        if (!fragment && !joined)
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field part has no encoded storage fragment"),
                 mlir::failure();
        const int64_t byteStride = std::max<int64_t>(1, owner.interleaveRows);
        if (joined) {
          int64_t group = field->access.getGroupSize();
          int64_t fields = field->access.getJoinFields();
          int64_t lowBits = field->access.getJoinLowBits();
          int64_t role = field->access.getJoinRole();
          llvm::StringRef order = field->access.getOrder();
          int64_t physicalRole =
              order == "lo_first" ? role : fields - 1 - role;
          std::string headByte =
              "(" + std::to_string(field->bitOffset / 8 + role * group) +
              " + (" + partLogicalIndex + "))";
          std::string tail =
              "((" + partLogicalIndex + ") - " + std::to_string(group) + ")";
          std::string lowByte =
              "(" + std::to_string(field->bitOffset / 8 + fields * group) +
              " + " + tail + ")";
          std::string highByte =
              "(" + std::to_string(field->bitOffset / 8 + role * group) +
              " + " + tail + ")";
          auto load = [&](llvm::StringRef byte) {
            return "*(const uint8_t *)(" + record + " + (" + byte.str() +
                   ") * " + std::to_string(byteStride) + ")";
          };
          std::string head = "(" + load(headByte) + " & " +
                             std::to_string((1u << logicalWidth) - 1) + ")";
          std::string low = "((" + load(lowByte) + " >> " +
                            std::to_string(physicalRole * lowBits) + ") & " +
                            std::to_string((1u << lowBits) - 1) + ")";
          std::string high = "((" + load(highByte) + " >> " +
                             std::to_string(logicalWidth) + ") & " +
                             std::to_string(
                                 (1u << (logicalWidth - lowBits)) - 1) +
                             ")";
          tuple.parts.push_back("((" + partLogicalIndex + ") < " +
                                std::to_string(group) + " ? " + head + " : (" +
                                low + " | (" + high + " << " +
                                std::to_string(lowBits) + ")))" );
          continue;
        }
        const std::string byteAddress =
            record + " + (" + fragment->byte + ") * " +
            std::to_string(byteStride);
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
            integer && integer.getWidth() < 8) {
          const unsigned width = integer.getWidth();
          const unsigned mask = (1u << width) - 1;
          std::string raw = "((*(const uint8_t *)(" + byteAddress + ") >> (" +
                            fragment->shift + ")) & " +
                            std::to_string(mask) + ")";
          if (integer.isUnsigned()) {
            tuple.parts.push_back(std::move(raw));
          } else {
            const unsigned sign = 1u << (width - 1);
            tuple.parts.push_back("((int32_t)((" + raw + " ^ " +
                                  std::to_string(sign) + ") - " +
                                  std::to_string(sign) + "))");
          }
        } else if (field->type.isF32())
          tuple.parts.push_back("weft_load_f32_le(" + byteAddress + ")");
        else if (field->type.isF16())
          tuple.parts.push_back("weft_load_f16_le(" + byteAddress + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 16)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "weft_load_u16_le(" :
                                                 "weft_load_i16_le(") +
              byteAddress + ")");
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 32)
          tuple.parts.push_back(
              std::string(integer.isSigned() ? "((int32_t)" : "") +
              "weft_load_u32_le(" + byteAddress + ")" +
              (integer.isSigned() ? ")" : ""));
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
                 integer && integer.getWidth() == 8)
          tuple.parts.push_back(
              std::string(integer.isUnsigned() ? "*(const uint8_t *)(" :
                                                 "*(const int8_t *)(") +
              byteAddress + ")");
        else
          return value.getDefiningOp()->emitError(
                     "selected scalar tuple field type has no intrinsic-C load"),
                 mlir::failure();
      }
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return value.getDefiningOp()->emitError(
                   "selected scalar tuple field has no intrinsic-C value type"),
               mlir::failure();
      for (std::string &part : tuple.parts)
        part = materializeScalarRead(*type, part);
      bindings[value] = tuple;
      return tuple;
    }
    auto expected = selectedBindingKind(value);
    const int64_t expectedParts =
        expected && *expected == Binding::Kind::Vector
            ? vectorPartCount(value)
            : expected && *expected == Binding::Kind::ScalarTuple
                  ? registerPartCount(value)
                  : 1;
    Binding owner = bindings.lookup(binding.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
      if (mlir::failed(record))
        return mlir::failure();
      owner = std::move(*record);
    }
    const bool materializeStreams =
        expected && *expected == Binding::Kind::Vector &&
        owner.kind == Binding::Kind::Record && owner.interleaveRows > 0 &&
        expectedParts > 1 && registerPartCount(value) == 1 &&
        expectedParts == streamPartCount(value);
    if (materializeStreams) {
      Binding streamed;
      streamed.kind = Binding::Kind::Vector;
      for (int64_t stream = 0; stream < expectedParts; ++stream) {
        Binding part =
            emitInterleavedField(value, binding, std::to_string(stream), stream);
        if (part.kind != Binding::Kind::Vector || part.parts.size() != 1) {
          value.getDefiningOp()->emitError(
              "encoded field stream has no single selected RVV value");
          return mlir::failure();
        }
        streamed.parts.push_back(std::move(part.parts.front()));
      }
      bindings[value] = streamed;
      return streamed;
    }
    Binding loaded = emitInterleavedField(value, binding, "0", 0);
    if (loaded.kind == Binding::Kind::None) {
      auto diagnostic = value.getDefiningOp()->emitError(
          "encoded field has no selected numeric load realization");
      diagnostic << " (field=" << binding.field.name;
      if (binding.field.storageAccess)
        diagnostic << ", mapping="
                   << binding.field.storageAccess.getMapping();
      if (binding.field.useAccess)
        diagnostic << ", form=" << binding.field.useAccess.getForm();
      diagnostic << ", value_type=" << value.getType()
                 << ", owner_type=" << binding.field.owner.getType();
      diagnostic << ", selected-carrier="
                 << static_cast<int>(expected.value_or(Binding::Kind::None))
                 << ")";
      return mlir::failure();
    }
    if (expected && loaded.kind != *expected) {
      value.getDefiningOp()->emitError(
          "encoded field load does not implement its selected carrier");
      return mlir::failure();
    }
    if ((loaded.kind == Binding::Kind::Vector ||
         loaded.kind == Binding::Kind::ScalarTuple) &&
        static_cast<int64_t>(loaded.parts.size()) != expectedParts) {
      value.getDefiningOp()->emitError(
          "encoded field load does not materialize every selected physical part");
      return mlir::failure();
    }
    if (loaded.kind == Binding::Kind::Scalar ||
        loaded.kind == Binding::Kind::ScalarTuple) {
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return value.getDefiningOp()->emitError(
                   "selected scalar field has no intrinsic-C value type"),
               mlir::failure();
      if (loaded.kind == Binding::Kind::Scalar)
        loaded.scalar = materializeScalarRead(*type, loaded.scalar);
      else
        for (std::string &part : loaded.parts)
          part = materializeScalarRead(*type, part);
    }
    bindings[value] = loaded;
    return loaded;
  }
  return binding;
}

mlir::LogicalResult Emitter::compileMemoryView(riscv::MemoryViewOp operation) {
  Binding base = bindings.lookup(operation.getBase());
  if (base.kind != Binding::Kind::Memory)
    return fail(operation, "memory view base has no physical pointer binding");
  const size_t rank = operation.getResult().getType().getShape().size();
  if (operation.getExtents().size() != rank ||
      operation.getStrides().size() != rank ||
      operation.getOrigins().size() != rank)
    return fail(operation, "memory view runtime metadata rank is incomplete");
  base.memory.extents.clear();
  base.memory.strides.clear();
  base.memory.origins.clear();
  auto appendScalars = [&](mlir::ValueRange values,
                           llvm::SmallVectorImpl<std::string> &target) {
    for (mlir::Value value : values) {
      Binding scalarValue = bindings.lookup(value);
      if (scalarValue.kind != Binding::Kind::Scalar)
        return false;
      target.push_back(scalarValue.scalar);
    }
    return true;
  };
  if (!appendScalars(operation.getExtents(), base.memory.extents) ||
      !appendScalars(operation.getStrides(), base.memory.strides) ||
      !appendScalars(operation.getOrigins(), base.memory.origins))
    return fail(operation,
                "memory view extent, stride, and origin must be explicit index SSA values");
  auto descriptor = operation.getResult().getType();
  base.memory.recordBytes = descriptor.getStorageBits() / 8;
  base.memory.logicalElements = descriptor.getElements();
  base.memory.interleaveRows = descriptor.getInterleaveRows();
  bindings[operation.getResult()] = std::move(base);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageLoad(
    riscv::StorageLoadOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.load.u8")
    return fail(operation, "artifact storage load has no closed byte-load leaf");
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (source.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage load requires explicit memory and byte-index bindings");
  bindings[operation.getResult()] =
      scalar(materializeScalarRead(
          "uint8_t", source.memory.name + "[(size_t)(" + index.scalar + ")]"));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageStore(
    riscv::StorageStoreOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.store.u8")
    return fail(operation, "artifact storage store has no closed byte-store leaf");
  Binding value = bindings.lookup(operation.getValue());
  Binding target = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (value.kind != Binding::Kind::Scalar ||
      target.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage store requires explicit value, memory, and byte index");
  line(target.memory.name + "[(size_t)(" + index.scalar + ")] = " +
       value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSlice(riscv::SliceOp slice) {
  Binding base = bindings.lookup(slice.getBase());
  if (base.kind == Binding::Kind::Field) {
    auto inputType = mlir::dyn_cast<riscv::ValueType>(slice.getBase().getType());
    const int64_t fieldStart =
        inputType ? static_cast<int64_t>(inputType.getShape().size()) -
                        base.field.logicalRank
                  : 0;
    size_t cursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size())
        return fail(slice, "field slice selector has no corresponding index");
      if (base.field.index)
        return fail(slice,
                    "intrinsic-C field projection supports one logical field axis");
      base.field.index = slice.getIndices()[cursor++];
      base.field.selector = selector.str();
      // A domain projection narrows an axis to the current Level partition but
      // keeps that logical axis in the result.  Only selectors that eliminate
      // the source axis reduce the encoded field rank.
      if (inputType && static_cast<int64_t>(position) >= fieldStart &&
          selector != "domain")
        --base.field.logicalRank;
    }
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(slice.getResult().getType()))
      base.field.shape.assign(resultType.getShape().asArrayRef().begin(),
                              resultType.getShape().asArrayRef().end());
    bindings[slice.getResult()] = std::move(base);
    return mlir::success();
  }
  Binding binding;
  binding.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Slice) {
    binding.slice = base.slice;
    auto baseType = mlir::cast<riscv::MemDescType>(slice.getBase().getType());
    const Binding &root = bindings.lookup(binding.slice.base);
    if (root.kind != Binding::Kind::Memory)
      return fail(slice, "nested slice has no root memory descriptor");
    if (binding.slice.selectors.size() > root.memory.axes.size())
      return fail(slice, "nested slice root selector rank exceeds its memory rank");
    binding.slice.selectors.resize(root.memory.axes.size(), "all");
    size_t cursor = 0;
    for (auto [position, selectorAttribute] : llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size() ||
          position >= baseType.getAxisIds().size())
        return fail(slice, "nested slice selector has no logical axis or index");
      int64_t axis = baseType.getAxisIds()[position];
      mlir::Value index = slice.getIndices()[cursor++];
      if (selector == "index") {
        // Plain scalar indices are offsets relative to the selected block.
        binding.slice.localOffsets.emplace_back(axis, index);
        continue;
      }
      if (selector != "domain" && selector != "group_index")
        return fail(slice, "nested slice has an unknown selector");
      auto found = llvm::find(root.memory.axes, axis);
      if (found == root.memory.axes.end())
        return fail(slice, "nested slice logical axis is absent from root memory");
      size_t rootDimension = static_cast<size_t>(
          std::distance(root.memory.axes.begin(), found));
      size_t rootCursor = 0;
      for (size_t dimension = 0; dimension < rootDimension; ++dimension)
        rootCursor += binding.slice.selectors[dimension] != "all";
      if (binding.slice.selectors[rootDimension] == "all")
        binding.slice.indices.insert(binding.slice.indices.begin() + rootCursor,
                                     index);
      else if (rootCursor < binding.slice.indices.size())
        binding.slice.indices[rootCursor] = index;
      else
        return fail(slice, "nested slice root selector has no matching index");
      // Physical points carry absolute coordinates in their canonical domain.
      // Domain and group-index projections therefore replace the root selector
      // for that axis; adding the child point to its parent would count the
      // parent base twice.
      binding.slice.selectors[rootDimension] = selector.str();
    }
  } else {
    binding.slice.base = slice.getBase();
    binding.slice.indices.assign(slice.getIndices().begin(), slice.getIndices().end());
    for (mlir::Attribute selector : slice.getSelectors())
      binding.slice.selectors.push_back(
          mlir::cast<mlir::StringAttr>(selector).getValue().str());
  }
  bindings[slice.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSubview(riscv::SubviewOp subview) {
  Binding base = bindings.lookup(subview.getBase());
  Binding result;
  result.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Memory) {
    result.slice.base = subview.getBase();
  } else if (base.kind == Binding::Kind::Slice) {
    result.slice = base.slice;
  } else {
    return fail(subview, "subview base has no memory-backed slice binding");
  }
  auto baseType = subview.getBase().getType();
  if (subview.getOffsets().size() != baseType.getAxisIds().size())
    return fail(subview, "subview offset rank disagrees with its descriptor");
  for (auto [axis, offset] :
       llvm::zip(baseType.getAxisIds().asArrayRef(),
                 subview.getOffsets()))
    if (offset != 0)
      result.slice.staticOffsets.emplace_back(axis, offset);
  bindings[subview.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReshape(riscv::ReshapeOp reshape) {
  Binding input = bindings.lookup(reshape.getInput());
  const int64_t inputParts =
      input.kind == Binding::Kind::Vector
          ? vectorPartCount(reshape.getInput())
          : input.kind == Binding::Kind::ScalarTuple
                ? scalarPartCount(reshape.getInput())
                : 1;
  const int64_t resultParts =
      input.kind == Binding::Kind::Vector
          ? vectorPartCount(reshape.getResult())
          : input.kind == Binding::Kind::ScalarTuple
                ? scalarPartCount(reshape.getResult())
                : 1;
  if (input.kind == Binding::Kind::None || inputParts <= 0 ||
      inputParts != resultParts ||
      ((input.kind == Binding::Kind::Vector ||
        input.kind == Binding::Kind::ScalarTuple) &&
       static_cast<int64_t>(input.parts.size()) != inputParts))
    return fail(reshape,
                "reshape input does not match its selected linear carrier partition");
  bindings[reshape.getResult()] = std::move(input);
  return mlir::success();
}

mlir::FailureOr<Binding> Emitter::recordForSlice(mlir::Value value) {
  Binding region = bindings.lookup(value);
  if (region.kind != Binding::Kind::Slice) {
    value.getDefiningOp()->emitError("encoded record requires a canonical slice region");
    return mlir::failure();
  }
  const Binding &base = bindings.lookup(region.slice.base);
  auto sliceType = mlir::dyn_cast<riscv::MemDescType>(value.getType());
  auto encoding = sliceType
                      ? mlir::dyn_cast<kernel::EncodingType>(sliceType.getEncoding())
                      : kernel::EncodingType();
  if (!encoding || encoding.getKind() == "dense") {
    value.getDefiningOp()->emitError("record slice must carry a concrete encoding");
    return mlir::failure();
  }
  if (base.kind != Binding::Kind::Memory) {
    value.getDefiningOp()->emitError("encoded record requires a memory-backed view");
    return mlir::failure();
  }
  std::string family = encoding.getFamily().str();
  if (base.memory.recordBytes <= 0 || base.memory.logicalElements <= 0 ||
      base.memory.strides.size() != base.memory.axes.size() ||
      base.memory.origins.size() != base.memory.axes.size())
    return value.getDefiningOp()->emitError(
               "encoded record descriptor has no explicit storage geometry"),
           mlir::failure();

  llvm::DenseMap<int64_t, PointInfo> points;
  llvm::SmallVector<std::string> coordinates(base.memory.origins.begin(),
                                             base.memory.origins.end());
  size_t cursor = 0;
  for (auto [dimension, selector] : llvm::enumerate(region.slice.selectors)) {
    if (selector == "all")
      continue;
    if (cursor >= region.slice.indices.size())
      return value.getDefiningOp()->emitError(
                 "encoded record selector has no index operand"),
             mlir::failure();
    const Binding &index = bindings.lookup(region.slice.indices[cursor++]);
    if (selector == "domain") {
      coordinates[dimension] = index.point.base;
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "group_index") {
      coordinates[dimension] = "(" + index.point.base + " / " +
                               std::to_string(index.point.physicalExtent) + ")";
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "index") {
      coordinates[dimension] = index.scalar;
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record has an unknown slice selector"),
             mlir::failure();
    }
  }
  for (auto [axis, offsetValue] : region.slice.localOffsets) {
    auto dimension = llvm::find(base.memory.axes, axis);
    if (dimension == base.memory.axes.end())
      return value.getDefiningOp()->emitError(
                 "encoded record local offset has no matching memory axis"),
             mlir::failure();
    const size_t position =
        static_cast<size_t>(std::distance(base.memory.axes.begin(), dimension));
    const Binding &offset = bindings.lookup(offsetValue);
    std::string expression;
    if (offset.kind == Binding::Kind::Scalar) {
      expression = offset.scalar;
    } else if (offset.kind == Binding::Kind::Point) {
      expression = offset.point.base;
      PointInfo point = offset.point;
      point.base = "(" + coordinates[position] + " + " + expression + ")";
      points[axis] = std::move(point);
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record local offset is neither scalar nor domain point"),
             mlir::failure();
    }
    coordinates[position] =
        "(" + coordinates[position] + " + " + expression + ")";
    if (auto point = points.find(axis); point != points.end())
      point->second.base = coordinates[position];
  }
  Binding result;
  result.kind = Binding::Kind::Record;
  const int64_t recordBytes = base.memory.recordBytes;
  const int64_t elements = base.memory.logicalElements;
  result.recordAxis = base.memory.axes.empty() ? 0 : base.memory.axes.back();
  result.recordElements = elements;
  result.recordStrideBytes = recordBytes;
  for (auto [dimension, axis] : llvm::enumerate(base.memory.axes))
    if (llvm::is_contained(sliceType.getAxisIds().asArrayRef(), axis))
      result.recordOrigins.emplace_back(axis, coordinates[dimension]);
  if (encoding.getKind() == "derived_instance" ||
      base.memory.interleaveRows > 0) {
    int64_t rows = base.memory.interleaveRows > 0
                       ? base.memory.interleaveRows
                       : sliceType.getInterleaveRows();
    if (rows <= 0 || base.memory.axes.size() != 2)
      return value.getDefiningOp()->emitError(
                 "derived interleave requires a two-axis view and selected rows"),
             mlir::failure();
    const PointInfo row = points.lookup(base.memory.axes[0]);
    const PointInfo block = points.lookup(base.memory.axes[1]);
    const std::string rowBase =
        base.memory.interleaveRows > 0
            ? "(" + row.base + " - " + base.memory.origins[0] + ")"
            : row.base;
    const std::string blockBase =
        base.memory.interleaveRows > 0
            ? "(" + block.base + " - " + base.memory.origins[1] + ")"
            : block.base;
    const std::string blocks = "(" + base.memory.extents[1] + " / " +
                               std::to_string(elements) + ")";
    result.interleaveRows = rows;
    result.recordStrideBytes = recordBytes * rows;
    result.recordByteStrides.emplace_back(base.memory.axes[0], "1");
    result.recordPointer = base.memory.name + " + (((" + rowBase + " / " +
                           std::to_string(rows) + ") * " + blocks +
                           " + (" + blockBase + " / " +
                           std::to_string(elements) + ")) * " +
                           std::to_string(recordBytes * rows) + ")";
  } else {
    if (base.memory.axes.empty())
      return value.getDefiningOp()->emitError(
                 "base encoded view has no logical axis"),
             mlir::failure();
    const size_t recordDimension = base.memory.axes.size() - 1;
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      result.recordByteStrides.emplace_back(
          base.memory.axes[dimension],
          "((" + base.memory.strides[dimension] + " / " +
              std::to_string(elements) + ") * " +
              std::to_string(recordBytes) + ")");
    }
    // Every dimension before the encoded element axis advances by whole
    // records. Keep that divisibility visible in C instead of flattening all
    // logical coordinates and dividing the sum: the latter hides the affine
    // record induction from the system compiler when extents are dynamic.
    std::string record = "0";
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      record = "(" + record + " + ((" + coordinates[dimension] + ") - (" +
               base.memory.origins[dimension] + ")) * ((" +
               base.memory.strides[dimension] + ") / " +
               std::to_string(elements) + "))";
    }
    const size_t elementDimension = recordDimension;
    record = "(" + record + " + (((" + coordinates[elementDimension] +
             ") - (" + base.memory.origins[elementDimension] + ")) * (" +
             base.memory.strides[elementDimension] + ")) / " +
             std::to_string(elements) + ")";
    result.recordPointer = base.memory.name + " + (" + record + ") * " +
                           std::to_string(recordBytes);
  }
  return result;
}

mlir::LogicalResult Emitter::compileAdmit(riscv::LoadOp admit) {
  Binding region = bindings.lookup(admit.getRegion());
  if (region.kind == Binding::Kind::Memory) {
    auto view = mlir::dyn_cast<riscv::MemDescType>(admit.getRegion().getType());
    auto encoding = view ? mlir::dyn_cast<kernel::EncodingType>(view.getEncoding())
                         : kernel::EncodingType();
    if (!encoding || encoding.getKind() != "dense")
      return fail(admit,
                  "whole-view admit is only defined for a dense read-only value");
    Binding slice;
    slice.kind = Binding::Kind::Slice;
    slice.slice.base = admit.getRegion();
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(slice));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(admit, "admit currently requires a canonical slice region");
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(
      mlir::cast<riscv::MemDescType>(admit.getRegion().getType()).getEncoding());
  if (!encoding)
    return fail(admit, "admit slice has no encoding");
  if (encoding.getKind() == "dense") {
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(region));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = recordForSlice(admit.getRegion());
  if (mlir::failed(result))
    return mlir::failure();
  if (mlir::Value allocation = admit.getSnapshotStorage()) {
    Binding storage = bindings.lookup(allocation);
    auto storageType = mlir::cast<riscv::LocalType>(allocation.getType());
    if (storage.kind != Binding::Kind::LocalArray)
      return fail(admit, "record snapshot has no selected local storage binding");
    if (mlir::failed(emitReadSnapshot(admit, admit.getLeaf(), storage.scalar,
                                     result->recordPointer,
                                     storageType.getSizeBytes())))
      return mlir::failure();
    result->recordPointer = storage.scalar;
  }
  bindings[admit.getResult()] = std::move(*result);
  return mlir::success();
}

std::optional<std::string> Emitter::byteReadAddress(mlir::Value value) {
  Binding source = bindings.lookup(value);
  std::optional<std::string> address;
  if (source.kind == Binding::Kind::Slice) {
    address = denseAddress(source.slice, {});
  } else if (source.kind == Binding::Kind::Memory) {
    address = source.memory.name;
  } else if (source.kind == Binding::Kind::Field) {
    Binding owner = bindings.lookup(source.field.owner);
    auto field = fieldFor(source);
    if (field && field->bitOffset % 8 == 0) {
      if (owner.kind == Binding::Kind::Record)
        address = "(" + owner.recordPointer + " + " +
                  std::to_string(field->bitOffset / 8) + ")";
      else if (owner.kind == Binding::Kind::Memory)
        address = "(" + owner.memory.name + " + " +
                  std::to_string(field->bitOffset / 8) + ")";
    }
  }
  return address;
}

mlir::LogicalResult Emitter::compileRegisterMaterialize(
    riscv::RegisterMaterializeOp materialize) {
  Binding input = bindings.lookup(materialize.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(materialize, "register materialize input has no emitted value");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> value =
        materializeNumeric(materialize.getResult(), std::move(input));
    if (mlir::failed(value))
      return mlir::failure();
    input = std::move(*value);
  }
  const bool scalarShare =
      materialize.getRealization() == "physical-share" ||
      (materialize.getRealization() == "share" &&
       (input.kind == Binding::Kind::Scalar ||
        input.kind == Binding::Kind::ScalarTuple));
  if (scalarShare) {
    auto type = scalarCType(
        riscv_internal::logicalElement(materialize.getResult().getType()));
    if (!type || (input.kind != Binding::Kind::Scalar &&
                  input.kind != Binding::Kind::ScalarTuple))
      return fail(materialize,
                  "physical scalar materialization has no closed intrinsic-C value");
    if (input.kind == Binding::Kind::Scalar) {
      std::string name = fresh("physical_share");
      line(*type + " " + name + " = " + input.scalar + ";");
      input.scalar = std::move(name);
    } else {
      for (std::string &part : input.parts) {
        std::string name = fresh("physical_share");
        line(*type + " " + name + " = " + part + ";");
        part = std::move(name);
      }
    }
  }
  bindings[materialize.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStagedView(riscv::StagedViewOp staged) {
  Binding source = bindings.lookup(staged.getSource());
  if (source.kind != Binding::Kind::Memory &&
      source.kind != Binding::Kind::Slice)
    return fail(staged, "reload staged view requires one emitted memory descriptor");
  bindings[staged.getResult()] = std::move(source);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalCapacityGuard(
    riscv::LocalCapacityGuardOp operation) {
  Binding bytes = bindings.lookup(operation.getSizeBytes());
  if (bytes.kind != Binding::Kind::Scalar ||
      operation.getLeaf().getInstruction() != "scalar.local-capacity-guard")
    return fail(operation, "dynamic local allocation has no closed capacity guard");
  line("if ((uint64_t)(" + bytes.scalar + ") > " +
       std::to_string(operation.getLimitBytes()) +
       "ULL) __builtin_trap();");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalAlloc(riscv::LocalAllocOp operation) {
  riscv::LocalType type = operation.getResult().getType();
  Binding size = bindings.lookup(operation.getSizeBytes());
  if (size.kind != Binding::Kind::Scalar || type.getAlignment() <= 0)
    return fail(operation, "local allocation has no closed byte size/alignment");
  std::string name = fresh(type.getPurpose() == "spill" ? "spill_slot"
                                                         : "local_object");
  line("_Alignas(" + std::to_string(type.getAlignment()) + ") uint8_t " + name +
       "[" + size.scalar + "];");
  Binding binding;
  binding.kind = Binding::Kind::LocalArray;
  binding.scalar = name;
  binding.localElements = size.scalar;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::emitReadSnapshot(
    mlir::Operation *operation, riscv::LeafAttr leaf,
    llvm::StringRef destination, llvm::StringRef source, int64_t bytes) {
  if (leaf.getInstruction() == "scalar.read-snapshot") {
    line("memcpy(" + destination.str() + ", " + source.str() + ", " +
         std::to_string(bytes) + ");");
    return mlir::success();
  }
  auto parameters = leaf.getParameters().asArrayRef();
  if (leaf.getInstruction() != "rvv.read-snapshot" ||
      parameters.size() != 2 || parameters[1] <= 0 || bytes <= 0)
    return fail(operation, "read snapshot has no exact byte-transfer leaf");
  const std::string group = "m" + std::to_string(parameters[0] / 8);
  for (int64_t offset = 0; offset < bytes;) {
    const int64_t width = std::min(parameters[1], bytes - offset);
    const std::string count = std::to_string(width);
    const std::string at = std::to_string(offset);
    const std::string value = fresh("snapshot_bytes");
    line("vuint8" + group + "_t " + value + " = __riscv_vle8_v_u8" + group + "((const uint8_t *)(" +
         source.str() + ") + " + at + ", " + count + ");");
    line("__riscv_vse8_v_u8" + group + "((uint8_t *)(" + destination.str() + ") + " + at +
         ", " + value + ", " + count + ");");
    offset += width;
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileDenseSnapshot(riscv::DenseSnapshotOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding storage = bindings.lookup(operation.getStorage());
  std::optional<std::string> address;
  if (source.kind == Binding::Kind::Slice)
    address = denseAddress(source.slice, {});
  else if (source.kind == Binding::Kind::Memory)
    address = source.memory.name;
  auto descriptor = operation.getResult().getType();
  auto element = denseElementType(descriptor.getEncoding());
  if (!address || !element || storage.kind != Binding::Kind::LocalArray ||
      storage.scalar.empty())
    return fail(operation, "dense snapshot has no complete memory/storage binding");
  if (mlir::failed(emitReadSnapshot(
          operation, operation.getLeaf(), storage.scalar, *address,
          operation.getStorage().getType().getSizeBytes())))
    return mlir::failure();
  Binding result;
  result.kind = Binding::Kind::Memory;
  result.memory.name = storage.scalar;
  result.memory.encoding = descriptor.getEncoding();
  result.memory.elementType = *element;
  result.memory.axes.assign(descriptor.getAxisIds().asArrayRef().begin(),
                             descriptor.getAxisIds().asArrayRef().end());
  result.memory.extents.push_back(std::to_string(descriptor.getShape()[0]));
  result.memory.strides.push_back("1");
  result.memory.origins.push_back("0");
  result.memory.recordBytes = descriptor.getStorageBits() / 8;
  result.memory.logicalElements = 1;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalBind(riscv::LocalBindOp operation) {
  Binding storage = bindings.lookup(operation.getStorage());
  Binding elements = bindings.lookup(operation.getElements());
  if (storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      elements.kind != Binding::Kind::Scalar)
    return fail(operation,
                "local bind requires one allocation and one explicit element extent");
  auto element = scalarCType(operation.getResult().getType().getElementType());
  if (!element)
    return fail(operation, "local bind element has no intrinsic-C type");
  storage.scalar = "((" + *element + " *)" + storage.scalar + ")";
  storage.localElements = elements.scalar;
  bindings[operation.getResult()] = std::move(storage);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalLoad(riscv::LocalLoadOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getResult().getType());
  if (source.kind != Binding::Kind::LocalArray || source.scalar.empty() ||
      index.kind != Binding::Kind::Scalar || !element ||
      operation.getLeaf().getInstruction() != "local.load.element")
    return fail(operation, "local element load has no closed physical binding");
  bindings[operation.getResult()] =
      scalar(materializeScalarRead(
          *element, "((const " + *element + " *)" + source.scalar + ")[" +
                        index.scalar + "]",
          "local_load"));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalStore(riscv::LocalStoreOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding destination = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getValue().getType());
  if (value.kind != Binding::Kind::Scalar ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || index.kind != Binding::Kind::Scalar ||
      !element || operation.getLeaf().getInstruction() != "local.store.element")
    return fail(operation, "local element store has no closed physical binding");
  line("((" + *element + " *)" + destination.scalar + ")[" + index.scalar +
       "] = " + value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLocalMaterialize(
    riscv::RVVLocalMaterializeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.local-materialize")
    return fail(operation,
                "RVV local materialize has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  Binding destination = bindings.lookup(operation.getDestination());
  auto element = scalarCType(operation.getInput().getType().getElementType());
  const int64_t parts = vectorPartCount(operation.getInput());
  const int64_t lanes = physicalLanes(operation.getInput());
  const unsigned bits = operation.getInput().getType().getLayout().getSew();
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || !element || parts <= 0 || lanes <= 0 ||
      input->parts.size() != static_cast<size_t>(parts))
    return fail(operation,
                "RVV local materialize has no complete vector/local binding");
  const std::string suffix = vectorSuffix(operation.getInput());
  for (int64_t part = 0; part < parts; ++part)
    line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix + "((" +
         *element + " *)" + destination.scalar + " + " +
         std::to_string(part * lanes) + ", " + input->parts[part] + ", " +
         partVL(operation.getInput(), part) + ");");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIndexMultipleGuard(
    riscv::IndexMultipleGuardOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "scalar.index-multiple-guard")
    return fail(operation, "index multiple guard has no exact selected leaf");
  Binding value = bindings.lookup(operation.getValue());
  if (value.kind != Binding::Kind::Scalar || operation.getMultiple() <= 0)
    return fail(operation,
                "index multiple guard requires one scalar value and positive divisor");
  line("if (((size_t)(" + value.scalar + ") % " +
       std::to_string(operation.getMultiple()) + ") != 0) __builtin_trap();");
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileEncodedLocalBind(riscv::EncodedLocalBindOp operation) {
  Binding source = bindings.lookup(operation.getInput());
  Binding storage = bindings.lookup(operation.getStorage());
  Binding blocks = bindings.lookup(operation.getBlockCount());
  riscv::LocalPackPlanAttr plan = operation.getPlan();
  if (source.kind != Binding::Kind::Record || source.interleaveRows != 0 ||
      source.recordElements != plan.getRecordElements() ||
      source.recordStrideBytes != plan.getRecordBytes() ||
      storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      blocks.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded local bind has no complete source, storage, and block-count bindings");

  Binding result;
  result.kind = Binding::Kind::Record;
  result.recordPointer = storage.scalar;
  result.interleaveRows = plan.getInterleaveRows();
  result.recordAxis = plan.getRecordAxis();
  result.recordElements = plan.getRecordElements();
  result.recordStrideBytes =
      plan.getRecordBytes() * plan.getInterleaveRows();
  result.recordGroupAxis = plan.getRowAxis();
  result.recordByteStrides.emplace_back(
      plan.getRowAxis(), "((" + blocks.scalar + ") * " +
                             std::to_string(plan.getRecordBytes()) + ")");
  result.recordOrigins = source.recordOrigins;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVEncodedLocalPackTransfer(
    riscv::RVVEncodedLocalPackTransferOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.local-pack.transfer.interleave")
    return fail(operation,
                "encoded local pack transfer has no exact selected leaf");
  Binding source = bindings.lookup(operation.getInput());
  Binding storage = bindings.lookup(operation.getStorage());
  Binding rowGroup = bindings.lookup(operation.getRowGroup());
  Binding rowBase = bindings.lookup(operation.getRowBase());
  Binding block = bindings.lookup(operation.getBlock());
  Binding blocks = bindings.lookup(operation.getBlockCount());
  Binding byte = bindings.lookup(operation.getByte());
  Binding transferVL = bindings.lookup(operation.getTransferVl());
  riscv::LocalPackPlanAttr plan = operation.getPlan();
  auto rowStride = llvm::find_if(
      source.recordByteStrides,
      [&](const auto &entry) { return entry.first == plan.getRowAxis(); });
  if (source.kind != Binding::Kind::Record || source.interleaveRows != 0 ||
      source.recordElements != plan.getRecordElements() ||
      source.recordStrideBytes != plan.getRecordBytes() ||
      rowStride == source.recordByteStrides.end() ||
      storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      rowGroup.kind != Binding::Kind::Scalar ||
      rowBase.kind != Binding::Kind::Scalar ||
      block.kind != Binding::Kind::Scalar ||
      blocks.kind != Binding::Kind::Scalar ||
      byte.kind != Binding::Kind::Scalar ||
      transferVL.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded local pack transfer has incomplete typed coordinates");

  const std::string transferSuffix =
      "u8" + lmulSpelling(operation.getTransferLayout().getLmulEighths());
  const std::string transferType =
      "vuint8" + lmulSpelling(operation.getTransferLayout().getLmulEighths()) +
      "_t";
  const std::string transferVLName = fresh("pack_vl");
  line("const size_t " + transferVLName + " = (size_t)(" +
       transferVL.scalar + ");");
  const std::string sourceAddress =
      source.recordPointer + " + (" + rowBase.scalar + ") * (" +
      rowStride->second + ") + (" + block.scalar + ") * " +
      std::to_string(plan.getRecordBytes()) + " + (" + byte.scalar + ")";
  const std::string targetAddress =
      "((((" + rowGroup.scalar + ") * (" + blocks.scalar + ") + (" +
      block.scalar + ")) * " + std::to_string(plan.getRecordBytes()) +
      " + (" + byte.scalar + ")) * " +
      std::to_string(plan.getInterleaveRows()) + ")";
  const std::string packed = fresh("pack_bytes");
  line(transferType + " " + packed + " = __riscv_vlse8_v_" +
       transferSuffix + "((const uint8_t *)(" + sourceAddress +
       "), (ptrdiff_t)(" + rowStride->second + "), " + transferVLName +
       ");");
  line("__riscv_vse8_v_" + transferSuffix + "((uint8_t *)(" + storage.scalar +
       " + " + targetAddress + "), " + packed + ", " + transferVLName +
       ");");
  line("if (" + transferVLName + " < " +
       std::to_string(plan.getInterleaveRows()) + ") {");
  ++indent;
  const std::string tail = fresh("pack_tail");
  const std::string zeros = fresh("pack_zeros");
  line("const size_t " + tail + " = " +
       std::to_string(plan.getInterleaveRows()) + " - " + transferVLName +
       ";");
  line(transferType + " " + zeros + " = __riscv_vmv_v_x_" + transferSuffix +
       "(0, " + tail + ");");
  line("__riscv_vse8_v_" + transferSuffix + "((uint8_t *)(" + storage.scalar +
       " + " + targetAddress + " + " + transferVLName + "), " + zeros +
       ", " + tail + ");");
  --indent;
  line("}");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSpill(riscv::SpillOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  Binding slot = bindings.lookup(operation.getSlot());
  const bool rvvSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "rvv.spill";
  const bool scalarTupleSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "scalar.tuple-spill";
  if ((!rvvSpill && !scalarTupleSpill) ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "spill destination has no typed local allocation");
  riscv::LocalType slotType = operation.getSlot().getType();
  mlir::Type element = riscv_internal::logicalElement(operation.getInput().getType());
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "spill value has no C element type");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (input.kind == Binding::Kind::Vector) {
    if (!rvvSpill)
      return fail(operation, "vector spill requires the selected RVV spill leaf");
    auto inputType =
        mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    const int64_t parts = vectorPartCount(operation.getInput());
    if (!inputType || parts <= 0 ||
        input.parts.size() != static_cast<size_t>(parts) ||
        slotType.getSizeBytes() % parts)
      return fail(operation, "vector spill byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = inputType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getInput());
    for (int64_t part = 0; part < parts; ++part)
      line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix +
           "((" + *scalarType + " *)(" + slot.scalar + " + " +
           std::to_string(part * bytes) + "), " + input.parts[part] + ", " +
           partVL(operation.getInput(), part) + ");");
    return mlir::success();
  }
  if (input.kind == Binding::Kind::Scalar ||
      input.kind == Binding::Kind::ScalarTuple) {
    if (!scalarTupleSpill)
      return fail(operation,
                  "scalar register spill requires the selected tuple-spill leaf");
    llvm::SmallVector<std::string> values;
    if (input.kind == Binding::Kind::Scalar)
      values.push_back(input.scalar);
    else
      values.assign(input.parts.begin(), input.parts.end());
    for (auto [index, value] : llvm::enumerate(values))
      line("((" + *scalarType + " *)" + slot.scalar + ")[" +
           std::to_string(index) + "] = " + value + ";");
    return mlir::success();
  }
  return fail(operation, "spill supports only typed RVV/scalar register values");
}

mlir::LogicalResult Emitter::compileReload(riscv::ReloadOp operation) {
  Binding slot = bindings.lookup(operation.getSlot());
  if (operation.getLeaf().getEngine() != "transfer" ||
      operation.getLeaf().getFamily() != "reload" ||
      operation.getLeaf().getInstruction() != "rvv.reload" ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "reload source has no typed local allocation");
  auto resultType = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!resultType)
    return fail(operation, "reload result has no typed physical value");
  auto kind = selectedBindingKind(operation.getResult());
  auto scalarType = scalarCType(resultType.getElementType());
  if (!kind || !scalarType)
    return fail(operation, "reload result has no terminal representation");
  Binding result;
  result.kind = *kind;
  if (*kind == Binding::Kind::Vector) {
    riscv::LocalType slotType = operation.getSlot().getType();
    const int64_t parts = vectorPartCount(operation.getResult());
    if (parts <= 0 || slotType.getSizeBytes() % parts)
      return fail(operation, "vector reload byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = resultType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    auto partIsUsed = [&](int64_t part) {
      for (mlir::OpOperand &use : operation.getResult().getUses()) {
        auto set = mlir::dyn_cast<riscv::RVVPartialSetOp>(use.getOwner());
        if (!set)
          return true;
        const size_t operand = use.getOperandNumber();
        const size_t lhsCount = set.getLhs().size();
        if (operand >= lhsCount + set.getRhs().size())
          return true;
        for (int64_t slot = 0; slot < set.getResult().getType().getSlots(); ++slot) {
          if (operand < lhsCount) {
            if (set.getLhsOperands()[slot] == static_cast<int64_t>(operand) &&
                set.getLhsParts()[slot] == part)
              return true;
          } else if (set.getRhsOperands()[slot] ==
                         static_cast<int64_t>(operand - lhsCount) &&
                     set.getRhsParts()[slot] == part) {
            return true;
          }
        }
      }
      return false;
    };
    for (int64_t part = 0; part < parts; ++part) {
      if (!partIsUsed(part)) {
        result.parts.emplace_back();
        continue;
      }
      std::string loaded = fresh("reload");
      line(type + " " + loaded + " = __riscv_vle" + std::to_string(bits) +
           "_v_" + suffix + "((const " + *scalarType + " *)(" + slot.scalar +
           " + " + std::to_string(part * bytes) + "), " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(loaded));
    }
  } else if (*kind == Binding::Kind::Scalar) {
    result.scalar = materializeScalarRead(
        *scalarType, "((const " + *scalarType + " *)" + slot.scalar + ")[0]",
        "reload");
  } else if (*kind == Binding::Kind::ScalarTuple) {
    for (int64_t part = 0; part < registerPartCount(operation.getResult()); ++part)
      result.parts.push_back(materializeScalarRead(
          *scalarType, "((const " + *scalarType + " *)" + slot.scalar +
                           ")[" + std::to_string(part) + "]",
          "reload"));
  } else {
    return fail(operation, "reload supports only typed RVV/scalar values");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCommit(riscv::StoreOp operation) {
  auto materialized = materializeNumeric(
      operation.getValue(), bindings.lookup(operation.getValue()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding value = std::move(*materialized);
  Binding region = bindings.lookup(operation.getRegion());
  riscv::AccessAttr selectedEdge = operation.getAccess();
  if (region.kind == Binding::Kind::Field) {
    auto field = fieldFor(region);
    if (!field || field->bitOffset % 8 ||
        field->access.getMapping() != "natural")
      return fail(operation,
                  "encoded field commit currently requires a byte-aligned natural field");
    Binding record = bindings.lookup(region.field.owner);
    if (record.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> materialized = recordForSlice(region.field.owner);
      if (mlir::failed(materialized))
        return mlir::failure();
      record = std::move(*materialized);
    }
    if (record.kind != Binding::Kind::Record)
      return fail(operation,
                  "encoded field commit owner has no selected record address");
    const unsigned width = riscv_internal::logicalBitWidth(field->type);
    std::string logicalIndex = "0";
    if (region.field.index) {
      const Binding &point = bindings.lookup(*region.field.index);
      if (point.kind != Binding::Kind::Point || record.recordElements <= 0)
        return fail(operation,
                    "encoded field commit index has no record-domain relation");
      logicalIndex = "(" + point.point.base + " % " +
                     std::to_string(record.recordElements) + ")";
      if (region.field.selector == "group_index")
        logicalIndex = "(" + logicalIndex + " / " +
                       std::to_string(point.point.physicalExtent) + ")";
    }
    const std::string address = record.recordPointer + " + " +
                                std::to_string(field->bitOffset / 8) + " + (" +
                                logicalIndex + ") * " +
                                std::to_string(width / 8);
    if (value.kind == Binding::Kind::Scalar) {
      mlir::Type element =
          riscv_internal::logicalElement(operation.getValue().getType());
      if (element.isF16())
        line(field->access.getAlignment() >= 2
                 ? "*(_Float16 *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_f16_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isF32())
        line(field->access.getAlignment() >= 4
                 ? "*(float *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_f32_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isInteger(16))
        line(field->access.getAlignment() >= 2
                 ? "*(int16_t *)(" + address + ") = " + value.scalar + ";"
                 : "weft_store_i16_le(" + address + ", " + value.scalar +
                       ");");
      else if (element.isInteger(8))
        line("*(" + *scalarCType(element) + " *)(" + address + ") = " +
             value.scalar + ";");
      else
        return fail(operation,
                    "encoded scalar field has no intrinsic-C store spelling");
      return mlir::success();
    }
    if (value.kind != Binding::Kind::Vector ||
        (width != 8 && width != 16 && width != 32))
      return fail(operation,
                  "encoded array field commit requires selected RVV values");
    for (auto [part, expression] : llvm::enumerate(value.parts)) {
      const std::string partAddress =
          address + " + " + partOffset(operation.getValue(), part) + " * " +
          std::to_string(width / 8);
      auto elementType = scalarCType(field->type);
      if (!elementType)
        return fail(operation,
                    "encoded array field element has no intrinsic-C type");
      line("__riscv_vse" + std::to_string(width) + "_v_" +
           vectorSuffix(operation.getValue()) + "((" + *elementType + " *)(" +
           partAddress + "), " +
           expression + ", " + partVL(operation.getValue(), part) + ");");
    }
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(operation, "commit requires an explicit View slice");
  const Binding &memory = bindings.lookup(region.slice.base);
  if (memory.kind != Binding::Kind::Memory)
    return fail(operation, "commit slice is not memory-backed");
  mlir::Type element =
      riscv_internal::logicalElement(operation.getValue().getType());
  if (!memory.memory.elementType || memory.memory.elementType != element)
    return fail(operation, "commit value and dense memory element types disagree");
  if (value.kind == Binding::Kind::Scalar) {
    auto address = denseAddress(region.slice, {});
    auto type = scalarCType(element);
    if (!address || !type)
      return fail(operation, "scalar commit has no dense address or C type");
    line("*(" + *type + " *)(" + *address + ") = " + value.scalar + ";");
    return mlir::success();
  }
  if (value.kind == Binding::Kind::ScalarTuple) {
    auto type = scalarCType(element);
    llvm::SmallVector<int64_t, 4> axes =
        registerAxesFor(operation.getValue());
    if (!type || value.parts.size() !=
                     static_cast<size_t>(registerPartCount(operation.getValue())))
      return fail(operation,
                  "dense scalar tuple commit has no complete physical mapping");
    for (size_t part = 0; part < value.parts.size(); ++part) {
      auto coordinates = registerCoordinates(operation.getValue(), part);
      if (!coordinates || coordinates->size() != axes.size())
        return fail(operation,
                    "dense scalar tuple commit has no coordinate projection");
      llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
      std::string active = "1";
      for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
        offsets.push_back({axis, std::to_string(coordinate)});
        auto scope = axisScopes.find(axis);
        if (scope != axisScopes.end() && !scope->second.empty())
          active += " && " + std::to_string(coordinate) + " < " +
                    scope->second.back().active;
      }
      auto address = denseAddress(region.slice, offsets);
      if (!address)
        return fail(operation,
                    "dense scalar tuple commit has no address relation");
      line("if (" + active + ") *(" + *type + " *)(" + *address + ") = " +
           value.parts[part] + ";");
    }
    return mlir::success();
  }
  if (value.kind != Binding::Kind::Vector)
    return fail(operation, "dense commit has no selected numeric handoff");
  size_t laneDimension =
      llvm::find(memory.memory.axes, laneAxisFor(operation.getValue())) -
      memory.memory.axes.begin();
  if (laneDimension >= memory.memory.axes.size())
    return fail(operation, "commit destination has no selected lane axis");
  const int64_t laneAxis = laneAxisFor(operation.getValue());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getValue());
  const int64_t streams = streamPartCount(operation.getValue());
  llvm::StringRef memoryForm =
      selectedEdge ? selectedEdge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided")
    return fail(operation,
                "dense commit has no pass-selected unit or strided form");
  for (size_t part = 0; part < value.parts.size(); ++part) {
    auto coordinates =
        registerCoordinates(operation.getValue(), part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "dense vector commit has no register-coordinate mapping");
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string active = "1";
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      offsets.push_back({axis, std::to_string(coordinate)});
      auto scope = axisScopes.find(axis);
      if (scope != axisScopes.end() && !scope->second.empty())
        active += " && " + std::to_string(coordinate) + " < " +
                  scope->second.back().active;
    }
    if (streams > 1)
      offsets.push_back({laneAxis, partOffset(operation.getValue(), part)});
    auto address = denseAddress(region.slice, offsets);
    if (!address)
      return fail(operation, "commit destination has no address relation");
    const std::string suffix = vectorSuffix(operation.getValue());
    const unsigned width =
        riscv_internal::logicalBitWidth(operation.getValue().getType());
    auto elementType = scalarCType(element);
    if (!width || !elementType)
      return fail(operation, "vector commit element has no fixed C representation");
    std::string statement;
    if (memoryForm == "unit")
      statement = "__riscv_vse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  value.parts[part] + ", " +
                  partVL(operation.getValue(), part) + ");";
    else
      statement = "__riscv_vsse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  memory.memory.strides[laneDimension] +
                  " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                  value.parts[part] + ", " +
                  partVL(operation.getValue(), part) + ");";
    if (!registerAxes.empty())
      line("if (" + active + ") " + statement);
    else
      line(statement);
  }
  return mlir::success();
}

} // namespace weft::riscv_emission
