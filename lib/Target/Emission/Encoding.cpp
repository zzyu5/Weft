#include "Emitter.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileField(riscv::FieldOp operation) {
  if (!operation.getAccess() || operation.getAccess().getMapping() == "opaque")
    return fail(operation,
                "field emission requires one selected memory-edge mapping");
  Binding binding;
  binding.kind = Binding::Kind::Field;
  binding.field.owner = operation.getOwner();
  binding.field.name = operation.getName().str();
  binding.field.storageAccess = operation.getAccess();
  binding.field.useAccess = operation.getAccess();
  binding.field.elementType =
      riscv_internal::logicalElement(operation.getResult().getType());
  if (mlir::isa<kernel::EncodingType>(binding.field.elementType)) {
    auto dense = denseElementType(binding.field.elementType);
    if (!dense)
      return fail(operation,
                  "encoded field result has no scalar storage element type");
    binding.field.elementType = *dense;
  }
  if (auto value =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType()))
    binding.field.shape.assign(value.getShape().asArrayRef().begin(),
                               value.getShape().asArrayRef().end());
  binding.field.logicalRank = riscv_internal::fieldFacts(operation).logicalRank;
  fieldProjections[operation.getResult()] = binding.field;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileExtract(riscv::ExtractOp operation) {
  Binding binding = bindings.lookup(operation.getInput());
  if (riscv::fieldReadProjection(operation.getResult())) {
    auto projection = fieldProjections.find(operation.getInput());
    if (projection == fieldProjections.end())
      return fail(operation, "encoded extract has no emitted storage projection");
    binding.kind = Binding::Kind::Field;
    binding.field = projection->second;
  }
  if (instructionOf(operation.getOperation()) == "rvv.extract.vrgather") {
    mlir::FailureOr<Binding> source =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(source))
      return mlir::failure();
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size())
        return fail(operation, "register extract selector has no index");
      if (selector == "gather") {
        if (gatherIndex)
          return fail(operation,
                      "register extract supports one shaped gather selector");
        gatherIndex = operation.getIndices()[cursor];
      }
      ++cursor;
    }
    if (!gatherIndex || source->kind != Binding::Kind::Vector ||
        source->parts.size() != 1)
      return fail(operation,
                  "register extract requires one complete source vector");
    mlir::FailureOr<Binding> indices = materializeNumeric(
        gatherIndex, bindings.lookup(gatherIndex));
    if (mlir::failed(indices) || indices->kind != Binding::Kind::Vector)
      return fail(operation, "register extract requires vector indices");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto indexPart = projectPart(gatherIndex, operation.getResult(), part);
      if (!indexPart || *indexPart >= indices->parts.size())
        return fail(operation,
                    "register extract index mapping requires an explicit layout conversion");
      std::string name = fresh("register_gather");
      line(type + " " + name + " = __riscv_vrgather_vv_" + suffix + "(" +
           source->parts.front() + ", " + indices->parts[*indexPart] + ", " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (instructionOf(operation.getOperation()) ==
      "rvv.extract.lane-to-replica") {
    mlir::FailureOr<Binding> source =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(source) || source->kind != Binding::Kind::Vector)
      return fail(operation,
                  "lane-to-replica extract requires one materialized RVV source");
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size() || gatherIndex ||
          selector != "gather")
        return fail(operation,
                    "lane-to-replica extract requires one typed gather index");
      gatherIndex = operation.getIndices()[cursor++];
    }
    auto laneCount = operation->getAttrOfType<mlir::IntegerAttr>(
        "lane_gather_count");
    auto sourceParts = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "lane_gather_source_parts");
    auto indexParts = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "lane_gather_index_parts");
    const int64_t resultParts = registerPartCount(operation.getResult());
    const Binding &indices = bindings.lookup(gatherIndex);
    mlir::Type element =
        riscv_internal::logicalElement(operation.getResult().getType());
    auto scalarType = scalarCType(element);
    if (!gatherIndex || cursor != operation.getIndices().size() || !laneCount ||
        laneCount.getInt() <= 1 || !sourceParts || !indexParts ||
        resultParts <= 0 ||
        sourceParts.size() != static_cast<size_t>(resultParts) ||
        indexParts.size() != static_cast<size_t>(resultParts) || !scalarType ||
        (indices.kind != Binding::Kind::Scalar &&
         indices.kind != Binding::Kind::ScalarTuple))
      return fail(operation,
                  "lane-to-replica extract has no complete pass-selected mapping");

    const std::string suffix = vectorSuffix(operation.getInput());
    Binding result;
    result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                   : Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < resultParts; ++part) {
      const int64_t sourcePart = sourceParts[part];
      const int64_t indexPart = indexParts[part];
      if (sourcePart < 0 ||
          sourcePart >= static_cast<int64_t>(source->parts.size()))
        return fail(operation,
                    "lane-to-replica extract source part exceeds its RVV value");
      std::string selected;
      if (indices.kind == Binding::Kind::Scalar) {
        if (indexPart != 0)
          return fail(operation,
                      "lane-to-replica scalar index has a nonzero selected part");
        selected = indices.scalar;
      } else {
        if (indexPart < 0 ||
            indexPart >= static_cast<int64_t>(indices.parts.size()))
          return fail(operation,
                      "lane-to-replica index part exceeds its scalar tuple");
        selected = indices.parts[static_cast<size_t>(indexPart)];
      }
      const std::string shifted =
          "__riscv_vslidedown_vx_" + suffix + "(" +
          source->parts[static_cast<size_t>(sourcePart)] + ", " + selected +
          ", " + std::to_string(laneCount.getInt()) + ")";
      std::string expression;
      if (mlir::isa<mlir::FloatType>(element)) {
        expression = "__riscv_vfmv_f_s_" + suffix + "_f" +
                     std::to_string(riscv_internal::logicalBitWidth(element)) +
                     "(" + shifted + ")";
      } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
        expression = "__riscv_vmv_x_s_" + suffix + "_" +
                     std::string(integer.isUnsigned() ? "u" : "i") +
                     std::to_string(integer.getWidth()) + "(" + shifted + ")";
      } else {
        return fail(operation,
                    "lane-to-replica extract has no intrinsic scalar spelling");
      }
      std::string name = fresh("lane_gather");
      line(*scalarType + " " + name + " = " + expression + ";");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(name);
      else
        result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Scalar &&
      !mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
    auto input = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    int64_t elements = 1;
    if (!input)
      return fail(operation,
                  "scalar physical extract has no shaped source value");
    for (int64_t extent : input.getShape().asArrayRef())
      elements *= extent;
    if (elements != 1)
      return fail(operation,
                  "scalar physical extract requires one logical source element");
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Record) {
    auto inputType = operation.getInput().getType();
    if (operation.getSelectors().size() != inputType.getAxisIds().size())
      return fail(operation,
                  "encoded record extract selector rank does not match its logical axes");
    size_t cursor = 0;
    llvm::SmallVector<std::pair<int64_t, std::string>> remainingStrides;
    llvm::SmallVector<std::pair<int64_t, std::string>> remainingOrigins;
    std::string pointer = binding.recordPointer;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      const int64_t axis = inputType.getAxisIds()[dimension];
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      if (!resultType)
        return fail(operation,
                    "encoded record extract result has no physical value type");
      const bool retainsAxis = llvm::is_contained(
          resultType.getAxisIds().asArrayRef(), axis);
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      auto stride = llvm::find_if(
          binding.recordByteStrides,
          [&](const auto &entry) { return entry.first == axis; });
      if (selector == "all") {
        if (stride != binding.recordByteStrides.end())
          remainingStrides.push_back(*stride);
        else if (axis != binding.recordAxis)
          return fail(operation,
                      "encoded record extract has no typed byte stride for one axis");
        if (auto origin = llvm::find_if(
                binding.recordOrigins,
                [&](const auto &entry) { return entry.first == axis; });
            origin != binding.recordOrigins.end())
          remainingOrigins.push_back(*origin);
        continue;
      }
      if (cursor >= operation.getIndices().size())
        return fail(operation,
                    "encoded record extract selector has no coordinate value");
      Binding coordinate = bindings.lookup(operation.getIndices()[cursor++]);
      std::string absoluteExpression;
      if (selector == "domain" && coordinate.kind == Binding::Kind::Point)
        absoluteExpression = coordinate.point.base;
      else if (selector == "group_index" &&
               coordinate.kind == Binding::Kind::Point)
        absoluteExpression = "(" + coordinate.point.base + " / " +
                             std::to_string(coordinate.point.physicalExtent) + ")";
      else if (selector == "index" && coordinate.kind == Binding::Kind::Scalar)
        absoluteExpression = coordinate.scalar;
      else
        return fail(operation,
                    "encoded record extract coordinate does not match its selector");
      std::string expression = absoluteExpression;
      if (coordinate.kind == Binding::Kind::Point)
        if (auto origin = llvm::find_if(
                binding.recordOrigins,
                [&](const auto &entry) { return entry.first == axis; });
            origin != binding.recordOrigins.end())
          expression = "((" + absoluteExpression + ") - (" +
                       origin->second + "))";
      if (stride != binding.recordByteStrides.end()) {
        const bool selectsInterleaveGroup =
            binding.recordGroupAxis == axis &&
            coordinate.kind == Binding::Kind::Point;
        if (selectsInterleaveGroup &&
            coordinate.point.physicalExtent != binding.interleaveRows)
          return fail(operation,
                      "encoded local-pack row projection is not aligned to its selected interleave cohort");
        pointer = "(" + pointer + " + (" + expression + ") * (" +
                  stride->second + "))";
        if (retainsAxis)
          remainingStrides.emplace_back(
              axis, selectsInterleaveGroup ? "1" : stride->second);
        if (selectsInterleaveGroup)
          binding.recordGroupAxis = 0;
      } else if (axis == binding.recordAxis && binding.recordElements > 0 &&
                 binding.recordStrideBytes > 0) {
        pointer = "(" + pointer + " + ((" + expression + ") / " +
                  std::to_string(binding.recordElements) + ") * " +
                  std::to_string(binding.recordStrideBytes) + ")";
        if (!retainsAxis)
          binding.recordAxis = 0;
      } else {
        return fail(operation,
                    "encoded record extract has no typed record coordinate relation");
      }
      if (retainsAxis && coordinate.kind == Binding::Kind::Point)
        remainingOrigins.emplace_back(axis, absoluteExpression);
    }
    if (cursor != operation.getIndices().size())
      return fail(operation,
                  "encoded record extract has unused coordinate operands");
    binding.recordPointer = std::move(pointer);
    binding.recordByteStrides = std::move(remainingStrides);
    binding.recordOrigins = std::move(remainingOrigins);
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::LocalArray) {
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType())) {
      if (operation.getLeaf().getInstruction() != "rvv.local-gather" ||
          operation.getAccess().getForm() != "indexed" ||
          resultType.getLayout().getCarrier() != "rvv")
        return fail(operation,
                    "shaped local extract has no selected RVV gather leaf");
      auto inputType = operation.getInput().getType();
      if (operation.getSelectors().size() != inputType.getShape().size())
        return fail(operation,
                    "local gather selector rank does not match its local value");
      size_t cursor = 0;
      size_t gatherDimension = inputType.getShape().size();
      mlir::Value indexValue;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(operation.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= operation.getIndices().size() || selector != "gather" ||
            indexValue)
          return fail(operation,
                      "local gather requires one shaped gather selector");
        gatherDimension = dimension;
        indexValue = operation.getIndices()[cursor++];
      }
      if (!indexValue || cursor != operation.getIndices().size())
        return fail(operation,
                    "local gather has no unique shaped index value");
      Binding indices = bindings.lookup(indexValue);
      auto indexLayout = layoutOf(indexValue);
      auto indexInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(indexValue.getType()));
      auto elementType = scalarCType(inputType.getElementType());
      if (indices.kind != Binding::Kind::Vector || !indexLayout ||
          !indexInteger || !indexInteger.isUnsigned() || !elementType)
        return fail(operation,
                    "local gather index and element types have no RVV spelling");

      llvm::SmallVector<std::string> elementStrides(inputType.getShape().size(),
                                                     "1");
      for (int64_t dimension =
               static_cast<int64_t>(inputType.getShape().size()) - 2;
           dimension >= 0; --dimension) {
        auto extent = extentForShape(inputType.getShape()[dimension + 1],
                                     inputType.getAxisIds()[dimension + 1]);
        if (!extent)
          return fail(operation,
                      "local gather source has no explicit physical extent");
        elementStrides[dimension] = "(" + elementStrides[dimension + 1] +
                                    " * (" + *extent + "))";
      }

      const std::string indexSuffix = vectorSuffix(indexValue);
      const std::string resultSuffix = vectorSuffix(operation.getResult());
      const std::string resultCType = vectorType(operation.getResult());
      const int64_t elementBytes =
          std::max<int64_t>(1, riscv_internal::logicalBitWidth(inputType) / 8);
      Binding gathered;
      gathered.kind = Binding::Kind::Vector;
      llvm::SmallVector<int64_t, 4> registerAxes =
          registerAxesFor(operation.getResult());
      const int64_t streams = streamPartCount(operation.getResult());
      for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
        auto indexPart = projectPart(indexValue, operation.getResult(), part);
        auto coordinates = registerCoordinates(operation.getResult(), part / streams);
        if (!indexPart || *indexPart >= indices.parts.size() || !coordinates ||
            coordinates->size() != registerAxes.size())
          return fail(operation,
                      "local gather index cannot be projected to its result layout");
        std::string baseElements = "0";
        for (size_t dimension = 0; dimension < inputType.getShape().size();
             ++dimension) {
          if (dimension == gatherDimension)
            continue;
          const int64_t axis = inputType.getAxisIds()[dimension];
          std::string coordinate = "0";
          auto registerAxis = llvm::find(registerAxes, axis);
          if (registerAxis != registerAxes.end())
            coordinate = std::to_string(
                (*coordinates)[registerAxis - registerAxes.begin()]);
          auto scope = axisScopes.find(axis);
          if (scope != axisScopes.end() && !scope->second.empty())
            coordinate = "(" + scope->second.back().base + " + " +
                         coordinate + ")";
          baseElements = "(" + baseElements + " + (" + coordinate +
                         ") * (" + elementStrides[dimension] + "))";
        }
        std::string offsets = indices.parts[*indexPart];
        const std::string byteScale =
            "(" + elementStrides[gatherDimension] + " * " +
            std::to_string(elementBytes) + ")";
        if (byteScale != "(1 * 1)")
          offsets = "__riscv_vmul_vx_" + indexSuffix + "(" + offsets +
                    ", " + byteScale + ", " +
                    partVL(operation.getResult(), part) + ")";
        std::string name = fresh("local_gather");
        line(resultCType + " " + name + " = __riscv_vluxei" +
             std::to_string(indexLayout.getSew()) + "_v_" + resultSuffix +
             "((const " + *elementType + " *)(" + binding.scalar + ") + (" +
             baseElements + "), " + offsets + ", " +
             partVL(operation.getResult(), part) + ");");
        gathered.parts.push_back(std::move(name));
      }
      bindings[operation.getResult()] = std::move(gathered);
      return mlir::success();
    }
    auto inputType = operation.getInput().getType();
    if (operation.getSelectors().size() != inputType.getShape().size())
      return fail(operation,
                  "local array extract selector rank does not match its value");
    std::string linear = "0";
    size_t cursor = 0;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        return fail(operation,
                    "scalar local array extract cannot retain a logical axis");
      if (selector != "index" || cursor >= operation.getIndices().size())
        return fail(operation,
                    "local array extract requires one scalar index per axis");
      Binding index = bindings.lookup(operation.getIndices()[cursor++]);
      if (index.kind != Binding::Kind::Scalar)
        return fail(operation, "local array index has no scalar representation");
      auto extent = extentForShape(inputType.getShape()[dimension],
                                   inputType.getAxisIds()[dimension]);
      if (!extent)
        return fail(operation, "local array axis has no physical extent");
      linear = "((" + linear + ") * (" + *extent + ") + (" + index.scalar + "))";
    }
    auto type = scalarCType(
        riscv_internal::logicalElement(operation.getResult().getType()));
    if (!type)
      return fail(operation, "scalar local array extract has no C value type");
    bindings[operation.getResult()] = scalar(materializeScalarRead(
        *type, binding.scalar + "[" + linear + "]", "local_load"));
    return mlir::success();
  }
  if (binding.kind == Binding::Kind::Field) {
    if (!operation.getAccess())
      return fail(operation,
                  "encoded extract has no pass-selected per-use memory edge");
    binding.field.useAccess = operation.getAccess();
    auto inputType =
        mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    const int64_t fieldStart =
        inputType ? static_cast<int64_t>(inputType.getShape().size()) -
                        binding.field.logicalRank
                  : 0;
    size_t cursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(operation.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (selector == "regular") {
        auto pattern = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "index_pattern");
        if (!pattern || pattern.size() != 3 || binding.field.regularRepeat)
          return fail(operation,
                      "regular encoded extract has no unique index pattern");
        binding.field.regularBase = pattern[0];
        binding.field.regularStride = pattern[1];
        binding.field.regularRepeat = pattern[2];
        binding.field.selector = "regular";
        continue;
      }
      if (cursor >= operation.getIndices().size())
        return fail(operation, "extract selector has no corresponding index");
      mlir::Value index = operation.getIndices()[cursor++];
      if (binding.field.index) {
        if (selector != "index")
          return fail(operation,
                      "nested encoded extract requires a relative scalar index");
        binding.field.relativeIndices.push_back(index);
      } else {
        binding.field.index = index;
        binding.field.selector = selector.str();
      }
      // Keep a domain-selected field axis available to the physical layout.
      // Its per-part time coordinate is needed when the selected sub-domain is
      // materialized as multiple sequential RVV values.
      if (inputType && static_cast<int64_t>(position) >= fieldStart &&
          selector != "domain")
        --binding.field.logicalRank;
    }
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType()))
      binding.field.shape.assign(resultType.getShape().asArrayRef().begin(),
                                 resultType.getShape().asArrayRef().end());
    fieldProjections[operation.getResult()] = binding.field;
    bindings[operation.getResult()] = std::move(binding);
    return mlir::success();
  }

  if (binding.kind == Binding::Kind::Slice) {
    if (auto inputType =
            mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
        inputType && mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
      // A shaped extract of a reloadable dense value is still an addressable
      // memory region.  Preserve that relation and attach the consumed logical
      // coordinates; the first numerical consumer materializes the result in
      // its own selected layout.  Loading the entire parent panel here would
      // destroy both reuse and the child operation's lane/register mapping.
      size_t cursor = 0;
      for (auto [position, selectorAttribute] :
           llvm::enumerate(operation.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= operation.getIndices().size() ||
            position >= inputType.getAxisIds().size())
          return fail(operation,
                      "local materialized extract has no logical axis or index");
        binding.slice.localOffsets.emplace_back(
            inputType.getAxisIds()[position], operation.getIndices()[cursor++]);
      }
      bindings[operation.getResult()] = std::move(binding);
      return mlir::success();
    }
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(binding));
    if (mlir::failed(materialized))
      return mlir::failure();
    auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
    int64_t elements = 1;
    for (int64_t extent : inputType.getShape().asArrayRef())
      elements *= extent;
    if (materialized->kind == Binding::Kind::Scalar && elements == 1 &&
        !mlir::isa<riscv::ValueType>(operation.getResult().getType())) {
      bindings[operation.getResult()] = std::move(*materialized);
      return mlir::success();
    }
    binding = std::move(*materialized);
  }

  if (binding.kind == Binding::Kind::ScalarTuple) {
    auto resultType =
        mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    const int64_t resultParts = registerPartCount(operation.getResult());
    auto arity = operation->getAttrOfType<mlir::IntegerAttr>(
        "replica_gather_arity");
    auto candidates = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "replica_gather_candidates");
    auto keys = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "replica_gather_keys");
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= operation.getIndices().size() || gatherIndex ||
          selector != "gather")
        return fail(operation,
                    "scalar tuple extract requires exactly one typed gather index");
      gatherIndex = operation.getIndices()[cursor++];
    }
    if (instructionOf(operation.getOperation()) != "scalar.replica-gather" ||
        operation.getAccess().getForm() != "register" || !arity ||
        arity.getInt() <= 0 || !candidates || !keys ||
        keys.size() != candidates.size() || !resultType || resultParts <= 0 ||
        static_cast<int64_t>(binding.parts.size()) !=
            registerPartCount(operation.getInput()) ||
        candidates.size() !=
            static_cast<size_t>(resultParts * arity.getInt()) || !gatherIndex ||
        cursor != operation.getIndices().size())
      return fail(operation,
                  "scalar tuple extract has no pass-selected replica candidate mapping");
    const Binding &index = bindings.lookup(gatherIndex);
    Binding tuple;
    tuple.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < resultParts; ++part) {
      std::string selected;
      if (index.kind == Binding::Kind::Scalar) {
        selected = index.scalar;
      } else if (index.kind == Binding::Kind::ScalarTuple) {
        auto indexPart = projectRegisterPart(gatherIndex, operation.getResult(), part);
        if (!indexPart || *indexPart >= index.parts.size())
          return fail(operation,
                      "scalar tuple extract index does not project to its result replica");
        selected = index.parts[*indexPart];
      } else {
        return fail(operation,
                    "scalar tuple extract index has no selected scalar representation");
      }
      const size_t row = static_cast<size_t>(part * arity.getInt());
      int64_t last = candidates[row + static_cast<size_t>(arity.getInt() - 1)];
      if (last < 0 || static_cast<size_t>(last) >= binding.parts.size())
        return fail(operation,
                    "scalar tuple extract candidate exceeds its source tuple");
      std::string expression = binding.parts[static_cast<size_t>(last)];
      if (arity.getInt() == 1)
        expression = "((void)(" + selected + "), " + expression + ")";
      for (int64_t candidate = arity.getInt() - 2; candidate >= 0; --candidate) {
        int64_t source = candidates[row + static_cast<size_t>(candidate)];
        if (source < 0 || static_cast<size_t>(source) >= binding.parts.size())
          return fail(operation,
                      "scalar tuple extract candidate exceeds its source tuple");
        const int64_t key = keys[row + static_cast<size_t>(candidate)];
        expression = "((" + selected + ") == " + std::to_string(key) +
                     " ? " + binding.parts[static_cast<size_t>(source)] + " : " +
                     expression + ")";
      }
      tuple.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(tuple);
    return mlir::success();
  }

  if (binding.kind != Binding::Kind::Vector)
    return fail(operation,
                "extract has no selected local vector or encoded-field input");
  mlir::Type extractElement =
      riscv_internal::logicalElement(operation.getInput().getType());
  if ((!mlir::isa<mlir::IntegerType>(extractElement) &&
       !mlir::isa<mlir::FloatType>(extractElement)) ||
      extractElement !=
          riscv_internal::logicalElement(operation.getResult().getType()))
    return fail(operation,
                "current local vector extract requires one unchanged numeric element type");
  std::optional<mlir::Value> pointValue;
  size_t cursor = 0;
  for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (cursor >= operation.getIndices().size() || pointValue)
      return fail(operation,
                  "current local vector extract supports one domain selector");
    if (selector != "domain")
      return fail(operation,
                  "current local vector extract requires a domain selector");
    pointValue = operation.getIndices()[cursor++];
  }
  if (!pointValue)
    return fail(operation, "local vector extract has no domain point");
  const Binding &point = bindings.lookup(*pointValue);
  const int64_t laneAxis = laneAxisFor(operation.getInput());
  if (point.kind != Binding::Kind::Point || point.point.axis != laneAxis)
    return fail(operation,
                "local vector extract point does not own the selected lane axis");

  auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
  int64_t logicalExtent = 0;
  for (auto [extent, axis] : llvm::zip(inputType.getShape().asArrayRef(),
                                       inputType.getAxisIds().asArrayRef()))
    if (axis == laneAxis)
      logicalExtent = extent;
  const int64_t inputLanes = physicalLanes(operation.getInput());
  const int64_t resultLanes = physicalLanes(operation.getResult());
  const int64_t inputStreams = streamPartCount(operation.getInput());
  const int64_t resultStreams = streamPartCount(operation.getResult());
  const int64_t inputRegisters = registerPartCount(operation.getInput());
  const int64_t resultRegisters = registerPartCount(operation.getResult());
  if (logicalExtent <= 0 || inputLanes <= 0 || resultLanes <= 0 ||
      inputLanes != resultLanes || inputStreams <= 0 || resultStreams <= 0 ||
      inputRegisters <= 0 || resultRegisters <= 0 ||
      point.point.physicalExtent <= 0 ||
      point.point.physicalExtent % resultLanes ||
      static_cast<int64_t>(binding.parts.size()) !=
          inputRegisters * inputStreams)
    return fail(operation,
                "selected local vector subview has incompatible strip geometry");

  llvm::SmallVector<size_t> sourceRegisters;
  for (int64_t resultRegister = 0; resultRegister < resultRegisters;
       ++resultRegister) {
    auto sourceRegister = projectRegisterPart(operation.getInput(),
                                              operation.getResult(),
                                              resultRegister);
    if (!sourceRegister)
      return fail(operation,
                  "local vector subview cannot project one register replica");
    sourceRegisters.push_back(*sourceRegister);
  }

  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const std::string relative = "(" + point.point.base + " % " +
                               std::to_string(logicalExtent) + ")";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t resultRegister = 0; resultRegister < resultRegisters;
       ++resultRegister) {
    const size_t sourceRegister = sourceRegisters[resultRegister];
    for (int64_t resultStream = 0; resultStream < resultStreams;
         ++resultStream) {
      const int64_t resultPart = resultRegister * resultStreams + resultStream;
      const std::string resultName = fresh("extract");
      const std::string splat =
          mlir::isa<mlir::FloatType>(extractElement) ? "__riscv_vfmv_v_f_"
                                                     : "__riscv_vmv_v_x_";
      line(resultType + " " + resultName + " = " + splat + resultSuffix +
           "(" +
           (mlir::isa<mlir::FloatType>(extractElement) ? "0.0" : "0") +
           ", " + partVL(operation.getResult(), resultPart) + ");");
      const std::string selectedStream =
          "(" + relative + " / " + std::to_string(inputLanes) + " + " +
          std::to_string(resultStream) + ")";
      for (int64_t sourceStream = 0; sourceStream < inputStreams;
           ++sourceStream) {
        const size_t sourcePart = sourceRegister * inputStreams + sourceStream;
        line("if (" + selectedStream + " == " +
             std::to_string(sourceStream) + ") " + resultName + " = " +
             binding.parts[sourcePart] + ";");
      }
      result.parts.push_back(resultName);
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileUpdate(riscv::UpdateOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  mlir::FailureOr<Binding> materialized = materializeNumeric(
      operation.getValue(), bindings.lookup(operation.getValue()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding value = std::move(*materialized);
  if (input.kind != Binding::Kind::LocalArray ||
      value.kind != Binding::Kind::Scalar)
    return fail(operation,
                "physical update currently requires a local array and scalar value");
  auto inputType = operation.getInput().getType();
  if (operation.getSelectors().size() != inputType.getShape().size())
    return fail(operation,
                "local array update selector rank does not match its value");
  std::string linear = "0";
  size_t cursor = 0;
  for (auto [dimension, selectorAttribute] :
       llvm::enumerate(operation.getSelectors())) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector != "index" || cursor >= operation.getIndices().size())
      return fail(operation,
                  "local array update requires one scalar index per axis");
    Binding index = bindings.lookup(operation.getIndices()[cursor++]);
    if (index.kind != Binding::Kind::Scalar)
      return fail(operation, "local array update index has no scalar representation");
    auto extent = extentForShape(inputType.getShape()[dimension],
                                 inputType.getAxisIds()[dimension]);
    if (!extent)
      return fail(operation, "local array update axis has no physical extent");
    linear = "((" + linear + ") * (" + *extent + ") + (" + index.scalar + "))";
  }
  line(input.scalar + "[" + linear + "] = " + value.scalar + ";");
  bindings[operation.getResult()] = std::move(input);
  return mlir::success();
}

std::optional<EncodingField>
Emitter::fieldFor(const Binding &fieldBinding) const {
  if (fieldBinding.kind != Binding::Kind::Field ||
      !fieldBinding.field.storageAccess || !fieldBinding.field.elementType)
    return std::nullopt;
  EncodingField field;
  field.name = fieldBinding.field.name;
  field.type = fieldBinding.field.elementType;
  field.shape = fieldBinding.field.shape;
  field.logicalRank = fieldBinding.field.logicalRank;
  field.access = fieldBinding.field.storageAccess;
  field.bitOffset = field.access.getBitOffset();
  field.storageBits = field.access.getStorageBits();
  return field;
}

Binding Emitter::emitInterleavedField(mlir::Value result,
                                      const Binding &fieldBinding,
                                      llvm::StringRef requestedIndex,
                                      size_t resultPart) {
  if (resultPart >= static_cast<size_t>(vectorPartCount(result)))
    return {};
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return {};
    owner = std::move(*record);
  }
  if (owner.kind != Binding::Kind::Record)
    return {};
  auto field = fieldFor(fieldBinding);
  if (!field)
    return {};
  unsigned logicalWidth = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type))
    logicalWidth = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(field->type))
    logicalWidth = floating.getWidth();
  if (!logicalWidth)
    return {};
  std::string logicalIndex = requestedIndex.str();
  int64_t logicalPartition = 0;
  if (fieldBinding.field.regularRepeat > 0) {
    logicalIndex = std::to_string(fieldBinding.field.regularBase);
    if (fieldBinding.field.regularRepeat == 1)
      logicalPartition = physicalLanes(result);
  } else if (fieldBinding.field.index) {
    const Binding &index = bindings.lookup(*fieldBinding.field.index);
    if (index.kind == Binding::Kind::Vector) {
      llvm::StringRef form = fieldBinding.field.useAccess
                                 ? fieldBinding.field.useAccess.getForm()
                                 : llvm::StringRef();
      if (form != "indexed" || owner.interleaveRows != 0 ||
          field->bitOffset % 8)
        return {};
      mlir::Value indexValue = *fieldBinding.field.index;
      auto indexLayout = layoutOf(indexValue);
      auto elementType = scalarCType(field->type);
      if (!indexLayout)
        return {};
      const std::string indexSuffix =
          "u" + std::to_string(indexLayout.getSew()) +
          lmulSpelling(indexLayout.getLmulEighths());
      const std::string indexType =
          "vuint" + std::to_string(indexLayout.getSew()) +
          lmulSpelling(indexLayout.getLmulEighths()) + "_t";
      const std::string resultSuffix = vectorSuffix(result);
      const std::string resultType = vectorType(result);
      llvm::StringRef mapping = field->access.getMapping();
      if (mapping == "grouped_layered") {
        auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
        auto resultLayout = layoutOf(result);
        const int64_t indexSEW = indexLayout.getSew();
        const int64_t group = field->access.getGroupSize();
        const int64_t layer = field->access.getLayerSize();
        const int64_t layers = layer > 0 ? group / layer : 0;
        llvm::StringRef order = field->access.getOrder();
        const uint64_t indexRange = uint64_t(1) << indexSEW;
        if (!integer || integer.isSigned() || !resultLayout ||
            (indexSEW != 8 && indexSEW != 16 && indexSEW != 32) ||
            resultLayout.getSew() != 8 ||
            logicalWidth == 0 || logicalWidth > 8 || group <= 0 ||
            layer <= 0 || group % layer || layers * logicalWidth > 8 ||
            static_cast<uint64_t>(group) > indexRange ||
            physicalLanes(indexValue) != physicalLanes(result))
          return {};

        const int64_t indexLMUL = indexLayout.getLmulEighths();
        const int64_t widthRatio = indexSEW / 8;
        if (indexLMUL != resultLayout.getLmulEighths() * widthRatio)
          return {};
        const uint64_t mask = (uint64_t(1) << logicalWidth) - 1;
        Binding gathered;
        gathered.kind = Binding::Kind::Vector;
        const int64_t resultStreams = streamPartCount(result);
        for (int64_t part = 0; part < vectorPartCount(result); ++part) {
          auto indexPart = projectPart(indexValue, result, part);
          auto coordinates =
              registerCoordinates(result, part / resultStreams);
          llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
          if (!indexPart || *indexPart >= index.parts.size() || !coordinates ||
              coordinates->size() != axes.size())
            return {};
          std::string record = "(" + owner.recordPointer;
          for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
            for (const auto &[recordAxis, stride] : owner.recordByteStrides)
              if (recordAxis == axis)
                record += " + " + std::to_string(coordinate) + " * " + stride;
          record += " + " + std::to_string(field->bitOffset / 8) + ")";
          const std::string vl = partVL(result, part);
          const std::string &indices = index.parts[*indexPart];
          std::string within = fresh("encoded_within");
          const bool oneRepresentableGroup =
              static_cast<uint64_t>(group) == indexRange;
          line(indexType + " " + within + " = " +
               (oneRepresentableGroup
                    ? indices
                    : "__riscv_vremu_vx_" + indexSuffix + "(" + indices +
                          ", " + std::to_string(group) + ", " + vl + ")") +
               ";");
          std::string bytes = fresh("encoded_bytes");
          std::string byteExpression =
              "__riscv_vremu_vx_" + indexSuffix + "(" + within + ", " +
              std::to_string(layer) + ", " + vl + ")";
          if (!oneRepresentableGroup) {
            std::string groups = fresh("encoded_group");
            line(indexType + " " + groups +
                 " = __riscv_vdivu_vx_" + indexSuffix + "(" + indices + ", " +
                 std::to_string(group) + ", " + vl + ");");
            byteExpression =
                "__riscv_vadd_vv_" + indexSuffix + "(__riscv_vmul_vx_" +
                indexSuffix + "(" + groups + ", " + std::to_string(layer) +
                ", " + vl + "), " + byteExpression + ", " + vl + ")";
          }
          line(indexType + " " + bytes + " = " + byteExpression + ";");
          std::string loaded = fresh("encoded_gather");
          line(resultType + " " + loaded + " = __riscv_vluxei" +
               std::to_string(indexSEW) + "_v_" +
               resultSuffix + "((const uint8_t *)(" + record + "), " + bytes +
               ", " + vl + ");");
          std::string layerWide = fresh("encoded_layer");
          std::string layerExpression =
              "__riscv_vdivu_vx_" + indexSuffix + "(" + within + ", " +
              std::to_string(layer) + ", " + vl + ")";
          if (order == "hi_first")
            layerExpression = "__riscv_vrsub_vx_" + indexSuffix + "(" +
                              layerExpression + ", " +
                              std::to_string(layers - 1) + ", " + vl + ")";
          line("vuint" + std::to_string(indexSEW) +
               lmulSpelling(indexLMUL) + "_t " + layerWide +
               " = __riscv_vmul_vx_" + indexSuffix + "(" + layerExpression +
               ", " + std::to_string(logicalWidth) + ", " + vl + ");");
          std::string layer8 = layerWide;
          if (indexSEW == 16) {
            layer8 = fresh("encoded_shift8");
            line(resultType + " " + layer8 + " = __riscv_vnsrl_wx_" +
                 resultSuffix + "(" + layerWide + ", 0, " + vl + ");");
          } else if (indexSEW == 32) {
            const int64_t shift16LMUL = indexLMUL / 2;
            const std::string shift16Suffix =
                "u16" + lmulSpelling(shift16LMUL);
            const std::string shift16Type =
                "vuint16" + lmulSpelling(shift16LMUL) + "_t";
            std::string layer16 = fresh("encoded_shift16");
            line(shift16Type + " " + layer16 + " = __riscv_vnsrl_wx_" +
                 shift16Suffix + "(" + layerWide + ", 0, " + vl + ");");
            layer8 = fresh("encoded_shift8");
            line(resultType + " " + layer8 + " = __riscv_vnsrl_wx_" +
                 resultSuffix + "(" + layer16 + ", 0, " + vl + ");");
          }
          std::string decoded = fresh("encoded_value");
          line(resultType + " " + decoded + " = __riscv_vand_vx_" +
               resultSuffix + "(__riscv_vsrl_vv_" + resultSuffix + "(" +
               loaded + ", " + layer8 + ", " + vl + "), " +
               std::to_string(mask) + ", " + vl + ");");
          gathered.parts.push_back(std::move(decoded));
        }
        return gathered;
      }
      if (mapping != "natural" || !elementType ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32))
        return {};
      Binding gathered;
      gathered.kind = Binding::Kind::Vector;
      const int64_t resultStreams = streamPartCount(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto indexPart = projectPart(indexValue, result, part);
        auto coordinates =
            registerCoordinates(result, part / resultStreams);
        llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
        if (!indexPart || *indexPart >= index.parts.size() || !coordinates ||
            coordinates->size() != axes.size())
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + " + std::to_string(field->bitOffset / 8) + ")";
        std::string offsets = index.parts[*indexPart];
        if (logicalWidth != 8)
          offsets = "__riscv_vmul_vx_" + indexSuffix + "(" + offsets +
                    ", " + std::to_string(logicalWidth / 8) + ", " +
                    partVL(result, part) + ")";
        std::string expression =
        "__riscv_vluxei" + std::to_string(indexLayout.getSew()) + "_v_" +
            resultSuffix + "((const " + *elementType + " *)(" + record +
            "), " + offsets + ", " + partVL(result, part) + ")";
        std::string loaded = fresh("gather");
        line(resultType + " " + loaded + " = " + expression + ";");
        gathered.parts.push_back(std::move(loaded));
      }
      return gathered;
    } else if (index.kind == Binding::Kind::Scalar) {
      logicalIndex = index.scalar;
    } else if (index.kind == Binding::Kind::Point) {
      const int64_t elements = owner.recordElements;
      if (elements <= 0)
        return {};
      logicalPartition = index.point.physicalExtent;
      if (fieldBinding.field.selector == "group_index")
        logicalIndex = "((" + index.point.base + " % " +
                       std::to_string(elements) + ") / " +
                       std::to_string(index.point.physicalExtent) + ")";
      else
        logicalIndex = "(" + index.point.base + " % " +
                       std::to_string(elements) + ")";
    } else {
      return {};
    }
  }
  for (mlir::Value relativeValue : fieldBinding.field.relativeIndices) {
    const Binding &relative = bindings.lookup(relativeValue);
    if (relative.kind != Binding::Kind::Scalar)
      return {};
    logicalIndex = "((" + logicalIndex + ") + (" + relative.scalar + "))";
  }
  if (owner.interleaveRows == 0) {
    auto resultLayout = layoutOf(result);
    const bool vectorResult = resultLayout && resultLayout.getCarrier() == "rvv";
    llvm::StringRef selectedForm = fieldBinding.field.useAccess
                                       ? fieldBinding.field.useAccess.getForm()
                                       : llvm::StringRef();
    std::string logicalProjectionFailure;
    auto logicalIndexForPart = [&](int64_t part)
        -> std::optional<std::string> {
      if (field->logicalRank == 0)
        return logicalIndex;
      auto value = mlir::dyn_cast<riscv::ValueType>(result.getType());
      if (!value || !resultLayout || field->logicalRank < 0 ||
          field->logicalRank > static_cast<int64_t>(field->shape.size()) ||
          field->logicalRank > static_cast<int64_t>(value.getAxisIds().size())) {
        logicalProjectionFailure = "field rank exceeds the physical value rank";
        return std::nullopt;
      }
      const size_t logicalRank = static_cast<size_t>(field->logicalRank);
      const size_t valueOuterRank = value.getAxisIds().size() - logicalRank;
      const size_t shapeOuterRank = field->shape.size() - logicalRank;
      int64_t linear = 0;
      for (size_t dimension = 0; dimension < logicalRank; ++dimension) {
        const int64_t axis = value.getAxisIds()[valueOuterRank + dimension];
        auto position = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
        if (position == resultLayout.getAxisIds().asArrayRef().end()) {
          logicalProjectionFailure = "field axis is absent from the physical layout";
          return std::nullopt;
        }
        const size_t layoutPosition = static_cast<size_t>(
            std::distance(resultLayout.getAxisIds().asArrayRef().begin(), position));
        if (resultLayout.getLaneFactors()[layoutPosition] != 1 ||
            resultLayout.getReplicaFactors()[layoutPosition] != 1 ||
            resultLayout.getFragmentFactors()[layoutPosition] != 1 ||
            resultLayout.getLocalFactors()[layoutPosition] != 1) {
          logicalProjectionFailure =
              "field axis is not represented solely by sequential time";
          return std::nullopt;
        }
        auto coordinate = timeCoordinate(result, part, axis);
        const int64_t logicalExtent = field->shape[shapeOuterRank + dimension];
        if (!coordinate || *coordinate < 0 ||
            *coordinate >= logicalExtent) {
          logicalProjectionFailure =
              "field time coordinate is incomplete or outside its logical extent";
          return std::nullopt;
        }
        linear = linear * logicalExtent + *coordinate;
      }
      return "((" + logicalIndex + ") + " + std::to_string(linear) + ")";
    };
    if (vectorResult && fieldBinding.field.regularRepeat > 1 &&
        selectedForm == "indexed" &&
        field->access.getMapping() == "natural") {
      auto elementType = scalarCType(field->type);
      const int64_t repeat = fieldBinding.field.regularRepeat;
      const int64_t stride = fieldBinding.field.regularStride;
      const int64_t lanes = physicalLanes(result);
      if (!elementType || field->bitOffset % 8 ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32) ||
          repeat <= 1 || stride <= 0 || lanes <= 0 ||
          (repeat % lanes && lanes % repeat))
        return {};
      Binding repeated;
      repeated.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        int64_t offset = 0;
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            llvm::StringRef(partOffset(result, part)).getAsInteger(10, offset) ||
            offset % lanes ||
            (repeat >= lanes ? offset % repeat + lanes > repeat
                             : offset % repeat))
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, byteStride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " +
                        byteStride;
        record += ")";
        auto address = [&](int64_t logicalOffset) {
          const int64_t sourceIndex =
              fieldBinding.field.regularBase +
              (logicalOffset / repeat) * stride;
          return "((const " + *elementType + " *)((const uint8_t *)(" +
                 record + ") + " + std::to_string(field->bitOffset / 8) +
                 "))[" + std::to_string(sourceIndex) + "]";
        };
        std::string name = fresh("field_repeat");
        const bool floating = mlir::isa<mlir::FloatType>(field->type);
        const std::string broadcast =
            "__riscv_" +
            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix;
        const std::string vl = partVL(result, part);
        std::string expression = broadcast + "(" + address(offset) + ", " +
                                 vl + ")";
        if (lanes > repeat)
          for (int64_t piece = 1; piece < lanes / repeat; ++piece) {
            const std::string next =
                broadcast + "(" + address(offset + piece * repeat) + ", " +
                vl + ")";
            expression = "__riscv_vslideup_vx_" + suffix + "(" + expression +
                         ", " + next + ", " + std::to_string(piece * repeat) +
                         ", " + vl + ")";
          }
        line(type + " " + name + " = " + expression + ";");
        repeated.parts.push_back(std::move(name));
      }
      return repeated;
    }
    if (vectorResult && field->access.getMapping() == "joined" &&
        !field->shape.empty()) {
      const int64_t laneAxis = laneAxisFor(result);
      auto laneStride = llvm::find_if(
          owner.recordByteStrides,
          [&](const auto &entry) { return entry.first == laneAxis; });
      const int64_t group = field->access.getGroupSize();
      const int64_t fields = field->access.getJoinFields();
      const int64_t lowBits = field->access.getJoinLowBits();
      const int64_t role = field->access.getJoinRole();
      llvm::StringRef order = field->access.getOrder();
      if (!laneAxis || laneStride == owner.recordByteStrides.end() || group <= 0 ||
          fields <= 1 || lowBits <= 0 || role < 0 || role >= fields ||
          logicalWidth <= lowBits || logicalWidth > 8)
        return {};
      const int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
      const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
      const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
      const uint64_t highMask =
          (uint64_t(1) << (logicalWidth - lowBits)) - 1;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      Binding decoded;
      decoded.kind = Binding::Kind::Vector;
      auto byteLoad = [&](llvm::StringRef record, llvm::StringRef offset,
                          llvm::StringRef stem, llvm::StringRef vl) {
        std::string name = fresh(stem);
        std::string statement =
            type + " " + name + " = __riscv_vlse8_v_" + suffix +
            "((const uint8_t *)(" + record.str() + " + (" + offset.str() +
            ")), (ptrdiff_t)(" + laneStride->second + "), " + vl.str() + ");";
        line(statement);
        return name;
      };
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        auto partLogical = logicalIndexForPart(part);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            !partLogical) {
          if (!partLogical)
            result.getDefiningOp()->emitError(
                "joined field has no closed per-part logical coordinate: ")
                << logicalProjectionFailure;
          return {};
        }
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + (" + partOffset(result, part) + ") * (" +
                  laneStride->second + "))";
        const std::string logical = "(" + *partLogical + ")";
        const std::string tail = "(" + logical + " - " +
                                 std::to_string(group) + ")";
        const std::string headOffset =
            std::to_string(field->bitOffset / 8 + role * group) + " + " + logical;
        const std::string lowOffset =
            std::to_string(field->bitOffset / 8 + fields * group) + " + " + tail;
        const std::string highOffset =
            std::to_string(field->bitOffset / 8 + role * group) + " + " + tail;
        const std::string vl = partVL(result, part);
        std::string head = byteLoad(record, headOffset, "joined_head", vl);
        head = "__riscv_vand_vx_" + suffix + "(" + head + ", " +
               std::to_string(logicalMask) + ", " + vl + ")";
        std::string low = byteLoad(record, lowOffset, "joined_low", vl);
        low = "__riscv_vand_vx_" + suffix + "(__riscv_vsrl_vx_" + suffix +
              "(" + low + ", " + std::to_string(physicalRole * lowBits) +
              ", " + vl + "), " + std::to_string(lowMask) + ", " + vl + ")";
        std::string high = byteLoad(record, highOffset, "joined_high", vl);
        high = "__riscv_vand_vx_" + suffix + "(__riscv_vsrl_vx_" + suffix +
               "(" + high + ", " + std::to_string(logicalWidth) +
               ", " + vl + "), " + std::to_string(highMask) + ", " + vl + ")";
        std::string assembled =
            order == "lo_first"
                ? "__riscv_vor_vv_" + suffix + "(" + low +
                      ", __riscv_vsll_vx_" + suffix + "(" + high + ", " +
                      std::to_string(lowBits) + ", " + vl + "), " + vl + ")"
                : "__riscv_vor_vv_" + suffix + "(" + high +
                      ", __riscv_vsll_vx_" + suffix + "(" + low + ", " +
                      std::to_string(logicalWidth - lowBits) + ", " + vl +
                      "), " + vl + ")";
        std::string name = fresh("joined_field");
        line(type + " " + name + " = (" + logical + " < " +
             std::to_string(group) + " ? " + head + " : " + assembled +
             ");");
        decoded.parts.push_back(std::move(name));
      }
      return decoded;
    }
    if (vectorResult && selectedForm == "strided" &&
        field->access.getMapping() == "natural" &&
        field->bitOffset % 8 == 0 &&
        (logicalWidth == 8 || logicalWidth == 16 || logicalWidth == 32)) {
      const int64_t laneAxis = laneAxisFor(result);
      auto laneStride = llvm::find_if(
          owner.recordByteStrides,
          [&](const auto &entry) { return entry.first == laneAxis; });
      auto elementType = scalarCType(field->type);
      if (!laneAxis || laneStride == owner.recordByteStrides.end() || !elementType)
        return {};
      Binding vectorResult;
      vectorResult.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      const int64_t streams = streamPartCount(result);
      llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        auto coordinates = registerCoordinates(result, part / streams);
        auto partLogical = logicalIndexForPart(part);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size() ||
            !partLogical) {
          if (!partLogical)
            result.getDefiningOp()->emitError(
                "natural field has no closed per-part logical coordinate: ")
                << logicalProjectionFailure;
          return {};
        }
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += " + (" + partOffset(result, part) + ") * (" +
                  laneStride->second + ") + " +
                  std::to_string(field->bitOffset / 8) + " + (" +
                  *partLogical + ") * " +
                  std::to_string(logicalWidth / 8) + ")";
        std::string loaded = fresh("field_strided");
        line(type + " " + loaded + " = __riscv_vlse" +
             std::to_string(logicalWidth) + "_v_" + suffix + "((const " +
             *elementType + " *)(" + record + "), (ptrdiff_t)(" +
             laneStride->second + "), " + partVL(result, part) + ");");
        vectorResult.parts.push_back(std::move(loaded));
      }
      return vectorResult;
    }
    if (vectorResult && field->access.getMapping() == "grouped_layered") {
      result.getDefiningOp()->emitError(
          "grouped/layered vector access reached emission without a typed physical window/stream operation");
      return {};
    }
    if (vectorResult && !field->shape.empty()) {
      if (field->bitOffset % 8 || field->access.getMapping() != "natural" ||
          (logicalWidth != 8 && logicalWidth != 16 && logicalWidth != 32))
        return {};
      Binding vectorResult;
      vectorResult.kind = Binding::Kind::Vector;
      const std::string suffix = vectorSuffix(result);
      const std::string type = vectorType(result);
      auto elementType = scalarCType(field->type);
      if (!elementType)
        return {};
      for (int64_t part = 0; part < vectorPartCount(result); ++part) {
        const int64_t streams = streamPartCount(result);
        auto coordinates = registerCoordinates(result, part / streams);
        llvm::SmallVector<int64_t, 4> axes = registerAxesFor(result);
        if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
          return {};
        std::string record = "(" + owner.recordPointer;
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
          for (const auto &[recordAxis, stride] : owner.recordByteStrides)
            if (recordAxis == axis)
              record += " + " + std::to_string(coordinate) + " * " + stride;
        record += ")";
        const std::string address =
            record + " + " + std::to_string(field->bitOffset / 8) +
            " + ((" + logicalIndex + ") + " + partOffset(result, part) + ") * " +
            std::to_string(logicalWidth / 8);
        std::string loaded = fresh("field");
        line(type + " " + loaded + " = __riscv_vle" +
             std::to_string(logicalWidth) + "_v_" + suffix + "((const " +
             *elementType + " *)(" + address + ")" +
             ", " + partVL(result, part) + ");");
        vectorResult.parts.push_back(std::move(loaded));
      }
      return vectorResult;
    }
    Binding scalarResult;
    scalarResult.kind = Binding::Kind::Scalar;
    llvm::StringRef kind = field->access.getMapping();
    if (kind == "joined") {
      int64_t group = field->access.getGroupSize();
      int64_t fields = field->access.getJoinFields();
      int64_t lowBits = field->access.getJoinLowBits();
      int64_t role = field->access.getJoinRole();
      llvm::StringRef order = field->access.getOrder();
      int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
      const std::string headByte =
          "(" + std::to_string(field->bitOffset / 8 + role * group) +
          " + (" + logicalIndex + "))";
      const std::string tail = "((" + logicalIndex + ") - " +
                               std::to_string(group) + ")";
      const std::string lowByte =
          "(" + std::to_string(field->bitOffset / 8 + fields * group) +
          " + " + tail + ")";
      const std::string highByte =
          "(" + std::to_string(field->bitOffset / 8 + role * group) +
          " + " + tail + ")";
      const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
      const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
      const uint64_t highMask =
          (uint64_t(1) << (logicalWidth - lowBits)) - 1;
      const std::string head =
          "((uint8_t)((" + owner.recordPointer + ")[" + headByte + "] & " +
          std::to_string(logicalMask) + "))";
      const std::string low =
          "((uint8_t)(((" + owner.recordPointer + ")[" + lowByte + "] >> " +
          std::to_string(physicalRole * lowBits) + ") & " +
          std::to_string(lowMask) + "))";
      const std::string high =
          "((uint8_t)(((" + owner.recordPointer + ")[" + highByte + "] >> " +
          std::to_string(logicalWidth) + ") & " +
          std::to_string(highMask) + "))";
      scalarResult.scalar = "((" + logicalIndex + ") < " +
                            std::to_string(group) + " ? " + head + " : (" +
                            low + " | (" + high + " << " +
                            std::to_string(lowBits) + ")))";
      return scalarResult;
    }
    auto fragment = singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    const std::string address = owner.recordPointer + " + (" + fragment->byte + ")";
    const bool naturallyByteAligned =
        kind == "natural" && field->bitOffset % 8 == 0 && logicalWidth % 8 == 0;
    if (field->type.isF32() && logicalWidth == 32 && naturallyByteAligned)
      scalarResult.scalar = field->access.getAlignment() >= 4
                                ? "*(const float *)(" + address + ")"
                                : "weft_load_f32_le(" + owner.recordPointer +
                                      " + " + fragment->byte + ")";
    else if (field->type.isF16() && logicalWidth == 16 && naturallyByteAligned)
      scalarResult.scalar = field->access.getAlignment() >= 2
                                ? "*(const _Float16 *)(" + address + ")"
                                : "weft_load_f16_le(" + owner.recordPointer +
                                      " + " + fragment->byte + ")";
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
             integer && integer.getWidth() <= 8) {
      const uint64_t mask = (uint64_t(1) << integer.getWidth()) - 1;
      std::string value = "((" + address + ")[0] >> (" + fragment->shift + ")) & " +
                          std::to_string(mask);
      if (integer.isSigned() && integer.getWidth() < 8) {
        const uint64_t sign = uint64_t(1) << (integer.getWidth() - 1);
        value = "((int8_t)((((" + value + ") ^ " + std::to_string(sign) +
                ") - " + std::to_string(sign) + ")))";
      } else if (integer.isSigned()) {
        value = "((int8_t)(" + value + "))";
      } else {
        value = "((uint8_t)(" + value + "))";
      }
      scalarResult.scalar = std::move(value);
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
               integer && integer.getWidth() == 16 && naturallyByteAligned) {
      scalarResult.scalar =
          std::string(integer.isUnsigned() ? "weft_load_u16_le(" :
                                             "weft_load_i16_le(") +
          owner.recordPointer + " + " + fragment->byte + ")";
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(field->type);
               integer && integer.getWidth() == 32 && naturallyByteAligned) {
      scalarResult.scalar =
          std::string(integer.isSigned() ? "((int32_t)" : "") +
          "weft_load_u32_le(" + owner.recordPointer + " + " +
          fragment->byte + ")" + (integer.isSigned() ? ")" : "");
    } else {
      return {};
    }
    return scalarResult;
  }

  // Byte assembly preserves the selected result lane count.  Its raw LMUL is
  // therefore the result LMUL scaled by the SEW ratio; this is a mechanical
  // projection of the selected value layout, not a second layout choice.
  auto rawCarrier = riscv::encodedFieldRawLMULEighths(
      mlir::dyn_cast<riscv::ValueType>(result.getType()));
  if (!rawCarrier)
    return {};
  const int64_t rawLMUL = *rawCarrier;
  const std::string rawSuffix = "u8" + lmulSpelling(rawLMUL);
  const std::string rawType = "vuint8" + lmulSpelling(rawLMUL) + "_t";
  const std::string vl = partVL(result, resultPart);
  const std::string offset = partOffset(result, resultPart);
  const std::string laneOffset = offset == "0" ? "" : " + " + offset;
  auto loadByte = [&](llvm::StringRef byte, llvm::StringRef prefix) {
    std::string loaded = fresh(prefix);
    line(rawType + " " + loaded + " = __riscv_vle8_v_" + rawSuffix + "(" +
         owner.recordPointer + " + (" + byte.str() + ") * " +
         std::to_string(owner.interleaveRows) + laneOffset + ", " + vl + ");");
    return loaded;
  };

  std::string raw;
  llvm::StringRef kind = field->access.getMapping();
  if (kind == "joined") {
    int64_t group = field->access.getGroupSize();
    int64_t fields = field->access.getJoinFields();
    int64_t lowBits = field->access.getJoinLowBits();
    int64_t role = field->access.getJoinRole();
    llvm::StringRef order = field->access.getOrder();
    int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
    raw = fresh("joined_value");
    line(rawType + " " + raw + ";");
    line("if ((" + logicalIndex + ") < " + std::to_string(group) + ") {");
    ++indent;
    std::string headByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                           std::to_string(role * group) + " + (" +
                           logicalIndex + "))";
    std::string head = loadByte(headByte, "joined_head");
    line(raw + " = __riscv_vand_vx_" + rawSuffix + "(" + head + ", " +
         std::to_string((1u << logicalWidth) - 1) + ", " + vl + ");");
    --indent;
    line("} else {");
    ++indent;
    std::string tail = "((" + logicalIndex + ") - " + std::to_string(group) + ")";
    std::string lowByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                          std::to_string(fields * group) + " + " + tail + ")";
    std::string highByte = "(" + std::to_string(field->bitOffset / 8) + " + " +
                           std::to_string(role * group) + " + " + tail + ")";
    std::string low = loadByte(lowByte, "joined_low");
    std::string high = loadByte(highByte, "joined_high");
    std::string lowPart = fresh("joined_low_bits");
    line(rawType + " " + lowPart + " = __riscv_vand_vx_" + rawSuffix +
         "(__riscv_vsrl_vx_" + rawSuffix + "(" + low + ", " +
         std::to_string(physicalRole * lowBits) + ", " + vl + "), " +
         std::to_string((1u << lowBits) - 1) + ", " + vl + ");");
    std::string highPart = fresh("joined_high_bits");
    line(rawType + " " + highPart + " = __riscv_vsll_vx_" + rawSuffix +
         "(__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" + rawSuffix +
         "(" + high + ", " + std::to_string(logicalWidth) + ", " + vl +
         "), " + std::to_string((1u << (logicalWidth - lowBits)) - 1) + ", " +
         vl + "), " + std::to_string(lowBits) + ", " + vl + ");");
    line(raw + " = __riscv_vor_vv_" + rawSuffix + "(" + lowPart + ", " +
         highPart + ", " + vl + ");");
    --indent;
    line("}");
  } else {
    auto fragment =
        singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    std::string loaded = loadByte(fragment->byte, "packed_byte");
    if (logicalWidth <= 8) {
      raw = fresh("packed_value");
      line(rawType + " " + raw + " = __riscv_vand_vx_" + rawSuffix +
           "(__riscv_vsrl_vx_" + rawSuffix + "(" + loaded + ", " +
           fragment->shift + ", " + vl + "), " +
           std::to_string((1u << logicalWidth) - 1) + ", " + vl + ");");
    } else {
      raw = std::move(loaded);
    }
  }
  Binding resultBinding;
  resultBinding.kind = Binding::Kind::Vector;
  if (logicalWidth <= 8) {
    resultBinding.parts.push_back(raw);
    return resultBinding;
  }
  if (logicalWidth == 16 && kind == "natural") {
    auto fragment =
        singleStorageFragment(*field, logicalIndex, logicalWidth);
    if (!fragment)
      return {};
    const std::string u16Suffix = "u16" + lmulSpelling(rawLMUL * 2);
    const std::string u16Type = "vuint16" + lmulSpelling(rawLMUL * 2) + "_t";
    std::string high = fresh("word_high");
    line(rawType + " " + high + " = __riscv_vle8_v_" + rawSuffix + "(" +
         owner.recordPointer + " + (" + fragment->byte + " + 1) * " +
         std::to_string(owner.interleaveRows) + laneOffset + ", " + vl + ");");
    std::string low16 = fresh("word_low16");
    std::string high16 = fresh("word_high16");
    line(u16Type + " " + low16 + " = __riscv_vzext_vf2_" + u16Suffix + "(" +
         raw + ", " + vl + ");");
    line(u16Type + " " + high16 + " = __riscv_vsll_vx_" + u16Suffix + "(" +
         "__riscv_vzext_vf2_" + u16Suffix + "(" + high + ", " + vl +
         "), 8, " + vl + ");");
    std::string word = fresh("word");
    line(u16Type + " " + word + " = __riscv_vor_vv_" + u16Suffix + "(" +
         low16 + ", " + high16 + ", " + vl + ");");
    const std::string resultSuffix = vectorSuffix(result);
    if (resultSuffix == u16Suffix) {
      resultBinding.parts.push_back(std::move(word));
      return resultBinding;
    }
    std::string converted = fresh("field");
    line(vectorType(result) + " " + converted + " = __riscv_vreinterpret_v_" +
         u16Suffix + "_" + resultSuffix + "(" + word + ");");
    resultBinding.parts.push_back(std::move(converted));
    return resultBinding;
  }
  return {};
}

} // namespace weft::riscv_emission
