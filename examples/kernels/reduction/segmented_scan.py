import weft
import weft.language as W


@weft.kernel
def segmented_inclusive_scan_f32(
    values: W.ptr[W.f32, W.readonly, W.noalias],
    segment_starts: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as position:
        value = W.load(values + position, other=W.f32(0.0))
        start = W.load(segment_starts + position, other=W.u8(0))
        boundary = start != W.u8(0)
        prefix = W.scan(
            value,
            op="add",
            identity=W.f32(0.0),
            inclusive=True,
            segment_start=boundary,
            order="ordered",
            acc_dtype=W.f32,
        )
        W.store(output + position, prefix)


@weft.kernel
def segmented_inclusive_scan_f32_equivalent(
    values: W.ptr[W.f32, W.readonly, W.noalias],
    segment_starts: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    value_base = values + begin
    start_base = segment_starts + begin
    output_base = output + begin
    extent = end - begin
    with W.vla(0, extent) as relative:
        value = W.load(value_base + relative, other=W.f32(0.0))
        start = W.load(start_base + relative, other=W.u8(0))
        boundary = W.u8(0) != start
        prefix = W.scan(
            value,
            op="add",
            identity=W.f32(0.0),
            inclusive=True,
            segment_start=boundary,
            order="ordered",
            acc_dtype=W.f32,
        )
        W.store(output_base + relative, prefix)
