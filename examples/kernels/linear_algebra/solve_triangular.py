import weft
import weft.language as W


@weft.kernel
def solve_lower_triangular_f32(
    matrix: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    solution: W.ptr[W.f32, W.noalias],
    batches: W.index,
    rows: W.index,
    rhs_columns: W.index,
) -> None:
    matrix_batch_stride = rows * rows
    rhs_batch_stride = rows * rhs_columns

    for batch in W.range(0, batches):
        matrix_batch = matrix + batch * matrix_batch_stride
        rhs_batch = rhs + batch * rhs_batch_stride
        solution_batch = solution + batch * rhs_batch_stride
        for row in W.range(0, rows):
            with W.vla(0, rhs_columns) as column:
                inner = W.axis(row)
                diagonal = W.load(matrix_batch + row * rows + row)
                coefficient = -W.load(matrix_batch + row * rows + inner) / diagonal
                previous = W.load(
                    solution_batch
                    + inner[None, :] * rhs_columns
                    + column[:, None]
                )
                result = W.vdot(
                    previous,
                    coefficient,
                    init=W.load(rhs_batch + row * rhs_columns + column)
                    / diagonal,
                    acc_dtype=W.f32,
                    order="relaxed",
                    math="native",
                )
                W.store(
                    solution_batch + row * rhs_columns + column,
                    result,
                )
