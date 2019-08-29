/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/segment_scan.cuh"


namespace gko {
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * @ingroup cuda
 */
namespace cuda {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;


namespace {


/**
 * The device function of COO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_line  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param b  the input dense vector
 * @param c  the output dense vector
 * @param scale  the function on the added value
 */
template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmv_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const auto column_id = blockIdx.y;
    size_type num = (nnz > start) * ceildiv(nnz - start, subwarp_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    for (; ind < ind_end; ind += subwarp_size) {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
            if (is_first_in_segment) {
                atomic_add(&(c[curr_row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_end;
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        // segmented scan
        bool is_first_in_segment =
            segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp_val));
        }
    }
}

template <bool force, typename Group, typename ValueType, typename IndexType,
          typename Closure>
__device__ void warp_spmm(
    const Group tile_block, const size_type nnz,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride, IndexType *__restrict__ curr_row,
    ValueType *__restrict__ temp, const size_type offset,
    const size_type column_id, const size_type end_col, Closure scale)
{
    const auto tidx = threadIdx.x;
    auto coo_val =
        (offset + tidx < nnz) ? val[offset + tidx] : zero<ValueType>();
    const auto col_id = (column_id < end_col) ? column_id : end_col - 1;
    auto coo_col = (offset + tidx < nnz) ? col[offset + tidx] : col[nnz - 1];
    const int end = min(static_cast<int>(nnz - offset), 32);
    for (int j = 0; j < end - 1; j++) {
        *temp += tile_block.shfl(coo_val, j) *
                 b[tile_block.shfl(coo_col, j) * b_stride + col_id];
        if (tile_block.shfl(*curr_row, j) !=
            tile_block.shfl(*curr_row, j + 1)) {
            const auto temp_row = tile_block.shfl(*curr_row, j);
            if (column_id < end_col) {
                atomic_add(&(c[temp_row * c_stride + col_id]), scale(*temp));
            }
            *temp = zero<ValueType>();
        }
    }
    *temp += tile_block.shfl(coo_val, end - 1) *
             b[tile_block.shfl(coo_col, end - 1) * b_stride + col_id];

    if (force) {
        const auto temp_row = tile_block.shfl(*curr_row, end - 1);
        if (column_id < end_col) {
            atomic_add(&(c[temp_row * c_stride + col_id]), scale(*temp));
        }
    } else {
        const auto next_row =
            (offset + 32 + tidx < nnz) ? row[offset + 32 + tidx] : row[nnz - 1];
        if (tile_block.shfl(next_row, 0) !=
            tile_block.shfl(*curr_row, end - 1)) {
            const auto temp_row = tile_block.shfl(*curr_row, end - 1);
            if (column_id < end_col) {
                atomic_add(&(c[temp_row * c_stride + col_id]), scale(*temp));
            }
            *temp = zero<ValueType>();
        }
        *curr_row = next_row;
    }
}

template <typename ValueType, typename IndexType, typename Closure>
__device__ void spmm_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const size_type start_col, const size_type end_col,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp = zero<ValueType>();
    const auto coo_idx =
        (static_cast<size_type>(blockDim.y) * blockIdx.x + threadIdx.y) *
        num_lines * 32;
    const auto tidx = threadIdx.x;
    const auto column_id = start_col + tidx;
    if (coo_idx < nnz) {
        const int lines = min(static_cast<int>(ceildiv(nnz - coo_idx, 32)),
                              static_cast<int>(num_lines));
        const auto tile_block =
            group::tiled_partition<32>(group::this_thread_block());
        auto curr_row =
            (coo_idx + tidx < nnz) ? row[coo_idx + tidx] : row[nnz - 1];
        for (int i = 0; i < lines - 1; i++) {
            warp_spmm<false>(tile_block, nnz, val, col, row, b, b_stride, c,
                             c_stride, &curr_row, &temp, coo_idx + i * 32,
                             column_id, end_col, scale);
        }
        warp_spmm<true>(tile_block, nnz, val, col, row, b, b_stride, c,
                        c_stride, &curr_row, &temp, coo_idx + (lines - 1) * 32,
                        column_id, end_col, scale);
    }
}

// template <typename ValueType, typename IndexType, typename Closure>
// __device__ void spmm_kernel(const size_type nnz, const size_type num_lines,
//                             const ValueType *__restrict__ val,
//                             const IndexType *__restrict__ col,
//                             const IndexType *__restrict__ row,
//                             const size_type start_col, const size_type
//                             end_col, const ValueType *__restrict__ b, const
//                             size_type b_stride, ValueType *__restrict__ c,
//                             const size_type c_stride, Closure scale)
// {
//     ValueType temp = zero<ValueType>();
//     const auto coo_idx =
//         (static_cast<size_type>(blockDim.y) * blockIdx.x + threadIdx.y) *
//         num_lines;
//     const auto column_id = start_col + threadIdx.x;
//     const auto tidx = threadIdx.x;
//     auto coo_end = coo_idx + num_lines;
//     coo_end = (coo_end > nnz) ? nnz : coo_end;
//     const auto tile_block =
//         group::tiled_partition<32>(group::this_thread_block());
//     if (column_id < end_col && coo_idx < nnz) {
//         IndexType curr_row;
//         IndexType next_row;
//         ValueType coo_val;
//         IndexType coo_col;
//         const auto num_col = end_col - start_col;
//         curr_row = row[coo_idx + tidx];
//         for (auto idx = coo_idx; idx < coo_end - 1; idx++) {
//             const auto mod = (idx - coo_idx) % num_col;
//             if (mod == 0 && (idx + tidx) < coo_end) {
//                 coo_val = val[idx + tidx];
//                 coo_col = col[idx + tidx];
//             }
//             temp += tile_block.shfl(coo_val, mod) *
//                     b[tile_block.shfl(coo_col, mod) * b_stride + column_id];
//             if (mod == num_col - 1) {
//                 if ((idx + tidx + 1) < coo_end) {
//                     next_row = row[idx + 1 + tidx];
//                 }
//                 if (tile_block.shfl(next_row, 0) !=
//                     tile_block.shfl(curr_row, num_col - 1)) {
//                     atomic_add(&(c[tile_block.shfl(curr_row, mod) * c_stride
//                     +
//                                    column_id]),
//                                scale(temp));
//                     temp = zero<ValueType>();
//                 }
//                 curr_row = next_row;
//             } else {
//                 if (tile_block.shfl(curr_row, mod) !=
//                     tile_block.shfl(curr_row, mod + 1)) {
//                     atomic_add(&(c[tile_block.shfl(curr_row, mod) * c_stride
//                     +
//                                    column_id]),
//                                scale(temp));
//                     temp = zero<ValueType>();
//                 }
//             }
//         }
//         const auto idx = coo_end - 1;
//         const auto mod = (idx - coo_idx) % num_col;
//         if (mod == 0 && (idx + tidx) < coo_end) {
//             coo_val = val[idx + tidx];
//             coo_col = col[idx + tidx];
//         }
//         temp += tile_block.shfl(coo_val, mod) *
//                 b[tile_block.shfl(coo_col, mod) * b_stride + column_id];
//         atomic_add(&(c[tile_block.shfl(curr_row, mod) * c_stride +
//         column_id]),
//                    scale(temp));
//     }
// }


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const size_type num_cols,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    for (size_type i = 0; i < num_cols; i += 32) {
        spmm_kernel(nnz, num_lines, val, col, row, i, min(i + 32, num_cols), b,
                    b_stride, c, c_stride,
                    [](const ValueType &x) { return x; });
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    spmv_kernel(nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
                [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType &x) { return scale_factor * x; });
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void set_zero(
    const size_type nnz, ValueType *__restrict__ val)
{
    const auto ind =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ind < nnz) {
        val[ind] = zero<ValueType>();
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto nnz = c->get_num_stored_elements();
    const dim3 grid(ceildiv(nnz, default_block_size));
    const dim3 block(default_block_size);
    set_zero<<<grid, block>>>(nnz, as_cuda_type(c->get_values()));

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    if (nwarps > 0) {
        if (b->get_size()[1] == 1) {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            abstract_spmv<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(32, warps_in_block, 1);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block));
            abstract_spmm<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                b->get_size()[1], as_cuda_type(b->get_const_values()),
                b->get_stride(), as_cuda_type(c->get_values()),
                c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
        const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b->get_size()[1]);
        abstract_spmv<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);

namespace kernel {

template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_row_idxs_to_ptrs(
    const IndexType *__restrict__ idxs, size_type num_nonzeros,
    IndexType *__restrict__ ptrs, size_type length)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx == 0) {
        ptrs[0] = 0;
        ptrs[length - 1] = num_nonzeros;
    }

    if (0 < tidx && tidx < num_nonzeros) {
        if (idxs[tidx - 1] < idxs[tidx]) {
            for (auto i = idxs[tidx - 1] + 1; i <= idxs[tidx]; i++) {
                ptrs[i] = tidx;
            }
        }
    }
}

}  // namespace kernel


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs, size_type length)
{
    const auto grid_dim = ceildiv(num_nonzeros, default_block_size);

    kernel::convert_row_idxs_to_ptrs<<<grid_dim, default_block_size>>>(
        as_cuda_type(idxs), num_nonzeros, as_cuda_type(ptrs), length);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Coo<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    const auto nnz = result->get_num_stored_elements();

    const auto source_row_idxs = source->get_const_row_idxs();

    convert_row_idxs_to_ptrs(exec, source_row_idxs, nnz, row_ptrs,
                             num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_CSR_KERNEL);


namespace kernel {


template <typename ValueType>
__global__
    __launch_bounds__(cuda_config::max_block_size) void initialize_zero_dense(
        size_type num_rows, size_type num_cols, size_type stride,
        ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type nnz, const IndexType *__restrict__ row_idxs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < nnz) {
        result[stride * row_idxs[tidx] + col_idxs[tidx]] = values[tidx];
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Coo<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(result->get_values()));

    const auto grid_dim = ceildiv(nnz, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        nnz, as_cuda_type(source->get_const_row_idxs()),
        as_cuda_type(source->get_const_col_idxs()),
        as_cuda_type(source->get_const_values()), stride,
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
