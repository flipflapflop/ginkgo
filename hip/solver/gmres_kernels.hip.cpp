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

#include "core/solver/gmres_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


template <typename ValueType>
void initialize_1(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status,
                  size_type krylov_dim) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums,
                  size_type krylov_dim) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const HipExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace hip
}  // namespace kernels
}  // namespace gko
