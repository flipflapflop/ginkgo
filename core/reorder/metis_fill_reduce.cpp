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

#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/reorder/metis_fill_reduce_kernels.hpp"


namespace gko {
namespace reorder {
namespace metis_fill_reduce {


GKO_REGISTER_OPERATION(get_permutation, metis_fill_reduce::get_permutation);


}  // namespace metis_fill_reduce


template <typename ValueType, typename IndexType>
void MetisFillReduce<ValueType, IndexType>::generate()
{
    const auto num_rows = this->system_matrix_->get_size()[0];
    exec->run(metis_fill_reduce::make_get_permutation(
        num_rows, this->system_matrix_->get_const_row_ptrs(),
        this->system_matrix_->get_const_col_idxs(),
        this->vertex_weights_.get_data(), this->permutation_.get_data(),
        this->inv_permutation_.get_data()));
}


template <typename ValueType, typename IndexType>
void MetisFillReduce<ValueType, IndexType>::permute() const
{
    using Vector = matrix::Dense<ValueType>;
    const auto exec = this->get_executor();

    exec->run(metis_fill_reduce::make_permute(gko::lend(system_matrix_),
                                              dense_b, dense_x));
}

#define GKO_DECLARE_METIS_FILL_REDUCE(ValueType, IndexType) \
    class MetisFillReduce<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_METIS_FILL_REDUCE);


}  // namespace reorder
}  // namespace gko
