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

#ifndef GKO_CORE_SOLVER_UPPER_TRS_HPP_
#define GKO_CORE_SOLVER_UPPER_TRS_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>


namespace gko {
namespace solver {


template <typename ValueType, typename IndexType>
class UpperTrs;


/**
 * This struct is used to pass parameters to the
 * EnableDefaultUpperTrsFactory::generate() method. It is the
 * ComponentsType of UpperTrsFactory.
 *
 * @tparam ValueType  precision of matrix elements
 */
template <typename ValueType>
struct UpperTrsArgs {
    std::shared_ptr<const LinOp> system_matrix;
    std::shared_ptr<const matrix::Dense<ValueType>> b;


    UpperTrsArgs(std::shared_ptr<const LinOp> system_matrix,
                 std::shared_ptr<const matrix::Dense<ValueType>> b)
        : system_matrix{system_matrix}, b{b}
    {}
};


/**
 * Declares an Abstract Factory specialized for UpperTrs solver.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 */
template <typename ValueType, typename IndexType>
using UpperTrsFactory =
    AbstractFactory<UpperTrs<ValueType, IndexType>, UpperTrsArgs<ValueType>>;


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of UpperTrsFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteUpperTrs  the concrete UpperTrs type which this factory
 *                           produces, needs to have a constructor which takes
 *                           a const ConcreteFactory *, and a
 *                           const UpperTrsArgs * as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 */
template <typename ConcreteFactory, typename ConcreteUpperTrs,
          typename ParametersType, typename ValueType, typename IndexType>
using EnableDefaultUpperTrsFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteUpperTrs, ParametersType,
                         UpperTrsFactory<ValueType, IndexType>>;


/**
 * This macro will generate a default implementation of a UpperTrsFactory for
 * the UpperTrs subclass it is defined in.
 *
 * This macro is very similar to the macro #ENABLE_LIN_OP_FACTORY(). A more
 * detailed description of the use of these type of macros can be found there.
 *
 * @param _upper_trs  concrete operator for which the factory is to be created
 *                    [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup solvers
 */
#define GKO_ENABLE_UPPER_TRS_FACTORY(_upper_trs, _parameters_name,             \
                                     _factory_name)                            \
public:                                                                        \
    const _parameters_name##_type &get_##_parameters_name() const              \
    {                                                                          \
        return _parameters_name##_;                                            \
    }                                                                          \
                                                                               \
    class _factory_name : public ::gko::solver::EnableDefaultUpperTrsFactory<  \
                              _factory_name, _upper_trs,                       \
                              _parameters_name##_type, ValueType, IndexType> { \
        friend class ::gko::EnablePolymorphicObject<                           \
            _factory_name,                                                     \
            ::gko::solver::UpperTrsFactory<ValueType, IndexType>>;             \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,    \
                                                   _factory_name>;             \
        using ::gko::solver::EnableDefaultUpperTrsFactory<                     \
            _factory_name, _upper_trs, _parameters_name##_type, ValueType,     \
            IndexType>::EnableDefaultUpperTrsFactory;                          \
    };                                                                         \
    friend ::gko::solver::EnableDefaultUpperTrsFactory<                        \
        _factory_name, _upper_trs, _parameters_name##_type, ValueType,         \
        IndexType>;                                                            \
                                                                               \
private:                                                                       \
    _parameters_name##_type _parameters_name##_;                               \
                                                                               \
public:                                                                        \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


/**
 * UpperTrs is the triangular solver which solves the system U x = b, when U is
 * an upper triangular matrix. It works best when passing in a matrix in CSR
 * format. If the matrix is not in CSR, then the generate step converts it into
 * a CSR matrix. The generation fails if the matrix is not convertible to CSR.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indices
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class UpperTrs : public EnableLinOp<UpperTrs<ValueType, IndexType>>,
                 public Preconditionable {
    friend class EnableLinOp<UpperTrs>;
    friend class EnablePolymorphicObject<UpperTrs, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Gets the system operator (CSR matrix) of the linear system.
     *
     * @return the system operator (CSR matrix)
     */
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> get_system_matrix()
        const
    {
        return system_matrix_;
    }

    /**
     * Gets the right hand side of the linear system.
     *
     * @return the right hand side
     */
    std::shared_ptr<const matrix::Dense<ValueType>> get_rhs() const
    {
        return b_;
    }

    /**
     * Returns the preconditioner operator used by the solver.
     *
     * @return the preconditioner operator used by the solver
     */
    std::shared_ptr<const LinOp> get_preconditioner() const override
    {
        return preconditioner_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(
            preconditioner, nullptr);
    };
    GKO_ENABLE_UPPER_TRS_FACTORY(UpperTrs, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Generates the analysis structure from the system matrix and the right
     * hand side needed for the level solver.
     */
    void generate();

    explicit UpperTrs(std::shared_ptr<const Executor> exec)
        : EnableLinOp<UpperTrs>(std::move(exec))
    {}

    explicit UpperTrs(const Factory *factory,
                      const UpperTrsArgs<ValueType> &args)
        : parameters_{factory->get_parameters()},
          EnableLinOp<UpperTrs>(factory->get_executor(),
                                transpose(args.system_matrix->get_size())),
          b_{std::move(args.b)},
          system_matrix_{}
    {
        using CsrMatrix = matrix::Csr<ValueType, IndexType>;

        GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);
        // This is needed because it does not make sense to call the copy and
        // convert if the existing matrix is empty.
        const auto exec = this->get_executor();
        if (!args.system_matrix->get_size()) {
            system_matrix_ = CsrMatrix::create(exec);
        } else {
            system_matrix_ =
                copy_and_convert_to<CsrMatrix>(exec, args.system_matrix);
        }
        if (parameters_.preconditioner) {
            preconditioner_ =
                parameters_.preconditioner->generate(system_matrix_);
        } else {
            preconditioner_ = matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]);
        }
        this->generate();
    }

private:
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> system_matrix_{};
    std::shared_ptr<const matrix::Dense<ValueType>> b_{};
    std::shared_ptr<const LinOp> preconditioner_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_UPPER_TRS_HPP
