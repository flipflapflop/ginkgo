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

#include <ginkgo/core/preconditioner/ilu.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Ilu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<value_type>;
    using l_solver_type = gko::solver::Bicgstab<value_type>;
    using u_solver_type = gko::solver::Bicgstab<value_type>;
    using default_ilu_prec_type = gko::preconditioner::Ilu<>;
    using ilu_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, false>;
    using ilu_rev_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, true>;
    using composition = gko::Composition<value_type>;

    Ilu()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{2, 1, 1}, {2, 5, 2}, {2, 5, 5}}, exec)),
          l_factor(
              gko::initialize<Mtx>({{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}, exec)),
          u_factor(
              gko::initialize<Mtx>({{2, 1, 1}, {0, 4, 1}, {0, 0, 3}}, exec)),
          l_u_composition(composition::create(l_factor, u_factor)),
          l_factory(
              l_solver_type::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-15)
                          .on(exec))
                  .on(exec)),
          u_factory(
              u_solver_type::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-15)
                          .on(exec))
                  .on(exec)),
          ilu_pre_factory(ilu_prec_type::build()
                              .with_l_solver_factory(l_factory)
                              .with_u_solver_factory(u_factory)
                              .on(exec)),
          ilu_rev_pre_factory(ilu_rev_prec_type::build()
                                  .with_l_solver_factory(l_factory)
                                  .with_u_solver_factory(u_factory)
                                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> l_factor;
    std::shared_ptr<Mtx> u_factor;
    std::shared_ptr<composition> l_u_composition;
    std::shared_ptr<l_solver_type::Factory> l_factory;
    std::shared_ptr<u_solver_type::Factory> u_factory;
    std::shared_ptr<ilu_prec_type::Factory> ilu_pre_factory;
    std::shared_ptr<ilu_rev_prec_type::Factory> ilu_rev_pre_factory;
};


TEST_F(Ilu, BuildsDefaultWithoutThrowing)
{
    auto ilu_pre_default_factory = ilu_prec_type::build().on(exec);

    ASSERT_NO_THROW(ilu_pre_default_factory->generate(l_u_composition));
}


TEST_F(Ilu, BuildsCustomWithoutThrowing)
{
    ASSERT_NO_THROW(ilu_pre_factory->generate(l_u_composition));
}


TEST_F(Ilu, BuildsCustomWithoutThrowing2)
{
    ASSERT_NO_THROW(ilu_pre_factory->generate(mtx));
}


TEST_F(Ilu, ThrowOnWrongCompositionInput)
{
    std::shared_ptr<composition> composition = composition::create(l_factor);

    ASSERT_THROW(ilu_pre_factory->generate(composition), gko::NotSupported);
}


TEST_F(Ilu, ThrowOnWrongCompositionInput2)
{
    std::shared_ptr<composition> composition =
        composition::create(l_factor, u_factor, l_factor);

    ASSERT_THROW(ilu_pre_factory->generate(composition), gko::NotSupported);
}


TEST_F(Ilu, SetsCorrectMatrices)
{
    auto ilu = ilu_pre_factory->generate(l_u_composition);
    auto internal_l_factor = ilu->get_l_solver()->get_system_matrix();
    auto internal_u_factor = ilu->get_u_solver()->get_system_matrix();

    // These convert steps are required since `get_system_matrix` usually
    // just returns `LinOp`, which `GKO_ASSERT_MTX_NEAR` can not use properly
    std::unique_ptr<Mtx> converted_l_factor{Mtx::create(exec)};
    std::unique_ptr<Mtx> converted_u_factor{Mtx::create(exec)};
    gko::as<gko::ConvertibleTo<Mtx>>(internal_l_factor.get())
        ->convert_to(converted_l_factor.get());
    gko::as<gko::ConvertibleTo<Mtx>>(internal_u_factor.get())
        ->convert_to(converted_u_factor.get());
    GKO_ASSERT_MTX_NEAR(converted_l_factor, l_factor, 0);
    GKO_ASSERT_MTX_NEAR(converted_u_factor, u_factor, 0);
}


TEST_F(Ilu, CanBeCopied)
{
    auto ilu = ilu_pre_factory->generate(l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto u_l_composition = composition::create(u_factor, l_factor);
    auto copied =
        ilu_prec_type::build().on(exec)->generate(gko::share(u_l_composition));

    copied->copy_from(ilu.get());

    ASSERT_EQ(before_l_solver.get(), copied->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), copied->get_u_solver().get());
}


TEST_F(Ilu, CanBeMoved)
{
    auto ilu = ilu_pre_factory->generate(l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto u_l_composition = composition::create(u_factor, l_factor);
    auto moved =
        ilu_prec_type::build().on(exec)->generate(gko::share(u_l_composition));

    moved->copy_from(std::move(ilu));

    ASSERT_EQ(before_l_solver.get(), moved->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), moved->get_u_solver().get());
}


TEST_F(Ilu, CanBeCloned)
{
    auto ilu = ilu_pre_factory->generate(l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();

    auto clone = ilu->clone();

    ASSERT_EQ(before_l_solver.get(), clone->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), clone->get_u_solver().get());
}


TEST_F(Ilu, SolvesDefaultSingleRhs)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner =
        default_ilu_prec_type::build().on(exec)->generate(mtx);
    preconditioner->apply(b.get(), x.get());

    // Since it uses TRS per default, the result should be accurate
    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesCustomTypeDefaultFactorySingleRhs)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner = ilu_prec_type::build().on(exec)->generate(mtx);
    preconditioner->apply(b.get(), x.get());

    // Since it uses Bicgstab with default parmeters, the result will not be
    // accurate
    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-1);
}


TEST_F(Ilu, SolvesSingleRhsWithParIlu)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(exec);
    auto par_ilu = par_ilu_fact->generate(mtx);

    auto preconditioner = ilu_pre_factory->generate(gko::share(par_ilu));
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesSingleRhsWithComposition)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner = ilu_pre_factory->generate(l_u_composition);
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesSingleRhsWithMtx)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner = ilu_pre_factory->generate(mtx);
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesReverseSingleRhs)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());
    auto preconditioner = ilu_rev_pre_factory->generate(l_u_composition);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.625, 0.875, 1.75}), 1e-14);
}


TEST_F(Ilu, SolvesAdvancedSingleRhs)
{
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, exec);
    auto preconditioner = ilu_pre_factory->generate(l_u_composition);

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.0, 2.0, -1.0}), 1e-14);
}


TEST_F(Ilu, SolvesAdvancedReverseSingleRhs)
{
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, exec);
    auto preconditioner = ilu_rev_pre_factory->generate(l_u_composition);

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.75, 6.25, 1.5}), 1e-14);
}


TEST_F(Ilu, SolvesMultipleRhs)
{
    const auto b =
        gko::initialize<Mtx>({{1.0, 8.0}, {3.0, 21.0}, {6.0, 24.0}}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 2});
    x->copy_from(b.get());
    auto preconditioner = ilu_pre_factory->generate(l_u_composition);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        1e-14);
}


TEST_F(Ilu, SolvesDifferentNumberOfRhs)
{
    const auto b1 = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x11 = Mtx::create(exec, gko::dim<2>{3, 1});
    auto x12 = Mtx::create(exec, gko::dim<2>{3, 1});
    x11->copy_from(b1.get());
    x12->copy_from(b1.get());
    const auto b2 =
        gko::initialize<Mtx>({{1.0, 8.0}, {3.0, 21.0}, {6.0, 24.0}}, exec);
    auto x2 = Mtx::create(exec, gko::dim<2>{3, 2});
    x2->copy_from(b2.get());
    auto preconditioner = ilu_pre_factory->generate(l_u_composition);

    preconditioner->apply(b1.get(), x11.get());
    preconditioner->apply(b2.get(), x2.get());
    preconditioner->apply(b1.get(), x12.get());

    GKO_ASSERT_MTX_NEAR(x11.get(), l({-3.0, 2.0, 1.0}), 1e-14);
    GKO_ASSERT_MTX_NEAR(x2.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        1e-14);
    GKO_ASSERT_MTX_NEAR(x12.get(), x11.get(), 1e-14);
}


TEST_F(Ilu, CanBeUsedAsPreconditioner)
{
    auto solver =
        gko::solver::Bicgstab<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .with_preconditioner(default_ilu_prec_type::build().on(exec))
            .on(exec)
            ->generate(mtx);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    x->copy_from(b.get());

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, CanBeUsedAsGeneratedPreconditioner)
{
    std::shared_ptr<default_ilu_prec_type> precond =
        default_ilu_prec_type::build().on(exec)->generate(mtx);
    auto solver =
        gko::solver::Bicgstab<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .with_generated_preconditioner(precond)
            .on(exec)
            ->generate(mtx);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    x->copy_from(b.get());

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


}  // namespace
