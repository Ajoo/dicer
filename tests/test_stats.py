import numpy as np
import pytest
from scipy.stats import norm, truncnorm, poisson, uniform
from scipy.integrate import quad

from dicer import domain as d
from dicer.stats import (
    FiniteOrderedSupportDistribution,
    ScipyContinuous,
    ScipyInteger,
    as_distribution,
    Delta,
    ReflectedDistribution,
    _deduplicate_support,
)


def test_finite_ordered_support_distribution_rvs():
    p = (FiniteOrderedSupportDistribution
         .from_support([0.1, 0.5, 0.3], [0.3, 0.2, 0.5])
    )
    x = p.rvs(1_000, random_state=42)
    i = p.domain.index(x)
    f = np.bincount(i)
    np.testing.assert_allclose(f / x.size, p.probabilities, atol=0.05)


def test_finite_ordered_support_distribution():
    p = (FiniteOrderedSupportDistribution
         .from_support([0.1, 0.5, 0.3], [0.3, 0.2, 0.5])
    )
    assert p.shape == tuple()

    x = np.array([0.02, 0.1, 0.25, 0.3, 1.0])
    pmf_expected = np.array([0., 0.3, 0.0, 0.5, 0.0])
    cdf_expected = np.array([0., 0.3, 0.3, 0.8, 1.0])
    
    np.testing.assert_array_equal(p.pdf(x), 0.)
    np.testing.assert_array_equal(p.pmf(x), pmf_expected)
    np.testing.assert_allclose(p.cdf(x), cdf_expected)


def test_finite_ordered_support_distribution_non_scalar():
    p = [
        [0.3, 0.2, 0.5],
        [0.1, 0.8, 0.1],
    ]
    p = (FiniteOrderedSupportDistribution
         .from_support([0.1, 0.5, 0.3], p)
    )
    assert p.shape == (2,)

    x = np.array([0.02, 0.1, 0.25, 0.3, 1.0])[:, None]
    pmf_expected = np.array([
        [0., 0.3, 0.0, 0.5, 0.0],
        [0., 0.1, 0.0, 0.1, 0.0]
    ]).T
    cdf_expected = np.array([
        [0., 0.3, 0.3, 0.8, 1.0],
        [0., 0.1, 0.1, 0.2, 1.0]
    ]).T
    
    np.testing.assert_array_equal(p.pdf(x), 0.)
    np.testing.assert_array_equal(p.pmf(x), pmf_expected)
    np.testing.assert_allclose(p.cdf(x), cdf_expected)


def test_integer_support_cdf_with_nonzero_lower_bound_and_scalar_input():
    p = FiniteOrderedSupportDistribution(
        d.IntegerColumn(10, 12),
        np.array([0.2, 0.3, 0.5]),
    )

    x = np.array([9.5, 10.0, 10.5, 11.9, 12.0, 99.0])
    np.testing.assert_allclose(p.cdf(x), [0.0, 0.2, 0.2, 0.5, 1.0, 1.0])
    assert p.cdf(10.5) == pytest.approx(0.2)
    assert p.pmf(11.0) == pytest.approx(0.3)


@pytest.mark.parametrize(
    'probabilities, match',
    [
        ([0.5, 0.4], 'sum to one'),
        ([1.1, -0.1], 'nonnegative'),
        ([0.5, np.nan], 'finite'),
        ([1.0], 'cardinality'),
    ],
)
def test_finite_support_distribution_rejects_invalid_probabilities(
    probabilities, match
):
    domain = d.FiniteOrderedColumn([0.0, 1.0])

    with pytest.raises(ValueError, match=match):
        FiniteOrderedSupportDistribution(domain, probabilities)


def test_as_distribution_wraps_scipy_and_existing_instances():
    frozen = norm(loc=1.0, scale=2.0)
    wrapped = as_distribution(frozen)
    assert isinstance(wrapped, ScipyContinuous)
    assert isinstance(wrapped.domain, d.FloatingPointColumn)

    wrapped_again = as_distribution(wrapped)
    assert isinstance(wrapped_again, ScipyContinuous)

    with pytest.raises(TypeError):
        as_distribution(object())


def test_batched_scipy_distribution_uses_scalar_enclosing_domain():
    frozen = uniform(
        loc=np.array([-2.0, 1.0]),
        scale=np.array([1.0, 3.0]),
    )

    wrapped = as_distribution(frozen)

    assert wrapped.domain == d.FloatingPointColumn(-2.0, 4.0)
    assert wrapped.support() == (-2.0, 4.0)

    # Zero lies inside the enclosing domain but outside both component
    # supports: above [-2, -1] and below [1, 4]. SciPy retains those
    # elementwise support semantics internally.
    np.testing.assert_array_equal(wrapped.pdf(0.0), [0.0, 0.0])
    np.testing.assert_array_equal(wrapped.cdf(0.0), [1.0, 0.0])


def test_reflected_distribution_full_api():
    source = uniform(loc=2.0, scale=3.0)
    reflected = ReflectedDistribution(source)

    assert reflected.support() == (-5.0, -2.0)
    assert reflected.shape == ()

    y = np.array([-4.5, -3.5, -2.5])
    np.testing.assert_allclose(reflected.pdf(y), source.pdf(-y))
    np.testing.assert_allclose(reflected.logpdf(y), source.logpdf(-y))
    np.testing.assert_allclose(reflected.cdf(y), source.sf(-y))
    np.testing.assert_allclose(reflected.logcdf(y), source.logsf(-y))
    np.testing.assert_allclose(reflected.sf(y), source.cdf(-y))
    np.testing.assert_allclose(reflected.logsf(y), source.logcdf(-y))
    np.testing.assert_array_equal(reflected.pmf(y), 0.0)

    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(reflected.ppf(q), -source.isf(q))
    np.testing.assert_allclose(reflected.isf(q), -source.ppf(q))

    np.testing.assert_allclose(reflected.mean(), -source.mean())
    np.testing.assert_allclose(reflected.var(), source.var())
    np.testing.assert_allclose(reflected.std(), source.std())
    np.testing.assert_allclose(reflected.moment(1), -source.moment(1))
    np.testing.assert_allclose(reflected.moment(2), source.moment(2))
    np.testing.assert_allclose(reflected.entropy(), source.entropy())
    np.testing.assert_allclose(reflected.expect(), -source.mean())

    mean, variance = reflected.stats(moments='mv')
    np.testing.assert_allclose(mean, -source.mean())
    np.testing.assert_allclose(variance, source.var())

    samples = reflected.rvs(size=20, random_state=42)
    expected = -as_distribution(source).rvs(size=20, random_state=42)
    np.testing.assert_allclose(samples, expected)


def test_distribution_base_methods_using_delta():
    domain = d.FiniteOrderedColumn([0.0])
    delta = Delta(domain=domain, value=0.0)

    x = np.array([0.0, 1.0])
    np.testing.assert_array_equal(delta.pdf(x), np.zeros_like(x, dtype=float))
    np.testing.assert_array_equal(delta.pmf(x), np.array([1.0, 0.0]))

    assert delta.pxf(0.0) == 1.0
    assert delta.logpmf(0.0) == 0.0
    assert delta.logpxf(0.0) == 0.0
    assert np.isneginf(delta.logpmf(1.0))
    assert np.isneginf(delta.logpxf(1.0))
    assert np.isneginf(delta.logpdf(1.0))
    assert delta.nnlf(np.array([0.0, 0.0])) == 0.0

    assert delta.sf(0.0) == 0.0
    assert delta.sf(-1.0) == 1.0
    assert delta.logsf(-1.0) == 0.0


def test_no_atoms_and_no_density_mixins_via_wrappers():
    continuous = as_distribution(norm())
    discrete = as_distribution(poisson(2.0))

    assert isinstance(continuous, ScipyContinuous)
    assert continuous.pdf(0.0) > 0.0
    assert continuous.pmf(0.0) == 0.0

    assert isinstance(discrete, ScipyInteger)
    assert discrete.pdf(0.0) == 0.0
    assert discrete.pmf(1) > 0.0

    samples_a = continuous.rvs(size=5, random_state=123)
    samples_b = continuous.rvs(size=5, random_state=123)
    np.testing.assert_allclose(samples_a, samples_b)


def test_deduplicate_support_merges_probabilities():
    support, probabilities = _deduplicate_support([2, 1, 2, 3], [0.3, 0.2, 0.1, 0.4])
    np.testing.assert_array_equal(support, np.array([1, 2, 3]))
    np.testing.assert_allclose(probabilities, np.array([0.2, 0.4, 0.4]))


# def test_ordered_discrete_distribution():
#     dist = OrderedDiscreteDistribution([3, 1, 3, 2], [0.4, 0.1, 0.2, 0.3])

#     assert isinstance(dist.domain, d.FiniteOrderedColumn)
#     np.testing.assert_array_equal(dist.domain.values, np.array([1, 2, 3]))

#     support = np.array([1, 2, 3, 4])
#     np.testing.assert_allclose(dist.pmf(support), np.array([0.1, 0.3, 0.6, 0.0]))
#     np.testing.assert_equal(dist.pdf(support), 0.)

#     test_points = np.array([1.0, 2.5, 3.1])
#     np.testing.assert_allclose(dist.pmf(test_points), np.array([0.1, 0.0, 0.0]))
#     np.testing.assert_equal(dist.pdf(test_points), 0.)

