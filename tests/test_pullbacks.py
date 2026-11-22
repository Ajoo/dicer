import numpy as np
import pytest
from scipy.stats import norm, truncnorm, uniform
from scipy.integrate import trapezoid

from dicer import domain as d
from dicer.stats import FiniteOrderedSupportDistribution, as_distribution
from dicer.transform._base import (
    ConfigurableSeriesTransformation,
    SeriesTransformation,
)
from dicer.transform.continuous import (
    AtomicScaledPushedForwardDistribution,
    ClippedPushedForwardDistribution,
    DiscretizedPushedForwardDistribution,
    ExplodedPushedForwardDistribution,
    GeneralizedPiecewiseAffine,
    RoundedPushedForwardDistribution,
    Round,
    Digitize,
    UnclippedPushedForwardDistribution,
    UniformDequantizedPushedForwardDistribution,
)


# ===========================================================================
# ClippedPushedForwardDistribution
# ===========================================================================

def test_clip_pushforward_has_boundary_atoms_and_interior_density():
    output_domain, clip = (
        d.FloatingPointColumn()
        .transform.clip(-1.0, 1.0)
        .unpack()
    )
    assert output_domain == d.FloatingPointColumn(-1.0, 1.0)
    assert clip.inverse is None

    clipped = clip.pushforward_distribution(norm())
    assert isinstance(clipped, ClippedPushedForwardDistribution)
    with pytest.raises(NotImplementedError, match='requires an inverse'):
        clip.pullback_distribution(clipped)

    y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        clipped.pdf(y),
        [0.0, norm.pdf(-1.0), norm.pdf(0.0), norm.pdf(1.0), 0.0],
    )
    np.testing.assert_allclose(
        clipped.pmf(y),
        [0.0, norm.cdf(-1.0), 0.0, norm.sf(1.0), 0.0],
    )
    np.testing.assert_allclose(
        clipped.cdf(y),
        [0.0, norm.cdf(-1.0), 0.5, 1.0, 1.0],
    )

    q = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(clipped.ppf(q), np.clip(norm.ppf(q), -1.0, 1.0))


def test_clip_pushforward_preserves_and_collapses_existing_atoms():
    source = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([0.1, 0.2, 0.3, 0.15, 0.25]),
    )
    _, clip = d.FloatingPointColumn().transform.clip(-1.0, 1.0).unpack()

    clipped = clip.pushforward_distribution(source)

    np.testing.assert_allclose(clipped.pmf([-1.0, 0.0, 1.0]), [0.3, 0.3, 0.4])
    np.testing.assert_allclose(clipped.cdf([-1.0, 0.0, 1.0]), [0.3, 0.6, 1.0])


def test_one_sided_and_degenerate_clip_pushforwards():
    _, lower_clip = d.FloatingPointColumn().transform.clip(lb=0.0).unpack()
    lower_clipped = lower_clip.pushforward_distribution(norm())
    assert lower_clipped.pmf(0.0) == pytest.approx(0.5)
    assert lower_clipped.cdf(0.0) == pytest.approx(0.5)

    output_domain, point_clip = (
        d.FloatingPointColumn()
        .transform.clip(0.0, 0.0)
        .unpack()
    )
    assert output_domain == d.FloatingPointColumn(0.0, 0.0)
    point_mass = point_clip.pushforward_distribution(norm())
    np.testing.assert_array_equal(point_mass.pmf([-1.0, 0.0, 1.0]), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(point_mass.cdf([-1.0, 0.0, 1.0]), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(point_mass.ppf([0.0, 0.5, 1.0]), 0.0)


def test_clip_rejects_reversed_bounds():
    with pytest.raises(ValueError, match='must not exceed'):
        d.FloatingPointColumn().transform.clip(1.0, -1.0)


def test_clip_pullback_reconstructs_continuous_tails():
    left_dist = truncnorm(-np.inf, -1.0)
    right_dist = truncnorm(1.0, np.inf)
    _, clip = (
        d.FloatingPointColumn()
        .transform.clip(
            -1.0,
            1.0,
            left_dist=left_dist,
            right_dist=right_dist,
        )
        .unpack()
    )
    assert clip.inverse.left_dist is left_dist
    assert clip.inverse.right_dist is right_dist

    clipped = clip.pushforward_distribution(norm())
    reconstructed = clip.pullback_distribution(clipped)
    assert isinstance(reconstructed, UnclippedPushedForwardDistribution)

    x = np.linspace(-3.0, 3.0, 13)
    np.testing.assert_allclose(reconstructed.pdf(x), norm.pdf(x))
    np.testing.assert_allclose(reconstructed.pmf(x), 0.0)
    np.testing.assert_allclose(reconstructed.cdf(x), norm.cdf(x))

    q = np.linspace(0.01, 0.99, 11)
    np.testing.assert_allclose(reconstructed.ppf(q), norm.ppf(q))

    samples = clip.inverse(np.array([-1, 0, 1]), random_state=42)
    assert np.issubdtype(samples.dtype, np.floating)
    assert samples[0] <= -1.0
    assert samples[1] == 0.0
    assert samples[2] >= 1.0


def test_clip_pullback_reconstructs_atomic_tails():
    source = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([0.1, 0.2, 0.3, 0.15, 0.25]),
    )
    left_dist = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-2.0, -1.0]),
        np.array([1.0 / 3.0, 2.0 / 3.0]),
    )
    right_dist = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([1.0, 2.0]),
        np.array([0.375, 0.625]),
    )
    _, clip = (
        d.FloatingPointColumn(-2.0, 2.0)
        .transform.clip(
            -1.0,
            1.0,
            left_dist=left_dist,
            right_dist=right_dist,
        )
        .unpack()
    )
    assert clip.inverse.left_dist is left_dist
    assert clip.inverse.right_dist is right_dist

    reconstructed = clip.pullback_distribution(
        clip.pushforward_distribution(source)
    )
    x = source.domain.values
    np.testing.assert_allclose(reconstructed.pdf(x), 0.0)
    np.testing.assert_allclose(reconstructed.pmf(x), source.pmf(x))
    np.testing.assert_allclose(reconstructed.cdf(x), source.cdf(x))

    q = np.array([0.05, 0.2, 0.45, 0.7, 0.9])
    np.testing.assert_array_equal(reconstructed.ppf(q), source.ppf(q))


def test_clip_rejects_reconstruction_distribution_crossing_bound():
    with pytest.raises(ValueError, match='left reconstruction distribution'):
        d.FloatingPointColumn().transform.clip(
            lb=-1.0,
            left_dist=uniform(-2.0, 2.0),
        )


def test_clip_requires_scalar_distribution_for_reconstruction():
    with pytest.raises(ValueError, match='must be scalar'):
        d.FloatingPointColumn().transform.clip(
            lb=-1.0,
            left_dist=uniform(loc=np.array([-3.0, -2.0]), scale=0.5),
        )


# ===========================================================================
# Fixed-point distribution mappings
# ===========================================================================

def test_round_to_fixed_point_has_lazy_atomic_pushforward():
    output_domain, transform = (
        d.FloatingPointColumn()
        .transform.round(2)
        .unpack()
    )
    assert output_domain == d.FixedPointColumn(2)

    rounded = transform.pushforward_distribution(norm())
    assert isinstance(rounded, RoundedPushedForwardDistribution)

    values = np.array([-0.01, 0.0, 0.01])
    expected_mass = norm.cdf(values + 0.005) - norm.cdf(values - 0.005)
    np.testing.assert_allclose(rounded.pdf(values), 0.0)
    np.testing.assert_allclose(rounded.pmf(values), expected_mass)
    assert rounded.pmf(0.001) == 0.0
    assert rounded.cdf(0.001) == pytest.approx(norm.cdf(0.005))

    q = np.array([0.1, 0.5, 0.9])
    expected_quantiles = np.ceil(norm.ppf(q) / 0.01 - 0.5) * 0.01
    np.testing.assert_allclose(rounded.ppf(q), expected_quantiles)

    samples = rounded.rvs(size=100, random_state=42)
    assert d.FixedPointColumn(2).check_decimals(samples)


def test_unbounded_zero_digit_round_has_lazy_atomic_pushforward():
    _, transform = d.FloatingPointColumn().transform.round().unpack()
    rounded = transform.pushforward_distribution(norm())

    assert isinstance(rounded, RoundedPushedForwardDistribution)
    assert rounded.domain == d.FixedPointColumn(0)
    values = np.array([-1.0, 0.0, 1.0])
    expected_mass = norm.cdf(values + 0.5) - norm.cdf(values - 0.5)
    np.testing.assert_allclose(rounded.pmf(values), expected_mass)
    assert rounded.pmf(0.25) == 0.0


def test_round_fixed_point_boundary_atoms_follow_ties_to_even():
    source = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([0.005, 0.015]),
        np.array([0.4, 0.6]),
    )
    transform = Round(d.FloatingPointColumn(0.0, 0.02), decimals=2)

    rounded = transform.pushforward_distribution(source)

    np.testing.assert_allclose(rounded.pmf([0.0, 0.01, 0.02]), [0.4, 0.0, 0.6])
    np.testing.assert_allclose(rounded.cdf([0.0, 0.01, 0.02]), [0.4, 0.4, 1.0])


def test_round_pullback_uniformly_dequantizes_atomic_distribution():
    _, transform = d.FloatingPointColumn().transform.round(2).unpack()
    rounded = transform.pushforward_distribution(norm())

    reconstructed = transform.pullback_distribution(rounded)
    assert isinstance(reconstructed, UniformDequantizedPushedForwardDistribution)
    np.testing.assert_allclose(reconstructed.pmf([-0.01, 0.0, 0.01]), 0.0)

    central_mass = norm.cdf(0.005) - norm.cdf(-0.005)
    assert reconstructed.pdf(0.0) == pytest.approx(central_mass / 0.01)
    np.testing.assert_allclose(
        reconstructed.cdf([-0.005, 0.005]),
        norm.cdf([-0.005, 0.005]),
    )
    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(reconstructed.cdf(reconstructed.ppf(q)), q)


def test_to_int_pushforward_and_atomic_pullback():
    _, transform = (
        d.FixedPointColumn(2, -1.0, 1.0)
        .transform.to_int()
        .unpack()
    )
    assert transform.is_lossy
    assert not transform.inverse.is_lossy

    integer = transform.pushforward_distribution(uniform(-1.0, 2.0))
    assert isinstance(integer, RoundedPushedForwardDistribution)
    np.testing.assert_allclose(integer.pmf([-1, 0, 1]), 0.005)

    source = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-1, 0, 2]),
        np.array([0.2, 0.3, 0.5]),
    )
    fixed = transform.pullback_distribution(source)
    assert isinstance(fixed, AtomicScaledPushedForwardDistribution)
    np.testing.assert_allclose(fixed.pdf([-0.01, 0.0, 0.02]), 0.0)
    np.testing.assert_allclose(
        fixed.pmf([-0.01, 0.0, 0.005, 0.02]),
        [0.2, 0.3, 0.0, 0.5],
    )
    np.testing.assert_allclose(fixed.cdf([-0.01, 0.0, 0.01, 0.02]), [0.2, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(fixed.ppf([0.1, 0.4, 0.8]), [-0.01, 0.0, 0.02])


def test_affine_on_fixed_point_propagates_continuous_and_atomic_measures():
    domain = d.FixedPointColumn(2, -1.0, 1.0)
    _, transform = domain.transform.affine(2.0, 1.0).unpack()
    assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
    raw_inverse = transform.inverse.resolve(fixed_point=False)
    lattice_inverse = transform.inverse.resolve(fixed_point=True)
    assert isinstance(lattice_inverse, SeriesTransformation)
    assert lattice_inverse.transformations[0] is raw_inverse
    assert isinstance(lattice_inverse.transformations[-1], Round)
    assert raw_inverse.inverse is transform
    assert not hasattr(transform, '_mathematical_inverse')
    assert transform.inverse.output_domain == domain
    assert transform.inverse.output_domain_for(
        fixed_point=False
    ) == d.FloatingPointColumn(-1.0, 1.0)

    y = np.array([0.004, 1.006])
    np.testing.assert_allclose(transform.inverse(y), [-0.5, 0.0])
    np.testing.assert_allclose(
        transform.inverse(y, fixed_point=False),
        [-0.498, 0.003],
    )

    continuous = transform.pushforward_distribution(uniform(-1.0, 2.0))
    output_values = np.array([-1.0, 1.0, 3.0])
    np.testing.assert_allclose(continuous.pdf(output_values), 0.25)
    np.testing.assert_allclose(continuous.pmf(output_values), 0.0)
    np.testing.assert_allclose(
        continuous.cdf(np.array([-0.01, 0.01])),
        np.array([0.2475, 0.2525]),
    )
    reconstructed = transform.pullback_distribution(continuous)
    assert reconstructed.domain == domain
    np.testing.assert_allclose(reconstructed.pdf([-0.5, 0.0, 0.5]), 0.0)
    np.testing.assert_allclose(reconstructed.pmf([-0.5, 0.0, 0.5]), 0.005)

    continuous_reconstruction = transform.pullback_distribution(
        continuous,
        fixed_point=False,
    )
    assert continuous_reconstruction.domain == d.FloatingPointColumn(-1.0, 1.0)
    x = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        continuous_reconstruction.pdf(x),
        0.5,
    )
    np.testing.assert_allclose(
        continuous_reconstruction.pmf(x),
        0.0,
    )

    pushed_through_inverse = transform.inverse.pushforward_distribution(continuous)
    np.testing.assert_allclose(
        pushed_through_inverse.pmf([-0.5, 0.0, 0.5]),
        reconstructed.pmf([-0.5, 0.0, 0.5]),
    )
    continuous_through_inverse = transform.inverse.pushforward_distribution(
        continuous,
        fixed_point=False,
    )
    np.testing.assert_allclose(
        continuous_through_inverse.pdf(x),
        continuous_reconstruction.pdf(x),
    )

    rounded = Round(d.FloatingPointColumn(-1.0, 1.0), 2).pushforward_distribution(
        uniform(-1.0, 2.0)
    )
    mapped_atoms = transform.pushforward_distribution(rounded)
    assert mapped_atoms.pdf(1.0) == 0.0
    assert mapped_atoms.pmf(1.0) == pytest.approx(rounded.pmf(0.0))
    pulled_atoms = transform.pullback_distribution(mapped_atoms)
    assert pulled_atoms.pmf(0.0) == pytest.approx(rounded.pmf(0.0))


def test_piecewise_affine_inverse_has_optional_fixed_point_projection():
    domain = d.FixedPointColumn(2, -1.0, 1.0)
    _, transform = domain.transform.piecewise_affine(
        np.array([-1.0, 0.0, 1.0]),
        np.array([-2.0, 0.0, 3.0]),
    ).unpack()

    assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
    raw_inverse = transform.inverse.resolve(fixed_point=False)
    lattice_inverse = transform.inverse.resolve(fixed_point=True)
    assert isinstance(lattice_inverse, SeriesTransformation)
    assert lattice_inverse.transformations[0] is raw_inverse
    assert isinstance(lattice_inverse.transformations[-1], Round)
    assert raw_inverse.inverse is transform
    assert transform.inverse.output_domain == domain

    output_distribution = uniform(-2.0, 5.0)
    pulled = transform.pullback_distribution(output_distribution)
    pushed_through_inverse = transform.inverse.pushforward_distribution(
        output_distribution
    )
    x = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(pulled.pdf(x), 0.0)
    np.testing.assert_allclose(
        pulled.pmf(x),
        pushed_through_inverse.pmf(x),
    )
    continuous_pulled = transform.pullback_distribution(
        output_distribution,
        fixed_point=False,
    )
    assert continuous_pulled.domain == d.FloatingPointColumn(-1.0, 1.0)
    np.testing.assert_allclose(continuous_pulled.pdf(x), [0.4, 0.6, 0.6])


def test_fixed_point_option_propagates_through_series():
    _, transform = (
        d.FixedPointColumn(2, 0.0, 1.0)
        .transform.affine(2.0, 1.0)
        .transform.exp()
        .unpack()
    )
    output_distribution = uniform(2.0, 18.0)

    lattice = transform.pullback_distribution(output_distribution)
    continuous = transform.pullback_distribution(
        output_distribution,
        fixed_point=False,
    )

    assert lattice.domain == d.FixedPointColumn(2, 0.0, 1.0)
    assert continuous.domain == d.FloatingPointColumn(0.0, 1.0)
    assert lattice.pdf(0.5) == 0.0
    assert lattice.pmf(0.5) > 0.0
    assert continuous.pdf(0.5) > 0.0
    assert continuous.pmf(0.5) == 0.0


def test_fixed_point_inverse_requires_boolean_fixed_point_option():
    _, transform = (
        d.FixedPointColumn(2, -1.0, 1.0)
        .transform.affine(2.0, 1.0)
        .unpack()
    )

    with pytest.raises(TypeError, match='fixed_point must be a boolean'):
        transform.inverse(0.0, fixed_point='yes')


# ===========================================================================
# Regression tests – pullback CDF correctness
# ===========================================================================

def test_log_pullback():
    size = 6
    _, t = d.FloatingPointColumn(0, np.inf).transform.log().unpack()

    py = norm(0, np.logspace(-1, 1, size))
    px = t.pullback_distribution(py)

    assert px.shape == (size,)

    quantile_values = np.linspace(0, 1, 11)[:, None]
    py_deciles = py.ppf(quantile_values)

    px_deciles = t.inverse(py_deciles)
    assert px_deciles.shape == (11, size)

    actual, desired = np.broadcast_arrays(px.cdf(px_deciles), quantile_values)
    np.testing.assert_allclose(actual, desired)


def test_exp_pullback():
    size = 6
    _, t = d.FloatingPointColumn().transform.exp().unpack()

    scale = np.logspace(-1, 0, size)
    py = truncnorm(0., np.inf, loc=0, scale=scale)
    px = t.pullback_distribution(py)

    assert px.shape == (size,)

    quantile_values = np.linspace(0, 1, 11)[:, None]
    py_deciles = py.ppf(quantile_values)

    px_deciles = t.inverse(py_deciles)
    assert px_deciles.shape == (11, size)

    actual, desired = np.broadcast_arrays(px.cdf(px_deciles), quantile_values)
    np.testing.assert_allclose(actual, desired)


def test_logit_pullback():
    size = 6
    _, t = d.FloatingPointColumn(0, 1).transform.logit().unpack()

    py = norm(0, np.logspace(-1, 1, size))
    px = t.pullback_distribution(py)

    assert px.shape == (size,)

    quantile_values = np.linspace(0, 1, 11)[:, None]
    py_deciles = py.ppf(quantile_values)

    px_deciles = t.inverse(py_deciles)
    assert px_deciles.shape == (11, size)

    actual, desired = np.broadcast_arrays(px.cdf(px_deciles), quantile_values)
    np.testing.assert_allclose(actual, desired)


def test_expit_pullback():
    size = 6
    _, t = d.FloatingPointColumn().transform.expit().unpack()

    scale = np.logspace(-1, 1, size)
    py = truncnorm(-1/scale/2, 1/scale/2, loc=0.5, scale=scale)
    px = t.pullback_distribution(py)

    assert px.shape == (size,)

    quantile_values = np.linspace(0, 1, 11)[:, None]
    py_deciles = py.ppf(quantile_values)

    px_deciles = t.inverse(py_deciles)
    assert px_deciles.shape == (11, size)

    actual, desired = np.broadcast_arrays(px.cdf(px_deciles), quantile_values)
    np.testing.assert_allclose(actual, desired)


def test_digitize_pushforward():
    bins = [-3, -2.,-1.,0., 3]
    left_tail = truncnorm(-np.inf, -3)
    right_tail = truncnorm(3, np.inf)

    _, t = (d.FloatingPointColumn()
        .transform.digitize(
            bins,
            left_tail=left_tail,
            right_tail=right_tail,
            )
        .unpack())

    px = norm(np.linspace(-5, 5, 11))
    py = t.pushforward_distribution(px)

    bins = np.r_[-np.inf, np.asarray(bins), np.inf]
    py_expected = np.diff(px.cdf(bins[:,None]), axis=0)
    np.testing.assert_allclose(py.probabilities.T, py_expected)


def test_digitize_pulback():
    #size = 6
    bins = [-3, -2.,-1.,0., 3]
    left_tail = truncnorm(-np.inf, -3)
    right_tail = truncnorm(3, np.inf)

    _, t = (d.FloatingPointColumn()
        .transform.digitize(
            bins,
            left_tail=left_tail,
            right_tail=right_tail,
            )
        .unpack())

    k = len(bins) + 1
    py = FiniteOrderedSupportDistribution(
        d.IntegerColumn(0, k-1),
        np.ones(k)/k
        )
    px = t.pullback_distribution(py)

    assert px.shape == tuple()

    quantile_values = np.linspace(0, 1, 11)
    py_deciles = py.ppf(quantile_values)

    Pminus = py.cdf(py_deciles - 1)
    Pplus = py.cdf(py_deciles)

    assert np.all(Pminus <= quantile_values)
    assert np.all(Pplus >= quantile_values - 1e-9)


# ===========================================================================
# InvertiblePushedForwardDistribution
# ===========================================================================

class TestInvertiblePushedForwardDistribution:
    """Full API coverage for InvertiblePushedForwardDistribution.

    Setup: Y = exp(X) where X ~ N(0,1), so Y ~ LogNormal(0,1).
    Change-of-variables: f_Y(y) = f_X(log y) / y  for y > 0.
    """

    @pytest.fixture
    def setup(self):
        _, t = d.FloatingPointColumn().transform.exp().unpack()
        px = as_distribution(norm(0, 1))
        py = t.pushforward_distribution(px)
        return px, t, py

    # --- shape ---

    def test_shape(self, setup):
        px, t, py = setup
        assert py.shape == ()

    # --- pdf / logpdf ---

    def test_pdf_change_of_variables(self, setup):
        """f_Y(y) = f_X(log y) / y  (log-normal density)."""
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0, np.e])
        expected = norm.pdf(np.log(y)) / y
        np.testing.assert_allclose(py.pdf(y), expected, rtol=1e-12)

    def test_logpdf_equals_log_pdf(self, setup):
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(py.logpdf(y), np.log(py.pdf(y)), rtol=1e-12)

    def test_pdf_integrates_to_one(self, setup):
        """Density integrates to 1 over the positive reals."""
        px, t, py = setup
        # Integrate to 30 to capture enough of the lognormal tail
        # (norm.sf(log(30)) ≈ 3e-4, so truncation error is negligible)
        y = np.linspace(0.001, 30.0, 50_000)
        integral = trapezoid(py.pdf(y), y)
        np.testing.assert_allclose(integral, 1.0, atol=1e-3)

    # --- pmf / logpmf ---

    def test_pmf_is_zero_for_continuous_input(self, setup):
        """Continuous pushforward has no atoms."""
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0])
        np.testing.assert_array_equal(py.pmf(y), np.zeros(3))

    def test_logpmf_is_neginf_for_continuous_input(self, setup):
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0])
        assert np.all(np.isneginf(py.logpmf(y)))

    # --- pxf / logpxf ---

    def test_pxf_equals_pdf_for_continuous(self, setup):
        """For a purely continuous distribution pxf = pdf (pmf = 0)."""
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(py.pxf(y), py.pdf(y), rtol=1e-12)

    def test_logpxf_equals_logpdf_for_continuous(self, setup):
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(py.logpxf(y), py.logpdf(y), rtol=1e-12)

    # --- cdf / logcdf ---

    def test_cdf_change_of_variables(self, setup):
        """F_Y(y) = F_X(log y)."""
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0, np.e])
        expected = norm.cdf(np.log(y))
        np.testing.assert_allclose(py.cdf(y), expected, rtol=1e-12)

    def test_logcdf_equals_log_cdf(self, setup):
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(py.logcdf(y), np.log(py.cdf(y)), rtol=1e-12)

    def test_cdf_is_monotone_and_bounded(self, setup):
        px, t, py = setup
        y = np.linspace(0.1, 5.0, 50)
        cdf = py.cdf(y)
        assert np.all(np.diff(cdf) >= 0)
        assert np.all((cdf >= 0) & (cdf <= 1))

    # --- rvs ---

    def test_rvs_are_positive(self, setup):
        """exp(x) > 0 for all x."""
        px, t, py = setup
        samples = py.rvs(size=200, random_state=0)
        assert np.all(samples > 0)

    def test_rvs_cdf_calibration(self, setup):
        """Empirical CDF of rvs matches theoretical CDF."""
        px, t, py = setup
        samples = np.sort(py.rvs(size=5_000, random_state=1))
        empirical = np.arange(1, len(samples) + 1) / len(samples)
        theoretical = py.cdf(samples)
        np.testing.assert_allclose(empirical, theoretical, atol=0.03)

    def test_sf_change_of_variables(self, setup):
        px, t, py = setup
        y = np.array([0.5, 1.0, 2.0, np.e])
        expected = 1 - norm.cdf(np.log(y))
        np.testing.assert_allclose(py.sf(y), expected, rtol=1e-12)

    def test_support(self, setup):
        px, t, py = setup
        assert py.support() == (0, np.inf)

    # --- ppf / isf / median / interval ---

    def test_ppf_inverts_cdf(self, setup):
        """cdf(ppf(q)) == q (fundamental PPF property)."""
        px, t, py = setup
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_allclose(py.cdf(py.ppf(q)), q, rtol=1e-6)

    def test_isf_equals_ppf_complement(self, setup):
        """isf(q) == ppf(1 - q) via OrderedDistribution."""
        px, t, py = setup
        q = np.array([0.1, 0.25, 0.5])
        # assert_array_equal treats NaN == NaN, which is needed here because
        # ppf currently returns NaN for inputs in (0, 1)
        np.testing.assert_array_equal(py.isf(q), py.ppf(1 - q))

    def test_median_equals_ppf_half(self, setup):
        """median() == ppf(0.5) via OrderedDistribution."""
        px, t, py = setup
        np.testing.assert_array_equal(py.median(), py.ppf(0.5))

    def test_interval_endpoints_are_ppf(self, setup):
        """interval(alpha) returns (ppf((1-alpha)/2), ppf((1+alpha)/2))."""
        px, t, py = setup
        alpha = 0.9
        lo, hi = py.interval(alpha)
        np.testing.assert_array_equal(lo, py.ppf((1 - alpha) / 2))
        np.testing.assert_array_equal(hi, py.ppf((1 + alpha) / 2))

    # --- methods with no implementation (not on Distribution base class) ---

    def test_mean_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.mean()

    def test_var_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.var()

    def test_std_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.std()

    def test_entropy_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.entropy()

    def test_moment_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.moment(1)

    def test_expect_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.expect(lambda y: y)

    def test_stats_not_implemented(self, setup):
        px, t, py = setup
        with pytest.raises(AttributeError):
            py.stats()


class TestDecreasingInvertiblePushedForwardDistribution:

    def test_continuous_cdf_and_ppf(self):
        _, transformation = (
            d.FloatingPointColumn()
            .transform.affine(-1.0, 0.0)
            .unpack()
        )
        pushed_forward = transformation.pushforward_distribution(norm())

        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(pushed_forward.cdf(y), norm.cdf(y))

        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_allclose(pushed_forward.ppf(q), norm.ppf(q))

    def test_atomic_cdf_and_ppf_include_jump_boundaries(self):
        source = FiniteOrderedSupportDistribution(
            d.FiniteOrderedColumn([0.0, 1.0, 2.0]),
            np.array([0.2, 0.3, 0.5]),
        )
        _, transformation = (
            d.FloatingPointColumn(0.0, 2.0)
            .transform.affine(-1.0, 0.0)
            .unpack()
        )
        pushed_forward = transformation.pushforward_distribution(source)

        y = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        expected_cdf = np.array([0.0, 0.5, 0.8, 1.0, 1.0])
        np.testing.assert_allclose(pushed_forward.cdf(y), expected_cdf)

        q = np.array([0.1, 0.5, 0.6, 0.8, 0.9])
        expected_ppf = np.array([-2.0, -2.0, -1.0, -1.0, 0.0])
        np.testing.assert_array_equal(pushed_forward.ppf(q), expected_ppf)


# ===========================================================================
# Empirical quantile distribution mapping
# ===========================================================================

class TestGeneralizedPiecewiseAffineDistributionMapping:

    @pytest.fixture
    def transform(self):
        _, transformation = (
            d.FloatingPointColumn()
            .transform.generalized_piecewise_affine(
                np.array([-1.0, 0.0, 1.0]),
                np.array([0.0, 0.5, 1.0]),
            )
            .unpack()
        )
        assert isinstance(transformation, GeneralizedPiecewiseAffine)
        assert isinstance(transformation.inverse, GeneralizedPiecewiseAffine)
        np.testing.assert_array_equal(
            transformation.xp,
            [-np.inf, -1.0, 0.0, 1.0, np.inf],
        )
        np.testing.assert_array_equal(
            transformation.yp,
            [0.0, 0.0, 0.5, 1.0, 1.0],
        )
        np.testing.assert_allclose(transformation.inverse([0.0, 1.0]), [-1.0, 1.0])
        return transformation

    def test_fixed_point_inverse_has_optional_round_projection(self):
        source_domain = d.FixedPointColumn(2, -10.0, 10.0)
        _, transform = source_domain.transform.generalized_piecewise_affine(
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        ).unpack()

        assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
        raw_inverse = transform.inverse.resolve(fixed_point=False)
        lattice_inverse = transform.inverse.resolve(fixed_point=True)
        assert isinstance(raw_inverse, GeneralizedPiecewiseAffine)
        assert isinstance(lattice_inverse, SeriesTransformation)
        assert lattice_inverse.transformations[0] is raw_inverse
        assert isinstance(lattice_inverse.transformations[-1], Round)
        assert raw_inverse.output_domain == d.FloatingPointColumn(-10.0, 10.0)
        assert transform.inverse.output_domain == source_domain
        assert transform.inverse.output_domain_for(
            fixed_point=False
        ) == d.FloatingPointColumn(-10.0, 10.0)
        np.testing.assert_allclose(transform.inverse([0.0, 1.0]), [-1.0, 1.0])

        pulled = transform.pullback_distribution(uniform())
        assert pulled.domain == source_domain
        assert pulled.pdf(0.0) == 0.0
        assert pulled.pmf(0.0) == pytest.approx(0.005)

        continuous = transform.pullback_distribution(
            uniform(),
            fixed_point=False,
        )
        assert continuous.domain == d.FloatingPointColumn(-10.0, 10.0)
        assert continuous.pdf(0.0) == pytest.approx(0.5)
        assert continuous.pmf(0.0) == 0.0

    def test_pushforward_preserves_density_and_clips_tail_mass(self, transform):
        pushed = transform.pushforward_distribution(norm())
        y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        np.testing.assert_allclose(
            pushed.cdf(y),
            [norm.cdf(-1.0), norm.cdf(-0.5), 0.5, norm.cdf(0.5), 1.0],
        )
        np.testing.assert_allclose(
            pushed.pmf(y),
            [norm.cdf(-1.0), 0.0, 0.0, 0.0, norm.sf(1.0)],
        )
        np.testing.assert_allclose(
            pushed.pdf(y[1:-1]),
            2.0 * norm.pdf([-0.5, 0.0, 0.5]),
        )

    def test_pushforward_maps_source_atoms_and_tail_intervals(self):
        source = FiniteOrderedSupportDistribution(
            d.FiniteOrderedColumn([-1.0, 0.0, 1.0, 2.0]),
            np.array([0.2, 0.3, 0.1, 0.4]),
        )
        _, transform = (
            d.FloatingPointColumn()
            .transform.generalized_piecewise_affine(
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.5, 1.0]),
            )
            .unpack()
        )
        pushed = transform.pushforward_distribution(source)

        np.testing.assert_allclose(pushed.pmf([0.0, 0.25, 1.0]), [0.2, 0.3, 0.5])
        np.testing.assert_allclose(pushed.cdf([0.0, 0.1, 0.25, 1.0]), [0.2, 0.2, 0.5, 1.0])

    def test_pullback_maps_probability_intervals_to_atoms(self):
        _, transform = (
            d.FloatingPointColumn()
            .transform.generalized_piecewise_affine(
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.5, 1.0]),
            )
            .unpack()
        )
        pulled = transform.pullback_distribution(uniform())

        np.testing.assert_allclose(pulled.pmf([0.0, 0.5]), [0.5, 0.0])
        np.testing.assert_allclose(pulled.cdf([0.0, 0.5, 1.0]), [0.5, 0.75, 1.0])
        np.testing.assert_allclose(pulled.pdf([0.25, 0.5, 0.75]), 0.5)


# ===========================================================================
# DiscretizedPushedForwardDistribution
# ===========================================================================

class TestDiscretizedPushedForwardDistribution:
    """Full API coverage for DiscretizedPushedForwardDistribution.

    Instantiated directly: Round N(0,1) on FloatingPointColumn(-10, 10).
    Y = round(X) for X ~ N(0,1).
    PMF:  P(Y=k) = Phi(k+0.5) - Phi(k-0.5)
    CDF:  F_Y(k) = Phi(k+0.5)

    Note: the lazy path (DiscretizedPushedForwardDistribution) is deliberately
    parked behind an infinite eager-output threshold. We instantiate the class
    directly to retain coverage of its current behavior.
    """

    @pytest.fixture
    def setup(self):
        round_t = Round(d.FloatingPointColumn(-10, 10))
        px = as_distribution(norm(0, 1))
        py = DiscretizedPushedForwardDistribution(px, round_t)
        return px, round_t, py

    # --- shape ---

    def test_shape(self, setup):
        px, round_t, py = setup
        assert py.shape == ()

    # --- pdf (zero density for discrete distribution) ---

    def test_pdf_is_zero(self, setup):
        px, round_t, py = setup
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(py.pdf(y), np.zeros(5))

    # --- pmf / logpmf ---

    def test_pmf_matches_cdf_difference(self, setup):
        """P(Y=k) = Phi(k+0.5) - Phi(k-0.5)."""
        px, round_t, py = setup
        k = np.arange(-4, 5, dtype=float)
        expected = norm.cdf(k + 0.5) - norm.cdf(k - 0.5)
        np.testing.assert_allclose(py.pmf(k), expected, rtol=1e-12)

    def test_pmf_sums_to_approximately_one(self, setup):
        px, round_t, py = setup
        k = np.arange(-10, 11, dtype=float)
        np.testing.assert_allclose(py.pmf(k).sum(), 1.0, atol=1e-6)

    def test_logpmf_equals_log_pmf(self, setup):
        px, round_t, py = setup
        k = np.arange(-4, 5, dtype=float)
        np.testing.assert_allclose(py.logpmf(k), np.log(py.pmf(k)), rtol=1e-12)

    # --- pxf / logpxf ---

    def test_pxf_equals_pmf(self, setup):
        """NoDensityMixin: pxf = pmf for a discrete distribution."""
        px, round_t, py = setup
        k = np.arange(-3, 4, dtype=float)
        np.testing.assert_allclose(py.pxf(k), py.pmf(k), rtol=1e-12)

    def test_logpxf_equals_logpmf(self, setup):
        """NoDensityMixin: logpxf = logpmf."""
        px, round_t, py = setup
        k = np.arange(-3, 4, dtype=float)
        np.testing.assert_allclose(py.logpxf(k), py.logpmf(k), rtol=1e-12)

    # --- cdf / logcdf ---

    def test_cdf_matches_norm_cdf(self, setup):
        """F_Y(k) = Phi(k + 0.5) (since Round.pullback_ub(k) = k + 0.5)."""
        px, round_t, py = setup
        k = np.arange(-4, 5, dtype=float)
        expected = norm.cdf(k + 0.5)
        np.testing.assert_allclose(py.cdf(k), expected, rtol=1e-12)

    def test_cdf_is_monotone_and_bounded(self, setup):
        px, round_t, py = setup
        k = np.arange(-8, 9, dtype=float)
        cdf = py.cdf(k)
        assert np.all(np.diff(cdf) >= 0), "CDF must be non-decreasing"
        assert np.all(cdf >= 0) and np.all(cdf <= 1)

    def test_logcdf_equals_log_cdf(self, setup):
        px, round_t, py = setup
        k = np.arange(-4, 5, dtype=float)
        np.testing.assert_allclose(py.logcdf(k), np.log(py.cdf(k)), rtol=1e-12)

    # --- rvs ---

    def test_rvs_are_integer_valued(self, setup):
        """Round.apply yields integer values."""
        px, round_t, py = setup
        samples = py.rvs(size=200, random_state=42)
        assert np.all(samples == np.floor(samples)), "Samples should be integers"

    def test_rvs_statistical_calibration(self, setup):
        """Empirical PMF should match theoretical PMF."""
        px, round_t, py = setup
        samples = py.rvs(size=10_000, random_state=7)
        k_range = np.arange(-5, 6)
        empirical = np.array([(samples == k).mean() for k in k_range])
        theoretical = py.pmf(k_range.astype(float))
        np.testing.assert_allclose(empirical, theoretical, atol=0.02)

    # --- sf / logsf (derived from cdf via OrderedDistribution) ---

    def test_sf_correctness(self, setup):
        """sf(k) = 1 - cdf(k) = norm.sf(k + 0.5)."""
        px, round_t, py = setup
        k = np.arange(-4, 5, dtype=float)
        expected = norm.sf(k + 0.5)
        # rtol slightly relaxed: 1 - cdf has floating-point cancellation near 0.5
        np.testing.assert_allclose(py.sf(k), expected, rtol=1e-10)

    def test_logsf_correctness(self, setup):
        px, round_t, py = setup
        k = np.arange(-4, 4, dtype=float)  # skip last to avoid log(0)
        np.testing.assert_allclose(py.logsf(k), np.log(py.sf(k)), rtol=1e-12)

    # --- support (from OrderedDistribution via output_domain) ---

    def test_support_returns_domain_bounds(self, setup):
        """support() returns (lb, ub) from the output domain of Round."""
        px, round_t, py = setup
        lb, ub = py.support()
        assert lb == round_t.output_domain.lb
        assert ub == round_t.output_domain.ub

    # --- unimplemented methods (ppf not implemented → NotImplementedError) ---

    def test_ppf_not_implemented(self, setup):
        px, round_t, py = setup
        with pytest.raises(NotImplementedError):
            py.ppf(0.5)

    def test_isf_not_implemented(self, setup):
        """isf delegates to ppf(1-q) → NotImplementedError."""
        px, round_t, py = setup
        with pytest.raises(NotImplementedError):
            py.isf(0.5)

    def test_median_not_implemented(self, setup):
        """median delegates to ppf(0.5) → NotImplementedError."""
        px, round_t, py = setup
        with pytest.raises(NotImplementedError):
            py.median()

    def test_interval_not_implemented(self, setup):
        """interval delegates to ppf → NotImplementedError."""
        px, round_t, py = setup
        with pytest.raises(NotImplementedError):
            py.interval(0.9)

    def test_mean_not_implemented(self, setup):
        px, round_t, py = setup
        with pytest.raises(AttributeError):
            py.mean()


# ===========================================================================
# FiniteOrderedSupportDistribution
# ===========================================================================

class TestFiniteOrderedSupportDistribution:
    """Full API coverage for FiniteOrderedSupportDistribution.

    Built from a Digitize pushforward of N(0,1) with bins [-2,-1,0,1,2]:
    6 bins (0..5) with probabilities:
      P(Y=0) = Phi(-2)
      P(Y=1) = Phi(-1) - Phi(-2)
      P(Y=2) = Phi(0)  - Phi(-1)
      P(Y=3) = Phi(1)  - Phi(0)
      P(Y=4) = Phi(2)  - Phi(1)
      P(Y=5) = 1 - Phi(2)
    """

    @pytest.fixture
    def setup(self):
        bins_inner = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        full_bins = np.r_[-np.inf, bins_inner, np.inf]
        dig = Digitize(d.FloatingPointColumn(), full_bins)
        px = as_distribution(norm(0, 1))
        py = dig.pushforward_distribution(px)
        assert isinstance(py, FiniteOrderedSupportDistribution)
        # Expected probabilities
        expected_probs = np.diff(norm.cdf(full_bins))
        return px, dig, py, full_bins, expected_probs

    # --- shape ---

    def test_shape(self, setup):
        px, dig, py, full_bins, probs = setup
        assert py.shape == ()

    # --- pdf (zero density) ---

    def test_pdf_is_zero(self, setup):
        """Discrete distribution has no Lebesgue density."""
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality, dtype=float)
        np.testing.assert_array_equal(py.pdf(idx), np.zeros(len(idx)))

    # --- pmf / logpmf ---

    def test_pmf_matches_expected(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        np.testing.assert_allclose(py.pmf(idx), probs, rtol=1e-12)

    def test_pmf_sums_to_one(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        np.testing.assert_allclose(py.pmf(idx).sum(), 1.0, rtol=1e-12)

    def test_digitize_boundary_atom_goes_to_upper_bin(self):
        px = FiniteOrderedSupportDistribution(
            d.FiniteOrderedColumn([0.5]),
            np.array([1.0]),
        )
        dig = Digitize(
            d.FloatingPointColumn(0.0, 1.0),
            np.array([0.0, 0.5, 1.0]),
        )

        py = dig.pushforward_distribution(px)

        assert dig(np.array(0.5)) == 1
        np.testing.assert_array_equal(
            dig._pushforward_boundary_atoms_go_right,
            [True],
        )
        np.testing.assert_array_equal(py.probabilities, [0.0, 1.0])

    def test_large_finite_digitize_remains_eager_while_lazy_path_is_parked(self):
        full_bins = np.r_[
            -np.inf,
            np.linspace(-4.0, 4.0, 300),
            np.inf,
        ]
        dig = Digitize(d.FloatingPointColumn(), full_bins)

        py = dig.pushforward_distribution(norm())

        assert dig.output_domain.cardinality > 256
        assert isinstance(py, FiniteOrderedSupportDistribution)

    def test_round_boundary_atoms_follow_numpy_ties_to_even(self):
        px = FiniteOrderedSupportDistribution(
            d.FiniteOrderedColumn([0.5, 1.5]),
            np.array([0.4, 0.6]),
        )
        round_t = Round(d.FloatingPointColumn(0.0, 2.0))

        py = round_t.pushforward_distribution(px)

        np.testing.assert_array_equal(round_t(np.array([0.5, 1.5])), [0, 2])
        np.testing.assert_array_equal(
            round_t._pushforward_boundary_atoms_go_right,
            [False, True],
        )
        np.testing.assert_allclose(py.pmf([0, 1, 2]), [0.4, 0.0, 0.6])

    def test_logpmf_equals_log_pmf(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        expected = np.log(py.pmf(idx))
        np.testing.assert_allclose(py.logpmf(idx), expected, rtol=1e-12)

    # --- pxf / logpxf ---

    def test_pxf_equals_pmf(self, setup):
        """NoDensityMixin.pxf delegates to pmf."""
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        np.testing.assert_allclose(py.pxf(idx), py.pmf(idx), rtol=1e-12)

    def test_logpxf_equals_log_pmf(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        expected = np.log(py.pmf(idx))
        np.testing.assert_allclose(py.logpxf(idx), expected, rtol=1e-12)

    # --- cdf / logcdf ---

    def test_cdf_is_monotone(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        cdf = py.cdf(idx)
        assert np.all(np.diff(cdf) >= 0)

    def test_cdf_reaches_one_at_last_bin(self, setup):
        px, dig, py, full_bins, probs = setup
        # Use an integer numpy array so IntegerColumn.index doesn't fail on .dtype
        idx = np.arange(py.domain.cardinality)
        np.testing.assert_allclose(py.cdf(idx)[-1], 1.0, rtol=1e-12)

    def test_cdf_is_cumsum_of_pmf(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        expected_cdf = np.cumsum(probs)
        np.testing.assert_allclose(py.cdf(idx), expected_cdf, rtol=1e-12)

    def test_logcdf_equals_log_cdf(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(1, py.domain.cardinality)  # skip idx=0 where cdf may be very small
        np.testing.assert_allclose(py.logcdf(idx), np.log(py.cdf(idx)), rtol=1e-12)

    # --- sf / logsf ---

    def test_sf_plus_cdf_equals_one(self, setup):
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality, dtype=float)
        np.testing.assert_allclose(py.sf(idx) + py.cdf(idx), np.ones(len(idx)), atol=1e-12)

    def test_logsf_equals_log_sf(self, setup):
        px, dig, py, full_bins, probs = setup
        # Only test where sf > 0 to avoid log(0)
        idx = np.arange(py.domain.cardinality - 1, dtype=float)
        np.testing.assert_allclose(py.logsf(idx), np.log(py.sf(idx)), rtol=1e-12)

    # --- ppf / isf ---

    def test_ppf_inverts_cdf_at_bin_probabilities(self, setup):
        """ppf(cdf(k)) should return k for k in the support."""
        px, dig, py, full_bins, probs = setup
        idx = np.arange(py.domain.cardinality)
        cdf_vals = py.cdf(idx.astype(float))
        recovered = py.ppf(cdf_vals)
        np.testing.assert_array_equal(recovered, idx)

    def test_ppf_monotone(self, setup):
        px, dig, py, full_bins, probs = setup
        q = np.linspace(0.01, 0.99, 50)
        vals = py.ppf(q)
        assert np.all(np.diff(vals) >= 0), "ppf must be non-decreasing"

    def test_isf_equals_ppf_complement(self, setup):
        px, dig, py, full_bins, probs = setup
        q = np.linspace(0.01, 0.99, 20)
        np.testing.assert_array_equal(py.isf(q), py.ppf(1.0 - q))

    # --- rvs ---

    def test_rvs_are_in_support(self, setup):
        px, dig, py, full_bins, probs = setup
        samples = py.rvs(size=500, random_state=5)
        valid = np.arange(py.domain.cardinality)
        assert np.all(np.isin(samples, valid))

    def test_rvs_statistical_calibration(self, setup):
        px, dig, py, full_bins, probs = setup
        samples = py.rvs(size=10_000, random_state=6)
        k_range = np.arange(py.domain.cardinality)
        empirical = np.array([(samples == k).mean() for k in k_range])
        np.testing.assert_allclose(empirical, probs, atol=0.02)

    # --- support / median / interval ---

    def test_support_returns_domain_bounds(self, setup):
        px, dig, py, full_bins, probs = setup
        lb, ub = py.support()
        assert lb == py.domain.lb
        assert ub == py.domain.ub

    def test_median_satisfies_cdf_criterion(self, setup):
        px, dig, py, full_bins, probs = setup
        m = py.median()
        assert py.cdf(m) >= 0.5

    def test_interval_covers_requested_probability(self, setup):
        px, dig, py, full_bins, probs = setup
        lo, hi = py.interval(0.8)
        assert lo <= hi
        # The interval [lo, hi] should cover at least 80% probability
        covered = py.cdf(hi) - (py.cdf(lo) - py.pmf(lo))
        assert covered >= 0.8 - 1e-10


# ===========================================================================
# ExplodedPushedForwardDistribution
# ===========================================================================

class TestExplodedPushedForwardDistribution:
    """Full API coverage for ExplodedPushedForwardDistribution.

    Built by pushing a uniform distribution over 6 bins through
    ContinuousInverseDigitize (the stochastic inverse of Digitize).
    """

    @pytest.fixture
    def setup(self):
        bins = [-3., -2., -1., 0., 3.]
        left_tail = truncnorm(-np.inf, -3.)
        right_tail = truncnorm(3., np.inf)

        _, dig = (d.FloatingPointColumn()
            .transform.digitize(bins, left_tail=left_tail, right_tail=right_tail)
            .unpack())
        # dig is the Digitize transformation; dig.inverse is ContinuousInverseDigitize
        cid = dig.inverse

        k = len(bins) + 1  # 6 bins
        py_discrete = FiniteOrderedSupportDistribution(
            d.IntegerColumn(0, k - 1),
            np.ones(k) / k,
        )

        px_continuous = cid.pushforward_distribution(py_discrete)
        assert isinstance(px_continuous, ExplodedPushedForwardDistribution)
        return py_discrete, cid, px_continuous

    # --- shape ---

    def test_shape(self, setup):
        py_discrete, cid, px = setup
        assert px.shape == ()

    # --- pmf / logpmf (NoAtomsMixin: always zero) ---

    def test_pmf_is_zero(self, setup):
        """Continuous distribution has no atoms."""
        py_discrete, cid, px = setup
        y = np.array([-2.5, -1.5, 0.5])
        np.testing.assert_array_equal(px.pmf(y), np.zeros(3))

    def test_logpmf_is_neginf(self, setup):
        py_discrete, cid, px = setup
        y = np.array([-2.5, -1.5, 0.5])
        assert np.all(np.isneginf(px.logpmf(y)))

    # --- pxf (NoAtomsMixin: pxf = pdf) ---

    def test_pxf_equals_pdf(self, setup):
        """NoAtomsMixin guarantees pxf == pdf regardless of correctness."""
        py_discrete, cid, px = setup
        y = np.array([-2.5, -0.5, 1.5])
        np.testing.assert_array_equal(px.pxf(y), px.pdf(y))

    # --- pdf correctness ---

    def test_pdf_is_positive_inside_support(self, setup):
        py_discrete, cid, px = setup
        # Points strictly inside the uniform bins should have positive density
        y = np.array([-2.5, -1.5, -0.5, 0.5])
        pdf_vals = px.pdf(y)
        assert np.all(pdf_vals > 0), f"Expected positive PDF, got {pdf_vals}"

    def test_logpdf_is_finite_inside_support(self, setup):
        py_discrete, cid, px = setup
        y = np.array([-2.5, -1.5, -0.5, 0.5])
        logpdf_vals = px.logpdf(y)
        assert np.all(np.isfinite(logpdf_vals)), f"Expected finite logpdf, got {logpdf_vals}"

    # --- cdf correctness ---

    def test_cdf_is_monotone_and_bounded(self, setup):
        py_discrete, cid, px = setup
        y = np.linspace(-3.0, 3.0, 10)
        cdf_vals = px.cdf(y)
        assert np.all(np.diff(cdf_vals) >= -1e-12)
        assert np.all((cdf_vals >= 0.0) & (cdf_vals <= 1.0))

    def test_pdf_and_cdf_outside_bounded_support(self):
        _, dig = (
            d.FloatingPointColumn(0.0, 1.0)
            .transform.digitize(np.array([0.5]))
            .unpack()
        )
        cid = dig.inverse
        py_discrete = FiniteOrderedSupportDistribution(
            d.IntegerColumn(0, 1),
            np.array([0.4, 0.6]),
        )
        px = cid.pushforward_distribution(py_discrete)

        y = np.array([-0.1, 1.1])
        np.testing.assert_array_equal(px.pdf(y), [0.0, 0.0])
        np.testing.assert_array_equal(px.cdf(y), [0.0, 1.0])

    # --- rvs (works correctly via PushedForwardDistribution.rvs) ---

    def test_rvs_are_finite(self, setup):
        """rvs draws from discrete distribution and maps through CID.apply."""
        py_discrete, cid, px = setup
        samples = px.rvs(size=200, random_state=42)
        assert samples.shape == (200,)
        assert np.all(np.isfinite(samples)), "All samples should be finite"

    def test_rvs_in_output_domain(self, setup):
        """ContinuousInverseDigitize output is bounded by the outermost bins
        only for the uniform pieces; tail pieces may extend beyond."""
        py_discrete, cid, px = setup
        samples = px.rvs(size=500, random_state=0)
        # All samples should lie within the support of the output domain
        lb, ub = cid.output_domain.lb, cid.output_domain.ub
        assert np.all(samples >= lb)
        assert np.all(samples <= ub)

    # --- support (from OrderedDistribution via output_domain, works correctly) ---

    def test_support_returns_domain_bounds(self, setup):
        """support() now works via OrderedDistribution; returns CID output bounds."""
        py_discrete, cid, px = setup
        lb, ub = px.support()
        assert lb == cid.output_domain.lb
        assert ub == cid.output_domain.ub

    # --- cdf-derived methods ---

    def test_logcdf_is_finite(self, setup):
        py_discrete, cid, px = setup
        vals = px.logcdf(np.linspace(-2.0, 2.0, 5))
        assert np.all(np.isfinite(vals))

    def test_sf_is_bounded(self, setup):
        py_discrete, cid, px = setup
        vals = px.sf(np.linspace(-2.0, 2.0, 5))
        assert np.all((vals >= 0.0) & (vals <= 1.0))

    def test_logsf_is_finite(self, setup):
        py_discrete, cid, px = setup
        vals = px.logsf(np.linspace(-2.0, 2.0, 5))
        assert np.all(np.isfinite(vals))

    # --- ppf ---

    def test_ppf_returns_values_in_support(self, setup):
        py_discrete, cid, px = setup
        q = np.array([0.1, 0.5, 0.9])
        y = px.ppf(q)
        lb, ub = px.support()
        assert np.all(y >= lb) and np.all(y <= ub)

    def test_ppf_is_monotone(self, setup):
        py_discrete, cid, px = setup
        q = np.linspace(0.01, 0.99, 20)
        y = px.ppf(q)
        assert np.all(np.diff(y) >= 0)

    def test_ppf_cdf_roundtrip(self, setup):
        """cdf(ppf(q)) ≈ q at interior quantiles."""
        py_discrete, cid, px = setup
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(px.cdf(px.ppf(q)), q, atol=1e-10)

    def test_isf_is_ppf_complement(self, setup):
        py_discrete, cid, px = setup
        q = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(px.isf(q), px.ppf(1 - q))

    def test_median_is_scalar(self, setup):
        py_discrete, cid, px = setup
        m = px.median()
        assert np.isscalar(m) or m.shape == ()

    def test_interval_covers_requested_probability(self, setup):
        py_discrete, cid, px = setup
        lo, hi = px.interval(0.8)
        assert lo <= hi

    def test_mean_is_finite(self, setup):
        py_discrete, cid, px = setup
        assert np.isfinite(px.mean())

    def test_var_is_non_negative(self, setup):
        py_discrete, cid, px = setup
        assert px.var() >= 0

    def test_moment_1_equals_mean(self, setup):
        py_discrete, cid, px = setup
        np.testing.assert_allclose(px.moment(1), px.mean())

    def test_moment_2_implies_var(self, setup):
        py_discrete, cid, px = setup
        np.testing.assert_allclose(px.var(), px.moment(2) - px.moment(1)**2)

    def test_expect_identity_equals_mean(self, setup):
        py_discrete, cid, px = setup
        np.testing.assert_allclose(px.expect(lambda y: y), px.mean(), rtol=1e-6)

    # --- multi-tail-bin regression (num_*_tail_bins=2 exposes cdf formula bug) ---

    @pytest.fixture
    def setup_multi_tail(self):
        """Setup with 2 bins in each tail to exercise lb>0 branches in cdf."""
        bins = [-3., -2., -1., 0., 3.]
        left_tail = truncnorm(-np.inf, -1.)
        right_tail = truncnorm(1., np.inf)

        _, dig = (d.FloatingPointColumn()
            .transform.digitize(
                bins,
                left_tail=left_tail,
                right_tail=right_tail,
                num_left_tail_bins=2,
                num_right_tail_bins=2,
            )
            .unpack())
        cid = dig.inverse

        k = len(bins) + 1  # 6 bins
        py_discrete = FiniteOrderedSupportDistribution(
            d.IntegerColumn(0, k - 1),
            np.ones(k) / k,
        )
        px = cid.pushforward_distribution(py_discrete)
        return py_discrete, cid, px

    def test_cdf_ppf_roundtrip_multi_tail(self, setup_multi_tail):
        """cdf(ppf(q)) ≈ q — catches the missing -lb bug in the tail cdf branch."""
        _, _, px = setup_multi_tail
        q = np.array([0.05, 0.15, 0.5, 0.85, 0.95])
        np.testing.assert_allclose(px.cdf(px.ppf(q)), q, atol=1e-10)

    # --- Python scalar inputs ---

    def test_pdf_python_scalar(self, setup):
        """pdf accepts a Python float and returns a scalar."""
        py_discrete, cid, px = setup
        y_arr = np.array([0.5])
        val = px.pdf(0.5)
        assert np.ndim(val) == 0
        np.testing.assert_allclose(val, px.pdf(y_arr)[0])

    def test_cdf_python_scalar(self, setup):
        """cdf accepts a Python float and returns a scalar."""
        py_discrete, cid, px = setup
        y_arr = np.array([0.5])
        val = px.cdf(0.5)
        assert np.ndim(val) == 0
        np.testing.assert_allclose(val, px.cdf(y_arr)[0])

    def test_ppf_python_scalar(self, setup):
        """ppf accepts a Python float and returns a scalar."""
        py_discrete, cid, px = setup
        q_arr = np.array([0.5])
        val = px.ppf(0.5)
        assert np.ndim(val) == 0
        np.testing.assert_allclose(val, px.ppf(q_arr)[0])

    # --- non-scalar px (batch of distributions) ---

    @pytest.fixture
    def setup_batch(self):
        bins = [-3., -2., -1., 0., 3.]
        left_tail = truncnorm(-np.inf, -3.)
        right_tail = truncnorm(3., np.inf)

        _, dig = (d.FloatingPointColumn()
            .transform.digitize(bins, left_tail=left_tail, right_tail=right_tail)
            .unpack())
        cid = dig.inverse

        k = len(bins) + 1  # 6 bins
        n = 3  # batch size

        # Three distributions with different concentrations
        probs = np.array([
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],   # left-tail heavy
            [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],      # uniform
            [0.05, 0.05, 0.1, 0.1, 0.1, 0.6],    # right-tail heavy
        ])

        py_discrete = FiniteOrderedSupportDistribution(
            d.IntegerColumn(0, k - 1),
            probs,
        )

        px_continuous = cid.pushforward_distribution(py_discrete)
        assert isinstance(px_continuous, ExplodedPushedForwardDistribution)
        return py_discrete, cid, px_continuous

    def test_shape_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        assert px.shape == (3,)

    def test_ppf_shape_batch(self, setup_batch):
        """ppf(q) where q.shape == batch.shape returns same shape."""
        py_discrete, cid, px = setup_batch
        q = np.array([0.2, 0.5, 0.8])
        y = px.ppf(q)
        assert y.shape == (3,)

    def test_ppf_in_support_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        q = np.array([0.1, 0.5, 0.9])
        y = px.ppf(q)
        lb, ub = px.support()
        assert np.all(y >= lb) and np.all(y <= ub)

    def test_ppf_monotone_per_distribution_batch(self, setup_batch):
        """For each fixed distribution in the batch, ppf is non-decreasing."""
        py_discrete, cid, px = setup_batch
        qs = np.linspace(0.05, 0.95, 10)
        # evaluate each distribution at the same set of quantiles
        ys = np.stack([px.ppf(np.full(3, q)) for q in qs])  # (10, 3)
        assert np.all(np.diff(ys, axis=0) >= 0)

    def test_ppf_cdf_roundtrip_batch(self, setup_batch):
        """cdf(ppf(q)) ≈ q for batch distributions."""
        py_discrete, cid, px = setup_batch
        q = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(px.cdf(px.ppf(q)), q, atol=1e-10)

    def test_rvs_shape_batch(self, setup_batch):
        """rvs draws one sample per distribution in the batch."""
        py_discrete, cid, px = setup_batch
        samples = px.rvs(random_state=0)
        assert samples.shape == (3,)

    def test_pdf_batch(self, setup_batch):
        """pdf is positive at interior points for each distribution."""
        py_discrete, cid, px = setup_batch
        y = np.array([-2.5, -0.5, 1.5])
        vals = px.pdf(y)
        assert vals.shape == (3,)
        assert np.all(vals > 0)

    def test_mean_shape_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        m = px.mean()
        assert m.shape == (3,)
        assert np.all(np.isfinite(m))

    def test_var_shape_and_sign_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        v = px.var()
        assert v.shape == (3,)
        assert np.all(v >= 0)

    def test_moment_1_shape_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        m = px.moment(1)
        assert m.shape == (3,)
        np.testing.assert_allclose(m, px.mean())

    def test_expect_shape_batch(self, setup_batch):
        py_discrete, cid, px = setup_batch
        result = px.expect(lambda y: y)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, px.mean(), rtol=1e-6)
