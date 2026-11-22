"""Comprehensive tests for dicer.infer — inference methods for floating-point domains."""

import numpy as np
import pytest
from scipy.stats import truncnorm, truncexpon

from dicer import Column
from dicer.domain import FixedPointColumn, FloatingPointColumn, IntegerColumn
from dicer.stats import ReflectedDistribution
from dicer.transform._base import (
    ConfigurableSeriesTransformation,
    SeriesTransformation,
)
from dicer.transform.continuous import (
    ContinuousInverseDigitize,
    GeneralizedPiecewiseAffine,
    Round,
    Unclip,
)
from dicer.tracking import Tracker
from dicer.infer import (
    quantile_discretize as _quantile_discretize,
    infer_genpareto_tail,
    infer_normal_tail,
)

RNG = np.random.default_rng(0)
N = 3000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_data():
    return RNG.uniform(scale=2.0, size=N)

@pytest.fixture
def exponential_data():
    return RNG.exponential(scale=2.0, size=N)

@pytest.fixture
def normal_data():
    return RNG.normal(loc=0.0, scale=1.0, size=N)

@pytest.fixture
def truncnorm_data():
    return truncnorm(-1, 2).rvs(size=N, random_state=1)

@pytest.fixture
def truncexpon_data():
    return truncexpon(5).rvs(size=N, random_state=2)

@pytest.fixture
def pareto_data():
    """Heavy-tailed Pareto data: shape α=1.5, bounded left at 1."""
    return RNG.pareto(1.5, size=N) + 1.0


# ---------------------------------------------------------------------------
# scalar inferences
# ---------------------------------------------------------------------------

class TestScalarInferences:

    @pytest.mark.parametrize(
        'data, match',
        [
            (np.array([]), 'non-empty'),
            (np.array([0.0, np.nan]), 'finite'),
            (np.array([0.0, np.inf]), 'finite'),
        ],
    )
    def test_rejects_empty_or_nonfinite_data(self, data, match):
        with pytest.raises(ValueError, match=match):
            FloatingPointColumn().infer.standardize(data=data)

    def test_fixed_point_keeps_float_domain_when_no_precision_matches(self):
        result = FloatingPointColumn().infer.fixed_point(
            data=np.array([0.1234567]),
            max_digits=2,
        )

        assert result.unpack('domain') == FloatingPointColumn()
        assert result.unpack('transformations') is None

    @pytest.mark.parametrize('method', ['standardize', 'min_max_scale'])
    def test_constant_scalers_use_unit_scale(self, method):
        data = np.full(5, 3.0)
        result = getattr(Tracker.from_series(data).infer, method)()
        np.testing.assert_array_equal(result.unpack('data'), np.zeros(5))

        transform = result.finish().transformation
        np.testing.assert_allclose(transform(np.array([2.0, 3.0, 4.0])), [-1, 0, 1])
        np.testing.assert_allclose(transform.inverse(np.array([-1.0, 0.0, 1.0])), [2, 3, 4])


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------

class TestClipInference:

    def test_auto_fits_normal_tails_on_unbounded_domain(self, normal_data):
        result = Tracker.from_series(normal_data).infer.clip(
            lb=-1.0,
            ub=1.0,
        )
        transform = result.unpack('transformations')

        assert transform.output_domain == FloatingPointColumn(-1.0, 1.0)
        assert transform.inverse.output_domain == FloatingPointColumn()
        np.testing.assert_allclose(
            transform.inverse.left_dist.support(),
            [-np.inf, -1.0],
        )
        np.testing.assert_allclose(
            transform.inverse.right_dist.support(),
            [1.0, np.inf],
        )
        np.testing.assert_allclose(
            result.unpack('data'),
            np.clip(normal_data, -1.0, 1.0),
        )

    def test_auto_uses_full_known_support_for_bounded_uniform_tails(self):
        data = np.linspace(0.0, 10.0, 101)
        transform = FloatingPointColumn(0.0, 10.0).infer.clip(
            data=data,
            lb=2.0,
            ub=8.0,
        ).unpack('transformations')

        np.testing.assert_allclose(
            transform.inverse.left_dist.support(),
            [0.0, 2.0],
        )
        np.testing.assert_allclose(
            transform.inverse.right_dist.support(),
            [8.0, 10.0],
        )

    def test_uniform_unbounded_tail_keeps_domain_without_assume_bounds(
        self,
        normal_data,
    ):
        transform = FloatingPointColumn().infer.clip(
            data=normal_data,
            lb=-1.0,
            left_tail_distribution='uniform',
        ).unpack('transformations')

        fitted_lb, fitted_ub = transform.inverse.left_dist.support()
        assert fitted_lb == -np.inf
        assert fitted_ub == -1.0
        empirical_lb = normal_data[normal_data <= -1.0].min()
        assert transform.inverse.left_dist.pdf(empirical_lb - 1.0) == 0.0
        assert transform.input_domain.lb == -np.inf
        assert transform.inverse.output_domain.lb == -np.inf

    def test_uniform_unbounded_tail_can_assume_empirical_bound(self, normal_data):
        transform = FloatingPointColumn().infer.clip(
            data=normal_data,
            lb=-1.0,
            left_tail_distribution='uniform',
            assume_bounds=True,
        ).unpack('transformations')

        fitted_lb = transform.inverse.left_dist.support()[0]
        assert transform.input_domain.lb == fitted_lb
        assert transform.inverse.output_domain.lb == fitted_lb

    def test_genpareto_preserves_unbounded_support_by_default(self, pareto_data):
        ub = float(np.quantile(pareto_data, 0.8))
        transform = FloatingPointColumn(1.0, np.inf).infer.clip(
            data=pareto_data,
            ub=ub,
            right_tail_distribution='genpareto',
        ).unpack('transformations')

        fitted_lb, fitted_ub = transform.inverse.right_dist.support()
        np.testing.assert_allclose(fitted_lb, ub)
        assert fitted_ub == np.inf
        assert transform.input_domain.ub == np.inf

    def test_requires_an_effective_finite_bound(self, normal_data):
        with pytest.raises(ValueError, match='requires lb, ub, or both'):
            FloatingPointColumn().infer.clip(data=normal_data)
        with pytest.raises(ValueError, match='finite scalar'):
            FloatingPointColumn().infer.clip(data=normal_data, lb=-np.inf)

    def test_parametric_tail_requires_enough_observations(self):
        with pytest.raises(ValueError, match='at least 3 observations'):
            FloatingPointColumn().infer.clip(
                data=np.array([0.0, 1.0, 2.0]),
                lb=0.5,
            )


# ---------------------------------------------------------------------------
# fixed-point inference registration
# ---------------------------------------------------------------------------

class TestFixedPointInferences:

    @pytest.fixture
    def data(self):
        return np.round(
            np.random.default_rng(123).normal(size=1000),
            2,
        )

    @staticmethod
    def tracker(data):
        return Tracker.from_series(data).assume.fixed_point(2)

    def test_scalar_inferences_and_fixed_point_noop(self):
        domain = FixedPointColumn(2)
        data = np.array([-1.25, 0.5, 2.0])

        assert domain.infer.fixed_point(data=data).unpack('domain') == domain
        assert domain.infer.lb(data=data).unpack('domain') == FixedPointColumn(
            2,
            -1.25,
            np.inf,
        )
        assert domain.infer.ub(data=data).unpack('domain') == FixedPointColumn(
            2,
            -np.inf,
            2.0,
        )
        assert domain.infer.nonnegative(
            data=np.array([0.0, 0.25]),
        ).unpack('domain') == FixedPointColumn(2, 0.0, np.inf)

    @pytest.mark.parametrize('method', ['standardize', 'min_max_scale'])
    def test_scalers_use_optional_fixed_point_projection(self, data, method):
        result = getattr(self.tracker(data).infer, method)()
        transform = result.unpack('transformations')

        assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
        lattice_inverse = transform.inverse.resolve(fixed_point=True)
        assert isinstance(lattice_inverse, SeriesTransformation)
        assert isinstance(lattice_inverse.transformations[-1], Round)
        reconstructed = transform.inverse(result.unpack('data'))
        np.testing.assert_allclose(reconstructed, data)
        assert FixedPointColumn(2).check_decimals(reconstructed)
        continuous = transform.inverse(
            result.unpack('data'),
            fixed_point=False,
        )
        np.testing.assert_allclose(continuous, data)

    def test_clip_fits_tails_and_rounds_inverse(self, data):
        result = self.tracker(data).infer.clip(lb=-1.0, ub=1.0)
        transform = result.unpack('transformations')

        assert transform.output_domain == FixedPointColumn(2, -1.0, 1.0)
        assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
        inverse = transform.inverse.resolve(fixed_point=False)
        lattice_inverse = transform.inverse.resolve(fixed_point=True)
        assert isinstance(inverse, Unclip)
        assert isinstance(lattice_inverse, SeriesTransformation)
        assert lattice_inverse.transformations[0] is inverse
        assert isinstance(lattice_inverse.transformations[-1], Round)
        reconstructed = transform.inverse(
            np.array([-1.0, 0.0, 1.0]),
            random_state=123,
        )
        assert FixedPointColumn(2).check_decimals(reconstructed)

    def test_to_quantiles_uses_optional_fixed_point_projection(self, data):
        result = self.tracker(data).infer.to_quantiles(num_quantiles=20)
        transform = result.unpack('transformations')

        assert transform.output_domain == FloatingPointColumn(0.0, 1.0)
        assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
        inverse = transform.inverse.resolve(fixed_point=False)
        lattice_inverse = transform.inverse.resolve(fixed_point=True)
        assert isinstance(inverse, GeneralizedPiecewiseAffine)
        assert isinstance(lattice_inverse, SeriesTransformation)
        assert lattice_inverse.transformations[0] is inverse
        assert isinstance(lattice_inverse.transformations[-1], Round)
        np.testing.assert_allclose(
            transform.inverse([0.0, 1.0]),
            [data.min(), data.max()],
        )

    def test_quantile_discretize_rounds_stochastic_inverse(self):
        data = np.round(np.linspace(-2.0, 2.0, 101), 2)
        tracker = (
            self.tracker(data)
            .assume.ge(-2.0)
            .assume.le(2.0)
        )
        result = tracker.infer.quantile_discretize(num_bins=5)
        transform = result.unpack('transformations')

        assert transform.output_domain == IntegerColumn(0, 4)
        assert isinstance(transform.inverse, ConfigurableSeriesTransformation)
        inverse = transform.inverse.resolve(fixed_point=False)
        lattice_inverse = transform.inverse.resolve(fixed_point=True)
        assert isinstance(inverse, ContinuousInverseDigitize)
        assert isinstance(lattice_inverse, SeriesTransformation)
        assert lattice_inverse.transformations[0] is inverse
        assert isinstance(lattice_inverse.transformations[-1], Round)
        reconstructed = transform.inverse(
            np.arange(5),
            random_state=123,
        )
        assert FixedPointColumn(2).check_decimals(reconstructed)
        continuous = transform.inverse(
            np.arange(5),
            random_state=123,
            fixed_point=False,
        )
        assert not FixedPointColumn(2).check_decimals(continuous)


# ---------------------------------------------------------------------------
# to_quantiles
# ---------------------------------------------------------------------------

class TestQuantileTransform:

    def test_output_domain_is_unit_interval(self, truncexpon_data):
        """Output domain should be FloatingPointColumn(0., 1.)."""
        col = Tracker.from_series(truncexpon_data).assume.nonnegative().assume.le(5)
        result = col.infer.to_quantiles(num_quantiles=50)
        assert result.unpack('domain') == FloatingPointColumn(0.0, 1.0)

    def test_from_explicit_data_kwarg(self, truncexpon_data):
        """infer.to_quantiles(data=...) should work on a bare domain."""
        col = Column.from_dtype(np.float64).assume.nonnegative().assume.le(5)
        result = col.infer.to_quantiles(data=truncexpon_data)
        assert result.unpack('domain') == FloatingPointColumn(0.0, 1.0)

    def test_assume_bounds_sets_domain_bounds(self, truncexpon_data):
        """assume_bounds=True should tighten the input domain to [data.min(), data.max()]."""
        col = Tracker.from_series(truncexpon_data)
        result = col.assume.nonnegative().infer.to_quantiles(assume_bounds=True)
        t = result.unpack('transformations')
        np.testing.assert_allclose(t.input_domain.lb, truncexpon_data.min())
        np.testing.assert_allclose(t.input_domain.ub, truncexpon_data.max())

    def test_clips_tails_and_inverse_uses_empirical_extrema(self):
        data = np.array([2.0, 3.0, 4.0])
        col = Tracker.from_series(data).assume.ge(0.0).assume.le(10.0)
        transform = (
            col.infer.to_quantiles(num_quantiles=2)
            .finish()
            .transformation
        )

        np.testing.assert_allclose(
            transform(np.array([0.0, 2.0, 3.0, 4.0, 10.0])),
            [0.0, 0.0, 0.5, 1.0, 1.0],
        )
        np.testing.assert_allclose(transform.inverse([0.0, 1.0]), [2.0, 4.0])

    def test_repeated_quantiles_map_to_probability_midpoint(self):
        data = np.array([0.0, 0.0, 0.0, 1.0])
        transform = (
            Tracker.from_series(data)
            .assume.ge(0.0)
            .assume.le(1.0)
            .infer.to_quantiles(num_quantiles=4)
            .finish()
            .transformation
        )

        assert transform(0.0) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# quantile_discretize infer method
# ---------------------------------------------------------------------------

class TestQuantileDiscretize:

    def test_num_bins_must_be_at_least_two(self, truncnorm_data):
        with pytest.raises(ValueError, match='greater than or equal to 2'):
            Tracker.from_series(truncnorm_data).infer.quantile_discretize(
                num_bins=1,
            )

    def test_basic_bounded_domain_integer_output(self, truncnorm_data):
        """Bounded domain with no tails → integer domain 0..num_bins-1."""
        num_bins = 10
        tracker = Tracker.from_series(truncnorm_data).assume.ge(-1.0).assume.le(2.0)
        result = tracker.infer.quantile_discretize(num_bins=num_bins)
        assert result.unpack('domain') == IntegerColumn(0, num_bins - 1)

    def test_basic_bounded_approx_uniform_counts(self, truncnorm_data):
        """Bin counts should be approximately equal for large data."""
        num_bins = 10
        tracker = Tracker.from_series(truncnorm_data).assume.ge(-1.0).assume.le(2.0)
        result = tracker.infer.quantile_discretize(num_bins=num_bins)
        counts = np.bincount(result.unpack('data'), minlength=num_bins)
        expected = len(truncnorm_data) / num_bins
        np.testing.assert_allclose(counts, expected, rtol=0.3)

    def test_assume_bounds(self, truncexpon_data):
        """Bin counts should be approximately equal for large data."""
        num_bins = 100
        t = (Tracker
            .from_series(truncexpon_data)
            .assume.nonnegative()
            .infer.quantile_discretize(
                num_bins=num_bins,
                min_prob_right_tail=0.1,
                right_tail_distribution='genpareto',
                assume_bounds=True
                )
            .unpack('transformations')
        )
        input_domain = t.input_domain
        assert isinstance(input_domain, FloatingPointColumn)
        assert input_domain.lb == 0
        assert input_domain.ub >= 5 and input_domain.is_bounded_right


    def test_tail_bins_have_roughly_correct_fraction(self, truncnorm_data):
        """Right-tail bins should hold approximately min_prob_right_tail of the data."""
        num_bins = 20
        tail_prob = 0.1
        tracker = Tracker.from_series(truncnorm_data).assume.ge(-1.0)
        result = tracker.infer.quantile_discretize(
            num_bins=num_bins,
            min_prob_right_tail=tail_prob,
            right_tail_distribution='normal',
        )
        bins = result.unpack('data')
        num_right_tail_bins = int(np.ceil(tail_prob * num_bins))
        fraction_in_right_tail = np.mean(bins >= num_bins - num_right_tail_bins)
        np.testing.assert_allclose(fraction_in_right_tail, tail_prob, atol=0.05)

    def test_auto_unbounded_tails_default_to_five_percent_normal(self, normal_data):
        transform = (
            Tracker.from_series(normal_data)
            .infer.quantile_discretize(num_bins=20)
            .unpack('transformations')
        )

        inverse = transform.inverse
        left_tail, central, right_tail = inverse.piece_distributions
        assert left_tail.dist.name == 'truncnorm'
        assert central is None
        assert right_tail.dist.name == 'truncnorm'
        assert inverse.piece_bins[1] == 1
        assert inverse.piece_bins[-2] == 19
        assert transform.input_domain.lb == -np.inf
        assert transform.input_domain.ub == np.inf

    def test_uniform_tail_on_unbounded_side_stops_at_empirical_max(
        self,
        exponential_data,
    ):
        transform = (
            Tracker.from_series(exponential_data)
            .assume.nonnegative()
            .infer.quantile_discretize(
                num_bins=20,
                right_tail_distribution='uniform',
            )
            .unpack('transformations')
        )

        right_tail = transform.inverse.piece_distributions[-1]
        _, fitted_ub = right_tail.support()
        assert fitted_ub == np.inf
        assert right_tail.pdf(exponential_data.max() + 1.0) == 0.0
        assert transform.input_domain.ub == np.inf

        bounded_transform = (
            Tracker.from_series(exponential_data)
            .assume.nonnegative()
            .infer.quantile_discretize(
                num_bins=20,
                right_tail_distribution='uniform',
                assume_bounds=True,
            )
            .unpack('transformations')
        )
        np.testing.assert_allclose(
            bounded_transform.input_domain.ub,
            exponential_data.max(),
        )

    @pytest.mark.parametrize('distribution', ['normal', 'genpareto'])
    def test_zero_probability_rejected_for_unbounded_nonuniform_tail(
        self,
        normal_data,
        distribution,
    ):
        with pytest.raises(ValueError, match='greater than zero'):
            Tracker.from_series(normal_data).infer.quantile_discretize(
                num_bins=20,
                min_prob_right_tail=0,
                right_tail_distribution=distribution,
            )

    def test_left_genpareto_tail_is_supported_and_preserves_unbounded_domain(
        self,
        exponential_data,
    ):
        data = -exponential_data
        transform = (
            Tracker.from_series(data)
            .assume.nonpositive()
            .infer.quantile_discretize(
                num_bins=20,
                left_tail_distribution='genpareto',
            )
            .unpack('transformations')
        )

        left_tail = transform.inverse.piece_distributions[0]
        fitted_lb, _ = left_tail.support()
        assert fitted_lb == -np.inf
        assert transform.input_domain.lb == -np.inf


# ---------------------------------------------------------------------------
# infer_genpareto_tail / infer_normal_tail helpers
# ---------------------------------------------------------------------------

class TestTailHelpers:

    @pytest.mark.parametrize(
        'fit_tail',
        [
            lambda vals: infer_normal_tail(vals, vals.min()),
            lambda vals: infer_genpareto_tail(vals, vals.min()),
        ],
        ids=['normal', 'genpareto'],
    )
    def test_parametric_tail_requires_three_observations(self, fit_tail):
        with pytest.raises(ValueError, match='at least 3 observations'):
            fit_tail(np.array([1.0, 2.0]))

    @pytest.mark.parametrize(
        'fit_tail',
        [
            lambda vals: infer_normal_tail(vals, vals.min()),
            lambda vals: infer_genpareto_tail(vals, vals.min()),
        ],
        ids=['normal', 'genpareto'],
    )
    def test_parametric_tail_requires_distinct_observations(self, fit_tail):
        with pytest.raises(ValueError, match='two distinct observations'):
            fit_tail(np.ones(3))

    def test_bound_inferences_allow_finite_genpareto_support(self, truncexpon_data):
        upper = FloatingPointColumn().infer.ub(
            data=truncexpon_data,
            tail_type='genpareto',
            tail_prob=0.1,
        ).unpack('domain')
        lower = FloatingPointColumn().infer.lb(
            data=-truncexpon_data,
            tail_type='genpareto',
            tail_prob=0.1,
        ).unpack('domain')

        assert np.isfinite(upper.ub)
        assert upper.ub >= truncexpon_data.max()
        assert np.isfinite(lower.lb)
        assert lower.lb <= -truncexpon_data.max()

    def test_bound_inferences_reject_unknown_tail_type(self, normal_data):
        with pytest.raises(ValueError, match='tail_type'):
            FloatingPointColumn().infer.ub(
                data=normal_data,
                tail_type='unknown',
            )

    def test_genpareto_tail_threshold_is_support_lower(self, pareto_data):
        """Fitted genpareto distribution should have support starting at the threshold."""
        threshold = float(np.quantile(pareto_data, 0.9))
        tail_vals = pareto_data[pareto_data >= threshold]
        dist = infer_genpareto_tail(tail_vals, threshold)
        lb, _ = dist.support()
        np.testing.assert_allclose(lb, threshold, atol=1e-6)

    def test_genpareto_tail_force_unbounded(self, pareto_data):
        """force_unbounded=True should produce a distribution with infinite upper support."""
        threshold = float(np.quantile(pareto_data, 0.9))
        tail_vals = pareto_data[pareto_data >= threshold]
        dist = infer_genpareto_tail(tail_vals, threshold, force_unbounded=True)
        _, ub = dist.support()
        assert not np.isfinite(ub), "With force_unbounded=True, upper support should be inf"

    def test_genpareto_left_tail_force_unbounded(self, pareto_data):
        vals = -pareto_data
        threshold = float(np.quantile(vals, 0.1))
        tail_vals = vals[vals <= threshold]
        dist = infer_genpareto_tail(
            tail_vals,
            threshold,
            force_unbounded=True,
            left=True,
        )
        lb, ub = dist.support()
        assert isinstance(dist, ReflectedDistribution)
        assert lb == -np.inf
        np.testing.assert_allclose(ub, threshold)

    def test_normal_right_tail_threshold_is_support_lower(self, normal_data):
        """Fitted truncnorm for right tail should have support starting at threshold."""
        threshold = float(np.quantile(normal_data, 0.9))
        tail_vals = normal_data[normal_data >= threshold]
        dist = infer_normal_tail(tail_vals, threshold, left=False)
        lb, _ = dist.support()
        np.testing.assert_allclose(lb, threshold, atol=1e-3)

    def test_normal_left_tail_threshold_is_support_upper(self, normal_data):
        """Fitted truncnorm for left tail should have support ending at threshold."""
        threshold = float(np.quantile(normal_data, 0.1))
        tail_vals = normal_data[normal_data <= threshold]
        dist = infer_normal_tail(tail_vals, threshold, left=True)
        _, ub = dist.support()
        np.testing.assert_allclose(ub, threshold, atol=1e-3)


def test_zero_digit_fixed_point_inference_preserves_float_representation():
    values = np.array([-2., 0., 3.])
    result = FloatingPointColumn(-2.7, 3.2).infer.fixed_point(data=values)
    assert result.unpack('domain') == FixedPointColumn(0, -2, 3)
    tracked = Tracker.from_series(values).infer.fixed_point()
    assert tracked.unpack('domain') == FixedPointColumn(0)
    assert tracked.unpack('data').dtype.kind == 'f'
    converted = tracked.transform.to_int()
    assert converted.unpack('domain') == IntegerColumn()
    assert converted.unpack('data').dtype.kind == 'i'
