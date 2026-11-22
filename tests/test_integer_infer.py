import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto, geom

from dicer import ColumnTracker, FrameTracker
from dicer.domain import IntegerColumn
from dicer.infer.integer import infer_integer_genpareto_tail
from dicer.stats import DiscreteGenPareto, FiniteOrderedSupportDistribution


@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('shape,scale', [(0, 2), (-1, 5), (-0.3, 2), (0.3, 2)])
def test_discrete_gpd_cell_probabilities_and_quantiles(left, shape, scale):
    tail = DiscreteGenPareto(shape, scale, threshold=10, left=left)
    distances = np.arange(30)
    x = 10 - distances if left else 10 + distances
    base = genpareto(shape, scale=scale)
    expected = base.sf(distances) - base.sf(distances + 1)
    np.testing.assert_allclose(tail.pmf(x), expected, atol=1e-14)
    if shape == 0:
        np.testing.assert_allclose(expected, geom.pmf(distances + 1, -np.expm1(-1 / scale)))
    if shape == -1:
        np.testing.assert_allclose(tail.pmf(x[:5]), 0.2)
    q = np.linspace(0.001, 0.999, 301)
    quantiles = tail.ppf(q)
    assert np.all(tail.cdf(quantiles) >= q - 1e-12)
    assert np.all(tail.cdf(quantiles - 1) < q + 1e-12)
    jump_q = tail.cdf(x)
    jumps = (jump_q > 0) & (jump_q < 1)
    np.testing.assert_array_equal(tail.ppf(jump_q[jumps]), x[jumps])
    np.testing.assert_array_equal(tail.pmf(x + 0.5), 0)
    samples = tail.rvs(size=5000, random_state=42)
    assert np.issubdtype(samples.dtype, np.integer)
    for point in x[:5]:
        assert np.mean(samples <= point) == pytest.approx(tail.cdf(point), abs=0.03)


def test_discrete_gpd_truncation_and_endpoint():
    tail = DiscreteGenPareto(0.2, 3, max_distance=5)
    assert tail.support() == (0, 5)
    assert tail.pmf(np.arange(6)).sum() == pytest.approx(1)
    np.testing.assert_array_equal(tail.ppf([0, 1]), [0, 5])
    assert DiscreteGenPareto(-1, 5.3).support() == (0, 5)


def test_integer_clip_empirical_smoothing_rounding_and_distribution_roundtrip():
    vals = np.array([-3, -3, -1, 0, 2, 4, 4])
    fit = ColumnTracker(IntegerColumn(-4, 5), vals).infer.clip(
        lb=-1.5, ub=2.5, smoothing=1,
    ).finish()
    transform = fit.transformation
    np.testing.assert_array_equal(fit.data, [-1, -1, -1, 0, 2, 2, 2])
    left, right = transform.inverse.left_dist, transform.inverse.right_dist
    np.testing.assert_allclose(left.pmf([-4, -3, -2, -1]), [1/7, 3/7, 1/7, 2/7])
    np.testing.assert_allclose(right.pmf([2, 3, 4, 5]), [2/7, 1/7, 3/7, 1/7])
    assert transform.inverse.output_domain == IntegerColumn(-4, 5)
    codes = np.tile(fit.data, 1000)
    reconstructed = transform.inverse(codes, random_state=3)
    assert np.issubdtype(reconstructed.dtype, np.integer)
    np.testing.assert_array_equal(transform(reconstructed), codes)
    source = FiniteOrderedSupportDistribution(IntegerColumn(-4, 5), np.full(10, 0.1))
    clipped = transform.pushforward_distribution(source)
    pulled = transform.pullback_distribution(clipped)
    assert isinstance(pulled, FiniteOrderedSupportDistribution)
    assert pulled.domain == source.domain
    assert pulled.pmf(np.arange(-4, 6)).sum() == pytest.approx(1)
    q = np.linspace(0.01, 0.99, 50)
    assert np.all(pulled.cdf(pulled.ppf(q)) >= q - 1e-12)
    np.testing.assert_allclose(
        transform.pushforward_distribution(pulled).pmf(np.arange(-1, 3)),
        clipped.pmf(np.arange(-1, 3)),
    )

    # Regression: conditional-q rescaling formerly returned 3 at this jump.
    assert pulled.ppf(pulled.cdf(2)) == 2
    np.testing.assert_array_equal(pulled.ppf([0, 1]), [-4, 5])


@pytest.mark.parametrize('tail_type', ['empirical', 'genpareto'])
def test_integer_clip_pullback_quantiles_at_jumps_and_gaps(tail_type):
    transform = ColumnTracker(
        IntegerColumn(-5, 5), np.tile(np.arange(-5, 6), 20),
    ).infer.clip(
        lb=-2, ub=2, left_tail_distribution=tail_type,
        right_tail_distribution=tail_type,
    ).finish().transformation
    # An interior gap, plus batches with one boundary carrying no mass.
    clipped = FiniteOrderedSupportDistribution(
        IntegerColumn(-2, 2),
        np.array([[.2, .3, 0, .1, .4], [0, .3, 0, .2, .5], [.4, .2, 0, .4, 0]]),
    )
    reconstructed = transform.pullback_distribution(clipped)
    assert isinstance(reconstructed, FiniteOrderedSupportDistribution)
    assert reconstructed.shape == (3,)
    values = reconstructed.domain.values
    np.testing.assert_allclose(reconstructed.probabilities.sum(axis=-1), 1)
    np.testing.assert_allclose(
        transform.pushforward_distribution(reconstructed).pmf(np.arange(-2, 3)[:, None]),
        clipped.pmf(np.arange(-2, 3)[:, None]),
        atol=2e-15,
    )
    for batch in range(3):
        cumulative = reconstructed.cdf(values[:, None])[:, batch]
        for jump in cumulative[(cumulative > 0) & (cumulative < cumulative[-1])]:
            for q in [np.nextafter(jump, 0), jump, np.nextafter(jump, 1)]:
                expected = values[np.searchsorted(cumulative, q, side='left')]
                assert reconstructed.ppf(q)[batch] == expected
    samples = reconstructed.rvs(size=(2000, 3), random_state=42)
    assert np.issubdtype(samples.dtype, np.integer)
    assert samples.shape == (2000, 3)


@pytest.mark.parametrize('left', [False, True])
def test_integer_clip_gpd_unbounded_tail(left):
    distances = np.floor(genpareto.rvs(0.2, scale=3, size=1000, random_state=42)).astype(int)
    vals = -distances if left else distances
    fit = ColumnTracker(IntegerColumn(), vals).infer.clip(
        **({'lb': -2} if left else {'ub': 2}),
    ).finish()
    transform = fit.transformation
    tail = transform.inverse.left_dist if left else transform.inverse.right_dist
    assert isinstance(tail, DiscreteGenPareto)
    assert tail.tail_shape >= 0
    assert transform.inverse.output_domain == IntegerColumn()
    np.testing.assert_array_equal(transform(transform.inverse(fit.data, random_state=0)), fit.data)


def test_empirical_assume_bounds_controls_declared_support():
    vals = np.array([0, 1, 2, 5, 5])
    for assume_bounds in (False, True):
        fit = ColumnTracker(IntegerColumn(0), vals).infer.clip(
            ub=2, right_tail_distribution='empirical', assume_bounds=assume_bounds,
        ).finish()
        expected = IntegerColumn(0, 5 if assume_bounds else np.inf)
        assert fit.transformation.input_domain == expected
        assert fit.transformation.inverse.output_domain == expected
        tail = fit.transformation.inverse.right_dist
        assert tail.pmf(6) == 0
        assert tail.pmf(3) > 0


def test_gpd_fit_uses_discrete_likelihood_and_can_learn_bounded_support():
    vals = np.tile(np.arange(8), 100)
    fitted = infer_integer_genpareto_tail(vals, 0, force_unbounded=False)
    assert fitted.tail_shape < 0
    assert fitted.domain.ub >= 7
    np.testing.assert_allclose(fitted.pmf(np.arange(8)), 1/8, atol=0.01)
    bounded = infer_integer_genpareto_tail(vals, 0, domain_bound=12)
    assert bounded.domain.ub == 12
    assert bounded.pmf(np.arange(13)).sum() == pytest.approx(1)


def test_integer_clip_frame_inference_replays():
    train = pd.DataFrame({'count': [0, 1, 2, 3, 5]})
    fit = (FrameTracker.from_dataframe(train).assume['count'].ge(0)
           .assume['count'].le(5).infer['count'].clip(ub=2.9).finish())
    np.testing.assert_array_equal(fit.data['count'], [0, 1, 2, 2, 2])
    np.testing.assert_array_equal(fit.transformation(pd.DataFrame({'count': [1, 4]}))['count'], [1, 2])


def test_empty_and_degenerate_clipped_tails():
    tracker = ColumnTracker(IntegerColumn(0, 10), np.array([2, 3]))
    fit = tracker.infer.clip(ub=5).finish()
    np.testing.assert_allclose(fit.transformation.inverse.right_dist.pmf(np.arange(5, 11)), 1/6)
    with pytest.raises(ValueError, match='observations'):
        tracker.infer.clip(ub=5, right_tail_distribution='genpareto')
    with pytest.raises(ValueError, match='strictly less'):
        tracker.infer.clip(lb=3, ub=3)


def test_bounded_gpd_clip_inference_preserves_original_interval():
    fit = ColumnTracker(IntegerColumn(0, 20), np.tile(np.arange(21), 20)).infer.clip(
        lb=5, ub=15, left_tail_distribution='genpareto',
        right_tail_distribution='genpareto',
    ).finish()
    assert fit.transformation.inverse.output_domain == IntegerColumn(0, 20)
    samples = fit.transformation.inverse(fit.data, random_state=42)
    np.testing.assert_array_equal(fit.transformation(samples), fit.data)


@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('assume_bounds', [False, True])
def test_uniform_clip_matches_continuous_support_policy(left, bounded, assume_bounds):
    vals = np.array([0, 1, 2, 2, 4, 6])
    if left:
        vals = -vals
    input_domain = IntegerColumn(-8, 0) if left else IntegerColumn(0, 8)
    if not bounded:
        input_domain = IntegerColumn(-np.inf, 0) if left else IntegerColumn(0)
    fit = ColumnTracker(input_domain, vals).infer.clip(
        **({'lb': -2.7, 'left_tail_distribution': 'uniform'} if left else
           {'ub': 2.7, 'right_tail_distribution': 'uniform'}),
        assume_bounds=assume_bounds,
    ).finish()
    t = fit.transformation
    tail = t.inverse.left_dist if left else t.inverse.right_dist
    endpoint = 8 if bounded else 6
    support = -np.arange(2, endpoint + 1) if left else np.arange(2, endpoint + 1)
    np.testing.assert_allclose(tail.pmf(support), 1 / len(support))
    assert tail.pmf(-endpoint-1 if left else endpoint+1) == 0
    expected_bound = endpoint if bounded or assume_bounds else np.inf
    expected_domain = IntegerColumn(-expected_bound, 0) if left else IntegerColumn(0, expected_bound)
    assert t.input_domain == expected_domain
    assert t.inverse.output_domain == expected_domain
    np.testing.assert_array_equal(t(t.inverse(fit.data, random_state=42)), fit.data)


def test_uniform_clip_empty_bounded_and_singleton_empirical_tail():
    t = ColumnTracker(IntegerColumn(0, 8), np.array([0, 1])).infer.clip(
        ub=3, right_tail_distribution='uniform',
    ).finish().transformation
    np.testing.assert_allclose(t.inverse.right_dist.pmf(np.arange(3, 9)), 1/6)
    t = ColumnTracker(IntegerColumn(0), np.array([0, 3, 3])).infer.clip(
        ub=3, right_tail_distribution='uniform',
    ).finish().transformation
    assert t.inverse.right_dist.pmf(3) == 1
    assert t.inverse.right_dist.pmf(4) == 0
    with pytest.raises(ValueError, match='requires observations'):
        ColumnTracker(IntegerColumn(0), np.array([0, 1])).infer.clip(
            ub=3, right_tail_distribution='uniform',
        )


def test_integer_empirical_bound_inferences():
    vals = np.array([2, 4, 7])
    source = IntegerColumn()
    assert source.infer.nonnegative(data=vals).unpack('domain') == IntegerColumn(0)
    assert source.infer.nonnegative(data=-vals).unpack('domain') == source
    assert source.infer.lb(data=vals).unpack('domain') == IntegerColumn(2)
    assert source.infer.ub(data=vals).unpack('domain') == IntegerColumn(upper_bound=7)
    bounded = IntegerColumn(1, 9)
    assert bounded.infer.nonnegative(data=vals).unpack('domain') == bounded
    tracked = ColumnTracker.from_series(vals).infer.lb().infer.ub()
    assert tracked.unpack('domain') == IntegerColumn(2, 7)
    np.testing.assert_array_equal(tracked.unpack('data'), vals)


@pytest.mark.parametrize('left', [False, True])
def test_integer_gpd_bound_inference(left):
    vals = np.tile(np.arange(20), 30)
    if left:
        vals = -vals
    method = 'lb' if left else 'ub'
    result = getattr(IntegerColumn().infer, method)(
        data=vals, tail_type='genpareto', tail_prob=0.27,
    ).unpack('domain')
    bound = result.lb if left else result.ub
    assert np.isfinite(bound)
    assert bound % 1 == 0
    assert result.lb <= vals.min() <= vals.max() <= result.ub


@pytest.mark.parametrize('method', ['lb', 'ub'])
def test_integer_bound_inference_invalid_options(method):
    infer = getattr(IntegerColumn().infer, method)
    with pytest.raises(ValueError, match='tail_type'):
        infer(data=np.arange(10), tail_type='unknown')
    with pytest.raises(ValueError, match='prob'):
        infer(data=np.arange(10), tail_type='genpareto', tail_prob=1.5)
