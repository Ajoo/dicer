import numpy as np
import pandas as pd
import pytest
from scipy.stats import geom, randint

from dicer import ColumnTracker, FrameTracker
from dicer.domain import IntegerColumn
from dicer.stats import DiscreteGenPareto, FiniteOrderedSupportDistribution as Finite


def test_multiple_bins_condition_one_empirical_piece_and_uniform_middle():
    base = Finite(IntegerColumn(0, 11), np.arange(1, 13) / 78)
    _, t = IntegerColumn(0, 11).transform.digitize(
        [2, 4, 6, 8, 10], left_tail=base, right_tail=base,
        num_left_tail_bins=2, num_right_tail_bins=2,
    ).unpack()
    np.testing.assert_array_equal(t.inverse.piece_bins, [0, 2, 4, 6])
    assert t.inverse.piece_distributions[1] is None
    for code in range(6):
        values = np.arange(2*code, 2*code+2)
        expected = np.full(2, .5) if code in (2, 3) else (values+1)/(values+1).sum()
        np.testing.assert_allclose(t.inverse.conditional(code).pmf(values), expected)
    codes = np.tile(np.arange(6), (3000, 1))
    samples = t.inverse(codes, random_state=42)
    np.testing.assert_array_equal(t(samples), codes)
    assert np.issubdtype(samples.dtype, np.integer)
    assert np.mean(samples[:, 0] == 1) == pytest.approx(2/3, abs=.025)
    py = Finite(IntegerColumn(0, 5), np.full((2, 6), 1/6))
    px = t.pullback_distribution(py)
    np.testing.assert_allclose(t.pushforward_distribution(px).probabilities, py.probabilities)


def test_custom_pieces_middle_base_and_unbounded_distribution_roundtrip():
    _, t = IntegerColumn().transform.digitize(
        [-4, 0, 3, 6], piece_bins=[0, 2, 3, 5],
        piece_distributions=[DiscreteGenPareto(.2, 3, threshold=-1, left=True),
                             randint(0, 3), geom(.4, loc=2)],
    ).unpack()
    codes = np.repeat(np.arange(5), 2000)
    np.testing.assert_array_equal(t(t.inverse(codes, random_state=1)), codes)
    py = Finite(IntegerColumn(0, 4), np.full(5, .2))
    px = t.pullback_distribution(py)
    np.testing.assert_allclose(t.pushforward_distribution(px).probabilities, py.probabilities)
    grid = np.arange(-20, 20)
    np.testing.assert_allclose(px.cdf(grid)-px.cdf(grid-1), px.pmf(grid), atol=1e-15)
    q = np.linspace(.01, .99, 100)
    assert np.all(px.cdf(px.ppf(q)) >= q-1e-12)


def test_explicit_piece_validation_and_zero_mass_fiber():
    for edges, bases in [([0, 3], [None]), ([0, 1, 1, 2], [None]*3)]:
        with pytest.raises(ValueError, match='partition'):
            IntegerColumn(0, 5).transform.digitize([3], piece_bins=edges, piece_distributions=bases)
    with pytest.raises(ValueError, match='zero mass'):
        IntegerColumn(0, 5).transform.digitize([3], middle_distribution=randint(0, 3))


@pytest.mark.parametrize('middle', ['uniform', 'empirical'])
def test_bounded_quantile_inference_and_empirical_middle(middle):
    values = np.tile(np.arange(20), 100)
    t = ColumnTracker(IntegerColumn(0, 19), values).infer.quantile_discretize(
        5, middle_distribution=middle,
    ).finish().transformation
    assert t.output_domain == IntegerColumn(0, 4)
    np.testing.assert_array_equal(np.bincount(t(values)), 400)
    np.testing.assert_array_equal(t.inverse.piece_bins, [0, 5])
    assert (t.inverse.piece_distributions[0] is None) == (middle == 'uniform')
    np.testing.assert_array_equal(t(t.inverse(t(values), random_state=42)), t(values))


def test_quantile_ties_and_constant_unbounded_data():
    for values in [np.array([0]*70+[1]*20+[5]*10), np.full(50, 3)]:
        t = ColumnTracker(IntegerColumn(), values).infer.quantile_discretize(
            20, left_tail_distribution='empirical', right_tail_distribution='empirical',
            middle_distribution='empirical',
        ).finish().transformation
        assert t.output_domain.cardinality < 20
        assert np.all(np.diff(t.cuts) > 0)
        np.testing.assert_array_equal(t(t.inverse(t(values), random_state=0)), t(values))


@pytest.mark.parametrize('name', ['genpareto', 'empirical', 'uniform'])
@pytest.mark.parametrize('assume_bounds', [False, True])
def test_quantile_unbounded_tails_multiple_bins(name, assume_bounds):
    values = np.tile(np.arange(-30, 31), 20)
    t = ColumnTracker(IntegerColumn(), values).infer.quantile_discretize(
        10, min_prob_left_tail=.2, min_prob_right_tail=.2,
        left_tail_distribution=name, right_tail_distribution=name,
        middle_distribution='empirical', assume_bounds=assume_bounds,
    ).finish().transformation
    np.testing.assert_array_equal(t.inverse.piece_bins, [0, 2, 8, 10])
    assert t.inverse.output_domain == t.input_domain
    if not assume_bounds:
        assert t.input_domain == IntegerColumn()
    elif name != 'genpareto':
        assert t.input_domain == IntegerColumn(-30, 30)
    np.testing.assert_array_equal(t(t.inverse(t(values), random_state=3)), t(values))


def test_quantile_frame_inference_replay():
    df = pd.DataFrame({'x': np.tile(np.arange(20), 10)})
    fit = (FrameTracker.from_dataframe(df).assume['x'].ge(0).assume['x'].le(19)
           .infer['x'].quantile_discretize(4, middle_distribution='empirical').finish())
    pd.testing.assert_frame_equal(fit.transformation(df), fit.data)


def test_constant_bounded_quantiles_keep_uniform_default():
    t = IntegerColumn(0, 5).infer.quantile_discretize(10, data=np.full(20, 3)).finish().transformation
    assert t.output_domain == IntegerColumn(0, 0)
    assert t.inverse.piece_distributions == [None]
    np.testing.assert_allclose(t.inverse.conditional(0).pmf(np.arange(6)), 1/6)


def test_uniform_unbounded_quantiles_with_zero_tail_probability():
    t = IntegerColumn(0).infer.quantile_discretize(
        5, data=np.arange(20), right_tail_distribution='uniform',
    ).finish().transformation
    np.testing.assert_array_equal(t.inverse.piece_bins, [0, 4, 5])
    assert t.input_domain == IntegerColumn(0)
    np.testing.assert_array_equal(t(t.inverse(np.arange(5), random_state=0)), np.arange(5))


def test_quantile_rejects_unusable_tail_configuration():
    with pytest.raises(ValueError, match='greater than zero'):
        IntegerColumn().infer.quantile_discretize(
            10, data=np.arange(20), min_prob_left_tail=0,
        )
    with pytest.raises(ValueError, match='sum to less'):
        IntegerColumn().infer.quantile_discretize(
            10, data=np.arange(20), min_prob_left_tail=.6, min_prob_right_tail=.5,
        )
