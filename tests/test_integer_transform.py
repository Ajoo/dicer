import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, randint, uniform

from dicer import ColumnTracker, FrameTracker
from dicer import domain as d
from dicer.stats import FiniteOrderedSupportDistribution, as_distribution
from dicer.transform.integer import (
    DiscreteInverseDigitize,
    IntegerAffine,
    IntegerClip,
    IntegerDigitize,
)


@pytest.mark.parametrize(
    ('input_domain', 'a', 'b', 'output_domain', 'values', 'expected'),
    [
        (
            d.IntegerColumn(-2, 4),
            1.0,
            np.int64(3),
            d.IntegerColumn(1, 7),
            [-2, 0, 4],
            [1, 3, 7],
        ),
        (
            d.IntegerColumn(-2, 4),
            np.int8(-1),
            3.0,
            d.IntegerColumn(-1, 5),
            [-2, 0, 4],
            [5, 3, -1],
        ),
    ],
)
def test_integer_affine_maps_bounds_values_and_inverse(
    input_domain,
    a,
    b,
    output_domain,
    values,
    expected,
):
    transformed_domain, transformation = (
        input_domain.transform.affine(a, b).unpack()
    )

    assert transformed_domain == output_domain
    assert transformation.output_domain == output_domain
    assert isinstance(transformation, IntegerAffine)
    assert transformation.a in (-1, 1)
    assert isinstance(transformation.a, int)
    assert isinstance(transformation.b, int)
    np.testing.assert_array_equal(transformation(values), expected)
    np.testing.assert_array_equal(transformation.inverse(expected), values)
    assert transformation.inverse.output_domain == input_domain


def test_integer_affine_supports_unbounded_intervals():
    transformed_domain = (
        d.IntegerColumn(2, np.inf)
        .transform.affine(-1, 5)
        .unpack('domain')
    )

    assert transformed_domain == d.IntegerColumn(-np.inf, 3)


@pytest.mark.parametrize('a', [0, 2, -2, 0.5, np.inf])
def test_integer_affine_rejects_unsupported_slopes(a):
    with pytest.raises(ValueError, match='slope a'):
        d.IntegerColumn(0, 3).transform.affine(a, 0)


@pytest.mark.parametrize('b', [0.5, np.inf])
def test_integer_affine_rejects_non_integer_intercepts(b):
    with pytest.raises(ValueError, match='intercept b must be integer-valued'):
        d.IntegerColumn(0, 3).transform.affine(1, b)


def test_integer_affine_pushes_forward_discrete_distributions():
    _, transformation = (
        d.IntegerColumn(2, 4).transform.affine(-1, 6).unpack()
    )
    pushed_forward = transformation.pushforward_distribution(randint(2, 5))

    values = np.array([2, 3, 4])
    np.testing.assert_allclose(pushed_forward.pmf(values), 1 / 3)
    np.testing.assert_allclose(pushed_forward.cdf(values), [1 / 3, 2 / 3, 1])
    np.testing.assert_array_equal(
        pushed_forward.ppf([0.2, 0.5, 0.9]),
        [2, 3, 4],
    )


def test_integer_clip_rounds_bounds_inward_and_preserves_dtype():
    input_domain = d.IntegerColumn(-5, 5)
    output_domain, transformation = (
        input_domain.transform.clip(-1.2, 2.8).unpack()
    )

    assert isinstance(transformation, IntegerClip)
    assert (transformation.lb, transformation.ub) == (-1, 2)
    assert output_domain == d.IntegerColumn(-1, 2)
    assert transformation.is_lossy

    values = np.array([-5, -1, 0, 2, 5], dtype=np.int16)
    clipped = transformation(values)
    np.testing.assert_array_equal(clipped, [-1, -1, 0, 2, 2])
    assert clipped.dtype == values.dtype


def test_integer_clip_pushforward_collapses_discrete_tail_mass():
    _, transformation = (
        d.IntegerColumn(-2, 2).transform.clip(-0.2, 1.8).unpack()
    )
    clipped = transformation.pushforward_distribution(randint(-2, 3))

    values = np.array([-1, 0, 1, 2])
    np.testing.assert_allclose(clipped.pmf(values), [0.0, 0.6, 0.4, 0.0])
    np.testing.assert_allclose(clipped.cdf(values), [0.0, 0.6, 1.0, 1.0])
    np.testing.assert_array_equal(clipped.ppf([0.1, 0.7]), [0, 1])


def test_integer_clip_is_lossless_when_bounds_do_not_tighten_domain():
    input_domain = d.IntegerColumn(-5, 5)

    same_domain, no_bounds = input_domain.transform.clip().unpack()
    wider_domain, wider_bounds = (
        input_domain.transform.clip(-5.2, 5.2).unpack()
    )

    assert same_domain == input_domain
    assert wider_domain == input_domain
    assert not no_bounds.is_lossy
    assert not wider_bounds.is_lossy


def test_integer_clip_rejects_bounds_with_empty_integer_interval():
    with pytest.raises(ValueError, match='must not exceed'):
        d.IntegerColumn(0, 5).transform.clip(1.2, 1.8)


def test_integer_digitize_maps_integer_fibers_and_samples_right_inverse():
    input_domain = d.IntegerColumn(0, 9)
    output_domain, transformation = (
        input_domain.transform.digitize([3.0, np.int64(7)]).unpack()
    )

    assert isinstance(transformation, IntegerDigitize)
    assert isinstance(transformation.inverse, DiscreteInverseDigitize)
    assert output_domain == d.IntegerColumn(0, 2)
    assert transformation.inverse.output_domain == input_domain
    np.testing.assert_array_equal(transformation.bins, [0, 3, 7, 10])
    assert transformation.inverse.bins is transformation.bins
    np.testing.assert_array_equal(transformation.lower, [0, 3, 7])
    np.testing.assert_array_equal(transformation.upper, [2, 6, 9])

    values = np.array([0, 2, 3, 6, 7, 9])
    np.testing.assert_array_equal(transformation(values), [0, 0, 1, 1, 2, 2])

    codes = np.repeat(np.arange(3), 200)
    reconstructed = transformation.inverse(codes, random_state=42)
    assert np.issubdtype(reconstructed.dtype, np.integer)
    np.testing.assert_array_equal(transformation(reconstructed), codes)
    assert transformation.inverse.conditional(1).support() == (3, 6)


def test_integer_digitize_has_exact_finite_pushforward_and_pullback():
    _, transformation = d.IntegerColumn(0, 9).transform.digitize([3, 7]).unpack()
    source = FiniteOrderedSupportDistribution(
        d.IntegerColumn(0, 9),
        np.full(10, 0.1),
    )

    digitized = transformation.pushforward_distribution(source)
    np.testing.assert_allclose(digitized.probabilities, [0.3, 0.4, 0.3])

    code_distribution = FiniteOrderedSupportDistribution(
        d.IntegerColumn(0, 2),
        np.array([0.3, 0.4, 0.3]),
    )
    reconstructed = transformation.pullback_distribution(code_distribution)
    np.testing.assert_allclose(reconstructed.probabilities, 0.1)


def test_integer_digitize_pushforward_preserves_distribution_batch_shape():
    _, transformation = d.IntegerColumn(0, 4).transform.digitize([2, 4]).unpack()
    source = FiniteOrderedSupportDistribution(
        d.IntegerColumn(0, 4),
        np.array([
            [0.1, 0.2, 0.3, 0.1, 0.3],
            [0.2, 0.1, 0.1, 0.4, 0.2],
        ]),
    )

    digitized = transformation.pushforward_distribution(source)

    assert digitized.shape == (2,)
    np.testing.assert_allclose(
        digitized.probabilities,
        [[0.3, 0.4, 0.3], [0.3, 0.5, 0.2]],
    )


@pytest.mark.parametrize(
    ('cuts', 'match'),
    [
        ([3.5], 'integer-valued'),
        ([3, 3], 'strictly increasing'),
        ([0, 3], 'input.lb < cuts <= input.ub'),
        ([3, 10], 'input.lb < cuts <= input.ub'),
        ([[3, 7]], 'one-dimensional'),
    ],
)
def test_integer_digitize_rejects_invalid_cuts(cuts, match):
    with pytest.raises(ValueError, match=match):
        d.IntegerColumn(0, 9).transform.digitize(cuts)


def test_integer_digitize_allows_one_bin_on_bounded_domain():
    output_domain, transformation = (
        d.IntegerColumn(-2, 2).transform.digitize([]).unpack()
    )

    assert output_domain == d.IntegerColumn(0, 0)
    np.testing.assert_array_equal(transformation([-2, 0, 2]), [0, 0, 0])
    samples = transformation.inverse(np.zeros(100, dtype=int), random_state=0)
    assert np.all((samples >= -2) & (samples <= 2))


def test_integer_digitize_uses_discrete_tail_for_unbounded_fiber():
    left_tail = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-4, -1]),
        np.array([0.25, 0.75]),
    )
    input_domain = d.IntegerColumn(-np.inf, 5)
    _, transformation = input_domain.transform.digitize(
        [0, 3],
        left_tail=left_tail,
    ).unpack()

    codes = np.repeat(np.arange(3), 200)
    reconstructed = transformation.inverse(codes, random_state=7)

    assert transformation.inverse.output_domain == input_domain
    assert set(np.unique(reconstructed[codes == 0])) <= {-4, -1}
    np.testing.assert_array_equal(transformation(reconstructed), codes)


@pytest.mark.parametrize(
    ('input_domain', 'cuts', 'match'),
    [
        (d.IntegerColumn(-np.inf, 5), [0], 'left-tail'),
        (d.IntegerColumn(0, np.inf), [3], 'right-tail'),
    ],
)
def test_integer_digitize_requires_distribution_for_unbounded_fiber(
    input_domain,
    cuts,
    match,
):
    with pytest.raises(ValueError, match=match):
        input_domain.transform.digitize(cuts)


def test_integer_digitize_rejects_invalid_tail_distribution():
    with pytest.raises(ValueError, match='ordered integer support'):
        d.IntegerColumn(-np.inf, 5).transform.digitize(
            [0],
            left_tail=norm(-2, 0.5),
        )

    crossing_tail = FiniteOrderedSupportDistribution(
        d.FiniteOrderedColumn([-1, 0]),
        np.array([0.5, 0.5]),
    )
    # A piece base may cross bin boundaries: each fiber conditions it.
    _, transformation = d.IntegerColumn(-np.inf, 5).transform.digitize(
        [0], left_tail=crossing_tail,
    ).unpack()
    np.testing.assert_array_equal(
        transformation.inverse(np.zeros(50, dtype=int), random_state=0), -1,
    )


def test_integer_encode_is_affine_shift_to_zero():
    input_domain = d.IntegerColumn(-3, 4)
    output_domain, transformation = input_domain.transform.encode().unpack()

    assert output_domain == d.IntegerColumn(0, 7)
    assert isinstance(transformation, IntegerAffine)
    assert (transformation.a, transformation.b) == (1, 3)
    np.testing.assert_array_equal(
        transformation([-3, 0, 4]),
        [0, 3, 7],
    )
    np.testing.assert_array_equal(
        transformation.inverse([0, 3, 7]),
        [-3, 0, 4],
    )


@pytest.mark.parametrize(
    'input_domain',
    [
        d.IntegerColumn(),
        d.IntegerColumn(0, np.inf),
        d.IntegerColumn(-np.inf, 3),
    ],
)
def test_integer_encode_requires_finite_bounds(input_domain):
    with pytest.raises(ValueError, match='finite lower and upper bounds'):
        input_domain.transform.encode()


def test_integer_encode_updates_tracked_data_and_replays():
    tracker = ColumnTracker(
        d.IntegerColumn(10, 12),
        data=np.array([12, 10, 11]),
    )

    fit = tracker.transform.encode().finish()

    assert fit.transformation.output_domain == d.IntegerColumn(0, 2)
    np.testing.assert_array_equal(fit.data, [2, 0, 1])
    np.testing.assert_array_equal(
        fit.transformation(np.array([10, 11, 12])),
        [0, 1, 2],
    )


def test_integer_encode_runs_in_dataframe_workflow():
    train = pd.DataFrame({'count': [12, 10, 11], 'label': ['c', 'a', 'b']})
    holdout = pd.DataFrame({'count': [10, 12], 'label': ['x', 'y']})

    fit = (
        FrameTracker.from_dataframe(train)
        .assume['count'].ge(10)
        .assume['count'].le(12)
        .transform['count'].encode()
        .finish()
    )

    assert fit.transformation.output_domain['count'] == d.IntegerColumn(0, 2)
    np.testing.assert_array_equal(fit.data['count'], [2, 0, 1])
    replayed = fit.transformation(holdout)
    np.testing.assert_array_equal(replayed['count'], [0, 2])
    np.testing.assert_array_equal(replayed['label'], ['x', 'y'])


@pytest.mark.parametrize('digits', [0, 2])
def test_integer_to_fixed_point(digits):
    source = d.IntegerColumn(-3, 5)
    output, transform = source.transform.to_fixed_point(digits).unpack()
    values = np.array([-3, 0, 5])
    converted = transform.apply(values)
    assert converted.dtype.kind == 'f'
    assert output == d.FixedPointColumn(digits, -3 * 10**-digits, 5 * 10**-digits)
    np.testing.assert_array_equal(transform.inverse.apply(converted), values)


def test_integer_dequantize_roundtrip_and_measure():
    source = d.IntegerColumn(-3, 5)
    output, transform = source.transform.dequantize().unpack()
    assert output == d.FloatingPointColumn(-3.5, 5.5)
    assert transform.inverse.output_domain == source
    values = np.tile(np.arange(-3, 6), 100)
    samples = transform.apply(values, random_state=42)
    np.testing.assert_array_equal(transform.inverse.apply(samples), values)
    mass = FiniteOrderedSupportDistribution(source, np.ones(9) / 9)
    continuous = transform.pushforward_distribution(mass)
    np.testing.assert_allclose(continuous.pdf(np.arange(-3, 6)), 1 / 9)
    restored = transform.inverse.pushforward_distribution(continuous)
    np.testing.assert_allclose(restored.pmf(np.arange(-3, 6)), 1 / 9)


def test_integer_as_unordered():
    source = d.IntegerColumn(0, 3)
    output, transform = source.transform.as_unordered().unpack()
    assert output == d.UnorderedIntegerColumn(4)
    values = np.arange(4)
    np.testing.assert_array_equal(transform.apply(values), values)
    np.testing.assert_array_equal(transform.inverse.apply(values), values)
    for invalid in [d.IntegerColumn(-1, 3), d.IntegerColumn(0)]:
        with pytest.raises(ValueError, match='integer codes'):
            invalid.transform.as_unordered()


@pytest.mark.parametrize('digits', [0, 1, 2])
@pytest.mark.parametrize('bounded', [True, False])
def test_fixed_point_shares_integer_dequantization(digits, bounded):
    h = 10**-digits
    source = d.FixedPointColumn(digits, -3*h, 5*h) if bounded else d.FixedPointColumn(digits)
    output, transform = source.transform.dequantize().unpack()
    assert transform.inverse.output_domain == source
    values = np.tile(np.arange(-3, 6)*h, 100)
    samples = transform.apply(values, random_state=42)
    restored = transform.inverse.apply(samples)
    assert restored.dtype.kind == 'f'
    np.testing.assert_allclose(restored, values)
    if bounded:
        np.testing.assert_allclose([output.lb, output.ub], [-3.5*h, 5.5*h])
        np.testing.assert_allclose(transform.inverse._pushforward_bins,
                                   np.arange(-3.5, 6)*h)
        continuous = as_distribution(uniform(output.lb, output.ub-output.lb))
        atomic = transform.inverse.pushforward_distribution(continuous)
        np.testing.assert_allclose(atomic.pmf(np.arange(-3, 6)*h), 1/9)


def test_shared_dequantization_custom_bounds():
    for source in [d.IntegerColumn(-3, 5), d.FixedPointColumn(0, -3, 5)]:
        output, transform = source.transform.dequantize(lb=-3.2, ub=5.1).unpack()
        assert output == d.FloatingPointColumn(-3.2, 5.1)
        assert transform.inverse.output_domain == source
        values = np.tile(np.arange(-3, 6), 100)
        np.testing.assert_array_equal(
            transform.inverse.apply(transform.apply(values, random_state=42)), values,
        )
