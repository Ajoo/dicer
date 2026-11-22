import numpy as np
import pytest
from scipy.stats import norm, uniform

from dicer.domain import FloatingPointColumn, FixedPointColumn, IntegerColumn
from dicer.transform.continuous import AddNoise, Round


def test_affine():
    domain = FloatingPointColumn(-1.0, 2.0)

    transformed_domain, transform = domain.transform.affine(3., -1.).unpack()
    assert transformed_domain == FloatingPointColumn(-4.0, 5.0)

    x = np.array([-1.0, 0.0, 1.5])
    expected = 3.0 * x - 1.0
    np.testing.assert_allclose(transform(x), expected)


def test_affine_rejects_zero_slope():
    with pytest.raises(ValueError, match='nonzero slope'):
        FloatingPointColumn(-1.0, 2.0).transform.affine(0.0, 1.0)


def test_map_without_inverse_rejects_distribution_pushforward():
    _, transform = (
        FloatingPointColumn(0.0, 1.0)
        .transform.map(np.square, lambda x: 2.0 * x)
        .unpack()
    )

    with pytest.raises(NotImplementedError, match='requires f_inv'):
        transform.pushforward_distribution(norm())


def test_transformation_validate_checks_input_domain():
    domain = FloatingPointColumn(0.0, 1.0)
    _, transform = domain.transform.affine(2.0, 0.0).unpack()

    values = np.array([0.0, 0.5, 1.0])
    assert transform.validate(values) is None
    with pytest.raises(ValueError, match='Values must lie within'):
        transform.validate(np.array([-0.1, 0.5]))


def test_decreasing_piecewise_affine():
    domain = FloatingPointColumn(0.0, 2.0)
    xp = np.array([0.0, 1.0, 2.0])
    yp = np.array([3.0, 1.0, 0.0])

    transformed_domain, transform = (
        domain.transform.piecewise_affine(xp, yp).unpack()
    )
    assert transformed_domain == FloatingPointColumn(0.0, 3.0)

    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    expected = np.array([3.0, 2.0, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(transform(x), expected)
    np.testing.assert_allclose(transform.inverse(expected), x)


def test_generalized_piecewise_affine_horizontal_and_vertical_pieces():
    domain = FloatingPointColumn(0.0, 2.0)
    xp = np.array([0.0, 1.0, 1.0, 2.0])
    yp = np.array([0.0, 0.0, 1.0, 2.0])

    transformed_domain, transform = (
        domain.transform.generalized_piecewise_affine(xp, yp).unpack()
    )

    assert transformed_domain == FloatingPointColumn(0.0, 2.0)
    np.testing.assert_allclose(transform([0.5, 1.0, 1.5]), [0.0, 0.5, 1.5])
    np.testing.assert_allclose(transform.inverse([0.0, 0.5, 1.5]), [1.0, 1.0, 1.5])
    assert transform.derivative(0.5) == 0.0
    assert np.isposinf(transform.derivative(1.0))

    with pytest.raises(ValueError, match='strictly monotone'):
        domain.transform.piecewise_affine(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0]),
        )


def test_exp():
    domain = FloatingPointColumn(-2.0, 1.0)

    transformed_domain, transform = domain.transform.exp().unpack()
    assert transformed_domain == FloatingPointColumn(np.exp(-2.), np.exp(1.))

    x = np.array([-2.0, -0.5, 1.0])
    expected = np.exp(x)
    np.testing.assert_allclose(transform(x), expected)


def test_probit():
    domain = FloatingPointColumn(0.0, 0.5)

    transformed_domain, transform = domain.transform.probit().unpack()
    assert transformed_domain == FloatingPointColumn(-float('inf'), 0.)

    x = np.array([0.5])
    expected = np.array([0.0])
    np.testing.assert_array_equal(transform(x), expected)


# TODO: add error and edge cases of no dist
def test_clip():
    domain = FloatingPointColumn(-5.0, 5.0)

    left_dist = uniform(-5.0, 1.0)
    right_dist = uniform(2.0, 3.0)
    transformed_domain, transform = (domain
        .transform.clip(-1., 2., left_dist=left_dist, right_dist=right_dist)
    ).unpack()
    assert transformed_domain == FloatingPointColumn(-1., 2.)

    x = np.array([-4.0, 0.0, 4.0])
    expected = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(transform(x), expected)


def test_digitize():
    domain = FloatingPointColumn(0.0, 10.0)

    bins = np.array([2.0, 5.0, 8.0])
    transformed_domain, transform = domain.transform.digitize(bins).unpack()
    assert transformed_domain == IntegerColumn(0, 3)

    x = np.array([0.5, 3.0, 6.5, 9.9])
    expected = np.digitize(x, bins)
    np.testing.assert_array_equal(transform(x), expected)


def test_round():
    domain = FloatingPointColumn(0.0, 10.0/3)

    transformed_domain, transform = domain.transform.round(2).unpack()
    assert transformed_domain == FixedPointColumn(2, 0.0, 3.33)

    x = np.random.default_rng(123).uniform(0.0, 10/3, size=1000)
    y = transform(x)
    np.testing.assert_array_equal(y, np.round(x, 2))

    x_new = transform.inverse(y, random_state=123)
    assert np.all(np.abs(x_new - y) <= 0.01/2)
    assert np.all(x_new >= 0)
    assert np.all(x_new <= 10.0/3)


def test_round_uses_decimal_precision_for_bin_boundaries():
    transform = Round(FloatingPointColumn(0.0, 0.02), decimals=2)

    np.testing.assert_allclose(
        transform._pushforward_bins,
        [0.0, 0.005, 0.015, 0.02],
    )
    np.testing.assert_allclose(transform.pullback_lb(np.array([0.0, 0.01, 0.02])),
                               [0.0, 0.005, 0.015])
    np.testing.assert_allclose(transform.pullback_ub(np.array([0.0, 0.01, 0.02])),
                               [0.005, 0.015, 0.02])


def test_add_noise_accepts_scalars_and_preserves_array_shape():
    transform = AddNoise(
        FloatingPointColumn(-1.0, 2.0),
        uniform(loc=-0.25, scale=0.5),
    )
    assert transform.output_domain == FloatingPointColumn(-1.25, 2.25)

    scalar = transform(1.0, random_state=42)
    assert np.ndim(scalar) == 0
    assert 0.75 <= scalar <= 1.25

    values = np.ones((2, 3))
    noisy = transform(values, random_state=42)
    assert noisy.shape == values.shape
    assert np.all((noisy >= 0.75) & (noisy <= 1.25))


def test_add_noise_is_not_registered_as_a_transformation():
    assert not hasattr(FloatingPointColumn().transform, 'add_noise')


# TODO: add tests of .inverse


def test_zero_digit_round_preserves_float_representation_until_to_int():
    source = FloatingPointColumn(-2.7, 3.2)
    output, transform = source.transform.round().unpack()
    assert output == FixedPointColumn(0, -3, 3)
    values = np.array([-2.7, -0.5, 1.5, 3.2])
    rounded = transform.apply(values)
    assert rounded.dtype.kind == 'f'
    np.testing.assert_array_equal(rounded, np.round(values))
    integer_domain, to_int = output.transform.to_int().unpack()
    assert integer_domain == IntegerColumn(-3, 3)
    integers = to_int.apply(rounded)
    assert integers.dtype.kind == 'i'
    np.testing.assert_array_equal(integers, rounded)
