import numpy as np
import pytest

from dicer.domain import Column, FloatingPointColumn, FixedPointColumn, IntegerColumn
from dicer.tracking import ColumnTracker


def test_fixed_point_assumptions_quantize_bounds():
    domain = Column.from_dtype(np.float64)
    assert isinstance(domain, FloatingPointColumn)

    fixed_domain = domain.assume.fixed_point(2)
    assert isinstance(fixed_domain, FixedPointColumn)
    assert fixed_domain.decimals == 2

    fixed_domain = fixed_domain.assume.ge(1.234)
    assert fixed_domain.lb == pytest.approx(1.24)

    fixed_domain = fixed_domain.assume.le(3.456)
    assert fixed_domain.ub == pytest.approx(3.45)


def test_floating_point_assumptions_bounds():
    domain = Column.from_dtype(np.float64)
    assert isinstance(domain, FloatingPointColumn)

    domain = domain.assume.ge(-1.5)
    assert domain.lb == -1.5

    domain = domain.assume.le(2.5)
    assert domain.ub == 2.5


def test_integer_assumptions_bounds():
    domain = Column.from_dtype(np.int64)
    assert isinstance(domain, IntegerColumn)

    domain = domain.assume.ge(-3)
    assert domain.lb == -3

    domain = domain.assume.le(5)
    assert domain.ub == 5


def test_assumptions_on_a_tracker():
    domain = Column.from_dtype(np.float64)
    tracker = ColumnTracker(domain)

    domain = domain.assume.ge(-1.5)
    tracker = tracker.assume.ge(-1.5)

    assert tracker.unpack('domain') == domain


def test_tracked_assumption_validates_data():
    tracker = ColumnTracker(
        Column.from_dtype(np.float64),
        data=np.array([-1.0, 0.0, 1.0]),
    )

    with pytest.raises(ValueError, match='Values must lie within'):
        tracker.assume.ge(0.0)

    # The failed assumption does not modify the persistent column tracker.
    assert tracker.unpack('domain').lb == -np.inf


def test_assumptions_reject_contradictory_bounds():
    domain = FloatingPointColumn().assume.ge(10.0)

    with pytest.raises(ValueError, match='Lower bound'):
        domain.assume.le(5.0)


@pytest.mark.parametrize(
    'make_domain',
    [
        lambda: FloatingPointColumn(2.0, 1.0),
        lambda: FixedPointColumn(2, 2.0, 1.0),
        lambda: IntegerColumn(2, 1),
    ],
)
def test_ordered_domain_constructors_reject_contradictory_bounds(make_domain):
    with pytest.raises(ValueError, match='Lower bound'):
        make_domain()


def test_ordered_domain_constructor_requires_scalar_bounds():
    with pytest.raises(ValueError, match='bound must be a scalar'):
        FloatingPointColumn(
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0]),
        )


@pytest.mark.parametrize('digits', [-1, 1.5])
def test_fixed_point_constructor_rejects_invalid_digits(digits):
    with pytest.raises(ValueError, match='digits'):
        FixedPointColumn(digits)


def test_fixed_point_constructor_accepts_integral_float_digits():
    assert FixedPointColumn(2.0).decimals == 2


def test_integer_constructor_rejects_noninteger_bounds():
    with pytest.raises(ValueError, match='bound must be an integer'):
        IntegerColumn(0.5, 2)


def test_fixed_point_domain_validates_precision():
    domain = FixedPointColumn(2, 0.0, 1.0)

    domain.validate(np.array([0.0, 0.25, 1.0]))
    with pytest.raises(ValueError, match='decimal places'):
        domain.validate(np.array([0.001]))
