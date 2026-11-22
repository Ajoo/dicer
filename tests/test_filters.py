"""Tests for BoolPropertyFilter, NumericRangeFilter, and their integration
with Frame descriptor __call__ syntax."""
import numpy as np
import pandas as pd
import pytest

from dicer import Frame, FrameTracker
from dicer.domain import (
    FloatingPointColumn,
    FixedPointColumn,
    IntegerColumn,
    FiniteOrderedColumn,
    FiniteUnorderedColumn,
)
from dicer.tracking import BoolPropertyFilter, NumericRangeFilter


# ── helpers ───────────────────────────────────────────────────────────────────

def make_frame():
    """Frame with float, int, and bounded-float columns."""
    return Frame.from_dtypes({
        'price': np.float64,   # FloatingPointColumn, unbounded
        'count': np.int32,     # IntegerColumn, semi-bounded (lb=0)
        'ratio': np.float64,   # FloatingPointColumn, unbounded
    })


def make_mixed_frame():
    """Frame with a fixed-point and finite-ordered column."""
    return Frame({
        'rate': FixedPointColumn(2, 0.0, 1.0),        # continuous, bounded, 2 decimals
        'score': IntegerColumn(0, 9),                   # discrete, bounded, cardinality=10
        'label': FiniteOrderedColumn([1, 2, 3, 4, 5]), # finite, ordered, cardinality=5
    })


# ── BoolPropertyFilter unit tests ─────────────────────────────────────────────

class TestBoolPropertyFilter:

    def test_matches_true_property(self):
        col = FloatingPointColumn()
        f = BoolPropertyFilter('continuous', True)
        assert f.matches('x', col) is True

    def test_no_match_wrong_value(self):
        col = FloatingPointColumn()
        f = BoolPropertyFilter('continuous', False)
        assert f.matches('x', col) is False

    def test_no_match_property_absent(self):
        from dicer.domain import GenericColumn
        col = GenericColumn(np.dtype('O'))
        f = BoolPropertyFilter('continuous', True)
        assert f.matches('x', col) is False

    def test_no_match_property_absent_false(self):
        from dicer.domain import GenericColumn
        col = GenericColumn(np.dtype('O'))
        f = BoolPropertyFilter('continuous', False)
        assert f.matches('x', col) is False

    def test_is_ordered_true(self):
        col = IntegerColumn()
        f = BoolPropertyFilter('ordered', True)
        assert f.matches('x', col) is True

    def test_is_ordered_false(self):
        col = FiniteUnorderedColumn([1, 2, 3])
        f = BoolPropertyFilter('ordered', False)
        assert f.matches('x', col) is True

    def test_is_bounded_true(self):
        col = FloatingPointColumn(0.0, 1.0)
        f = BoolPropertyFilter('bounded', True)
        assert f.matches('x', col) is True

    def test_is_bounded_false_unbounded(self):
        col = FloatingPointColumn()
        f = BoolPropertyFilter('bounded', False)
        assert f.matches('x', col) is True

    def test_is_bounded_left(self):
        col = FloatingPointColumn(0.0)  # lb=0, ub=inf
        f_left = BoolPropertyFilter('bounded_left', True)
        f_right = BoolPropertyFilter('bounded_right', True)
        assert f_left.matches('x', col) is True
        assert f_right.matches('x', col) is False

    def test_finite_finite_column(self):
        col = IntegerColumn(0, 9)
        f = BoolPropertyFilter('finite', True)
        assert f.matches('x', col) is True

    def test_finite_unbounded_integer(self):
        col = IntegerColumn()
        f = BoolPropertyFilter('finite', False)
        assert f.matches('x', col) is True

    def test_finite_column_always_bounded(self):
        col = FiniteOrderedColumn([1, 2, 3])
        assert BoolPropertyFilter('bounded', True).matches('x', col) is True
        assert BoolPropertyFilter('bounded_left', True).matches('x', col) is True
        assert BoolPropertyFilter('bounded_right', True).matches('x', col) is True


# ── NumericRangeFilter unit tests ─────────────────────────────────────────────

class TestNumericRangeFilter:

    def test_cardinality_in_range(self):
        col = IntegerColumn(0, 9)  # cardinality = 10
        f = NumericRangeFilter('cardinality', max_val=100)
        assert f.matches('x', col) is True

    def test_cardinality_out_of_range(self):
        col = IntegerColumn(0, 9)  # cardinality = 10
        f = NumericRangeFilter('cardinality', max_val=5)
        assert f.matches('x', col) is False

    def test_cardinality_continuous_excluded(self):
        col = FloatingPointColumn()  # cardinality = -1
        f = NumericRangeFilter('cardinality', max_val=100)
        assert f.matches('x', col) is False

    def test_cardinality_property_absent(self):
        from dicer.domain import GenericColumn
        col = GenericColumn(np.dtype('O'))
        f = NumericRangeFilter('cardinality', max_val=100)
        assert f.matches('x', col) is False

    def test_min_cardinality(self):
        col = IntegerColumn(0, 4)  # cardinality = 5
        assert NumericRangeFilter('cardinality', min_val=3).matches('x', col) is True
        assert NumericRangeFilter('cardinality', min_val=6).matches('x', col) is False

    def test_both_bounds(self):
        col = IntegerColumn(0, 9)  # cardinality = 10
        assert NumericRangeFilter('cardinality', min_val=5, max_val=15).matches('x', col) is True
        assert NumericRangeFilter('cardinality', min_val=11, max_val=20).matches('x', col) is False

    def test_decimals_in_range(self):
        col = FixedPointColumn(2, 0.0, 1.0)  # decimals = 2
        assert NumericRangeFilter('decimals', max_val=3).matches('x', col) is True
        assert NumericRangeFilter('decimals', min_val=3).matches('x', col) is False

    def test_decimals_absent(self):
        col = FloatingPointColumn()  # no decimals attribute
        f = NumericRangeFilter('decimals', max_val=5)
        assert f.matches('x', col) is False


# ── Integration: Frame.assume(__call__ filters) ───────────────────────────────

class TestFrameAssumeBoolFilter:

    def test_continuous_only_updates_floats(self):
        d = make_frame()
        d2 = d.assume(continuous=True).ge(0.0)
        assert d2['price'].lb == 0.0
        assert d2['ratio'].lb == 0.0
        assert d2['count'].lb == -np.inf  # IntegerColumn not touched

    def test_ordered_updates_both_float_and_int(self):
        d = make_frame()
        d2 = d.assume(ordered=True).ge(0.0)
        assert d2['price'].lb == 0.0
        assert d2['ratio'].lb == 0.0
        assert d2['count'].lb == 0.0

    def test_bounded_false_selects_unbounded_columns(self):
        d = make_frame()
        d2 = d.assume(bounded=False).ge(0.0).assume(bounded=False).le(1.0)
        # All float columns start unbounded; after ge+le they become bounded
        assert d2['price'].lb == 0.0
        assert d2['price'].ub == 1.0

    def test_no_match_returns_unchanged_frame(self):
        d = make_frame()
        # Request continuous=False — none of our columns match (all are float or int)
        # Actually IntegerColumn has is_continuous=False, so count matches
        d2 = d.assume(continuous=False).ge(5)
        assert d2['count'].lb == 5
        assert d2['price'].lb == -np.inf

    def test_combined_filters(self):
        d = make_frame()
        # continuous AND bounded_left=False → only float cols with lb=-inf
        d2 = d.assume(continuous=True, bounded_left=False).ge(0.0)
        assert d2['price'].lb == 0.0
        assert d2['ratio'].lb == 0.0

    def test_name_and_bool_filter_combined(self):
        d = make_frame()
        d2 = d.assume('price', continuous=True).ge(0.0)
        assert d2['price'].lb == 0.0
        assert d2['ratio'].lb == -np.inf  # ratio excluded by name filter


class TestFrameAssumeNumericFilter:

    def test_max_cardinality_selects_small_discrete(self):
        d = Frame({'a': IntegerColumn(0, 4), 'b': IntegerColumn(0, 99)})
        # a: cardinality=5 (passes), b: cardinality=100 (excluded)
        d2 = d.assume(max_cardinality=10).ge(2)
        assert d2['a'].lb == 2
        assert d2['b'].lb == 0

    def test_min_cardinality(self):
        d = Frame({'a': IntegerColumn(0, 4), 'b': IntegerColumn(0, 99)})
        d2 = d.assume(min_cardinality=50).ge(10)
        assert d2['a'].lb == 0   # cardinality=5, excluded
        assert d2['b'].lb == 10  # cardinality=100, passes

    def test_max_decimals(self):
        d = Frame({
            'x': FixedPointColumn(1, 0.0, 10.0),  # decimals=1
            'y': FixedPointColumn(3, 0.0, 10.0),  # decimals=3
        })
        d2 = d.assume(max_decimals=2).ge(1.0)
        assert d2['x'].lb == 1.0   # 1 decimal, passes
        assert d2['y'].lb == 0.0   # 3 decimals, excluded

    def test_min_decimals(self):
        d = Frame({
            'x': FixedPointColumn(1, 0.0, 10.0),
            'y': FixedPointColumn(3, 0.0, 10.0),
        })
        d2 = d.assume(min_decimals=2).ge(1.0)
        assert d2['x'].lb == 0.0   # 1 decimal, excluded
        assert d2['y'].lb == 1.0   # 3 decimals, passes

    def test_continuous_excluded_from_cardinality_filter(self):
        d = make_frame()
        # No float column should be selected; only int if cardinality allows
        d2 = d.assume(max_cardinality=100).ge(1)
        assert d2['price'].lb == -np.inf
        assert d2['ratio'].lb == -np.inf


# ── Integration: FrameInferences ──────────────────────────────────────────────

class TestFrameInferences:

    def test_infer_nonnegative_continuous_only(self):
        ft = FrameTracker.from_dataframe(pd.DataFrame({
            'price': [1.0, 2.0, 3.0],
            'count': [1, 2, 3],
            'ratio': [0.1, 0.5, 0.9],
        }))
        ft2 = ft.infer(continuous=True).nonnegative()
        domain = ft2.unpack('domain')
        assert domain['price'].lb == 0.0
        assert domain['ratio'].lb == 0.0
        # count is IntegerColumn (is_continuous=False) — not selected by filter, untouched
        assert domain['count'].lb == -np.inf

    def test_infer_requires_tracker_with_data(self):
        ft = FrameTracker.from_dtypes({'price': np.float64})
        with pytest.raises(Exception):
            ft.infer(continuous=True).nonnegative()

    def test_transforming_inference_updates_data_and_pipeline_without_mutating_source(self):
        source = pd.DataFrame({
            'price': [1.0, 2.0, 3.0],
            'count': [10, 20, 30],
        })
        ft = FrameTracker.from_dataframe(source)

        fitted = ft.infer(continuous=True).standardize()

        with pytest.raises(RuntimeError, match='already been consumed'):
            ft.unpack('data')
        pd.testing.assert_frame_equal(source, pd.DataFrame({
            'price': [1.0, 2.0, 3.0],
            'count': [10, 20, 30],
        }))
        np.testing.assert_allclose(
            fitted.unpack('data')['price'],
            [-np.sqrt(1.5), 0.0, np.sqrt(1.5)],
        )
        np.testing.assert_array_equal(
            fitted.unpack('data')['count'],
            source['count'],
        )
        assert [
            operation.target for operation in fitted.transformations
        ] == ['price']

        fit = fitted.finish()
        pd.testing.assert_frame_equal(
            fit.transformation.apply(source),
            fit.data,
        )

    def test_unsupported_domain_inference_leaves_tracker_usable(self):
        ft = FrameTracker.from_dataframe(pd.DataFrame({
            'price': [1.0, 2.0, 3.0],
        }))
        transformed = ft.transform['price'].affine(2.0, 0.0)

        with pytest.raises(NotImplementedError, match='cannot propagate'):
            transformed.infer(continuous=True).nonnegative()
        assert len(transformed.transformations) == 1
        np.testing.assert_allclose(transformed.unpack('data')['price'], [2.0, 4.0, 6.0])
        assert transformed.unpack('domain')['price'].lb == -np.inf
        transformed.finish()

    def test_quantile_discretize_preserves_frame_structure_and_categorical_data(self):
        source = pd.DataFrame({
            'x': np.linspace(0.0, 1.0, 20),
            'label': pd.Categorical(['a', 'b'] * 10),
        }, index=np.arange(100, 120))
        ft = (FrameTracker
            .from_dataframe(source)
            .assume['x'].ge(0.0)
            .assume['x'].le(1.0)
        )

        fitted = ft.infer['x'].quantile_discretize(num_bins=4)
        transformed = fitted.unpack('data')

        with pytest.raises(RuntimeError, match='already been consumed'):
            ft.unpack('data')
        pd.testing.assert_series_equal(
            source['x'],
            pd.Series(np.linspace(0.0, 1.0, 20), index=np.arange(100, 120), name='x'),
        )
        assert transformed.index.equals(source.index)
        assert list(transformed.columns) == list(source.columns)
        pd.testing.assert_series_equal(transformed['label'], source['label'])
        assert transformed['x'].between(0, 3).all()
        assert [operation.target for operation in fitted.transformations] == ['x']

        fit = fitted.finish()
        replayed = fit.transformation.apply(source)
        pd.testing.assert_frame_equal(replayed, fit.data)
        pd.testing.assert_series_equal(replayed['label'], source['label'])


def test_zero_decimal_filter_and_explicit_integer_conversion():
    frame = Frame({
        'whole': FixedPointColumn(0, 0, 10),
        'fractional': FixedPointColumn(2, 0, 10),
        'float': FloatingPointColumn(0, 10),
        'int': IntegerColumn(0, 10),
    })
    selected = frame.transform(continuous=True, max_decimals=0).to_int().unpack('domain')
    assert selected['whole'] == IntegerColumn(0, 10)
    assert selected['fractional'] == frame['fractional']
    assert selected['float'] == frame['float']
    assert selected['int'] == frame['int']

    all_fixed = frame.transform(continuous=True, min_decimals=0).to_int().unpack('domain')
    assert all_fixed['whole'] == IntegerColumn(0, 10)
    assert all_fixed['fractional'] == IntegerColumn(0, 1000)
    assert all_fixed['float'] == frame['float']
