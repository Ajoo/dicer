import numpy as np
import pandas as pd
import pytest
from scipy.stats import randint

from dicer import ColumnTracker, FrameTracker
from dicer.domain import FixedPointColumn, IntegerColumn
from dicer.transform.continuous import FixedPointToInt, IntToFixedPoint


@pytest.mark.parametrize('digits', [0, 2])
def test_to_int_bounds_propagate_to_source_and_inverse(digits):
    h = 10**-digits
    source = FixedPointColumn(digits)
    original = ColumnTracker(source, np.array([1., 2., 3.])*h).transform.to_int()
    fit = original.infer.lb().infer.ub().finish()
    transformation = fit.transformation
    assert transformation.input_domain == FixedPointColumn(digits, h, 3*h)
    assert transformation.output_domain == IntegerColumn(1, 3)
    assert transformation.inverse.input_domain == transformation.output_domain
    assert transformation.inverse.output_domain == transformation.input_domain
    assert original.transformations.input_domain == source
    np.testing.assert_allclose(transformation.inverse.apply(fit.data), np.array([1, 2, 3])*h)
    pulled = transformation.pullback_distribution(randint(1, 4))
    np.testing.assert_allclose(pulled.pmf(np.array([1, 2, 3])*h), 1/3)


def test_to_fixed_point_bounds_propagate_through_chain():
    tracker = ColumnTracker(IntegerColumn(), np.array([12, 23, 34]))
    tracker = tracker.transform.to_fixed_point(2).transform.to_int()
    fit = tracker.assume.ge(10).assume.le(40).finish()
    assert fit.transformation.input_domain == IntegerColumn(10, 40)
    assert fit.transformation.output_domain == IntegerColumn(10, 40)
    np.testing.assert_array_equal(fit.transformation.apply([12, 23, 34]), fit.data)


def test_series_restriction():
    a = IntToFixedPoint(IntegerColumn(), 2)
    b = FixedPointToInt(a.output_domain)
    restricted = (b * a).restrict_output_domain(IntegerColumn(10, 40))
    assert restricted.input_domain == IntegerColumn(10, 40)
    assert restricted.output_domain == IntegerColumn(10, 40)
    assert restricted.transformations[0].output_domain == restricted.transformations[1].input_domain


def test_frame_bounds_after_to_int_replay_and_validate():
    source = pd.DataFrame({'x': [1., 2., 3.], 'y': [5., 6., 7.]})
    fit = (FrameTracker.from_dataframe(source)
           .infer(continuous=True).fixed_point()
           .transform(continuous=True).to_int()
           .infer().lb().infer().ub().finish())
    assert fit.transformation.input_domain['x'] == FixedPointColumn(0, 1, 3)
    assert fit.transformation.output_domain['y'] == IntegerColumn(5, 7)
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)
    with pytest.raises(ValueError):
        fit.transformation.validate(pd.DataFrame({'x': [0.], 'y': [5.]}))
    assert fit.transformation['x'].input_domain == FixedPointColumn(0, 1, 3)


def test_derived_field_restriction_updates_sibling_conversions():
    source = pd.DataFrame({'x': [12, 23, 34]})
    fit = (FrameTracker.from_dataframe(source)
           .transform('x', into='a').to_fixed_point(2)
           .transform('x', into='b').to_fixed_point(1)
           .assume['a'].ge(.1).assume['a'].le(.4)
           .finish())
    assert fit.transformation.input_domain['x'] == IntegerColumn(10, 40)
    assert fit.transformation.output_domain['x'] == IntegerColumn(10, 40)
    assert fit.transformation.output_domain['a'] == FixedPointColumn(2, .1, .4)
    assert fit.transformation.output_domain['b'] == FixedPointColumn(1, 1, 4)
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)


def test_unsupported_sibling_restriction_is_atomic():
    source = pd.DataFrame({'x': [1, 2, 3]})
    tracker = (FrameTracker.from_dataframe(source)
               .transform('x', into='a').to_fixed_point()
               .transform('x', into='b').affine(1, 1))
    with pytest.raises(NotImplementedError, match='cannot propagate'):
        tracker.assume['a'].ge(0)
    assert tracker.unpack('domain')['x'] == IntegerColumn()
    tracker.finish()


def test_unsupported_multi_column_inference_is_atomic():
    source = pd.DataFrame({'x': [1., 2.], 'y': [3., 4.]})
    tracker = (FrameTracker.from_dataframe(source)
               .assume['x'].fixed_point(0)
               .transform['x'].to_int()
               .transform['y'].affine(2., 0.))
    with pytest.raises(NotImplementedError, match='cannot propagate'):
        tracker.infer().lb()
    assert tracker.unpack('domain')['x'] == IntegerColumn()
    tracker.finish()


def test_invalid_assumption_does_not_modify_column_pipeline():
    tracker = ColumnTracker(FixedPointColumn(0), np.array([1., 2.])).transform.to_int()
    with pytest.raises(ValueError):
        tracker.assume.ge(2)
    assert tracker.finish().transformation.input_domain == FixedPointColumn(0)
