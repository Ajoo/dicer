import numpy as np
import pandas as pd
import pytest

from dicer import FrameTracker
from dicer import domain as d


def test_frame_transform_into_retains_source_and_replays_derived_column():
    source = pd.DataFrame({'x': [1.0, 2.0], 'z': [3.0, 4.0]})

    fit = (
        FrameTracker.from_dataframe(source)
        .transform('x', into='twice_x').affine(2.0, 0.0)
        .finish()
    )

    assert list(fit.data.columns) == ['x', 'z', 'twice_x']
    np.testing.assert_allclose(fit.data['x'], source['x'])
    np.testing.assert_allclose(fit.data['twice_x'], [2.0, 4.0])
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)
    assert not fit.transformation.is_lossy


def test_derived_column_observes_operation_order_before_source_replacement():
    source = pd.DataFrame({'x': [1.0, 2.0]})

    fit = (
        FrameTracker.from_dataframe(source)
        .transform('x', into='twice_x').affine(2.0, 0.0)
        .transform('x').affine(3.0, 0.0)
        .finish()
    )

    np.testing.assert_allclose(fit.data['x'], [3.0, 6.0])
    np.testing.assert_allclose(fit.data['twice_x'], [2.0, 4.0])
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)


def test_one_hot_assumption_groups_physical_columns_as_one_logical_column():
    source = pd.DataFrame({
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'value': [10.0, 20.0, 30.0],
    })

    tracker = (
        FrameTracker.from_dataframe(source)
        .assume(['red', 'green', 'blue'], into='color').one_hot()
    )
    frame_domain = tracker.unpack('domain')

    assert list(frame_domain.column_names) == ['color', 'value']
    assert frame_domain.components('color') == ('red', 'green', 'blue')
    assert frame_domain.physical_columns == ('red', 'green', 'blue', 'value')
    assert frame_domain['color'] == d.OneHotColumn(
        ['red', 'green', 'blue'],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        frame_domain.extract(tracker.unpack('data'), 'color'),
        source[['red', 'green', 'blue']],
    )


def test_invalid_group_assumption_does_not_consume_tracker():
    source = pd.DataFrame({'a': [1, 1], 'b': [0, 1]})
    tracker = FrameTracker.from_dataframe(source)

    with pytest.raises(ValueError, match='exactly one active'):
        tracker.assume(['a', 'b'], into='category').one_hot()

    pd.testing.assert_frame_equal(tracker.unpack('data'), source)


def test_one_hot_can_be_converted_to_categorical_in_place():
    source = pd.DataFrame({
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
    })

    fit = (
        FrameTracker.from_dataframe(source)
        .assume(['red', 'green', 'blue'], into='color').one_hot()
        .transform('color').to_categorical()
        .finish()
    )

    assert list(fit.data.columns) == ['color']
    assert isinstance(fit.data['color'].dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(fit.data['color'], ['red', 'green', 'blue'])
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)


def test_categorical_can_derive_one_hot_with_default_component_names():
    dtype = pd.CategoricalDtype(['red', 'green', 'blue'])
    source = pd.DataFrame({
        'color': pd.Categorical(['blue', 'red'], dtype=dtype),
    })

    fit = (
        FrameTracker.from_dataframe(source)
        .transform('color', into='color_one_hot').one_hot_encode(dtype=np.int8)
        .finish()
    )

    assert list(fit.data.columns) == [
        'color',
        'color_one_hot.0',
        'color_one_hot.1',
        'color_one_hot.2',
    ]
    np.testing.assert_array_equal(
        fit.data[['color_one_hot.0', 'color_one_hot.1', 'color_one_hot.2']],
        [[0, 0, 1], [1, 0, 0]],
    )
    assert fit.transformation.output_domain['color_one_hot'] == d.OneHotColumn(
        ['red', 'green', 'blue'],
        dtype=np.int8,
    )


@pytest.mark.parametrize(
    ('assumption', 'expected_type'),
    [
        ('simplex', d.SimplexColumn),
        ('vector', d.VectorColumn),
    ],
)
def test_structured_group_assumptions(assumption, expected_type):
    if assumption == 'simplex':
        source = pd.DataFrame({'x': [0.25], 'y': [0.75]})
    else:
        source = pd.DataFrame({'x': [1.0], 'y': [2.0]})

    descriptor = FrameTracker.from_dataframe(source).assume(
        ['x', 'y'],
        into='point',
    )
    tracker = getattr(descriptor, assumption)()

    assert isinstance(tracker.unpack('domain')['point'], expected_type)
    assert tracker.unpack('domain').components('point') == ('x', 'y')


def test_unlabeled_simplex_argmax_returns_unordered_integer_codes():
    simplex = d.SimplexColumn(3)

    output_domain, transformation = simplex.transform.argmax().unpack()

    assert output_domain == d.UnorderedIntegerColumn(3)
    np.testing.assert_array_equal(
        transformation([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]),
        [1, 0],
    )


def test_labeled_simplex_argmax_returns_vertex_labels_and_uses_first_tie():
    simplex = d.SimplexColumn(3, labels=('red', 'green', 'blue'))

    output_domain, transformation = simplex.transform.argmax().unpack()

    assert output_domain == d.FiniteUnorderedColumn(
        ['red', 'green', 'blue']
    )
    np.testing.assert_array_equal(
        transformation([[0.1, 0.7, 0.2], [0.5, 0.5, 0.0]]),
        ['green', 'red'],
    )


def test_frame_simplex_argmax_can_derive_labeled_prediction():
    source = pd.DataFrame({
        'p0': [0.1, 0.8],
        'p1': [0.7, 0.1],
        'p2': [0.2, 0.1],
    })

    fit = (
        FrameTracker.from_dataframe(source)
        .assume(['p0', 'p1', 'p2'], into='probabilities').simplex(
            labels=('red', 'green', 'blue')
        )
        .transform('probabilities', into='prediction').argmax()
        .finish()
    )

    assert list(fit.data.columns) == ['p0', 'p1', 'p2', 'prediction']
    np.testing.assert_array_equal(fit.data['prediction'], ['green', 'red'])
    pd.testing.assert_frame_equal(fit.transformation.apply(source), fit.data)


def test_conditional_rates_uses_named_fit_observation_and_defaults_components():
    group_dtype = pd.CategoricalDtype(['a', 'b', 'unseen'])
    outcome_dtype = pd.CategoricalDtype(['u', 'v'])
    train = pd.DataFrame({
        'group': pd.Categorical(['a', 'a', 'b', 'b', 'b'], dtype=group_dtype),
        'outcome': pd.Categorical(['u', 'v', 'v', 'v', 'u'], dtype=outcome_dtype),
    })
    holdout = pd.DataFrame({
        'group': pd.Categorical(['unseen', 'a', 'b'], dtype=group_dtype),
        'outcome': pd.Categorical(['v', 'v', 'u'], dtype=outcome_dtype),
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .infer(
            'group',
            into='outcome_rates',
            using={'outcome': 'outcome'},
        ).conditional_rates(smoothing=1.0)
        .finish()
    )

    assert fit.transformation.output_domain['outcome_rates'] == d.SimplexColumn(
        2,
        labels=('u', 'v'),
    )
    assert list(fit.data.columns) == [
        'group', 'outcome', 'outcome_rates.0', 'outcome_rates.1'
    ]
    replayed = fit.transformation.apply(holdout)
    np.testing.assert_allclose(
        replayed[['outcome_rates.0', 'outcome_rates.1']],
        [[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]],
    )


def test_conditional_rates_accepts_explicit_output_components():
    group_dtype = pd.CategoricalDtype(['a', 'b'])
    outcome_dtype = pd.CategoricalDtype(['no', 'yes'])
    train = pd.DataFrame({
        'group': pd.Categorical(['a', 'a', 'b'], dtype=group_dtype),
        'outcome': pd.Categorical(['yes', 'no', 'yes'], dtype=outcome_dtype),
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .infer(
            'group',
            into='rates',
            components=('rate_no', 'rate_yes'),
            using={'outcome': 'outcome'},
        ).conditional_rates()
        .finish()
    )

    assert list(fit.data.columns) == [
        'group', 'outcome', 'rate_no', 'rate_yes'
    ]


def test_conditional_rates_accepts_multiple_named_using_columns():
    group_dtype = pd.CategoricalDtype(['a'])
    outcome_dtype = pd.CategoricalDtype(['u', 'v'])
    train = pd.DataFrame({
        'group': pd.Categorical(['a', 'a'], dtype=group_dtype),
        'outcome': pd.Categorical(['u', 'v'], dtype=outcome_dtype),
        'weight': [3.0, 1.0],
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .infer(
            'group',
            into='rates',
            using={'outcome': 'outcome', 'weights': 'weight'},
        ).conditional_rates(smoothing=0.0)
        .finish()
    )

    np.testing.assert_allclose(
        fit.data[['rates.0', 'rates.1']],
        [[0.75, 0.25], [0.75, 0.25]],
    )


def test_into_rejects_multiple_selected_sources():
    tracker = FrameTracker.from_dataframe(pd.DataFrame({
        'x': [1.0],
        'y': [2.0],
    }))

    with pytest.raises(ValueError, match='exactly one'):
        tracker.transform(['x', 'y'], into='z').affine(2.0, 0.0)


def test_components_must_match_structured_output_width():
    dtype = pd.CategoricalDtype(['a', 'b'])
    tracker = FrameTracker.from_dataframe(pd.DataFrame({
        'category': pd.Categorical(['a'], dtype=dtype),
    }))

    with pytest.raises(ValueError, match='requires 2 physical components'):
        tracker.transform(
            'category',
            into='encoded',
            components=('only_one',),
        ).one_hot_encode()
