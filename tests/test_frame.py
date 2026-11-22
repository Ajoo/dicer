import numpy as np
import pandas as pd
import pytest
from scipy.stats import uniform

from dicer import FitResult, Frame, FrameTracker, FrameTransformation
from dicer.domain import FixedPointColumn, FloatingPointColumn, IntegerColumn
from dicer.tracking import NameFilter, Tracker


# ── helpers ───────────────────────────────────────────────────────────────────

DTYPES = {
    'price': np.float64,
    'count': np.int32,
    'ratio': np.float64,
}


def make_frame():
    return Frame.from_dtypes(DTYPES)


def make_ft():
    return FrameTracker.from_dtypes(DTYPES)


def make_ft_with_data():
    return FrameTracker.from_dataframe(
        pd.DataFrame({
            'price': [30.0, 12.0, 1.1],
            'count': [3, 5, 23],
            'ratio': [1.1, 3.4, 0.78],
        })
    )

# ── Frame.from_dtypes ─────────────────────────────────────────────────────────

def test_from_dtypes_dict():
    d = make_frame()
    assert isinstance(d['price'], FloatingPointColumn)
    assert isinstance(d['count'], IntegerColumn)
    assert isinstance(d['ratio'], FloatingPointColumn)


def test_from_dtypes_pandas_series():
    df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in DTYPES.items()})
    d = Frame.from_dtypes(df.dtypes)
    assert isinstance(d['price'], FloatingPointColumn)
    assert isinstance(d['count'], IntegerColumn)


def test_with_domains_preserves_layout_and_rejects_new_logical_columns():
    frame = make_frame()
    refined = FloatingPointColumn(0.0, 10.0)

    updated = frame.with_domains({'price': refined})

    assert updated['price'] == refined
    assert updated.layout == frame.layout
    with pytest.raises(KeyError, match='Unknown logical columns'):
        frame.with_domains({'new': refined})


# ── Frame.assume — no filter (applies to all matching columns) ────────────────

def test_assume_no_filter_ge_updates_floats():
    d = make_frame()
    d2 = d.assume.ge(0.0)
    # Both float columns updated
    assert d2['price'].lb == 0.0
    assert d2['ratio'].lb == 0.0
    # Integer column also has ge for integers (int32 starts at -inf)
    assert d2['count'].lb == 0


def test_assume_ge_does_not_mutate_original():
    d = make_frame()
    _ = d.assume.ge(0.0)
    assert d['price'].lb == -np.inf


# ── Frame.assume — filter by exact name ───────────────────────────────────────

def test_assume_by_name_updates_only_named_column():
    d = make_frame()
    d2 = d.assume['price'].ge(0.0)
    assert d2['price'].lb == 0.0
    # ratio untouched
    assert d2['ratio'].lb == -np.inf


def test_assume_by_name_getitem_exact():
    d = make_frame()
    d2 = d.assume['count'].ge(1)
    assert d2['count'].lb == 1
    assert d2['price'].lb == -np.inf


def test_assume_call_with_name_kwarg():
    d = make_frame()
    d2 = d.assume(name='price').ge(0.0)
    assert d2['price'].lb == 0.0
    assert d2['ratio'].lb == -np.inf


# ── Frame.assume — filter by pattern ──────────────────────────────────────────

def test_assume_getitem_wildcard_pattern():
    d = make_frame()
    d2 = d.assume['p*'].ge(0.0)
    assert d2['price'].lb == 0.0
    assert d2['ratio'].lb == -np.inf  # ratio not matched by 'p*'
    assert d2['count'].lb == -np.inf  # count not matched by 'p*'


def test_assume_call_with_pattern_kwarg():
    d = make_frame()
    d2 = d.assume('*rice').ge(0.0)
    assert d2['price'].lb == 0.0
    assert d2['ratio'].lb == -np.inf


def test_assume_pattern_matches_all():
    d = make_frame()
    d2 = d.assume(pattern='*').ge(0.0)
    # All numeric columns updated; label (generic) silently skipped
    assert d2['price'].lb == 0.0
    assert d2['ratio'].lb == 0.0
    assert d2['count'].lb == 0


def test_assume_explicit_name_no_match_raises():
    d = make_frame()
    with pytest.raises(KeyError, match='No logical columns match'):
        d.assume['zzz_nomatch'].ge(0.0)


def test_assume_explicit_pattern_no_match_raises():
    d = make_frame()
    with pytest.raises(KeyError, match='No logical columns match'):
        d.assume(pattern='zzz_*').ge(0.0)


def test_assume_property_only_no_match_remains_a_noop():
    d = make_frame()
    assert d.assume(finite=True).ge(0.0) == d


def test_existing_name_filtered_out_by_property_remains_a_noop():
    d = make_frame()
    assert d.assume('price', finite=True).ge(0.0) == d


def test_explicit_transform_selection_no_match_raises_without_consuming_tracker():
    tracker = make_ft_with_data()

    with pytest.raises(KeyError, match='No logical columns match'):
        tracker.transform['zzz_*'].affine(2.0, 0.0)

    assert list(tracker.unpack('domain').column_names) == [
        'price',
        'count',
        'ratio',
    ]


def test_property_only_transform_selection_no_match_remains_a_noop():
    source = pd.DataFrame({'x': [1.0, 2.0]})

    fit = (
        FrameTracker.from_dataframe(source)
        .transform(finite=True).one_hot_encode()
        .finish()
    )

    pd.testing.assert_frame_equal(fit.data, source)
    assert fit.transformation.operations == ()


# ── Frame.assume — chaining ───────────────────────────────────────────────────

def test_assume_chaining():
    d = make_frame()
    d2 = d.assume['price'].ge(0.0).assume['price'].le(1000.0)
    assert d2['price'].lb == 0.0
    assert d2['price'].ub == 1000.0
    assert d2['ratio'].lb == -np.inf


# ── Frame.transform — domain updated, no tracking ─────────────────────────────

def test_transform_affine_updates_domain():
    d = make_frame()
    # price is unbounded float; affine(2, 0) → 2x, domain stays unbounded
    d2 = d.transform['price'].affine(2.0, 0.0).unpack('domain')
    assert isinstance(d2['price'], FloatingPointColumn)
    # ratio unchanged
    assert d2['ratio'] == d['ratio']


def test_transform_logit_updates_domain():
    # logit requires [0, 1] domain
    d = Frame.from_dtypes({'ratio': np.float64})
    d = d.assume['ratio'].ge(0.0).assume['ratio'].le(1.0)
    d2 = d.transform['ratio'].logit().unpack('domain')
    # logit maps (0,1) -> (-inf, inf); output domain should be unbounded
    assert d2['ratio'].lb == -np.inf
    assert d2['ratio'].ub == np.inf


# ── FrameTracker.from_dtypes ─────────────────────────────────────

def test_frametracker_from_dtypes():
    ft = make_ft()
    frame = ft.unpack('domain')
    assert isinstance(frame['price'], FloatingPointColumn)
    assert isinstance(frame['count'], IntegerColumn)


def test_frametracker_does_not_return_detached_column_trackers():
    ft = make_ft()

    with pytest.raises(TypeError, match='not subscriptable'):
        ft['price']

# ── FrameTracker.assume — no transformation recorded ─────────────────────────

def test_frametracker_assume_updates_domain():
    ft = make_ft()
    ft2 = ft.assume['price'].ge(0.0)
    assert ft2.unpack('domain')['price'].lb == 0.0


def test_frametracker_assume_consumes_original():
    ft = make_ft_with_data()
    owned_data = ft.unpack('data')
    ft2 = ft.assume['price'].ge(0.0)
    with pytest.raises(RuntimeError, match='already been consumed'):
        ft.unpack()
    assert ft2.unpack('data') is owned_data
    assert ft2.unpack('domain')['price'].lb == 0.0


def test_frametracker_assume_chaining():
    ft = make_ft()
    ft2 = ft.assume['ratio'].ge(0.0).assume['ratio'].le(1.0)
    ratio_domain = ft2.unpack('domain')['ratio']
    assert ratio_domain.lb == 0.0
    assert ratio_domain.ub == 1.0


def test_frametracker_invalid_assumption_does_not_consume_tracker():
    source = pd.DataFrame({'value': [-1.0, 1.0]})
    tracker = FrameTracker.from_dataframe(source)

    with pytest.raises(ValueError, match="Column 'value'"):
        tracker.assume['value'].ge(0.0)

    pd.testing.assert_frame_equal(tracker.unpack('data'), source)

# ── FrameTracker.transform — transformation recorded ─────────────────────────

def test_frametracker_transform_records_transformation():
    ft = make_ft()
    ft2 = ft.transform['price'].affine(2.0, 0.0)
    operations = ft2.transformations
    assert len(operations) == 1
    assert operations[0].source == 'price'
    assert operations[0].target == 'price'


def test_frametracker_transform_does_not_affect_other_columns():
    ft = make_ft()
    ft2 = ft.transform['price'].affine(2.0, 0.0)
    assert all(operation.target != 'ratio' for operation in ft2.transformations)


def test_frametracker_transform_consumes_original():
    ft = make_ft_with_data()
    ft2 = ft.transform['price'].affine(2.0, 0.0)
    with pytest.raises(RuntimeError, match='already been consumed'):
        ft.transform['ratio'].affine(3.0, 0.0)
    assert len(ft2.transformations) == 1


def test_from_dataframe_copies_input_once_before_mutating_owned_data():
    source = pd.DataFrame({
        'price': [30.0, 12.0, 1.1],
        'count': [3, 5, 23],
        'ratio': [1.1, 3.4, 0.78],
    })
    expected = source.copy()
    ft = FrameTracker.from_dataframe(source)
    owned_data = ft.unpack('data')
    assert owned_data is not source
    ft2 = (ft
        .transform['price']
        .affine(2.0, 0.0)
    )
    pd.testing.assert_frame_equal(source, expected)
    assert ft2.unpack('data') is owned_data
    np.testing.assert_allclose(ft2.unpack('data')['price'], [60.0, 24.0, 2.2])


def test_frametracker_rejects_branching_and_allows_linear_chaining():
    ft0 = make_ft_with_data()
    ft1 = ft0.transform['price'].affine(2.0, 0.0)

    with pytest.raises(RuntimeError, match='already been consumed'):
        ft0.assume['ratio'].ge(0.0)

    ft2 = ft1.transform['price'].affine(3.0, 0.0)
    with pytest.raises(RuntimeError, match='already been consumed'):
        ft1.unpack()
    assert len(ft2.transformations) == 2
    np.testing.assert_allclose(ft2.unpack('data')['price'], [180.0, 72.0, 6.6])


def test_frametracker_copy_is_rejected():
    ft = make_ft_with_data()
    with pytest.raises(RuntimeError, match='branching is not supported'):
        ft.copy()


def test_column_tracker_remains_branchable_and_keeps_independent_data():
    source = pd.Series([1.0, 2.0, 3.0])
    tracker = Tracker.from_series(source)

    doubled = tracker.transform.affine(2.0, 0.0)
    tripled = tracker.transform.affine(3.0, 0.0)

    np.testing.assert_allclose(tracker.unpack('data'), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(doubled.unpack('data'), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(tripled.unpack('data'), [3.0, 6.0, 9.0])


def test_column_tracker_finish_uses_shared_fit_result_without_consuming_tracker():
    tracker = Tracker.from_series(pd.Series([1.0, 2.0, 3.0]))
    transformed = tracker.transform.affine(2.0, 1.0)

    result = transformed.finish()

    assert isinstance(result, FitResult)
    assert result._fields == ('data', 'transformation')
    np.testing.assert_allclose(result.data, [3.0, 5.0, 7.0])
    np.testing.assert_allclose(
        result.transformation.apply(np.array([4.0, 5.0])),
        [9.0, 11.0],
    )
    # Column trackers remain persistent and usable after finishing.
    np.testing.assert_allclose(transformed.unpack('data'), [3.0, 5.0, 7.0])


def test_column_tracker_finish_without_transforms_returns_identity():
    tracker = Tracker.from_series(pd.Series([1.0, 2.0, 3.0]))

    result = tracker.finish()

    assert result.transformation.input_domain == tracker.unpack('domain')
    assert result.transformation.output_domain == tracker.unpack('domain')
    assert result.transformation.inverse is result.transformation
    assert result.transformation.apply(result.data) is result.data

# ── FrameTracker.unpack ───────────────────────────────────────────────────────

def test_unpack_returns_frame_and_transforms():
    ft = make_ft()
    ft2 = ft.transform['price'].affine(2.0, 0.0)
    output_frame, transforms = ft2.unpack()
    assert isinstance(output_frame, Frame)
    assert [(operation.source, operation.target) for operation in transforms] == [
        ('price', 'price')
    ]


def test_unpack_output_domain_correct():
    ft = FrameTracker.from_dtypes({'price': np.float64})
    ft2 = ft.assume['price'].ge(0.0).assume['price'].le(1.0).transform['price'].logit()
    output_frame, transforms = ft2.unpack()
    assert output_frame['price'].lb == -np.inf
    assert output_frame['price'].ub == np.inf
    assert [operation.target for operation in transforms] == ['price']


def test_unpack_with_no_transforms():
    ft = make_ft()
    ft2 = ft.assume['price'].ge(0.0)
    output_frame, transforms = ft2.unpack()
    assert transforms == ()
    assert output_frame['price'].lb == 0.0


# ── FrameTracker chained pipeline ────────────────────────────────────────────

def test_chained_assume_and_transform():
    ft = FrameTracker.from_dtypes({'ratio': np.float64, 'count': np.int32})
    output_frame, transforms = (
        ft
        .assume['ratio'].ge(0.0)
        .assume['ratio'].le(1.0)
        .transform['ratio'].logit()
        .unpack()
    )
    assert [operation.target for operation in transforms] == ['ratio']
    assert output_frame['ratio'].lb == -np.inf


# ── FrameTransformation replay ────────────────────────────────────────────────

def test_finish_returns_data_and_frame_transformation_and_consumes_tracker():
    source = pd.DataFrame({
        'price': [1.0, 2.0, 3.0],
        'count': [10, 20, 30],
    })
    tracker = (FrameTracker
        .from_dataframe(source)
        .assume['price'].ge(0.0)
        .transform['price'].affine(2.0, 1.0)
    )

    result = tracker.finish()

    assert isinstance(result, FitResult)
    assert isinstance(result, tuple)
    assert result._fields == ('data', 'transformation')
    assert isinstance(result.transformation, FrameTransformation)
    assert result.transformation.output_domain['price'].lb == 1.0
    np.testing.assert_allclose(result.data['price'], [3.0, 5.0, 7.0])
    with pytest.raises(RuntimeError, match='already been consumed'):
        tracker.unpack()


def test_frame_transformation_replays_pipeline_without_mutating_input():
    source = pd.DataFrame({
        'price': [1.0, 2.0, 3.0],
        'count': [10, 20, 30],
        'ratio': [0.1, 0.5, 0.9],
    }, index=[4, 7, 9])
    original = source.copy(deep=True)
    fit = (FrameTracker
        .from_dataframe(source)
        .assume['price'].ge(0.0)
        .transform['price'].affine(2.0, 1.0)
        .transform['price'].affine(3.0, -1.0)
        .transform['ratio'].affine(10.0, 0.0)
        .finish()
    )

    replayed = fit.transformation.apply(source)

    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(replayed, fit.data)
    assert replayed is not source
    assert replayed.index.equals(source.index)
    assert list(replayed.columns) == list(source.columns)


def test_frame_transformation_getitem_returns_composed_column_lineage():
    source = pd.DataFrame({
        'x': [1.0, 2.0],
        'z': [3.0, 4.0],
    })
    fit = (
        FrameTracker.from_dataframe(source)
        .transform('x').affine(3.0, 1.0)
        .transform('x', into='y').affine(2.0, -1.0)
        .transform('x').affine(-1.0, 4.0)
        .finish()
    )

    assert not hasattr(fit.transformation, '_column_transformations')

    np.testing.assert_allclose(
        fit.transformation['x'](source['x'].to_numpy()),
        fit.data['x'],
    )
    np.testing.assert_allclose(
        fit.transformation['y'](source['x'].to_numpy()),
        fit.data['y'],
    )
    np.testing.assert_allclose(
        fit.transformation['z'](source['z'].to_numpy()),
        fit.data['z'],
    )

    pulled = fit.transformation['y'].pullback_distribution(
        uniform(loc=1.0, scale=6.0),
    )
    assert pulled.pdf(0.5) == pytest.approx(1.0)

    with pytest.raises(KeyError, match="Unknown output column 'missing'"):
        fit.transformation['missing']


def test_frame_column_pullback_can_disable_fixed_point_atomization():
    source = pd.DataFrame({'a': [0.0, 0.5, 1.0]})
    fit = (
        FrameTracker.from_dataframe(source)
        .assume['a'].fixed_point(2)
        .assume['a'].ge(0.0)
        .assume['a'].le(1.0)
        .transform['a'].affine(2.0, 1.0)
        .finish()
    )
    transformation = fit.transformation['a']
    output_distribution = uniform(loc=1.0, scale=2.0)

    lattice = transformation.pullback_distribution(output_distribution)
    continuous = transformation.pullback_distribution(
        output_distribution,
        fixed_point=False,
    )

    assert lattice.domain == FixedPointColumn(2, 0.0, 1.0)
    assert continuous.domain == FloatingPointColumn(0.0, 1.0)
    assert lattice.pdf(0.5) == 0.0
    assert lattice.pmf(0.5) == pytest.approx(0.01)
    assert continuous.pdf(0.5) == pytest.approx(1.0)
    assert continuous.pmf(0.5) == 0.0


def test_frame_transformation_accepts_reordered_columns_and_restores_domain_order():
    source = pd.DataFrame({
        'price': [1.0, 2.0],
        'count': [10, 20],
        'ratio': [0.1, 0.2],
    })
    fit = (FrameTracker
        .from_dataframe(source)
        .transform['price'].affine(2.0, 0.0)
        .finish()
    )

    replayed = fit.transformation.apply(source[['ratio', 'price', 'count']])

    assert list(replayed.columns) == ['price', 'count', 'ratio']
    np.testing.assert_allclose(replayed['price'], [2.0, 4.0])


@pytest.mark.parametrize(
    'invalid, match',
    [
        (pd.DataFrame({'price': [1.0], 'count': [1]}), 'missing columns'),
        (
            pd.DataFrame({
                'price': [1.0], 'count': [1], 'ratio': [0.1], 'extra': [0],
            }),
            'unexpected columns',
        ),
        (
            pd.DataFrame({
                'price': [1.0], 'count': [1.0], 'ratio': [0.1],
            }),
            'dtype',
        ),
    ],
)
def test_frame_transformation_validate_rejects_incompatible_frames(invalid, match):
    source = pd.DataFrame({
        'price': [1.0],
        'count': [1],
        'ratio': [0.1],
    })
    fit = FrameTracker.from_dataframe(source).finish()

    with pytest.raises((TypeError, ValueError), match=match):
        fit.transformation.validate(invalid)


def test_frame_transformation_validate_checks_input_assumptions():
    source = pd.DataFrame({'value': [0.0, 1.0]})
    fit = (
        FrameTracker.from_dataframe(source)
        .assume['value'].ge(0.0)
        .finish()
    )

    with pytest.raises(ValueError, match="Column 'value'"):
        fit.transformation.validate(pd.DataFrame({'value': [-1.0]}))


def test_frame_fit_replays_fitted_parameters_on_holdout_dataframe():
    category_dtype = pd.CategoricalDtype(['a', 'b'])
    train = pd.DataFrame({
        'value': [1.0, 2.0, 3.0],
        'count': [10, 20, 30],
        'label': pd.Categorical(['a', 'b', 'a'], dtype=category_dtype),
    }, index=[2, 4, 6])
    holdout = pd.DataFrame({
        'value': [10.0, 20.0],
        'count': [40, 50],
        'label': pd.Categorical(['b', 'a'], dtype=category_dtype),
    }, index=[11, 13])
    original_train = train.copy(deep=True)
    original_holdout = holdout.copy(deep=True)

    fit = (
        FrameTracker.from_dataframe(train)
        .infer['value'].standardize()
        .finish()
    )

    fit.transformation.validate(holdout)
    replayed = fit.transformation.apply(holdout)

    mean = train['value'].mean()
    std = train['value'].std(ddof=0)
    np.testing.assert_allclose(replayed['value'], (holdout['value'] - mean) / std)
    pd.testing.assert_series_equal(replayed['count'], holdout['count'])
    pd.testing.assert_series_equal(replayed['label'], holdout['label'])
    assert replayed.index.equals(holdout.index)
    assert list(replayed.columns) == list(holdout.columns)
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(holdout, original_holdout)


# ── Filter classes ────────────────────────────────────────────────────────────

def test_name_filter_single():
    col = FloatingPointColumn()
    f = NameFilter('price')
    assert f.matches('price', col)
    assert not f.matches('ratio', col)


def test_name_filter_set():
    col = FloatingPointColumn()
    f = NameFilter(['price', 'ratio'])
    assert f.matches('price', col)
    assert f.matches('ratio', col)
    assert not f.matches('count', col)


def test_pattern_filter():
    col = FloatingPointColumn()
    f = NameFilter('p*')
    assert f.matches('price', col)
    assert not f.matches('ratio', col)
