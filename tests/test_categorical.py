import numpy as np
import pandas as pd
import pytest

from dicer import ColumnTracker, FrameTracker
from dicer import domain as d
from dicer.stats import (
    FiniteOrderedSupportDistribution,
    FiniteSupportDistribution,
    FiniteUnorderedSupportDistribution,
)
from dicer.tracking import Tracker


def test_has_support_refines_domain_without_changing_representation():
    values = pd.Series(['blue', 'red', 'blue'], dtype=object)

    supported = Tracker.from_series(values).assume.has_support(['red', 'blue'])

    supported_domain = supported.unpack('domain')
    assert supported_domain == d.FiniteUnorderedColumn(
        ['red', 'blue'],
        dtype=values.dtype,
    )
    assert supported_domain.dtype == values.dtype
    np.testing.assert_array_equal(supported.unpack('data'), values.to_numpy())


def test_has_support_validates_tracked_values_before_updating_domain():
    tracker = Tracker.from_series(pd.Series(['red', 'unknown'], dtype=object))

    with pytest.raises(ValueError, match='finite domain'):
        tracker.assume.has_support(['red', 'blue'])

    assert isinstance(tracker.unpack('domain'), d.GenericColumn)


def test_finite_unordered_column_preserves_codebook_order():
    domain = d.FiniteUnorderedColumn(['blue', 'red'], dtype=object)

    np.testing.assert_array_equal(domain.values, ['blue', 'red'])
    np.testing.assert_array_equal(
        domain.index(['red', 'missing', 'blue']),
        [1, -1, 0],
    )
    np.testing.assert_array_equal(domain.at([1, 0]), ['red', 'blue'])
    assert not domain.ordered
    assert not hasattr(domain, 'lb')

    with pytest.raises(ValueError, match='no predecessor'):
        domain.index('red', last=True)
    with pytest.raises(TypeError, match='ordered sequence'):
        d.FiniteUnorderedColumn({'red', 'blue'})

    tuple_domain = d.FiniteUnorderedColumn(
        [('red', 1), ('blue', 2)],
        dtype=object,
    )
    assert tuple_domain.index(('blue', 2)) == 1


def test_unordered_integer_column_is_the_unordered_index_set():
    domain = d.UnorderedIntegerColumn(3)

    assert domain.cardinality == 3
    assert not domain.ordered
    np.testing.assert_array_equal(domain.values, [0, 1, 2])
    np.testing.assert_array_equal(domain.index([0, 2, 3, -1]), [0, 2, -1, -1])
    np.testing.assert_array_equal(domain.at([2, 0]), [2, 0])

    with pytest.raises(ValueError, match=r'\[0, 3\)'):
        domain.validate([0, 3])


def test_to_categorical_is_lossless_and_retains_requested_order_flag():
    input_domain = d.FiniteUnorderedColumn(['low', 'high'], dtype=object)

    output_domain, transformation = (
        input_domain.transform.to_categorical(ordered=True).unpack()
    )

    assert isinstance(output_domain, d.PandasCategoricalColumn)
    assert output_domain.ordered
    assert list(output_domain.values) == ['low', 'high']
    assert not transformation.is_lossy

    transformed = transformation(np.array(['high', 'low'], dtype=object))
    assert isinstance(transformed, pd.Categorical)
    assert transformed.dtype == output_domain.dtype
    np.testing.assert_array_equal(transformed.codes, [1, 0])
    np.testing.assert_array_equal(
        transformation.inverse(transformed),
        ['high', 'low'],
    )


def test_encode_maps_unordered_support_to_unordered_integer_column():
    input_domain = d.FiniteUnorderedColumn(['red', 'blue'], dtype=object)

    output_domain, transformation = input_domain.transform.encode().unpack()

    assert output_domain == d.UnorderedIntegerColumn(2)
    np.testing.assert_array_equal(
        transformation(np.array(['blue', 'red', 'blue'], dtype=object)),
        [1, 0, 1],
    )
    np.testing.assert_array_equal(
        transformation.inverse(np.array([1, 0, 1])),
        ['blue', 'red', 'blue'],
    )


@pytest.mark.parametrize('ordered', [False, True])
def test_encode_pandas_categorical_preserves_order_semantics(ordered):
    input_domain = d.PandasCategoricalColumn(
        pd.CategoricalDtype(['low', 'high'], ordered=ordered)
    )

    output_domain, transformation = input_domain.transform.encode().unpack()

    expected_type = d.IntegerColumn if ordered else d.UnorderedIntegerColumn
    assert isinstance(output_domain, expected_type)
    assert output_domain.ordered is ordered
    np.testing.assert_array_equal(
        transformation(
            pd.Categorical(['high', 'low'], dtype=input_domain.dtype)
        ),
        [1, 0],
    )


def test_as_ordered_promotes_unordered_integer_codes():
    input_domain = d.UnorderedIntegerColumn(3)

    output_domain, transformation = input_domain.transform.as_ordered().unpack()

    assert output_domain == d.IntegerColumn(0, 2)
    np.testing.assert_array_equal(transformation([2, 0, 1]), [2, 0, 1])
    np.testing.assert_array_equal(
        transformation.inverse([2, 0, 1]),
        [2, 0, 1],
    )


def test_unordered_categorical_workflow_can_promote_codes_as_ordered():
    input_domain = d.PandasCategoricalColumn(
        pd.CategoricalDtype(['low', 'high'], ordered=False)
    )

    encoded = input_domain.transform.encode()
    assert encoded.unpack('domain') == d.UnorderedIntegerColumn(2)

    ordered = encoded.transform.as_ordered()
    assert ordered.unpack('domain') == d.IntegerColumn(0, 1)
    np.testing.assert_array_equal(
        ordered.transformations(pd.Categorical(
            ['high', 'low'],
            dtype=input_domain.dtype,
        )),
        [1, 0],
    )


def test_frame_categorical_encode_permute_as_ordered_replays_full_pipeline():
    dtype = pd.CategoricalDtype(['low', 'high'], ordered=False)
    train = pd.DataFrame({
        'grade': pd.Categorical(['high', 'low', 'high'], dtype=dtype),
    })
    holdout = pd.DataFrame({
        'grade': pd.Categorical(['low', 'high'], dtype=dtype),
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .transform['grade'].encode()
        .transform['grade'].permute([1, 0])
        .transform['grade'].as_ordered()
        .finish()
    )

    assert fit.transformation.output_domain['grade'] == d.IntegerColumn(0, 1)
    np.testing.assert_array_equal(fit.data['grade'], [0, 1, 0])
    fit.transformation.validate(holdout)
    replayed = fit.transformation.apply(holdout)
    np.testing.assert_array_equal(replayed['grade'], [1, 0])


def test_rank_by_frequency_permutes_codes_and_promotes_them_to_ordered():
    input_domain = d.UnorderedIntegerColumn(4, dtype=np.int8)
    values = np.array([0, 0, 0, 1, 1, 2], dtype=np.int8)

    ranked = (
        ColumnTracker(input_domain, data=values)
        .infer.rank_by_frequency(ascending=True)
    )

    assert ranked.unpack('domain') == d.IntegerColumn(0, 3)
    np.testing.assert_array_equal(
        ranked.unpack('data'),
        [3, 3, 3, 2, 2, 1],
    )
    permutation = ranked._tracked_transformations[0]
    assert permutation.permutation.dtype == input_domain.dtype
    np.testing.assert_array_equal(permutation.permutation, [3, 2, 1, 0])


@pytest.mark.parametrize(
    ('ascending', 'expected'),
    [
        (True, [1, 1, 2, 2, 0]),
        (False, [0, 0, 1, 1, 2]),
    ],
)
def test_rank_by_frequency_has_deterministic_ties(ascending, expected):
    input_domain = d.UnorderedIntegerColumn(3)
    values = np.array([0, 0, 1, 1, 2])

    ranked = input_domain.infer.rank_by_frequency(
        data=values,
        ascending=ascending,
    )

    np.testing.assert_array_equal(ranked.transformations(values), expected)


def test_frame_rank_by_frequency_fits_and_replays_categorical_ranking():
    dtype = pd.CategoricalDtype(['low', 'medium', 'high'], ordered=False)
    train = pd.DataFrame({
        'grade': pd.Categorical(
            ['high', 'low', 'high', 'medium', 'high', 'medium'],
            dtype=dtype,
        ),
    })
    holdout = pd.DataFrame({
        'grade': pd.Categorical(['low', 'high', 'medium'], dtype=dtype),
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .transform['grade'].encode()
        .infer['grade'].rank_by_frequency(ascending=True)
        .finish()
    )

    assert fit.transformation.output_domain['grade'] == d.IntegerColumn(0, 2)
    np.testing.assert_array_equal(fit.data['grade'], [2, 0, 2, 1, 2, 1])
    replayed = fit.transformation.apply(holdout)
    np.testing.assert_array_equal(replayed['grade'], [0, 2, 1])


@pytest.mark.parametrize(
    ('values', 'match'),
    [
        (np.array([], dtype=int), 'non-empty'),
        (np.array([[0], [1]]), 'one-dimensional'),
        (np.array([0, 3]), r'\[0, 3\)'),
    ],
)
def test_rank_by_frequency_rejects_unsupported_data(values, match):
    input_domain = d.UnorderedIntegerColumn(3)

    with pytest.raises(ValueError, match=match):
        input_domain.infer.rank_by_frequency(data=values)


def test_rank_by_frequency_requires_boolean_ascending():
    input_domain = d.UnorderedIntegerColumn(2)

    with pytest.raises(TypeError, match='ascending must be a boolean'):
        input_domain.infer.rank_by_frequency(data=[0, 1], ascending='yes')


def test_frequency_encode_maps_codes_to_relative_frequencies():
    input_domain = d.UnorderedIntegerColumn(4, dtype=np.int8)
    values = np.array([0, 0, 0, 1, 1, 2], dtype=np.int8)

    encoded = ColumnTracker(
        input_domain,
        data=values,
    ).infer.frequency_encode()

    assert encoded.unpack('domain') == d.FiniteOrderedColumn(
        [0.0, 1 / 6, 1 / 3, 1 / 2]
    )
    np.testing.assert_allclose(
        encoded.unpack('data'),
        [0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 6],
    )


def test_frame_frequency_encode_fits_and_replays_categorical_frequencies():
    dtype = pd.CategoricalDtype(['a', 'b', 'unseen'], ordered=False)
    train = pd.DataFrame({
        'category': pd.Categorical(['a', 'a', 'a', 'b'], dtype=dtype),
    })
    holdout = pd.DataFrame({
        'category': pd.Categorical(['unseen', 'b', 'a'], dtype=dtype),
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .transform['category'].encode()
        .infer['category'].frequency_encode()
        .finish()
    )

    assert fit.transformation.output_domain['category'] == d.FiniteOrderedColumn(
        [0.0, 0.25, 0.75]
    )
    np.testing.assert_allclose(fit.data['category'], [0.75, 0.75, 0.75, 0.25])
    replayed = fit.transformation.apply(holdout)
    np.testing.assert_allclose(replayed['category'], [0.0, 0.25, 0.75])


@pytest.mark.parametrize(
    ('values', 'match'),
    [
        (np.array([], dtype=int), 'non-empty'),
        (np.array([[0], [1]]), 'one-dimensional'),
        (np.array([0, 3]), r'\[0, 3\)'),
    ],
)
def test_frequency_encode_rejects_unsupported_data(values, match):
    input_domain = d.UnorderedIntegerColumn(3)

    with pytest.raises(ValueError, match=match):
        input_domain.infer.frequency_encode(data=values)


def test_permute_relabels_unordered_integer_codes_and_attaches_inverse():
    input_domain = d.UnorderedIntegerColumn(3, dtype=np.int8)

    output_domain, transformation = (
        input_domain.transform.permute([2, 0, 1]).unpack()
    )

    assert output_domain == input_domain
    assert transformation.permutation.dtype == input_domain.dtype
    transformed = transformation(np.array([0, 1, 2], dtype=np.int8))
    assert transformed.dtype == input_domain.dtype
    np.testing.assert_array_equal(transformed, [2, 0, 1])
    assert transformation.inverse.permutation.dtype == input_domain.dtype
    np.testing.assert_array_equal(
        transformation.inverse([2, 0, 1]),
        [0, 1, 2],
    )


@pytest.mark.parametrize(
    ('permutation', 'error', 'match'),
    [
        ([[0, 1, 2]], ValueError, 'one-dimensional'),
        ([0, 1], ValueError, 'length'),
        ([0.0, 1.0, 2.0], TypeError, 'integers'),
        ([0, 0, 2], ValueError, 'each code'),
        ([0, 1, 3], ValueError, 'each code'),
    ],
)
def test_permute_rejects_invalid_permutations(permutation, error, match):
    domain = d.UnorderedIntegerColumn(3)

    with pytest.raises(error, match=match):
        domain.transform.permute(permutation)


def test_frame_categorical_fit_and_replay_preserve_extension_dtype():
    train = pd.DataFrame({
        'color': pd.Series(['blue', 'red', 'blue'], dtype=object),
        'value': [1, 2, 3],
    })
    holdout = pd.DataFrame({
        'color': pd.Series(['red', 'blue'], dtype=object),
        'value': [4, 5],
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .assume['color'].has_support(['red', 'blue'])
        .transform['color'].to_categorical(ordered=True)
        .finish()
    )

    assert isinstance(fit.data['color'].dtype, pd.CategoricalDtype)
    assert fit.data['color'].dtype.ordered
    fit.transformation.validate(holdout)
    replayed = fit.transformation.apply(holdout)
    assert replayed['color'].dtype == fit.data['color'].dtype
    pd.testing.assert_series_equal(replayed['value'], holdout['value'])

    invalid = holdout.copy()
    invalid.loc[0, 'color'] = 'unknown'
    with pytest.raises(ValueError, match="Column 'color'"):
        fit.transformation.validate(invalid)


@pytest.mark.parametrize('ordered', [False, True])
def test_to_categorical_inference_uses_pandas_category_inference(ordered):
    values = pd.Series(['zebra', 'ant', 'zebra', 'moose'], dtype=object)
    expected = values.astype('category')
    if ordered:
        expected = expected.cat.as_ordered()

    inferred = Tracker.from_series(values).infer.to_categorical(ordered=ordered)

    output_domain = inferred.unpack('domain')
    transformed = pd.Series(inferred.unpack('data'))
    assert isinstance(output_domain, d.PandasCategoricalColumn)
    assert output_domain.dtype == expected.dtype
    assert transformed.dtype == expected.dtype
    np.testing.assert_array_equal(transformed.cat.codes, expected.cat.codes)


def test_frame_to_categorical_inference_fits_and_replays_support():
    train = pd.DataFrame({
        'color': pd.Series(['blue', 'red', 'blue'], dtype=object),
        'value': [1, 2, 3],
    })
    holdout = pd.DataFrame({
        'color': pd.Series(['red', 'blue'], dtype=object),
        'value': [4, 5],
    })

    fit = (
        FrameTracker.from_dataframe(train)
        .infer['color'].to_categorical(ordered=True)
        .finish()
    )

    expected_dtype = train['color'].astype('category').cat.as_ordered().dtype
    assert fit.data['color'].dtype == expected_dtype
    replayed = fit.transformation.apply(holdout)
    assert replayed['color'].dtype == expected_dtype

    invalid = holdout.copy()
    invalid.loc[0, 'color'] = 'unknown'
    with pytest.raises(ValueError, match="Column 'color'"):
        fit.transformation.validate(invalid)


@pytest.mark.parametrize(
    ('values', 'match'),
    [
        (np.array([], dtype=object), 'non-empty'),
        (np.array(['red', None], dtype=object), 'missing values'),
        (np.array([['red'], ['blue']], dtype=object), 'one-dimensional'),
    ],
)
def test_to_categorical_inference_rejects_unsupported_data(values, match):
    domain = d.GenericColumn(np.dtype(object))

    with pytest.raises(ValueError, match=match):
        domain.infer.to_categorical(data=values)


def test_to_categorical_inference_requires_boolean_ordered():
    domain = d.GenericColumn(np.dtype(object))

    with pytest.raises(TypeError, match='ordered must be a boolean'):
        domain.infer.to_categorical(data=['red', 'blue'], ordered='yes')


def test_encode_pushes_and_pulls_finite_distribution_exactly():
    input_domain = d.FiniteUnorderedColumn(['red', 'blue'], dtype=object)
    probabilities = np.array([
        [0.25, 0.75],
        [0.60, 0.40],
    ])
    source = FiniteSupportDistribution(input_domain, probabilities)
    _, transformation = input_domain.transform.encode().unpack()

    encoded = transformation.pushforward_distribution(source)

    assert isinstance(encoded, FiniteUnorderedSupportDistribution)
    assert encoded.domain == d.UnorderedIntegerColumn(2)
    np.testing.assert_allclose(encoded.probabilities, probabilities)

    reconstructed = transformation.pullback_distribution(encoded)
    assert isinstance(reconstructed, FiniteUnorderedSupportDistribution)
    assert reconstructed.domain == input_domain
    np.testing.assert_allclose(reconstructed.probabilities, source.probabilities)


def test_one_hot_encode_distribution_transport_is_not_implemented():
    input_domain = d.FiniteUnorderedColumn(['red', 'blue'], dtype=object)
    source = FiniteSupportDistribution(input_domain, np.array([0.25, 0.75]))
    _, transformation = input_domain.transform.one_hot_encode().unpack()

    with pytest.raises(NotImplementedError, match='grouped columns'):
        transformation.pushforward_distribution(source)
    with pytest.raises(NotImplementedError, match='grouped columns'):
        transformation.pullback_distribution(source)
    with pytest.raises(NotImplementedError, match='grouped columns'):
        transformation.conditional('red')


def test_simplex_distribution_transport_is_not_implemented():
    simplex = d.SimplexColumn(2)
    _, transformation = simplex.transform.argmax().unpack()

    with pytest.raises(NotImplementedError, match='pushforward'):
        transformation.pushforward_distribution(object())
    with pytest.raises(NotImplementedError, match='pullback'):
        transformation.pullback_distribution(object())


def test_permute_pushes_and_pulls_finite_distribution_exactly():
    domain = d.UnorderedIntegerColumn(3)
    source = FiniteSupportDistribution(domain, np.array([0.1, 0.2, 0.7]))
    _, transformation = domain.transform.permute([2, 0, 1]).unpack()

    permuted = transformation.pushforward_distribution(source)

    assert isinstance(permuted, FiniteUnorderedSupportDistribution)
    np.testing.assert_allclose(permuted.probabilities, [0.2, 0.7, 0.1])

    reconstructed = transformation.pullback_distribution(permuted)
    np.testing.assert_allclose(reconstructed.probabilities, source.probabilities)


def test_frequency_encode_pushforward_aggregates_tied_frequencies():
    input_domain = d.UnorderedIntegerColumn(4)
    source = FiniteSupportDistribution(
        input_domain,
        np.array([0.1, 0.2, 0.3, 0.4]),
    )
    _, transformation = input_domain.infer.frequency_encode(
        data=[0, 0, 1, 2]
    ).unpack()

    encoded = transformation.pushforward_distribution(source)

    assert encoded.domain == d.FiniteOrderedColumn([0.0, 0.25, 0.5])
    np.testing.assert_allclose(encoded.probabilities, [0.4, 0.5, 0.1])


def test_ordered_categorical_encode_produces_ordered_distribution():
    input_domain = d.PandasCategoricalColumn(
        pd.CategoricalDtype(['low', 'high'], ordered=True)
    )
    source = FiniteSupportDistribution(input_domain, np.array([0.4, 0.6]))
    encoded_domain, transformation = input_domain.transform.encode().unpack()

    encoded = transformation.pushforward_distribution(source)

    assert isinstance(encoded, FiniteOrderedSupportDistribution)
    assert encoded_domain == d.IntegerColumn(0, 1)
    assert encoded.domain == encoded_domain
    np.testing.assert_allclose(encoded.probabilities, [0.4, 0.6])


def test_ordered_categorical_distribution_has_ordinal_support():
    input_domain = d.FiniteUnorderedColumn(['low', 'high'], dtype=object)
    source = FiniteSupportDistribution(input_domain, np.array([0.4, 0.6]))
    _, transformation = (
        input_domain.transform.to_categorical(ordered=True).unpack()
    )

    categorical = transformation.pushforward_distribution(source)

    assert isinstance(categorical, FiniteOrderedSupportDistribution)
    assert categorical.support() == ('low', 'high')
    np.testing.assert_allclose(categorical.cdf(['low', 'high']), [0.4, 1.0])
