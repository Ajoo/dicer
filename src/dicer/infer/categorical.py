import numpy as np
import pandas as pd

from dicer import domain
from dicer.tracking import (
    ColumnInferences,
    I,
    ObservedColumn,
    register_infer,
)
from dicer.transform.categorical import CategoryRateEncode, FrequencyEncode


@register_infer(domain.GenericColumn)
class GenericColumnInferences(ColumnInferences):

    @I
    def to_categorical(domain, vals, ordered=False):
        """Infer finite support and convert values to pandas categorical.

        Categories are inferred using pandas' ``Series.astype('category')``
        behavior. ``ordered=True`` retains that inferred category sequence and
        marks the resulting categorical dtype as ordered.
        """
        if not isinstance(ordered, (bool, np.bool_)):
            raise TypeError('ordered must be a boolean.')

        vals = np.asarray(vals)
        if vals.ndim != 1:
            raise ValueError('Categorical inference data must be one-dimensional.')
        if vals.size == 0:
            raise ValueError('Categorical inference requires non-empty data.')

        categorical = pd.Series(vals, copy=False).astype('category')
        if categorical.isna().any():
            raise ValueError('Categorical inference does not support missing values.')
        if ordered:
            categorical = categorical.cat.as_ordered()

        return (
            domain
            .assume.has_support(categorical.cat.categories)
            .transform.to_categorical(ordered=categorical.cat.ordered)
        )


@register_infer(domain.FiniteColumn)
class FiniteColumnInferences(ColumnInferences):

    @I
    def conditional_rates(
        domain,
        vals,
        *,
        outcome: ObservedColumn,
        weights: ObservedColumn | None = None,
        smoothing=1.0,
    ):
        """Fit the empirical conditional distribution of an outcome category."""
        if not isinstance(outcome, ObservedColumn):
            raise TypeError('outcome must be supplied as an ObservedColumn.')
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError('smoothing must be finite and nonnegative.')

        input_domain = domain.unpack('domain')
        vals = np.asarray(vals)
        outcome_vals = np.asarray(outcome.values)
        if vals.ndim != 1 or outcome_vals.ndim != 1:
            raise ValueError('conditional_rates requires one-dimensional categories.')
        if vals.size == 0:
            raise ValueError('conditional_rates requires non-empty data.')
        if vals.shape != outcome_vals.shape:
            raise ValueError('Primary and outcome observations must be aligned.')
        input_domain.validate(vals)
        outcome.domain.validate(outcome_vals)
        if not getattr(outcome.domain, 'finite', False):
            raise TypeError('conditional_rates outcome must have finite support.')

        input_indexes = np.asarray(input_domain.index(vals), dtype=np.intp)
        outcome_indexes = np.asarray(
            outcome.domain.index(outcome_vals),
            dtype=np.intp,
        )
        if np.any(input_indexes < 0) or np.any(outcome_indexes < 0):
            raise ValueError('Observed values must belong to their finite domains.')

        if weights is None:
            sample_weights = np.ones(vals.size, dtype=np.float64)
        else:
            if not isinstance(weights, ObservedColumn):
                raise TypeError('weights must be supplied as an ObservedColumn.')
            sample_weights = np.asarray(weights.values, dtype=np.float64)
            if sample_weights.shape != vals.shape:
                raise ValueError('weights must be aligned with the primary data.')
            if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0):
                raise ValueError('weights must be finite and nonnegative.')

        counts = np.zeros(
            (input_domain.cardinality, outcome.domain.cardinality),
            dtype=np.float64,
        )
        np.add.at(
            counts,
            (input_indexes, outcome_indexes),
            sample_weights,
        )
        totals = counts.sum(axis=1, keepdims=True)
        denominator = totals + smoothing * outcome.domain.cardinality
        if np.any(denominator == 0):
            raise ValueError(
                'Some input categories have no observations; use positive smoothing.'
            )
        rates = (counts + smoothing) / denominator

        labels = getattr(outcome.domain, 'values', None)
        if labels is None:
            labels = getattr(outcome.domain, 'categories', None)
        return domain._apply_transform(
            CategoryRateEncode(input_domain, rates, labels=labels)
        )


@register_infer(domain.UnorderedIntegerColumn)
class UnorderedIntegerInferences(FiniteColumnInferences):

    @I
    def frequency_encode(domain, vals):
        """Replace each code by its relative frequency in the fitting data."""
        vals = np.asarray(vals)
        if vals.ndim != 1:
            raise ValueError(
                'Frequency-encoding inference data must be one-dimensional.'
            )
        if vals.size == 0:
            raise ValueError(
                'Frequency-encoding inference requires non-empty data.'
            )
        input_domain = domain.unpack('domain')
        input_domain.validate(vals)
        indexes = np.asarray(input_domain.index(vals), dtype=np.intp)
        counts = np.bincount(indexes, minlength=input_domain.cardinality)
        frequencies = counts.astype(np.float64) / vals.size
        return domain._apply_transform(
            FrequencyEncode(input_domain, frequencies)
        )

    @I
    def rank_by_frequency(domain, vals, ascending=False):
        """Order unordered codes by their observed frequency.

        ``ascending=True`` assigns rank zero to the least frequent code, so
        the resulting integer values increase with frequency. Ties are broken
        by the original code, and codes absent from the data participate with
        frequency zero.
        """
        if not isinstance(ascending, (bool, np.bool_)):
            raise TypeError('ascending must be a boolean.')

        vals = np.asarray(vals)
        if vals.ndim != 1:
            raise ValueError(
                'Frequency-ranking inference data must be one-dimensional.'
            )
        if vals.size == 0:
            raise ValueError(
                'Frequency-ranking inference requires non-empty data.'
            )
        domain.validate(vals)

        counts = np.bincount(vals, minlength=domain.cardinality)
        original_codes = np.arange(domain.cardinality, dtype=domain.dtype)
        frequencies = counts if ascending else -counts
        codes_by_rank = np.lexsort((original_codes, frequencies))

        permutation = np.empty(domain.cardinality, dtype=domain.dtype)
        permutation[codes_by_rank] = np.arange(
            domain.cardinality,
            dtype=domain.dtype,
        )
        return domain.transform.permute(permutation).transform.as_ordered()
