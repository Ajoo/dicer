from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from dicer import domain
from dicer.stats import FiniteSupportDistribution, as_distribution
from dicer.tracking import ColumnTransformations, T, register_transform
from dicer.transform._base import Transformation


def _finite_probabilities(distribution, input_domain):
    distribution = as_distribution(distribution)
    positions = np.arange(input_domain.cardinality, dtype=np.intp)
    input_values = np.asarray(input_domain.at(positions))
    expand = (slice(None),) + (None,) * distribution.ndim
    probabilities = distribution.pmf(input_values[expand])
    return np.moveaxis(probabilities, 0, -1)


@dataclass
class FiniteRecode(Transformation):
    """A bijection between finite domains sharing one stable enumeration."""

    target_domain: domain.Column
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False

    def __post_init__(self):
        if self.input_domain.cardinality != self.target_domain.cardinality:
            raise ValueError('Finite recoding requires equal cardinalities.')

    def apply(self, x, random_state=None):
        indexes = self.input_domain.index(x)
        return self.target_domain.at(indexes)

    @property
    def output_domain(self):
        return self.target_domain

    def _require_distribution_support(self):
        if any(
            isinstance(column, domain.VectorColumn)
            for column in (self.input_domain, self.output_domain)
        ):
            raise NotImplementedError(
                'Distribution transport involving grouped columns is not '
                'implemented.'
            )

    def pushforward_distribution(self, distribution):
        self._require_distribution_support()
        probabilities = _finite_probabilities(distribution, self.input_domain)
        return FiniteSupportDistribution(self.output_domain, probabilities)

    def pullback_distribution(self, distribution, **options):
        self._require_distribution_support()
        if self.inverse is None:
            raise NotImplementedError(
                'pullback_distribution requires an inverse transformation.'
            )
        inverse = self.inverse.resolve(**options)
        return inverse.pushforward_distribution(distribution)


@dataclass
class Permutation(Transformation):
    """Relabel unordered integer codes with ``new_code = p[old_code]``."""

    permutation: np.ndarray
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False

    def __post_init__(self):
        if not isinstance(self.input_domain, domain.UnorderedIntegerColumn):
            raise TypeError(
                'Permutation requires an UnorderedIntegerColumn input domain.'
            )

        permutation = np.asarray(self.permutation)
        if permutation.ndim != 1:
            raise ValueError('permutation must be one-dimensional.')
        if permutation.size != self.input_domain.cardinality:
            raise ValueError(
                'permutation length must match the domain cardinality.'
            )
        if not np.issubdtype(permutation.dtype, np.integer):
            raise TypeError('permutation must contain integers.')

        permutation = permutation.astype(self.input_domain.dtype, copy=True)
        expected = np.arange(
            self.input_domain.cardinality,
            dtype=self.input_domain.dtype,
        )
        if not np.array_equal(np.sort(permutation), expected):
            raise ValueError(
                'permutation must contain each code in [0, cardinality) once.'
            )
        permutation.setflags(write=False)
        self.permutation = permutation

    @property
    def output_domain(self):
        return domain.UnorderedIntegerColumn(
            self.input_domain.cardinality,
            dtype=self.input_domain.dtype,
        )

    def apply(self, x, random_state=None):
        return self.permutation[np.asarray(x)]

    def pushforward_distribution(self, distribution):
        if self.inverse is None:
            raise NotImplementedError(
                'pushforward_distribution requires an inverse transformation.'
            )
        return self.inverse.pullback_distribution(distribution)

    def pullback_distribution(self, distribution, **options):
        probabilities = _finite_probabilities(distribution, self.output_domain)
        return FiniteSupportDistribution(
            self.input_domain,
            probabilities[..., self.permutation],
        )


@dataclass
class CategoryRateEncode(Transformation):
    """Map each finite input value to a fitted probability vector."""

    rates: np.ndarray
    labels: tuple | None = None
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True

    def __post_init__(self):
        rates = np.asarray(self.rates, dtype=np.float64)
        if (
            rates.ndim != 2
            or rates.shape[0] != self.input_domain.cardinality
            or rates.shape[1] == 0
        ):
            raise ValueError(
                'rates must have shape (input cardinality, output dimension), '
                f'got {rates.shape!r}.'
            )
        if self.labels is not None:
            self.labels = tuple(self.labels)
        self.output_domain.validate(rates)
        rates = rates.copy()
        rates.setflags(write=False)
        self.rates = rates

    @property
    def output_domain(self):
        return domain.SimplexColumn(
            self.rates.shape[1],
            dtype=self.rates.dtype,
            labels=self.labels,
        )

    def apply(self, x, random_state=None):
        indexes = np.asarray(self.input_domain.index(x))
        if np.any(indexes < 0):
            raise ValueError('Values must belong to the fitted input domain.')
        return self.rates[indexes]


@dataclass
class FrequencyEncode(Transformation):
    """Map unordered integer codes to fitted relative frequencies."""

    frequencies: np.ndarray
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True

    def __post_init__(self):
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        expected = (self.input_domain.cardinality,)
        if frequencies.shape != expected:
            raise ValueError(
                f'frequencies must have shape {expected!r}, got '
                f'{frequencies.shape!r}.'
            )
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0):
            raise ValueError('frequencies must be finite and nonnegative.')
        frequencies = frequencies.copy()
        frequencies.setflags(write=False)
        self.frequencies = frequencies

    @property
    def output_domain(self):
        return domain.FiniteOrderedColumn(np.unique(self.frequencies))

    def apply(self, x, random_state=None):
        indexes = np.asarray(self.input_domain.index(x))
        if np.any(indexes < 0):
            raise ValueError('Values must belong to the fitted input domain.')
        return self.frequencies[indexes]

    def pushforward_distribution(self, distribution):
        probabilities = _finite_probabilities(distribution, self.input_domain)
        output_indexes = self.output_domain.index(self.frequencies)
        output_probabilities = np.zeros(
            (*probabilities.shape[:-1], self.output_domain.cardinality),
            dtype=np.float64,
        )
        for input_index, output_index in enumerate(output_indexes):
            output_probabilities[..., output_index] += probabilities[..., input_index]
        return FiniteSupportDistribution(
            self.output_domain,
            output_probabilities,
        )


@dataclass
class SimplexArgmax(Transformation):
    """Select the first maximum-probability simplex vertex."""

    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True

    @property
    def output_domain(self):
        if self.input_domain.labels is None:
            return domain.UnorderedIntegerColumn(self.input_domain.dimension)
        return domain.FiniteUnorderedColumn(self.input_domain.labels)

    def apply(self, x, random_state=None):
        indexes = np.argmax(np.asarray(x), axis=-1)
        return self.output_domain.at(indexes)


def _recode_pair(input_domain, output_domain):
    return (
        FiniteRecode(input_domain, output_domain),
        FiniteRecode(output_domain, input_domain),
    )



def _encoded_domain(input_domain):
    if input_domain.ordered:
        return domain.IntegerColumn(0, input_domain.cardinality - 1)
    return domain.UnorderedIntegerColumn(input_domain.cardinality)


@register_transform(domain.FiniteUnorderedColumn)
class FiniteUnorderedTransformations(ColumnTransformations):

    @T
    def to_categorical(input_domain, ordered=False):
        dtype = pd.CategoricalDtype(
            categories=input_domain.values,
            ordered=ordered,
        )
        output_domain = domain.PandasCategoricalColumn(dtype)
        return _recode_pair(input_domain, output_domain)

    @T
    def encode(input_domain):
        return _recode_pair(input_domain, _encoded_domain(input_domain))

    @T
    def one_hot_encode(input_domain, dtype=np.dtype(np.int64)):
        output_domain = domain.OneHotColumn(input_domain.values, dtype=dtype)
        return _recode_pair(input_domain, output_domain)


@register_transform(domain.UnorderedIntegerColumn)
class UnorderedIntegerTransformations(FiniteUnorderedTransformations):

    @T
    def permute(input_domain, permutation):
        forward = Permutation(input_domain, permutation)
        inverse = Permutation(
            forward.output_domain,
            np.argsort(forward.permutation),
        )
        return forward, inverse

    @T
    def as_ordered(input_domain):
        output_domain = domain.IntegerColumn(
            0,
            input_domain.cardinality - 1,
        )
        return _recode_pair(input_domain, output_domain)


@register_transform(domain.PandasCategoricalColumn)
class PandasCategoricalTransformations(ColumnTransformations):

    @T
    def encode(input_domain):
        return _recode_pair(input_domain, _encoded_domain(input_domain))

    @T
    def one_hot_encode(input_domain, dtype=np.dtype(np.int64)):
        output_domain = domain.OneHotColumn(input_domain.values, dtype=dtype)
        return _recode_pair(input_domain, output_domain)


@register_transform(domain.SimplexColumn)
class SimplexTransformations(ColumnTransformations):

    @T
    def argmax(input_domain):
        return SimplexArgmax(input_domain)


@register_transform(domain.OneHotColumn)
class OneHotTransformations(SimplexTransformations):

    @T
    def encode(input_domain):
        return _recode_pair(input_domain, _encoded_domain(input_domain))

    @T
    def to_categorical(input_domain, ordered=False):
        output_domain = domain.PandasCategoricalColumn(
            pd.CategoricalDtype(input_domain.categories, ordered=ordered)
        )
        return _recode_pair(input_domain, output_domain)
