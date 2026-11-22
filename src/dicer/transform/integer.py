from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import ClassVar

import numpy as np
from scipy.stats import randint

from dicer import domain
from dicer.stats import (
    Distribution,
    DiscreteOrderedDistribution,
    FiniteOrderedSupportDistribution,
    NoDensityMixin,
    as_distribution,
)
from dicer.tracking import ColumnTransformations, T, register_transform
from dicer.transform._base import Transformation
from dicer.transform.continuous import (
    Clip, InvertibleTransformation, Unclip, PushedForwardDistribution,
    IntToFixedPoint, FixedPointToInt, FixedPointTransformations,
)
from dicer.transform.categorical import _recode_pair


@dataclass
class IntegerAffine(InvertibleTransformation):
    """A bijective affine map between contiguous integer intervals."""

    a: int
    b: int
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False

    def __post_init__(self):
        if self.a not in (-1, 1):
            raise ValueError('Integer affine slope a must be either -1 or 1.')
        if not np.isfinite(self.b) or self.b % 1 != 0:
            raise ValueError('Integer affine intercept b must be integer-valued.')
        self.a = int(self.a)
        self.b = int(self.b)

    def apply(self, x, random_state=None):
        return self.a * np.asarray(x) + self.b

    def log_abs_det_jacobian(self, x):
        return np.zeros_like(x, dtype=float)

    @property
    def output_domain(self):
        if self.a == 1:
            lb = self.input_domain.lb + self.b
            ub = self.input_domain.ub + self.b
        else:
            lb = -self.input_domain.ub + self.b
            ub = -self.input_domain.lb + self.b
        return domain.IntegerColumn(lb, ub)


@dataclass
class IntegerClip(Clip):
    """Clip integers after rounding bounds inward onto the integer lattice."""

    def __post_init__(self):
        if self.lb is not None:
            self.lb = ceil(self.lb)
        if self.ub is not None:
            self.ub = floor(self.ub)
        super().__post_init__()

    @property
    def is_lossy(self):
        return (
            self.lb is not None and self.lb > self.input_domain.lb
        ) or (
            self.ub is not None and self.ub < self.input_domain.ub
        )


class IntegerUnclip(Unclip):
    """Reconstruct clipped integer tails using discrete distributions."""

    def pushforward_distribution(self, distribution):
        reconstructed = super().pushforward_distribution(distribution)
        output_domain = self.output_domain
        if not output_domain.finite:
            return reconstructed

        # Tabulate the complete mixture so CDF and quantiles use the same
        # cumulative masses, without rescaling q into a conditional tail.
        expand = (slice(None),) + (None,) * reconstructed.ndim
        probabilities = reconstructed.pmf(output_domain.values[expand])
        return FiniteOrderedSupportDistribution(
            output_domain,
            np.moveaxis(probabilities, 0, -1),
        )

    @property
    def output_domain(self):
        output = super().output_domain
        return domain.IntegerColumn(output.lb, output.ub)

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        x = np.asarray(x)
        y = x.astype(np.int64, copy=True)
        for bound, tail in ((self.lb, self.left_dist), (self.ub, self.right_dist)):
            if tail is not None:
                mask = x == bound
                y[mask] = tail.rvs(size=np.count_nonzero(mask), random_state=prng)
        return y


@dataclass
class IntegerDigitize(Transformation):
    """Map a contiguous integer interval to ordered bin codes."""

    bins: np.ndarray
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True

    def __post_init__(self):
        self.bins = np.asarray(self.bins)

    @property
    def cuts(self):
        return self.bins[1:-1]

    @property
    def lower(self):
        return self.bins[:-1]

    @property
    def upper(self):
        return self.bins[1:] - 1

    @property
    def output_domain(self):
        return domain.IntegerColumn(0, self.bins.size - 2)

    def apply(self, x, random_state=None):
        return np.digitize(x, self.cuts)

    def pullback_lb(self, y):
        return self.lower[np.asarray(y, dtype=np.intp)]

    def pullback_ub(self, y):
        return self.upper[np.asarray(y, dtype=np.intp)]

    def pushforward_distribution(self, distribution):
        distribution = as_distribution(distribution)
        expand = (slice(None),) + (None,) * distribution.ndim
        probabilities = (
            distribution.cdf(self.upper[expand])
            - distribution.cdf((self.lower - 1)[expand])
        )
        probabilities = np.moveaxis(probabilities, 0, -1)
        return FiniteOrderedSupportDistribution(
            self.output_domain,
            probabilities,
        )

    def pullback_distribution(self, distribution, **options):
        return self.inverse.pushforward_distribution(distribution)


@dataclass
class DiscreteInverseDigitize(Transformation):
    """Sample an integer from the fiber of an ordered bin code."""

    bins: np.ndarray
    piece_bins: np.ndarray
    piece_distributions: list[Distribution | None]
    is_deterministic: ClassVar[bool] = False
    is_lossy: ClassVar[bool] = False

    def __post_init__(self):
        self.bins = np.asarray(self.bins)

        self.piece_bins = np.asarray(self.piece_bins)
        if (
            self.piece_bins.ndim != 1
            or self.piece_bins.size != len(self.piece_distributions) + 1
            or self.piece_bins[0] != 0
            or self.piece_bins[-1] != self.bins.size - 1
            or np.any(self.piece_bins % 1 != 0)
            or np.any(np.diff(self.piece_bins) <= 0)
        ):
            raise ValueError('piece_bins must partition all bin codes into nonempty pieces.')
        self.piece_bins = self.piece_bins.astype(np.intp)
        self.piece_distributions = [
            None if dist is None else self._integer_distribution(dist)
            for dist in self.piece_distributions
        ]
        if not np.isfinite(self.lower[0]) and self.piece_distributions[0] is None:
            raise ValueError(
                'An unbounded first integer fiber requires a left-tail distribution.'
            )
        if not np.isfinite(self.upper[-1]) and self.piece_distributions[-1] is None:
            raise ValueError(
                'An unbounded last integer fiber requires a right-tail distribution.'
            )
        self._fibers = []
        for code, (lower, upper) in enumerate(zip(self.lower, self.upper)):
            piece = np.searchsorted(self.piece_bins[1:], code, side='right')
            base = self.piece_distributions[piece]
            if base is None:
                fiber = as_distribution(randint(int(lower), int(upper) + 1))
            elif np.isfinite(lower) and np.isfinite(upper):
                fiber_domain = domain.IntegerColumn(int(lower), int(upper))
                probabilities = base.pmf(fiber_domain.values)
                mass = probabilities.sum()
                if mass <= 0:
                    raise ValueError(f'Base distribution assigns zero mass to bin {code}.')
                fiber = FiniteOrderedSupportDistribution(fiber_domain, probabilities / mass)
            else:
                fiber = ConditionalIntegerDistribution(base, lower, upper)
            self._fibers.append(fiber)

    @property
    def lower(self):
        return self.bins[:-1]

    @property
    def upper(self):
        return self.bins[1:] - 1

    @staticmethod
    def _integer_distribution(distribution):
        distribution = as_distribution(distribution)
        tail_domain = distribution.domain
        integer_domain = isinstance(tail_domain, domain.IntegerColumn) or (
            tail_domain.finite
            and np.all(np.asarray(tail_domain.values) % 1 == 0)
        )
        if not tail_domain.ordered or not integer_domain:
            raise ValueError(
                'Base distribution must have ordered integer support.'
            )
        if distribution.shape:
            raise ValueError('Base distribution must be scalar.')
        return distribution

    @staticmethod
    def _tail_distribution(distribution, lower, upper, side):
        distribution = DiscreteInverseDigitize._integer_distribution(distribution)
        tail_domain = distribution.domain
        if tail_domain.lb < lower or tail_domain.ub > upper:
            raise ValueError(
                f'The {side}-tail distribution must be supported inside its fiber.'
            )
        return distribution

    @property
    def output_domain(self):
        return domain.IntegerColumn(self.bins[0], self.bins[-1] - 1)

    def _fiber_distribution(self, code):
        return self._fibers[code]

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        x = np.asarray(x)
        y = np.empty(x.shape, dtype=np.int64)
        for code in range(self.input_domain.cardinality):
            mask = x == code
            count = np.count_nonzero(mask)
            if count:
                y[mask] = self._fiber_distribution(code).rvs(
                    size=count,
                    random_state=prng,
                )
        return y

    def conditional(self, x):
        return self._fiber_distribution(int(x))

    def pushforward_distribution(self, distribution):
        if not self.output_domain.finite:
            return DiscreteDigitizeReconstruction(distribution, self)

        distribution = as_distribution(distribution)
        values = self.output_domain.values
        probabilities = 0.0
        for code in range(self.input_domain.cardinality):
            weight = np.asarray(distribution.pmf(code))[..., None]
            probabilities = (
                probabilities
                + weight * self._fiber_distribution(code).pmf(values)
            )
        return FiniteOrderedSupportDistribution(
            self.output_domain,
            probabilities,
        )


class ConditionalIntegerDistribution(DiscreteOrderedDistribution):
    """A base distribution conditioned on an unbounded integer fiber."""

    def __init__(self, base, lower, upper):
        self.base = base
        self.domain = domain.IntegerColumn(lower, upper)
        self._cdf_lower = base.cdf(lower - 1)
        self._sf_upper = base.sf(upper)
        self._use_sf = self._cdf_lower > 0.5
        self._mass = (base.sf(lower - 1) - self._sf_upper if self._use_sf
                      else base.cdf(upper) - self._cdf_lower)
        if self._mass <= 0:
            raise ValueError('Base distribution assigns zero mass to an integer fiber.')

    @property
    def shape(self):
        return ()

    def pmf(self, x):
        x = np.asarray(x)
        return np.where((x >= self.domain.lb) & (x <= self.domain.ub),
                        self.base.pmf(x) / self._mass, 0.0)

    def cdf(self, x):
        x = np.clip(np.asarray(x), self.domain.lb - 1, self.domain.ub)
        mass = (self.base.sf(self.domain.lb - 1) - self.base.sf(x) if self._use_sf
                else self.base.cdf(x) - self._cdf_lower)
        return np.clip(mass / self._mass, 0, 1)

    def ppf(self, q):
        q = np.asarray(q)
        if self._use_sf:
            value = self.base.isf(self._sf_upper + (1 - q) * self._mass)
        else:
            value = self.base.ppf(self._cdf_lower + q * self._mass)
        return np.where((q < 0) | (q > 1), np.nan,
                        np.clip(value, self.domain.lb, self.domain.ub))

    def rvs(self, size=None, random_state=None):
        q = np.random.default_rng(random_state).uniform(size=size)
        q = np.maximum(q, np.nextafter(0.0, 1.0))
        return self.ppf(q).astype(np.int64)


class DiscreteDigitizeReconstruction(NoDensityMixin, PushedForwardDistribution):
    """Mixture of conditional integer fibers for an unbounded source domain."""

    def pmf(self, x):
        return sum(self.dist.pmf(code) * fiber.pmf(x)
                   for code, fiber in enumerate(self.transformation._fibers))

    def cdf(self, x):
        return sum(self.dist.pmf(code) * fiber.cdf(x)
                   for code, fiber in enumerate(self.transformation._fibers))

    def ppf(self, q):
        q = np.asarray(q)
        codes = self.dist.ppf(q)
        result = np.zeros(np.broadcast_shapes(q.shape, self.shape))
        # Analytical conditional probabilities retain rounding ambiguity at
        # exact jumps. Finite reconstruction uses a single probability table.
        for code, fiber in enumerate(self.transformation._fibers):
            mass = self.dist.pmf(code)
            with np.errstate(divide='ignore', invalid='ignore'):
                local_q = (q - self.dist.cdf(code - 1)) / mass
            result = np.where(codes == code, fiber.ppf(np.clip(local_q, 0, 1)), result)
        return np.where((q < 0) | (q > 1), np.nan, result)


@register_transform(domain.IntegerColumn)
class IntegerTransformations(ColumnTransformations):

    @T
    def to_fixed_point(input_domain, digits=0):
        forward = IntToFixedPoint(input_domain, digits)
        return forward, FixedPointToInt(forward.output_domain)

    dequantize = FixedPointTransformations.dequantize

    @T
    def as_unordered(input_domain):
        if input_domain.lb != 0 or not input_domain.is_bounded:
            raise ValueError('as_unordered requires integer codes 0 through K - 1.')
        return _recode_pair(
            input_domain, domain.UnorderedIntegerColumn(int(input_domain.cardinality)),
        )

    @T
    def affine(input_domain, a, b):
        forward = IntegerAffine(input_domain, a, b)
        inverse = IntegerAffine(
            forward.output_domain,
            forward.a,
            -forward.a * forward.b,
        )
        return forward, inverse

    @T
    def clip(input_domain, lb=None, ub=None, left_dist=None, right_dist=None):
        forward = IntegerClip(input_domain, lb, ub)
        lb, ub = forward.lb, forward.ub
        clips_left = lb is not None and lb > input_domain.lb
        clips_right = ub is not None and ub < input_domain.ub
        if (clips_left and left_dist is None) or (clips_right and right_dist is None):
            return forward
        if clips_left:
            left_dist = DiscreteInverseDigitize._tail_distribution(left_dist, input_domain.lb, lb, 'left')
        if clips_right:
            right_dist = DiscreteInverseDigitize._tail_distribution(right_dist, ub, input_domain.ub, 'right')
        inverse = IntegerUnclip(
            forward.output_domain,
            lb if clips_left else None,
            ub if clips_right else None,
            left_dist if clips_left else None,
            right_dist if clips_right else None,
        )
        return forward, inverse

    @T
    def digitize(
        input_domain, cuts, left_tail=None, right_tail=None,
        num_left_tail_bins=1, num_right_tail_bins=1,
        *, middle_distribution=None, piece_bins=None, piece_distributions=None,
    ):
        cuts = np.asarray(cuts)
        if cuts.ndim != 1:
            raise ValueError('Integer digitization cuts must be one-dimensional.')
        if np.any(~np.isfinite(cuts)) or np.any(cuts % 1 != 0):
            raise ValueError('Integer digitization cuts must be integer-valued.')
        if np.any(np.diff(cuts) <= 0):
            raise ValueError('Integer digitization cuts must be strictly increasing.')
        if cuts.size and (
            cuts[0] <= input_domain.lb
            or cuts[-1] > input_domain.ub
        ):
            raise ValueError(
                'Integer digitization cuts must satisfy input.lb < cuts <= input.ub.'
            )

        bins = np.r_[input_domain.lb, cuts, input_domain.ub + 1]
        forward = IntegerDigitize(input_domain, bins)
        num_bins = bins.size - 1
        if piece_bins is None and piece_distributions is None:
            left = num_left_tail_bins if left_tail is not None else 0
            right = num_bins - num_right_tail_bins if right_tail is not None else num_bins
            if not 0 <= left <= right <= num_bins:
                raise ValueError('Tail pieces must not overlap or exceed the number of bins.')
            piece_bins, piece_distributions = [0], []
            for start, stop, base in (
                (0, left, left_tail), (left, right, middle_distribution),
                (right, num_bins, right_tail),
            ):
                if start < stop:
                    piece_bins.append(stop)
                    piece_distributions.append(base)
        elif piece_bins is None or piece_distributions is None:
            raise ValueError('Supply both piece_bins and piece_distributions.')
        elif left_tail is not None or right_tail is not None or middle_distribution is not None:
            raise ValueError('Use either explicit pieces or tail/middle arguments.')
        inverse = DiscreteInverseDigitize(
            forward.output_domain,
            bins,
            piece_bins,
            piece_distributions,
        )
        return forward, inverse

    def encode(self):
        if not self.domain.is_bounded:
            raise ValueError(
                'Encoding an IntegerColumn requires finite lower and upper bounds.'
            )
        return self.affine(1, -self.domain.lb)
