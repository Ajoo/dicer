from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np
from scipy.integrate import quad
from scipy.stats import uniform
from scipy.special import logit, expit
from scipy.stats import norm

from dicer import domain
from dicer.tracking import register_transform, ColumnTransformations, T
from dicer.stats import (
    ContinuousDistribution,
    FiniteOrderedSupportDistribution,
    Distribution,
    OrderedDistribution,
    as_distribution,
    NoDensityMixin,
    NoAtomsMixin,
)
from dicer.transform._base import (
    OptionalTransformationStep,
    Transformation,
    PushforwardMixin,
)

# TODO: should all functions (apply and dist methods) assert
#       that the input comes from the appropriate domain?
#       also call np.asarray() on it


def dlogit(x):
    x = x * (1 - x)
    return np.where(x == 0, np.inf, 1/x)


def dexpit(y):
    p = expit(y)
    return p * (1 - p)


## MAPPED DISTRIBUTIONS

@dataclass(init=False)
class PushedForwardDistribution(OrderedDistribution):
    dist: Distribution
    transformation: Transformation

    def __init__(self, dist, transformation):
        self.dist = as_distribution(dist)
        self.transformation = transformation

    @property
    def domain(self):
        return self.transformation.output_domain

    @property
    def shape(self):
        return self.dist.shape

    def rvs(self, size=None, random_state=None):
        prng = np.random.default_rng(random_state)
        x = self.dist.rvs(size=size, random_state=prng)
        return self.transformation.apply(x, random_state=prng)


class InvertiblePushedForwardDistribution(PushedForwardDistribution):

    @property
    def _inverse(self):
        return self.transformation.inverse.resolve(
            fixed_point=False,
        )

    def __post_init__(self):
        assert self.transformation.is_deterministic
        assert not self.transformation.is_lossy
        assert self.transformation.inverse is not None

    def pdf(self, y):
        return np.exp(self.logpdf(y))

    def logpdf(self, y):
        x = self._inverse(y)
        return (
            self.dist.logpdf(x)
            - self.transformation.log_abs_det_jacobian(x)
        )

    def pmf(self, y):
        x = self._inverse(y)
        return self.dist.pmf(x)

    @property
    def _is_increasing(self):
        transformation = self.transformation
        input_domain = transformation.input_domain
        return (
            transformation(input_domain.lb)
            < transformation(input_domain.ub)
        )

    def cdf(self, y):
        x = self._inverse(y)
        if self._is_increasing:
            return self.dist.cdf(x)

        # For a decreasing transformation, f(X) <= y iff X >= f^-1(y).
        # sf(x) is P(X > x), so include any atom at x explicitly.
        return self.dist.sf(x) + self.dist.pmf(x)

    def ppf(self, q):
        if self._is_increasing:
            x = self.dist.ppf(q)
        else:
            q = np.asarray(q)
            x = self.dist.ppf(1.0 - q)

            # At a CDF jump, the decreasing pushforward needs the greatest
            # source value whose inclusive upper tail still contains q. A
            # standard PPF returns the value on the lower side of that jump.
            atom_mass = self.dist.pmf(x)
            if np.any(atom_mass > 0):
                next_x = self.dist.ppf(
                    np.nextafter(self.dist.cdf(x), 1.0)
                )
                next_inclusive_tail = (
                    self.dist.sf(next_x) + self.dist.pmf(next_x)
                )
                use_next = (atom_mass > 0) & (next_inclusive_tail >= q)
                x = np.where(use_next, next_x, x)

        return self.transformation(x)


class GeneralizedPiecewiseAffinePushedForwardDistribution(PushedForwardDistribution):
    """Pushforward through a nondecreasing generalized affine graph."""

    def pdf(self, y):
        y = np.asarray(y)
        transformation = self.transformation
        lower, upper = transformation.preimage_bounds(y)
        inverse = transformation.graph_inverse
        x = inverse(y)
        inverse_slope = inverse.derivative(y)
        regular = (
            (lower == upper)
            & (inverse_slope > 0)
            & np.isfinite(inverse_slope)
        )
        density = self.dist.pdf(x) * inverse_slope
        valid = (
            (y >= transformation.image_lb)
            & (y <= transformation.image_ub)
        )
        return np.where(valid & regular, density, 0.0)

    def pmf(self, y):
        y = np.asarray(y)
        transformation = self.transformation
        lower, upper = transformation.preimage_bounds(y)
        same_point = lower == upper
        inverse_slope = transformation.graph_inverse.derivative(y)
        regular = same_point & (inverse_slope > 0) & np.isfinite(inverse_slope)
        singleton_maps = regular | (transformation(lower) == y)

        singleton_mass = np.where(
            singleton_maps,
            self.dist.pmf(lower),
            0.0,
        )
        interval_mass = (
            self.dist.cdf(upper)
            - self.dist.cdf(lower)
            + self.dist.pmf(lower)
        )
        interval_mass = interval_mass - np.where(
            transformation(lower) == y,
            0.0,
            self.dist.pmf(lower),
        )
        interval_mass = interval_mass - np.where(
            transformation(upper) == y,
            0.0,
            self.dist.pmf(upper),
        )
        mass = np.where(same_point, singleton_mass, interval_mass)

        valid = (
            (y >= transformation.image_lb)
            & (y <= transformation.image_ub)
        )
        return np.where(valid, mass, 0.0)

    def cdf(self, y):
        y = np.asarray(y)
        transformation = self.transformation
        lower, upper = transformation.preimage_bounds(y)
        inverse_slope = transformation.graph_inverse.derivative(y)
        regular = (
            (lower == upper)
            & (inverse_slope > 0)
            & np.isfinite(inverse_slope)
        )
        cdf = self.dist.cdf(upper) - np.where(
            (~regular) & (transformation(upper) > y),
            self.dist.pmf(upper),
            0.0,
        )
        cdf = np.where(y < transformation.image_lb, 0.0, cdf)
        return np.where(y >= transformation.image_ub, 1.0, cdf)

    def ppf(self, q):
        return self.transformation(self.dist.ppf(q))


class ClippedPushedForwardDistribution(PushedForwardDistribution):
    """Pushforward through a monotone clip, including boundary atoms."""

    def pdf(self, y):
        y = np.asarray(y)
        density = self.dist.pdf(y)
        valid = (y >= self.domain.lb) & (y <= self.domain.ub)
        density = np.where(valid, density, 0.0)
        return np.where(np.isnan(y), np.nan, density)

    def pmf(self, y):
        y = np.asarray(y)
        transformation = self.transformation

        if (
            transformation.lb is not None
            and transformation.ub is not None
            and transformation.lb == transformation.ub
        ):
            mass = np.where(y == transformation.lb, 1.0, 0.0)
            return np.where(np.isnan(y), np.nan, mass)

        mass = self.dist.pmf(y)
        if transformation.lb is not None:
            mass = np.where(
                y == transformation.lb,
                self.dist.cdf(transformation.lb),
                mass,
            )
        if transformation.ub is not None:
            upper_mass = (
                self.dist.sf(transformation.ub)
                + self.dist.pmf(transformation.ub)
            )
            mass = np.where(y == transformation.ub, upper_mass, mass)

        valid = (y >= self.domain.lb) & (y <= self.domain.ub)
        mass = np.where(valid, mass, 0.0)
        return np.where(np.isnan(y), np.nan, mass)

    def cdf(self, y):
        y = np.asarray(y)
        cumulative = self.dist.cdf(y)
        cumulative = np.where(y < self.domain.lb, 0.0, cumulative)
        cumulative = np.where(y >= self.domain.ub, 1.0, cumulative)
        return np.where(np.isnan(y), np.nan, cumulative)

    def ppf(self, q):
        return self.transformation(self.dist.ppf(q))


class UnclippedPushedForwardDistribution(PushedForwardDistribution):
    """Replace clipping-boundary atoms with reconstruction distributions."""

    @staticmethod
    def _tail_pdf(distribution, x):
        pdf = getattr(distribution, 'pdf', None)
        if callable(pdf):
            return pdf(x)
        return np.zeros_like(x, dtype=float)

    @staticmethod
    def _tail_pmf(distribution, x):
        pmf = getattr(distribution, 'pmf', None)
        if callable(pmf):
            return pmf(x)
        return np.zeros_like(x, dtype=float)

    @property
    def _left_mass(self):
        transformation = self.transformation
        if transformation.lb is None:
            return 0.0
        return self.dist.pmf(transformation.lb)

    @property
    def _right_mass(self):
        transformation = self.transformation
        if transformation.ub is None:
            return 0.0
        return self.dist.pmf(transformation.ub)

    def _interior(self, x):
        transformation = self.transformation
        interior = (
            (x >= transformation.input_domain.lb)
            & (x <= transformation.input_domain.ub)
        )
        if transformation.lb is not None:
            interior &= x > transformation.lb
        if transformation.ub is not None:
            interior &= x < transformation.ub
        return interior

    def pdf(self, x):
        x = np.asarray(x)
        transformation = self.transformation
        density = np.where(self._interior(x), self.dist.pdf(x), 0.0)
        if transformation.left_dist is not None:
            density = density + self._left_mass * self._tail_pdf(
                transformation.left_dist,
                x,
            )
        if transformation.right_dist is not None:
            density = density + self._right_mass * self._tail_pdf(
                transformation.right_dist,
                x,
            )
        return np.where(np.isnan(x), np.nan, density)

    def pmf(self, x):
        x = np.asarray(x)
        transformation = self.transformation
        mass = np.where(self._interior(x), self.dist.pmf(x), 0.0)
        if transformation.left_dist is not None:
            mass = mass + self._left_mass * self._tail_pmf(
                transformation.left_dist,
                x,
            )
        if transformation.right_dist is not None:
            mass = mass + self._right_mass * self._tail_pmf(
                transformation.right_dist,
                x,
            )
        return np.where(np.isnan(x), np.nan, mass)

    def cdf(self, x):
        x = np.asarray(x)
        transformation = self.transformation
        cumulative = self.dist.cdf(x)

        if transformation.left_dist is not None:
            cumulative = cumulative - np.where(
                x >= transformation.lb,
                self._left_mass,
                0.0,
            )
            cumulative = (
                cumulative
                + self._left_mass * transformation.left_dist.cdf(x)
            )

        if transformation.right_dist is not None:
            cumulative = cumulative - np.where(
                x >= transformation.ub,
                self._right_mass,
                0.0,
            )
            cumulative = (
                cumulative
                + self._right_mass * transformation.right_dist.cdf(x)
            )

        return np.where(np.isnan(x), np.nan, np.clip(cumulative, 0.0, 1.0))

    def ppf(self, q):
        q = np.asarray(q)
        transformation = self.transformation
        value = self.dist.ppf(q)

        # Rescaling q into a conditional tail can round across a CDF jump
        # when that tail has atoms (even for continuous-column transforms).
        # Finite integer unclip distributions avoid this by tabulating the
        # complete mixture; this analytical path retains that limitation.
        if transformation.left_dist is not None:
            left_mass = self._left_mass
            with np.errstate(divide='ignore', invalid='ignore'):
                left_q = q / left_mass
            from_left = (left_mass > 0.0) & (q <= left_mass)
            value = np.where(
                from_left,
                transformation.left_dist.ppf(np.clip(left_q, 0.0, 1.0)),
                value,
            )

        if transformation.right_dist is not None:
            right_mass = self._right_mass
            right_start = 1.0 - right_mass
            with np.errstate(divide='ignore', invalid='ignore'):
                right_q = (q - right_start) / right_mass
            from_right = (right_mass > 0.0) & (
                (q > right_start)
                | ((right_start == 0.0) & (q == 0.0))
            )
            value = np.where(
                from_right,
                transformation.right_dist.ppf(np.clip(right_q, 0.0, 1.0)),
                value,
            )

        return np.where((q < 0.0) | (q > 1.0), np.nan, value)


class DiscretizedPushedForwardDistribution(NoDensityMixin, PushedForwardDistribution):
    """Lazy pushforward through an ordered atomizing transformation.

    This path is parked while finite output domains are handled eagerly. Before
    enabling it, it needs:

    * PMF validation for non-support values and values outside the domain;
    * CDF handling between support values and at infinities/NaN;
    * exact lower/upper boundary ownership when the source has atoms;
    * numerically stable tail masses using the survival function;
    * a PPF implementation, including boundary corrections; and
    * effective construction validation (``__post_init__`` is not currently
      called by ``PushedForwardDistribution.__init__``).

    The ``probabilities`` property should either be removed from this lazy
    representation or defined only for finite output domains.
    """

    def __post_init__(self):
        assert self.transformation.is_deterministic
        assert self.transformation.is_lossy
        #assert self.transformation.inverse is not None

    @property
    def probabilities(self):
        raise NotImplementedError("TODO")

    def pmf(self, y):
        lb = self.transformation.pullback_lb(y)
        ub = self.transformation.pullback_ub(y)
        return self.dist.cdf(ub) - self.dist.cdf(lb)

    def logpmf(self, y):
        return np.log(self.pmf(y))

    def cdf(self, y):
        x = self.transformation.pullback_ub(y)
        return self.dist.cdf(x)

    def logcdf(self, y):
        return np.log(self.cdf(y))


class RoundedPushedForwardDistribution(NoDensityMixin, PushedForwardDistribution):
    """Lazy atomic pushforward through rounding onto a regular lattice."""

    @property
    def _precision(self):
        return self.domain.precision

    def _is_output_value(self, y):
        y = np.asarray(y)
        nearest = np.rint(y / self._precision) * self._precision
        valid = np.isclose(
            y,
            nearest,
            rtol=getattr(self.domain, 'rtol', 0.0),
            atol=getattr(self.domain, 'atol', 0.0),
        )
        return valid & (y >= self.domain.lb) & (y <= self.domain.ub)

    def _floor_output(self, y):
        y = np.asarray(y)
        finite = np.isfinite(y)
        safe_y = np.where(finite, y, 0.0)
        scaled = np.nextafter(safe_y / self._precision, np.inf)
        floored = np.floor(scaled) * self._precision
        return np.where(finite, floored, y)

    def pmf(self, y):
        y = np.asarray(y)
        transformation = self.transformation
        lower = transformation.pullback_lb(y)
        upper = transformation.pullback_ub(y)

        lower_cdf = self.dist.cdf(lower)
        upper_cdf = self.dist.cdf(upper)
        mass = np.where(
            upper_cdf <= 0.5,
            upper_cdf - lower_cdf,
            self.dist.sf(lower) - self.dist.sf(upper),
        )
        mass = mass + np.where(
            transformation(lower) == y,
            self.dist.pmf(lower),
            0.0,
        )
        mass = mass - np.where(
            transformation(upper) == y,
            0.0,
            self.dist.pmf(upper),
        )
        mass = np.where(self._is_output_value(y), mass, 0.0)
        return np.where(np.isnan(y), np.nan, mass)

    def cdf(self, y):
        y = np.asarray(y)
        transformation = self.transformation
        output = self._floor_output(y)
        safe_output = np.where(np.isfinite(output), output, 0.0)
        upper = transformation.pullback_ub(safe_output)
        mapped_upper = transformation(upper)
        cumulative = self.dist.cdf(upper) - np.where(
            mapped_upper > y,
            self.dist.pmf(upper),
            0.0,
        )
        cumulative = np.where((y < self.domain.lb) | np.isneginf(y), 0.0, cumulative)
        cumulative = np.where((y >= self.domain.ub) | np.isposinf(y), 1.0, cumulative)
        return np.where(np.isnan(y), np.nan, cumulative)

    def ppf(self, q):
        q = np.asarray(q)
        q_eval = np.clip(
            q,
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        candidate = self.transformation(self.dist.ppf(q_eval))
        previous = candidate - self._precision
        value = np.where(self.cdf(previous) >= q, previous, candidate)
        value = np.where(q == 0.0, self.domain.lb, value)
        value = np.where(q == 1.0, self.domain.ub, value)
        return np.where((q < 0.0) | (q > 1.0), np.nan, value)


class AtomicScaledPushedForwardDistribution(
    NoDensityMixin,
    PushedForwardDistribution,
):
    """Scale integer atoms onto a fixed-point lattice without enumeration."""

    @property
    def _precision(self):
        return self.domain.precision

    def _is_output_value(self, y):
        nearest = np.rint(y / self._precision) * self._precision
        return (
            np.isclose(
                y,
                nearest,
                rtol=self.domain.rtol,
                atol=self.domain.atol,
            )
            & (y >= self.domain.lb)
            & (y <= self.domain.ub)
        )

    def pmf(self, y):
        y = np.asarray(y)
        index = np.rint(y / self._precision)
        mass = self.dist.pmf(index)
        mass = np.where(self._is_output_value(y), mass, 0.0)
        return np.where(np.isnan(y), np.nan, mass)

    def cdf(self, y):
        y = np.asarray(y)
        finite = np.isfinite(y)
        safe_y = np.where(finite, y, 0.0)
        index = np.floor(
            np.nextafter(safe_y / self._precision, np.inf)
        )
        cumulative = self.dist.cdf(index)
        cumulative = np.where((y < self.domain.lb) | np.isneginf(y), 0.0, cumulative)
        cumulative = np.where((y >= self.domain.ub) | np.isposinf(y), 1.0, cumulative)
        return np.where(np.isnan(y), np.nan, cumulative)

    def ppf(self, q):
        return self.dist.ppf(q) * self._precision


class UniformDequantizedPushedForwardDistribution(
    NoAtomsMixin,
    PushedForwardDistribution,
):
    """Spread each finite-precision atom uniformly over its rounding cell."""

    def _cell(self, x):
        transformation = self.transformation
        input_domain = transformation.input_domain
        value = np.round(x, input_domain.decimals)
        half_width = input_domain.precision / 2
        lower = np.maximum(value - half_width, transformation.lb)
        upper = np.minimum(value + half_width, transformation.ub)
        return value, lower, upper

    def pdf(self, x):
        x = np.asarray(x)
        safe_x = np.where(np.isfinite(x), x, 0.0)
        value, lower, upper = self._cell(safe_x)
        width = upper - lower
        density = np.divide(
            self.dist.pmf(value),
            width,
            out=np.zeros_like(np.broadcast_arrays(x, width)[0], dtype=float),
            where=width > 0,
        )
        valid = (x >= self.domain.lb) & (x <= self.domain.ub)
        return np.where(np.isnan(x), np.nan, np.where(valid, density, 0.0))

    def cdf(self, x):
        x = np.asarray(x)
        safe_x = np.where(np.isfinite(x), x, 0.0)
        value, lower, upper = self._cell(safe_x)
        mass = self.dist.pmf(value)
        cumulative_below = self.dist.cdf(value) - mass
        fraction = np.divide(
            safe_x - lower,
            upper - lower,
            out=np.zeros_like(np.broadcast_arrays(x, mass)[0], dtype=float),
            where=upper > lower,
        )
        cumulative = cumulative_below + mass * np.clip(fraction, 0.0, 1.0)
        cumulative = np.where(x <= self.domain.lb, 0.0, cumulative)
        cumulative = np.where(x >= self.domain.ub, 1.0, cumulative)
        return np.where(np.isnan(x), np.nan, cumulative)

    def ppf(self, q):
        q = np.asarray(q)
        q_eval = np.clip(
            q,
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        value = self.dist.ppf(q_eval)
        mass = self.dist.pmf(value)
        cumulative_below = self.dist.cdf(value) - mass
        _, lower, upper = self._cell(value)
        fraction = np.divide(
            q - cumulative_below,
            mass,
            out=np.zeros_like(np.broadcast_arrays(q, mass)[0], dtype=float),
            where=mass > 0,
        )
        result = lower + np.clip(fraction, 0.0, 1.0) * (upper - lower)
        result = np.where(q == 0.0, self.domain.lb, result)
        result = np.where(q == 1.0, self.domain.ub, result)
        return np.where((q < 0.0) | (q > 1.0), np.nan, result)


# TODO: make it so forwarded methods are only defined in
# __init__ and only when self.transformation has that attribute
class DelegatedToTransformMixin:
    def pdf(self, y):
        return self.transformation.pdf(self.dist, y)

    def cdf(self, y):
        return self.transformation.cdf(self.dist, y)
    
    def ppf(self, y):
        return self.transformation.ppf(self.dist, y)
    
    def expect(self, func):
        return self.transformation.expect(self.dist, func)

    def moment(self, order: int):
        return self.transformation.moment(self.dist, order)

    def mean(self):
        return self.moment(1)

    def var(self):
        return self.moment(2) - self.moment(1)**2


class ExplodedPushedForwardDistribution(NoAtomsMixin, DelegatedToTransformMixin, PushedForwardDistribution):

    def __post_init__(self):
        assert not self.transformation.is_deterministic
        assert not self.transformation.is_lossy
        #assert self.transformation.inverse is not None


## TRANSFORMATION SPECIALISATIONS

class InvertibleTransformation(PushforwardMixin, Transformation):
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False
    _Pushforward: ClassVar[type] = InvertiblePushedForwardDistribution


class AtomizingTransformation(Transformation):
    # continuous -> discrete
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True
    # Keep every finite output eager until the lazy implementation above
    # satisfies the full ordered-distribution contract.
    _CARDINALITY_FOR_EAGER_PUSHFORWARD = np.inf

    @property
    def _pushforward_bins(self):
        ...

    def pullback_lb(self, y):
        ...

    def pullback_ub(self, y):
        ...

    @property
    def _pushforward_boundary_atoms_go_right(self):
        boundaries = self._pushforward_bins[1:-1]
        boundary_indices = np.arange(boundaries.size)
        left_values = self.output_domain.at(boundary_indices)
        right_values = self.output_domain.at(
            boundary_indices + 1
        )
        mapped_values = self.apply(boundaries)
        valid = (mapped_values == left_values) | (mapped_values == right_values)
        if not np.all(valid):
            raise ValueError(
                'Each atomizing boundary must map to an adjacent output value.'
            )
        return mapped_values == right_values

    def _pushforward_distribution(self, distribution: Distribution):
        distribution = as_distribution(distribution)

        expand = (slice(None),) + (None,) * distribution.ndim
        bins = self._pushforward_bins[1:-1][expand]
        atoms_go_right = self._pushforward_boundary_atoms_go_right[expand]
        P_inner = distribution.cdf(bins) - (
            atoms_go_right * distribution.pmf(bins)
        )
        endpoint_shape = (1, *distribution.shape)
        P = np.concatenate([
            np.zeros(endpoint_shape),
            P_inner,
            np.ones(endpoint_shape),
        ])
        p = np.diff(np.moveaxis(P, 0, -1), axis=-1)
        return FiniteOrderedSupportDistribution(self.output_domain, p)

    def pushforward_distribution(self, distribution):
        if self.output_domain.cardinality < self._CARDINALITY_FOR_EAGER_PUSHFORWARD:
            return self._pushforward_distribution(distribution)

        return DiscretizedPushedForwardDistribution(distribution, self)

    def pullback_distribution(
        self,
        distribution: Distribution,
        **options,
    ):
        inverse = self.inverse.resolve(**options)
        return inverse.pushforward_distribution(distribution)




## DESCRIPTORS

@register_transform(domain.FloatingPointColumn)
class FloatingPointTransformations(ColumnTransformations):

    @T
    def digitize(
            domain: domain.Column,
            bins: np.ndarray,
            left_tail: Distribution | None = None,
            right_tail: Distribution | None = None,
            num_left_tail_bins: int = 1,
            num_right_tail_bins: int = 1,
            ):
        bins = np.asarray(bins)
        domain.check_dtype(bins.dtype)
        bins = np.r_[domain.lb, bins, domain.ub]

        num_bins = bins.size - 1
        if left_tail is not None and right_tail is not None:
            piece_bins = np.array([num_left_tail_bins, num_bins - num_right_tail_bins])
            piece_distributions = [left_tail, None, right_tail]
        elif left_tail is not None:
            piece_bins = np.array([num_left_tail_bins])
            piece_distributions = [left_tail, None]
        elif right_tail is not None:
            piece_bins = np.array([num_bins - num_right_tail_bins])
            piece_distributions = [None, right_tail]
        else:
            piece_bins = np.array([], dtype=np.int64)
            piece_distributions = [None]

        piece_bins = np.r_[0, piece_bins, bins.size]

        t = Digitize(domain, bins)
        t_inv = ContinuousInverseDigitize(
            t.output_domain,
            bins,
            piece_bins=piece_bins,
            piece_distributions=piece_distributions,
            )
        return t, t_inv

    @T
    def map(
            domain: domain.Column,
            f: Callable,
            df: Callable,
            f_inv: Callable | None=None,
            df_inv: Callable | None=None,
            ):
        if f_inv is not None and df_inv is None:
            df_inv = lambda y: 1/df(f_inv(y))

        t = InvertibleMap(domain, f, df)
        if f_inv is None:
            return t

        t_inv = InvertibleMap(t.output_domain, f_inv, df_inv)
        return t, t_inv

    def affine(
            self,
            a: float,
            b: float,
            ):
        if a == 0:
            raise ValueError(
                'An affine transformation must have nonzero slope. '
                'Use a generalized or atomizing transformation for a '
                'constant map.'
            )
        f = lambda x: a*x + b
        df = lambda x: a
        f_inv = lambda y: (y - b)/a
        df_inv = lambda y: 1/a

        return self.map(f, df, f_inv, df_inv)

    def log(self):
        assert self.domain.lb >= 0, (
            "log input domain should be nonnegative"
        )
        return self.map(np.log, np.reciprocal, np.exp, np.exp)

    def exp(self):
        return self.map(np.exp, np.exp, np.log, np.reciprocal)

    def logit(self):
        msg = "logit input domain should be ⊆ [0, 1]"
        assert self.domain.lb >= 0, msg
        assert self.domain.ub <= 1, msg

        return self.map(logit, dlogit, expit, dexpit)

    def expit(self):
        return self.map(expit, dexpit, logit, dlogit)

    def probit(self, loc=0.0, scale=1.0):
        msg = "probit input domain should be ⊆ [0, 1]"
        assert self.domain.lb >= 0, msg
        assert self.domain.ub <= 1, msg

        n = norm(loc, scale)
        dppf = lambda x: 1/n.pdf(n.ppf(x))
        return self.map(n.ppf, dppf, n.cdf, n.pdf)

    def invprobit(self, loc=0.0, scale=1.0):
        n = norm(loc, scale)
        return self.map(n.cdf, n.pdf, n.ppf)

    def softsign(self):
        f = lambda x: np.where(np.isfinite(x), x / (1 + np.abs(x)), np.sign(x))
        df = lambda x: 1 / (1 + np.abs(x))**2
        f_inv = lambda y: y / (1 - np.abs(y))
        df_inv = lambda y: 1 / (1 - np.abs(y))**2
        return self.map(f, df, f_inv, df_inv)

    def tanh(self):
        df = lambda x: 1 - np.tanh(x)**2
        df_inv = lambda y: 1 / (1 - y**2)
        return self.map(np.tanh, df, np.atanh, df_inv)

    def atanh(self):
        msg = "atanh input domain should be ⊆ [-1, 1]"
        assert self.domain.lb >= -1, msg
        assert self.domain.ub <= 1, msg

        df_inv = lambda x: 1 - np.tanh(x)**2
        df = lambda y: 1 / (1 - y**2)
        return self.map(np.atanh, df, np.tanh, df_inv)

    @T
    def clip(
            domain: domain.Column,
            lb: float | None=None,
            ub: float | None=None,
            left_dist: Distribution=None,
            right_dist: Distribution=None,
            ):
        t = Clip(domain, lb, ub)

        clips_left = lb is not None and lb > domain.lb
        clips_right = ub is not None and ub < domain.ub
        if (
            (clips_left and left_dist is None)
            or (clips_right and right_dist is None)
        ):
            return t

        t_inv = Unclip(
            t.output_domain,
            lb if clips_left else None,
            ub if clips_right else None,
            left_dist if clips_left else None,
            right_dist if clips_right else None,
        )
        return t, t_inv

    @T
    def round(domain: domain.Column, decimals: int = 0):
        t = Round(domain, decimals)
        t_inv = UniformDequantize(t.output_domain, domain.lb, domain.ub)
        return t, t_inv

    # Parked until additive convolution pushforward is implemented.
    # @T
    # def add_noise(domain, distribution: ContinuousDistribution):
    #     return AddNoise(domain, distribution)

    # TODO: this should handle the case where transformation doesn't cover the whole domain
    @T
    def piecewise_affine(domain, xp: np.ndarray, yp: np.ndarray):
        """Piecewise affine map from xp reference values to fp reference values.
        """
        xp = np.asarray(xp)
        yp = np.asarray(yp)

        assert xp[0] == domain.lb
        assert xp[-1] == domain.ub

        t = PiecewiseAffine(domain, xp, yp)
        if yp[0] < yp[-1]:
            inverse_xp, inverse_yp = yp, xp
        else:
            # numpy.interp requires increasing reference x-coordinates.
            inverse_xp, inverse_yp = yp[::-1], xp[::-1]

        t_inv = PiecewiseAffine(t.output_domain, inverse_xp, inverse_yp)
        return t, t_inv

    @T
    def generalized_piecewise_affine(
            input_domain: domain.Column,
            xp: np.ndarray,
            yp: np.ndarray,
            ):
        """Nondecreasing affine graph allowing horizontal and vertical pieces."""
        xp = np.asarray(xp)
        yp = np.asarray(yp)
        if xp.ndim != 1 or yp.ndim != 1:
            raise ValueError('Generalized piecewise-affine knots must be one-dimensional.')
        if xp.size != yp.size or xp.size < 2:
            raise ValueError('Generalized piecewise-affine knots must have equal length >= 2.')
        if xp[0] < input_domain.lb or xp[-1] > input_domain.ub:
            raise ValueError(
                'Generalized piecewise-affine knots must lie within the input domain.'
            )
        if xp[0] > input_domain.lb:
            xp = np.r_[input_domain.lb, xp]
            yp = np.r_[yp[0], yp]
        if xp[-1] < input_domain.ub:
            xp = np.r_[xp, input_domain.ub]
            yp = np.r_[yp, yp[-1]]

        t = GeneralizedPiecewiseAffine(input_domain, xp, yp)
        t_inv = GeneralizedPiecewiseAffine(
            t.output_domain,
            yp,
            xp,
            select_boundary_interior=True,
        )
        return t, t_inv


@register_transform(domain.FixedPointColumn)
class FixedPointTransformations(FloatingPointTransformations):

    def _inverse_projection(self, other_domain):
        if self.domain == other_domain:
            return
        assert isinstance(other_domain, domain.FloatingPointColumn)
        return OptionalTransformationStep(
            Round(other_domain, self.domain.decimals),
            option='fixed_point',
            default=True,
        )

    @T
    def dequantize(input_domain, lb=None, ub=None):
        forward = UniformDequantize(input_domain, lb, ub)
        inverse = DequantizeRound(
            forward.output_domain, input_domain.decimals, input_domain,
        )
        return forward, inverse

    @T
    def to_int(domain):
        t = FixedPointToInt(domain)
        t_inv = IntToFixedPoint(t.output_domain, domain.decimals)
        return t, t_inv

    def clip(
            self,
            lb: float|None=None,
            ub: float|None=None,
            left_dist: Distribution=None,
            right_dist: Distribution=None,
            ):
        if lb is not None and not self.domain.check_decimals(lb):
            raise NotImplementedError(
                "clip on a fixed-point domain with a non fixed-point lower bound"
                f"is not supported. Got {lb} which does not have {self.domain.decimals} decimals."
            )

        if ub is not None and not self.domain.check_decimals(ub):
            raise NotImplementedError(
                "clip on a fixed-point domain with a non fixed-point upper bound"
                f"is not supported. Got {ub} which does not have {self.domain.decimals} decimals."
            )

        return super().clip(lb, ub, left_dist, right_dist)


## CONCRETE TRANSFORMS — FLOATING POINT INPUT

@dataclass
class GeneralizedPiecewiseAffine(PushforwardMixin, Transformation):
    """Nondecreasing piecewise-affine graph with a midpoint selector.

    Repeated ``xp`` values define vertical pieces and evaluate to the midpoint
    of their output interval. Repeated ``yp`` values define horizontal pieces.
    When ``select_boundary_interior`` is true, vertical pieces at the graph
    boundaries select the endpoint toward the graph interior rather than the
    midpoint. Interior vertical pieces always select their midpoint.
    """
    xp: np.ndarray
    yp: np.ndarray
    select_boundary_interior: bool = False
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True
    _Pushforward: ClassVar[type] = (
        GeneralizedPiecewiseAffinePushedForwardDistribution
    )

    def __post_init__(self):
        self.xp = np.asarray(self.xp)
        self.yp = np.asarray(self.yp)
        if self.xp.ndim != 1 or self.yp.ndim != 1:
            raise ValueError('Generalized piecewise-affine knots must be one-dimensional.')
        if self.xp.size != self.yp.size or self.xp.size < 2:
            raise ValueError('Generalized piecewise-affine knots must have equal length >= 2.')

        dx = np.diff(self.xp)
        dy = np.diff(self.yp)
        if np.any(dx < 0) or np.any(dy < 0):
            raise ValueError('Generalized piecewise-affine knots must be nondecreasing.')
        if np.any((dx == 0) & (dy == 0)):
            raise ValueError('Consecutive generalized knots must not be identical.')

        starts = np.r_[0, np.flatnonzero(dx) + 1]
        stops = np.r_[starts[1:], self.xp.size]
        self._x = self.xp[starts]
        self._entry_y = self.yp[starts]
        self._exit_y = self.yp[stops - 1]
        self._mid_y = 0.5 * (self._entry_y + self._exit_y)
        self._vertical = self._entry_y != self._exit_y
        self._selected_y = self._mid_y.copy()
        if self.select_boundary_interior and self._x.size > 1:
            if self._vertical[0]:
                self._selected_y[0] = self._exit_y[0]
            if self._vertical[-1]:
                self._selected_y[-1] = self._entry_y[-1]
        if self._x.size > 1:
            self._segment_slopes = (
                (self._entry_y[1:] - self._exit_y[:-1])
                / np.diff(self._x)
            )
        else:
            self._segment_slopes = np.empty(0)

    @property
    def image_lb(self):
        return float(self.yp[0])

    @property
    def image_ub(self):
        return float(self.yp[-1])

    @property
    def output_domain(self):
        return domain.FloatingPointColumn(self.image_lb, self.image_ub)

    @property
    def graph_inverse(self):
        return self.inverse.resolve(fixed_point=False)

    def apply(self, x, random_state=None):
        x = np.asarray(x)
        if self._x.size == 1:
            result = np.where(
                x < self._x[0],
                self._entry_y[0],
                np.where(x > self._x[0], self._exit_y[0], self._selected_y[0]),
            )
            return np.where(np.isnan(x), np.nan, result)

        x_eval = np.clip(x, self._x[0], self._x[-1])
        block = np.searchsorted(self._x, x_eval, side='right') - 1
        segment = block.clip(0, self._x.size - 2)
        segment_dy = self._entry_y[segment + 1] - self._exit_y[segment]
        with np.errstate(invalid='ignore'):
            fraction = (
                (x_eval - self._x[segment])
                / (self._x[segment + 1] - self._x[segment])
            )
        result = (
            self._exit_y[segment]
            + fraction * segment_dy
        )
        result = np.where(segment_dy == 0, self._exit_y[segment], result)

        exact_block = block.clip(0, self._x.size - 1)
        exact = x_eval == self._x[exact_block]
        result = np.where(exact, self._selected_y[exact_block], result)
        result = np.where(x < self._x[0], self._entry_y[0], result)
        result = np.where(x > self._x[-1], self._exit_y[-1], result)
        return np.where(np.isnan(x), np.nan, result)

    def derivative(self, x):
        x = np.asarray(x)
        if self._x.size == 1:
            result = np.zeros_like(x, dtype=float)
            result = np.where(
                x == self._x[0],
                np.inf if self._vertical[0] else 0.0,
                result,
            )
            return np.where(np.isnan(x), np.nan, result)

        block = np.searchsorted(self._x, x, side='right') - 1
        segment = block.clip(0, self._segment_slopes.size - 1)
        result = self._segment_slopes[segment]
        result = np.where((x < self._x[0]) | (x > self._x[-1]), 0.0, result)

        exact_block = block.clip(0, self._x.size - 1)
        vertical = (x == self._x[exact_block]) & self._vertical[exact_block]
        result = np.where(vertical, np.inf, result)
        return np.where(np.isnan(x), np.nan, result)

    def fiber_bounds(self, x):
        x = np.asarray(x)
        value = self(x)
        lower = np.array(value, dtype=float, copy=True)
        upper = np.array(value, dtype=float, copy=True)
        block = np.searchsorted(self._x, x, side='right') - 1
        exact_block = block.clip(0, self._x.size - 1)
        exact = x == self._x[exact_block]
        lower = np.where(exact, self._entry_y[exact_block], lower)
        upper = np.where(exact, self._exit_y[exact_block], upper)
        return lower, upper

    def preimage_bounds(self, y):
        lower, upper = self.graph_inverse.fiber_bounds(y)
        lower = np.where(y == self.image_lb, self.input_domain.lb, lower)
        upper = np.where(y == self.image_ub, self.input_domain.ub, upper)
        return lower, upper

# TODO: make sure this works with unbounded input and/or output domains
@dataclass
class PiecewiseAffine(InvertibleTransformation):
    """Piecewise affine map defined by reference points (xp -> fp).

    Uses ``numpy.interp`` internally. Values outside ``[xp[0], xp[-1]]``
    are clamped to the boundary output values.

    Parameters
    ----------
    input_domain : FloatingPointColumn
    xp : ndarray
        Sorted input reference points.
    yp : ndarray
        Corresponding output reference points (same length as *xp*).
    """
    xp: np.ndarray
    yp: np.ndarray

    @property
    def output_domain(self):
        return domain.FloatingPointColumn(
            float(min(self.yp[0], self.yp[-1])),
            float(max(self.yp[0], self.yp[-1])),
        )

    def __post_init__(self):
        self.xp = np.asarray(self.xp)
        self.yp = np.asarray(self.yp)
        if self.xp.ndim != 1 or self.yp.ndim != 1:
            raise ValueError('Piecewise-affine knots must be one-dimensional.')
        if self.xp.size != self.yp.size or self.xp.size < 2:
            raise ValueError('Piecewise-affine knots must have equal length >= 2.')

        dx = np.diff(self.xp)
        dy = np.diff(self.yp)
        if np.any(dx <= 0):
            raise ValueError('Piecewise-affine input knots must be strictly increasing.')
        if not (np.all(dy > 0) or np.all(dy < 0)):
            raise ValueError('Piecewise-affine output knots must be strictly monotone.')
        self.log_abs_dydx = np.log(np.abs(dy)) - np.log(dx)

    def apply(self, x, random_state=None):
        return np.interp(x, self.xp, self.yp)

    def log_abs_det_jacobian(self, x):
        idx = np.searchsorted(self.xp, x, side='right') - 1
        idx = idx.clip(0, self.log_abs_dydx.size - 1)
        return self.log_abs_dydx[idx]


# TODO: Adapt to fixed point
@dataclass
class InvertibleMap(InvertibleTransformation):
    f: Callable
    df: Callable
    is_lossy: ClassVar[bool] = False
    is_deterministic: ClassVar[bool] = True

    def apply(self, x, random_state=None):
        return self.f(x)

    def log_abs_det_jacobian(self, x):
        return np.log(np.abs(self.df(x)))

    def pushforward_distribution(self, distribution):
        if self.inverse is None:
            raise NotImplementedError(
                'Pushing a distribution through map requires f_inv.'
            )
        return super().pushforward_distribution(distribution)

    @property
    def output_domain(self):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            lb = self.f(self.input_domain.lb)
            ub = self.f(self.input_domain.ub)
        if ub < lb:
            lb, ub = ub, lb

        return domain.FloatingPointColumn(lower_bound=lb, upper_bound=ub)


@dataclass
class Digitize(AtomizingTransformation):
    bins: np.ndarray

    def apply(self, x, random_state=None):
        return np.digitize(x, self.bins[1:-1])

    @property
    def output_domain(self):
        return domain.IntegerColumn(0, self.bins.size - 2)

    def pullback_lb(self, y):
        return self.bins[y]

    def pullback_ub(self, y):
        return self.bins[y+1]

    @property
    def _pushforward_boundary_atoms_go_right(self):
        return np.ones(self.bins.size - 2, dtype=bool)

    @property
    def _pushforward_bins(self):
        return self.bins


# TODO: how to handle combination of continuous and discrete
# TODO: adapt to fixed point
@dataclass
class Clip(PushforwardMixin, Transformation):
    lb: float|int|None=None
    ub: float|int|None=None
    is_lossy: ClassVar[bool] = True
    is_deterministic: ClassVar[bool] = True
    _Pushforward: ClassVar[type] = ClippedPushedForwardDistribution

    def __post_init__(self):
        if self.lb is not None and self.ub is not None and self.lb > self.ub:
            raise ValueError('Clip lower bound must not exceed its upper bound.')

    def apply(self, x, random_state=None):
        if self.lb is None and self.ub is None:
            return x
        return np.clip(x, self.lb, self.ub)

    @property
    def output_domain(self):
        lb, ub = self.input_domain.lb, self.input_domain.ub
        if self.lb is not None:
            lb = max(lb, self.lb)
        if self.ub is not None:
            ub = min(ub, self.ub)
        return self.input_domain.with_new_bounds(lb, ub)


# TODO: default left and right dist as Uniform if output domain is bounded
@dataclass
class Unclip(PushforwardMixin, Transformation):
    lb: float|int|None=None
    ub: float|int|None=None
    left_dist: Distribution|None=None
    right_dist: Distribution|None=None
    is_lossy: ClassVar[bool] = False
    is_deterministic: ClassVar[bool] = False
    _Pushforward: ClassVar[type] = UnclippedPushedForwardDistribution

    def __post_init__(self):
        if self.lb is not None and self.ub is not None and self.lb == self.ub:
            raise ValueError(
                'Cannot reconstruct two clipped tails from one boundary atom.'
            )
        if self.lb is None and self.left_dist is not None:
            raise ValueError(
                'A left reconstruction distribution requires a lower bound.'
            )
        if self.ub is None and self.right_dist is not None:
            raise ValueError(
                'A right reconstruction distribution requires an upper bound.'
            )
        if self.lb is not None:
            if self.left_dist is None:
                raise ValueError('A left reconstruction distribution is required.')
            self._validate_tail_distribution(self.left_dist, 'left')
            left_support_ub = self.left_dist.support()[1]
            if (
                left_support_ub > self.lb
                and not np.isclose(
                    left_support_ub,
                    self.lb,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    'The left reconstruction distribution must be supported '
                    'at or below the lower clipping bound.'
                )
        if self.ub is not None:
            if self.right_dist is None:
                raise ValueError('A right reconstruction distribution is required.')
            self._validate_tail_distribution(self.right_dist, 'right')
            right_support_lb = self.right_dist.support()[0]
            if (
                right_support_lb < self.ub
                and not np.isclose(
                    right_support_lb,
                    self.ub,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    'The right reconstruction distribution must be supported '
                    'at or above the upper clipping bound.'
                )

    @staticmethod
    def _validate_tail_distribution(distribution, side):
        support = tuple(distribution.support())
        scalar_values = (*support, distribution.cdf(0.0), distribution.ppf(0.5))
        if any(np.ndim(value) != 0 for value in scalar_values):
            raise ValueError(
                f'The {side} reconstruction distribution must be scalar.'
            )

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        x = np.asarray(x)
        y = np.asarray(x, dtype=float).copy()

        if self.lb is not None:
            at_lb = x <= self.lb
            y[at_lb] = self.left_dist.rvs(
                size=at_lb.sum(),
                random_state=prng,
            )
        if self.ub is not None:
            at_ub = x >= self.ub
            y[at_ub] = self.right_dist.rvs(
                size=at_ub.sum(),
                random_state=prng,
            )
        return y

    # TODO: need to make sure that the Round that gets appended to match
    # fixed point handles rdist and ldist
    @property
    def output_domain(self):
        lb, ub = self.input_domain.lb, self.input_domain.ub
        if self.left_dist is not None:
            lb = min(lb, self.left_dist.support()[0])
        if self.right_dist is not None:
            ub = max(ub, self.right_dist.support()[1])
        return domain.FloatingPointColumn(lb, ub)


 # TODO: adapt to fixed point
 # TODO: add bayesian deconvolution inverse (need p(x))
@dataclass
class AddNoise(Transformation):
    distribution: ContinuousDistribution | None = None
    is_lossy: ClassVar[bool] = True
    is_deterministic: ClassVar[bool] = False

    def __post_init__(self):
        if self.distribution is None:
            raise ValueError('AddNoise requires a noise distribution.')
        self.distribution = as_distribution(self.distribution)

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        x = np.asarray(x)
        noise = self.distribution.rvs(size=x.shape, random_state=prng)
        return x + noise

    @property
    def output_domain(self):
        lb, ub = self.distribution.support()
        lb += self.input_domain.lb
        ub += self.input_domain.ub
        return domain.FloatingPointColumn(lb, ub)

    def conditional(self, x):
        # TODO: Need to implement shifted transformation
        # return x + distribution
        raise NotImplementedError


@dataclass
class Round(AtomizingTransformation):
    decimals: int = 0
    is_lossy: ClassVar[bool] = True
    is_deterministic: ClassVar[bool] = True

    def apply(self, x, random_state=None):
        return np.round(x, self.decimals)

    @property
    def output_domain(self):
        return domain.FixedPointColumn(
            self.decimals,
            np.round(self.input_domain.lb, self.decimals),
            np.round(self.input_domain.ub, self.decimals)
            )

    @property
    def _pushforward_bins(self):
        h = 10**-self.decimals
        lb = np.round(self.input_domain.lb, self.decimals)
        ub = np.round(self.input_domain.ub, self.decimals)
        num_midpoints = int(np.rint((ub - lb) / h))
        midpoints = lb + (np.arange(num_midpoints) + 0.5) * h
        return np.r_[self.input_domain.lb, midpoints, self.input_domain.ub]

    @property
    def _pushforward_boundary_atoms_go_right(self):
        h = 10**-self.decimals
        boundaries = self._pushforward_bins[1:-1]
        right_values = np.round(boundaries + h/2, self.decimals)
        return self.apply(boundaries) == right_values

    def pushforward_distribution(self, distribution):
        output_domain = self.output_domain
        if (
            isinstance(output_domain, domain.IntegerColumn)
            and output_domain.is_bounded
            and output_domain.cardinality < self._CARDINALITY_FOR_EAGER_PUSHFORWARD
        ):
            return super().pushforward_distribution(distribution)
        return RoundedPushedForwardDistribution(distribution, self)

    def pullback_lb(self, y):
        h = 10**-self.decimals
        return np.maximum(self.input_domain.lb, y - h/2)

    def pullback_ub(self, y):
        h = 10**-self.decimals
        return np.minimum(self.input_domain.ub, y + h/2)


@dataclass
class DequantizeRound(Round):
    original_domain: domain.Column = None

    @property
    def output_domain(self):
        # Rounding half-cell endpoints can expand the original lattice bounds.
        return self.original_domain

    def apply(self, x, random_state=None):
        y = super().apply(x, random_state)
        if isinstance(self.original_domain, domain.IntegerColumn):
            return y.astype(np.int64)
        return y

    @property
    def _pushforward_bins(self):
        h = self.original_domain.precision
        lb, ub = self.original_domain.lb, self.original_domain.ub
        count = int(np.rint((ub - lb) / h))
        midpoints = lb + (np.arange(count) + 0.5) * h
        return np.r_[self.input_domain.lb, midpoints, self.input_domain.ub]


## CONCRETE TRANSFORMS — DISCRETE / FIXED-POINT INPUT
## (inverses of the continuous transforms above)

def _check_bound_restriction(original, restricted):
    if (type(original) is not type(restricted)
            or original.decimals != restricted.decimals):
        raise NotImplementedError('Only bounds can be restricted through fixed-point conversion.')
    if restricted.lb < original.lb or restricted.ub > original.ub:
        raise ValueError('A domain restriction cannot widen the bounds.')


@dataclass
class FixedPointToInt(AtomizingTransformation):
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = True

    def restrict_input_domain(self, input_domain):
        _check_bound_restriction(self.input_domain, input_domain)
        forward = FixedPointToInt(input_domain)
        return forward.with_inverse(IntToFixedPoint(forward.output_domain, self.decimals))

    def restrict_output_domain(self, output_domain):
        _check_bound_restriction(self.output_domain, output_domain)
        h = 10**self.decimals
        return self.restrict_input_domain(domain.FixedPointColumn(
            self.decimals, output_domain.lb / h, output_domain.ub / h,
        ))

    def apply(self, x, random_state=None):
        h = 10**self.decimals
        return np.round(x*h).astype(np.int64)

    def pushforward_distribution(self, distribution):
        return RoundedPushedForwardDistribution(distribution, self)

    def pullback_lb(self, y):
        h = 10**-self.decimals
        return np.maximum(self.input_domain.lb, y*h - h/2)

    def pullback_ub(self, y):
        h = 10**-self.decimals
        return np.minimum(self.input_domain.ub, y*h + h/2)

    @property
    def output_domain(self):
        h = 10**self.decimals
        lb = self.input_domain.lb
        if np.isfinite(lb):
            lb = round(lb * h)

        ub = self.input_domain.ub
        if np.isfinite(ub):
            ub = round(ub * h)
        return domain.IntegerColumn(lb, ub)

    @property
    def decimals(self):
        return self.input_domain.decimals


@dataclass
class IntToFixedPoint(PushforwardMixin, Transformation):
    decimals: int
    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False
    _Pushforward: ClassVar[type] = AtomicScaledPushedForwardDistribution

    def restrict_input_domain(self, input_domain):
        _check_bound_restriction(self.input_domain, input_domain)
        forward = IntToFixedPoint(input_domain, self.decimals)
        return forward.with_inverse(FixedPointToInt(forward.output_domain))

    def restrict_output_domain(self, output_domain):
        _check_bound_restriction(self.output_domain, output_domain)
        h = 10**self.decimals
        lb = round(output_domain.lb * h) if np.isfinite(output_domain.lb) else output_domain.lb
        ub = round(output_domain.ub * h) if np.isfinite(output_domain.ub) else output_domain.ub
        return self.restrict_input_domain(domain.IntegerColumn(lb, ub))

    def apply(self, x, random_state=None):
        h = 10**-self.decimals
        return np.asarray(x, dtype=float)*h

    @property
    def output_domain(self):
        h = 10**-self.decimals
        lb = self.input_domain.lb * h
        ub = self.input_domain.ub * h
        return domain.FixedPointColumn(self.decimals, lb, ub)


@dataclass
class UniformDequantize(PushforwardMixin, Transformation):
    lb: float | None = None
    ub: float | None = None
    is_lossy: ClassVar[bool] = True
    is_deterministic: ClassVar[bool] = False
    _Pushforward: ClassVar[type] = UniformDequantizedPushedForwardDistribution

    def __post_init__(self):
        assert isinstance(self.input_domain, domain.FinitePrecisionMixin)
        h = self.input_domain.precision

        if self.lb is None:
            self.lb = self.input_domain.lb - h/2
        if self.ub is None:
            self.ub = self.input_domain.ub + h/2

        assert self.ub <= self.input_domain.ub + h/2
        assert self.lb >= self.input_domain.lb - h/2

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        h = self.input_domain.precision
        lb = np.maximum(x - h/2, self.lb)
        ub = np.minimum(x + h/2, self.ub)
        return prng.uniform(lb, ub)

    @property
    def output_domain(self):
        return domain.FloatingPointColumn(self.lb, self.ub)

    def conditional(self, x):
        h = self.input_domain.precision
        lb = np.maximum(x - h/2, self.lb)
        ub = np.minimum(x + h/2, self.ub)
        return as_distribution(uniform(lb, ub - lb))


class ContinuousInverseDigitize(PushforwardMixin, Transformation):
    bins: np.ndarray
    piece_bins: np.ndarray = np.array([])
    piece_distributions: list[ContinuousDistribution] = [None]
    is_deterministic: ClassVar[bool] = False
    is_lossy: ClassVar[bool] = False
    _Pushforward: ClassVar[type] = ExplodedPushedForwardDistribution

    def __init__(self, input_domain: domain.ContinuousColumn, bins, piece_bins=np.array([]), piece_distributions=[None]):
        assert np.isfinite(bins[0]) or piece_distributions[0] is not None
        assert np.isfinite(bins[-1]) or piece_distributions[-1] is not None

        self.input_domain = input_domain
        self.bins = bins
        self.piece_bins = piece_bins
        self.piece_distributions = piece_distributions
        self._bin_moments = {}

        self._compute_cumulative_bins()

    @property
    def output_domain(self):
        return domain.FloatingPointColumn(self.bins[0], self.bins[-1])

    def _compute_cumulative_bins(self):
        self.cumulative_bins = []

        for i, distr in enumerate(self.piece_distributions):
            start = self.piece_bins[i]
            stop = self.piece_bins[i+1] + 1

            if distr is None:  # uniform
                self.cumulative_bins.append(None)
            else:
                self.cumulative_bins.append(distr.cdf(self.bins[start:stop]))

    @property
    def num_pieces(self):
        return len(self.piece_distributions)

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        x = np.asarray(x)
        # TODO: check input domain assumptions

        piece_indexes = np.digitize(x, self.piece_bins[1:-1])

        y = np.empty(x.shape, dtype=self.output_domain.dtype)
        for i, distr in enumerate(self.piece_distributions):
            mask = piece_indexes == i
            xi = x[mask]

            if distr is None:  # uniform
                lb = self.bins[xi]
                ub = self.bins[xi + 1]
                y[mask] = prng.uniform(lb, ub)
            else:  # truncated inverse cdf sampling
                xi -= self.piece_bins[i]  # TODO: check if not i+1
                lb = self.cumulative_bins[i][xi]
                ub = self.cumulative_bins[i][xi + 1]
                u = prng.uniform(lb, ub)
                y[mask] = distr.ppf(u)

        return y

    def pdf(self, px: FiniteOrderedSupportDistribution, y):
        is_scalar = np.ndim(y) == 0
        y = np.atleast_1d(y)
        x = np.digitize(y, self.bins[1:-1])
        piece_indexes = np.digitize(x, self.piece_bins[1:-1])

        p = px.pmf(x)
        for i, distr in enumerate(self.piece_distributions):
            mask = piece_indexes == i
            xi = x[mask]

            if distr is None:  # uniform
                lb = self.bins[xi]
                ub = self.bins[xi + 1]
                p[mask] *= 1/(ub - lb)
            else:  # truncated inverse cdf sampling
                yi = y[mask]
                xi -= self.piece_bins[i]  # TODO: check if not i+1
                lb = self.cumulative_bins[i][xi]
                ub = self.cumulative_bins[i][xi + 1]
                p[mask] *= distr.pdf(yi)/(ub - lb)
        valid = (y >= self.bins[0]) & (y <= self.bins[-1])
        p = np.where(valid, p, 0.0)
        p = np.where(np.isnan(y), np.nan, p)
        return p[0] if is_scalar else p

    def cdf(self, px: Distribution, y):
        is_scalar = np.ndim(y) == 0
        y = np.atleast_1d(y)
        x = np.digitize(y, self.bins[1:-1])
        piece_indexes = np.digitize(x, self.piece_bins[1:-1])

        p = px.pmf(x)
        P = px.cdf(x) - p
        for i, distr in enumerate(self.piece_distributions):
            mask = piece_indexes == i
            xi = x[mask]
            yi = y[mask]

            if distr is None:  # uniform
                lb = self.bins[xi]
                ub = self.bins[xi + 1]
                P[mask] += p[mask] * (yi - lb)/(ub - lb)
            else:  # truncated inverse cdf sampling
                xi -= self.piece_bins[i]  # TODO: check if not i+1
                lb = self.cumulative_bins[i][xi]
                ub = self.cumulative_bins[i][xi + 1]
                P[mask] += p[mask] * (distr.cdf(yi) - lb)/(ub - lb)
        P = np.where(y < self.bins[0], 0.0, P)
        P = np.where(y >= self.bins[-1], 1.0, P)
        P = np.where(np.isnan(y), np.nan, P)
        return P[0] if is_scalar else P

    def ppf(self, px: Distribution, q):
        x = px.ppf(q) # bin shape: broadcast_shapes(q.shape, px.shape)
        
        is_scalar = np.ndim(x) == 0
        x = np.atleast_1d(x)
        
        piece_indexes = np.digitize(x, self.piece_bins[1:-1])

        q0, q1 = px.cdf(x-1), px.cdf(x) # mass accounted for
        #dq = np.atleast_1d((q - q0) / (q1 - q0))
        dq = (q - q0) / (q1 - q0)

        y, y1 = self.bins[x], self.bins[x+1]
        dy = y1 - y
        for i, distr in enumerate(self.piece_distributions):
            mask = piece_indexes == i
            dqi = dq[mask]
            if dqi.size == 0:
                continue
            if distr is None:
                y[mask] += dqi * dy[mask]
            else:
                xi = x[mask] - self.piece_bins[i]
                qi0 = self.cumulative_bins[i][xi]
                qi1 = self.cumulative_bins[i][xi + 1]
                y[mask] = distr.ppf(qi0 + dqi * (qi1 - qi0))
        
        return y[0] if is_scalar else y

    def moment(self, px: FiniteOrderedSupportDistribution, order: int):
        m = self.bin_moments(order)
        p = px.probabilities
        return p @ m
    
    def bin_moments(self, order):
        if order in self._bin_moments:
            return self._bin_moments[order]
        
        num_bins = self.bins.size - 1
        m = np.empty(num_bins)
        
        for start_bin, stop_bin, distribution in zip(
            self.piece_bins[:-1],
            self.piece_bins[1:],
            self.piece_distributions
            ):
            if distribution is None:
                p = order + 1
                im = self.bins[start_bin:stop_bin+1] ** p / p
                m[start_bin:stop_bin] = np.diff(im)/np.diff(self.bins[start_bin:stop_bin+1])
            else:
                b = self.bins[start_bin:stop_bin+1]
                for i, (lb, ub) in enumerate(zip(b[:-1], b[1:])):
                    m[start_bin+i] = distribution.expect(
                        lambda x: x**order,
                        lb = lb, ub = ub,
                        conditional = True
                        )
            
        self._bin_moments[order] = m
        return m
    
    def expect(self, px:FiniteOrderedSupportDistribution, func):
        num_bins = self.bins.size - 1
        efunc = np.empty(num_bins)
        
        for start_bin, stop_bin, distribution in zip(
            self.piece_bins[:-1],
            self.piece_bins[1:],
            self.piece_distributions
            ):
            if distribution is None:
                b = self.bins[start_bin:stop_bin+1]
                for i, (lb, ub) in enumerate(zip(b[:-1], b[1:])):
                    efunc[start_bin+i] = quad(func, lb, ub)[0] / (ub - lb)
            else:
                b = self.bins[start_bin:stop_bin+1]
                for i, (lb, ub) in enumerate(zip(b[:-1], b[1:])):
                    efunc[start_bin+i] = distribution.expect(
                        func,
                        lb = lb, ub = ub,
                        conditional = True
                        )
            
        return px.probabilities @ efunc
