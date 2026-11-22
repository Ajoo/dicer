"""Distribution utilities aligned with SciPy's ``rv_generic`` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, TypeAlias

import numpy as np
from scipy.stats._distn_infrastructure import (
    rv_generic,
    rv_frozen, 
    rv_continuous_frozen, 
    rv_discrete_frozen,
)

from dicer import domain as d


ScipyFrozen: TypeAlias = rv_frozen
ScipyContinuousFrozen: TypeAlias = rv_continuous_frozen
ScipyDiscreteFrozen: TypeAlias = rv_discrete_frozen


_supported_methods = [
    'pdf', 'logpdf', 'pmf', 'logpmf', 'pxf', 'logpxf',
    'cdf', 'logcdf', 'ppf', 'sf', 'logsf', 'isf', 'rvs',
    # TODO
    'stats', 'median', 'mean', 'var', 'std', 'moment',
    'entropy', 'interval', 'expect', 'support',
]


def as_distribution(dist):
    if isinstance(dist, Distribution):
        return dist
    
    if isinstance(dist, rv_generic):
        dist = dist() # freeze args
    
    if not isinstance(dist, rv_frozen):
        raise TypeError(
            f"dist argument should be a dicer Distribution or a scipy distribution."
            f"Got {type(dist)}."
        )
    return ScipyWrapped(dist)


@dataclass
class Distribution:
    domain: d.Column

    # at minimum must implement ppf and pdf / pmf
    def pdf(self, x):
        raise NotImplementedError
    
    def pmf(self, x):
        raise NotImplementedError

    # generic pdf + pmf
    def pxf(self, x):
        return self.pdf(x) + self.pmf(x)
    
    def logpxf(self, x):
        return np.logaddexp(self.logpdf(x), self.logpmf(x))
    
    def nnlf(self, x):
        return -np.sum(self.logpxf(x))
    
    # pdf/pmf derived methods
    def logpdf(self, x):
        with np.errstate(divide='ignore'):
            return np.log(self.pdf(x))
    
    def logpmf(self, x):
        with np.errstate(divide='ignore'):
            return np.log(self.pmf(x))
    
    # TODO: probably need to adjust size
    def rvs(self, size=None, random_state=None):
        prng = np.random.default_rng(random_state)
        u = prng.uniform(size=size)
        return self.ppf(u)
    
    @property
    def shape(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)
    

class OrderedDistribution(Distribution):
    domain: d.OrderedColumn

    def support(self):
        return self.domain.lb, self.domain.ub

    def cdf(self, x):
        raise NotImplementedError
    
    def ppf(self, q):
        raise NotImplementedError

    # cdf derived methods
    def logcdf(self, x):
        return np.log(self.cdf(x))
    
    def sf(self, x):
        return 1.0-self.cdf(x)
    
    def logsf(self, x):
        return np.log(self.sf(x))
    
    # ppf derived methods
    def isf(self, q):
        return self.ppf(1.0-q)
    
    def median(self):
        return self.ppf(0.5)
    
    def interval(self, confidence):
        confidence = np.asarray(confidence)
        q1 = (1.0-confidence)/2
        q2 = (1.0+confidence)/2
        return self.ppf(q1), self.ppf(q2)
    
    # alias
    def quantile(self, q):
        return self.ppf(q)
    
    @property
    def shape(self):
        raise NotImplementedError


class NoAtomsMixin:
    """ Adds pmf and pxf methods to a continuous distribution
    """

    def pmf(self, x):
        return np.zeros_like(x, dtype=np.float64)
    
    def logpmf(self, x):
        return np.full_like(x, -np.inf, dtype=np.float64)
    
    def pxf(self, x):
        return self.pdf(x)
    
    def logpxf(self, x):
        return self.logpdf(x)


class NoDensityMixin:
    """ Adds pdf and pxf methods to a discrete distribution
    """

    def pdf(self, x):
        return np.zeros_like(x, dtype=np.float64)
    
    def logpdf(self, x):
        return np.full_like(x, -np.inf, dtype=np.float64)
    
    def pxf(self, x):
        return self.pmf(x)
    
    def logpxf(self, x):
        return self.logpmf(x)


class UnorderedDistribution(NoDensityMixin, Distribution):
    domain: d.FiniteColumn

    def __post_init__(self):
        assert not self.domain.ordered

    def support(self):
        return self.domain.values


class ContinuousDistribution(NoAtomsMixin, OrderedDistribution):
    pass


class DiscreteOrderedDistribution(NoDensityMixin, OrderedDistribution):
    pass


class DiscreteGenPareto(DiscreteOrderedDistribution):
    """Floor-discretized GPD distances, optionally truncated and reflected.

    ``X = threshold + D`` (or ``threshold - D`` when left), where
    ``D = floor(Z)`` and ``Z`` is a zero-location continuous GPD.
    ``max_distance`` conditions D on a known finite integer bound.
    """

    def __init__(self, shape, scale, threshold=0, left=False, max_distance=np.inf):
        from scipy.stats import genpareto

        if scale <= 0:
            raise ValueError('GPD scale must be positive.')
        self.tail_shape = shape
        self.scale = scale
        self.threshold = threshold
        self.left = left
        self.base = genpareto(shape, scale=scale)
        endpoint = np.ceil(-scale / shape) - 1 if shape < 0 else np.inf
        self.max_distance = min(max_distance, endpoint)
        self._normalizer = self.base.cdf(self.max_distance + 1)
        self.domain = d.IntegerColumn(
            threshold - self.max_distance if left else threshold,
            threshold if left else threshold + self.max_distance,
        )

    @property
    def shape(self):
        return ()

    def logpmf(self, x):
        distance = self.threshold - np.asarray(x) if self.left else np.asarray(x) - self.threshold
        lo = self.base.logsf(distance)
        hi = self.base.logsf(distance + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = lo + np.log(-np.expm1(hi - lo)) - np.log(self._normalizer)
            valid = (distance >= 0) & (distance <= self.max_distance) & (distance % 1 == 0)
        return np.where(valid, result, -np.inf)

    def pmf(self, x):
        return np.exp(self.logpmf(x))

    def cdf(self, x):
        x = np.floor(np.asarray(x))
        if self.left:
            result = (
                self.base.sf(self.threshold - x)
                - self.base.sf(self.max_distance + 1)
            ) / self._normalizer
        else:
            result = self.base.cdf(x - self.threshold + 1) / self._normalizer
        return np.clip(result, 0, 1)

    def ppf(self, q):
        q = np.asarray(q)
        if self.left:
            z = self.base.isf(q * self._normalizer + self.base.sf(self.max_distance + 1))
            distance = np.floor(z)
        else:
            distance = np.ceil(self.base.ppf(q * self._normalizer)) - 1
        distance = np.clip(distance, 0, self.max_distance)
        result = self.threshold - distance if self.left else self.threshold + distance
        # Inverting the continuous CDF can land one ULP across an integer.
        # Enforce the discrete quantile inequalities at jump boundaries.
        result = np.where(self.cdf(result - 1) >= q, result - 1, result)
        result = np.where(self.cdf(result) < q, result + 1, result)
        result = np.where(q == 0, self.domain.lb, result)
        result = np.where(q == 1, self.domain.ub, result)
        return np.where((q < 0) | (q > 1), np.nan, result)

    def rvs(self, size=None, random_state=None):
        prng = np.random.default_rng(random_state)
        q = prng.uniform(size=size)
        distance = np.floor(self.base.ppf(q * self._normalizer)).astype(np.int64)
        return self.threshold - distance if self.left else self.threshold + distance


class ReflectedDistribution(ContinuousDistribution):
    """Distribution of ``-X`` for a continuous distribution ``X``."""

    def __init__(self, distribution):
        self.distribution = as_distribution(distribution)
        lb, ub = self.distribution.support()
        self.domain = d.FloatingPointColumn(-ub, -lb)

    @property
    def shape(self):
        return self.distribution.shape

    def pdf(self, x):
        return self.distribution.pdf(-np.asarray(x))

    def logpdf(self, x):
        return self.distribution.logpdf(-np.asarray(x))

    def cdf(self, x):
        return self.distribution.sf(-np.asarray(x))

    def logcdf(self, x):
        return self.distribution.logsf(-np.asarray(x))

    def sf(self, x):
        return self.distribution.cdf(-np.asarray(x))

    def logsf(self, x):
        return self.distribution.logcdf(-np.asarray(x))

    def ppf(self, q):
        return -self.distribution.isf(np.asarray(q))

    def isf(self, q):
        return -self.distribution.ppf(np.asarray(q))

    def rvs(self, size=None, random_state=None):
        return -self.distribution.rvs(
            size=size,
            random_state=random_state,
        )

    def mean(self, *args, **kwargs):
        return -self.distribution.mean(*args, **kwargs)

    def var(self, *args, **kwargs):
        return self.distribution.var(*args, **kwargs)

    def std(self, *args, **kwargs):
        return self.distribution.std(*args, **kwargs)

    def moment(self, order, *args, **kwargs):
        result = self.distribution.moment(order, *args, **kwargs)
        return -result if order % 2 else result

    def entropy(self, *args, **kwargs):
        return self.distribution.entropy(*args, **kwargs)

    def expect(self, func=None, *args, **kwargs):
        lb = kwargs.pop('lb', None)
        ub = kwargs.pop('ub', None)
        if ub is not None:
            kwargs['lb'] = -ub
        if lb is not None:
            kwargs['ub'] = -lb
        if func is None:
            reflected_func = lambda x: -x
        else:
            reflected_func = lambda x: func(-x)
        return self.distribution.expect(reflected_func, *args, **kwargs)

    def stats(self, *args, moments='mv', **kwargs):
        result = self.distribution.stats(
            *args,
            moments=moments,
            **kwargs,
        )
        values = (result,) if len(moments) == 1 else result
        reflected = tuple(
            -value if moment in ('m', 's') else value
            for moment, value in zip(moments, values)
        )
        return reflected[0] if len(reflected) == 1 else reflected


class ScipyWrapped(rv_frozen):

    def __new__(cls, dist, *args, **kwds):
        if cls is ScipyWrapped:
            if isinstance(dist, rv_continuous_frozen):
                return super().__new__(ScipyContinuous)
            elif isinstance(dist, rv_discrete_frozen):
                return super().__new__(ScipyInteger)
        return super().__new__(cls)

    def __init__(
        self,
        dist: rv_frozen, 
        domain: d.Column=None
        ):

        if domain is None:
            lb, ub = dist.support()
            domain = self._DefaultDomain(np.min(lb), np.max(ub))

        self.domain = domain
        self.dist = dist.dist
        self.args = dist.args
        self.kwds = dist.kwds

    @property
    def shape(self):
        return np.broadcast_shapes(
            *[arg.shape for arg in self.args if not np.isscalar(arg)], 
            *[arg.shape for arg in self.kwds.values() if not np.isscalar(arg)])

    @property
    def ndim(self):
        return len(self.shape)

    def support(self):
        return self.domain.lb, self.domain.ub

    def rvs(self, size=None, random_state=None):
        return super().rvs(size=size, random_state=random_state)


class ScipyContinuous(NoAtomsMixin, ScipyWrapped, rv_continuous_frozen):
    _DefaultDomain = d.FloatingPointColumn
        

class ScipyInteger(NoDensityMixin, ScipyWrapped, rv_discrete_frozen):
    _DefaultDomain = d.IntegerColumn


ContinuousFrozen: TypeAlias = ScipyContinuousFrozen | ContinuousDistribution
DiscreteFrozen: TypeAlias = ScipyDiscreteFrozen | DiscreteOrderedDistribution


def _deduplicate_support(support, probabilities):
    support, indices, inverse = np.unique(support, return_index=True, return_inverse=True)
    probabilities = np.asarray(probabilities)

    if support.size == inverse.size:
        return support, probabilities[..., indices]
    
    if probabilities.ndim == 1:
        probabilities = np.bincount(inverse, weights=probabilities)
    else:
        P = inverse[:, None] == np.arange(support.size)
        probabilities = np.tensordot(probabilities, P, 1)

    return support, probabilities


class FiniteDomain(d.MaybeFinite, d.Column):
    pass


# def _take_over_last(a, i):
#     adim, idim = a.ndim, i.ndim
#     a = a[(None,) *idim]
#     i = i[(Ellipsis,) + (None,) * adim]
#     ai = np.take_along_axis(a, i, axis=-1)[..., 0]
#     return np.where(i[...,0]==-1, 0., ai)

def _take_over_last(a, i):
    ai = np.take_along_axis(*np.broadcast_arrays(a, i[..., None]), axis=-1)
    return np.where(i==-1, 0., ai[..., 0])


def _sample_categorical(p, size=None, random_state=None):
    prng = np.random.default_rng(random_state)
    
    size = () if size is None else size
    shape = np.broadcast_shapes(p.shape[:-1], size)
    u = prng.uniform(size=shape)[..., None]
    P = np.cumsum(p, -1)

    return np.sum(u > P, -1) # np.argmin(u > P, -1)


# TODO: consider allowing this distribution to have an arbitrary domain
# independent from the support (as long as it contains it)
# in this case, we would add a values field and a Delta distribution would
# just be a special case with a single support point
@dataclass
class FiniteSupportDistribution(NoDensityMixin):
    domain: FiniteDomain
    probabilities: np.ndarray

    def __new__(cls, domain: FiniteDomain, *args, **kwargs):
        if cls is not FiniteSupportDistribution:
            return super().__new__(cls)
        
        if domain.ordered:
            return super().__new__(FiniteOrderedSupportDistribution)
        return super().__new__(FiniteUnorderedSupportDistribution)

    def __post_init__(self):
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if not self.domain.finite:
            raise ValueError('domain must be finite.')
        if self.probabilities.ndim == 0:
            raise ValueError('probabilities must have a support axis.')
        if self.domain.cardinality != self.probabilities.shape[-1]:
            raise ValueError(
                'The final probability axis must match the domain cardinality.'
            )
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError('probabilities must be finite.')
        if np.any(self.probabilities < 0):
            raise ValueError('probabilities must be nonnegative.')
        if not np.allclose(self.probabilities.sum(axis=-1), 1.0):
            raise ValueError('probabilities must sum to one.')

    @classmethod
    def from_support(
        cls,
        support: Sequence[float | int],
        probabilities: np.ndarray,
    ):
        support, probabilities = _deduplicate_support(support, probabilities)
        domain = cls._Column(support)

        return cls(domain, probabilities)

    def pmf(self, x):
        idx = self.domain.index(x)
        return _take_over_last(self.probabilities, idx)
    
    def rvs(self, size=None, random_state=None):
        """Draw random variates from the finite support distribution.

        Parameters
        ----------
        size : int or tuple of int, optional
            Number of samples to draw, or the sample shape. If omitted, a single
            sample is returned.
        random_state : None, int, numpy.random.Generator, or RandomState
            Random state for reproducible draws, following SciPy's API.

        Returns
        -------
        numpy.ndarray
            Samples from the support with shape ``size + self.shape``.
        """
        idx = _sample_categorical(
            self.probabilities,
            size=size,
            random_state=random_state,
        )
        return  self.domain.at(idx)

    @property
    def shape(self):
        return self.probabilities.shape[:-1]


class FiniteOrderedSupportDistribution(FiniteSupportDistribution, OrderedDistribution):
    _Column = d.FiniteOrderedColumn

    def cdf(self, x):
        idx = self.domain.index(x, last=True)
        P = np.cumsum(self.probabilities, -1)
        return _take_over_last(P, idx)
    
    def ppf(self, q):
        P = np.cumsum(self.probabilities, -1)
        q = np.asarray(q)

        idx = np.sum(P[...,:-1] < q[..., None], -1)
        return self.domain.values[idx]

        #return np.interp(q, P, self.domain.values)
        

class FiniteUnorderedSupportDistribution(FiniteSupportDistribution, UnorderedDistribution):
    _Column = d.FiniteUnorderedColumn


@dataclass
class Delta(NoDensityMixin, Distribution):
    """Degenerate distribution concentrated at a single point."""
    value: np.ndarray

    def __new__(cls, domain: FiniteDomain, *args, **kwargs):
        if cls is not Delta:
            return super().__new__(cls)
        
        if domain.ordered:
            return super().__new__(OrderedDelta)
        return super().__new__(cls)

    def __post_init__(self):
        self.value = np.asarray(self.value)

    def pmf(self, x):
        return x == self.value
    
    def rvs(self, size=None, random_state=None):
        if size is None:
            return self.value
        return np.broadcast_to(self.value, size)

    @property
    def shape(self):
        return self.value.shape


class OrderedDelta(Delta, OrderedDistribution):
    def cdf(self, x):
        return x >= self.value
    
    def ppf(self, q):
        return np.broadcast_to(self.value, q.shape)
