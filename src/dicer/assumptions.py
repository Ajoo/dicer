from dataclasses import dataclass
import numpy as np
from typing import ClassVar, Callable
from numbers import Number
import operator as op
from functools import partial


class Assumption:
    def check(self, vals):
        raise NotImplementedError
    

@dataclass
class ComparisonAssumption(Assumption):
    value: Number
    operator: ClassVar[Callable]

    def check(self, vals):
        return np.all(self.operator(vals, self.value))

@dataclass
class GreaterThan(ComparisonAssumption):
    value: Number
    operator: ClassVar[Callable] = op.gt
    
    
@dataclass
class GreaterOrEqualThan(ComparisonAssumption):
    value: Number
    operator: ClassVar[Callable] = op.ge


@dataclass
class LesserOrEqualThan(ComparisonAssumption):
    value: Number
    operator: ClassVar[Callable] = op.le


@dataclass
class LesserThan(ComparisonAssumption):
    value: Number
    operator: ClassVar[Callable] = op.lt


@dataclass
class IsIn(ComparisonAssumption):
    value: np.ndarray
    operator: ClassVar[Callable] = partial(np.isin, assume_unique=True)

    def __post_init__(self):
        self.value = np.unique(np.asarray(self.value))


class IsUnordered(Assumption):

    def check(self, vals):
        return np.True_
    
@dataclass
class IsFinitePrecision(Assumption):
    decimals: int
    rtol: float = 1e-3
    atol: float = 1e-8

    def check(self, vals):
        d = 10**-self.decimals
        r = vals % d
        r = np.minimum(r, d-r)
        return np.all(r <= self.rtol * d + self.atol)
