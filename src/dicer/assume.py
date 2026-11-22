from math import floor, ceil, isfinite

from dicer import domain as d
from dicer.tracking import register_assume, ColumnAssumptions, A


@register_assume(d.FloatingPointColumn)
class FloatingPointAssumptions(ColumnAssumptions):

    @A
    def fixed_point(domain, digits):
        return d.FixedPointColumn.from_floating(domain, digits)

    def nonnegative(self):
        return self.ge(0.)
    
    def nonpositive(self):
        return self.le(0.)

    @A
    def ge(domain, lb):
        if lb >= domain.lb:
            return domain.with_new_bounds(lower_bound=lb)
        return domain

    @A
    def le(domain, ub):
        if ub <= domain.ub:
            return domain.with_new_bounds(upper_bound=ub)
        return domain


@register_assume(d.FixedPointColumn)
class FixedPointAssumptions(ColumnAssumptions):

    def nonnegative(self):
        return self.ge(0.)

    def nonpositive(self):
        return self.le(0.)

    @staticmethod
    def _quantize_lower_bound(value, decimals):
        if not isfinite(value):
            return value
        h = 10 ** decimals
        return round(ceil(value * h) / h, decimals)

    @staticmethod
    def _quantize_upper_bound(value, decimals):
        if not isfinite(value):
            return value
        h = 10 ** decimals
        return round(floor(value * h) / h, decimals)

    @A
    def ge(domain, lb):
        if lb >= domain.lb:
            lb = FixedPointAssumptions._quantize_lower_bound(lb, domain.decimals)
            return domain.with_new_bounds(lower_bound=lb)
        return domain

    @A
    def le(domain, ub):
        if ub <= domain.ub:
            ub = FixedPointAssumptions._quantize_upper_bound(ub, domain.decimals)
            return domain.with_new_bounds(upper_bound=ub)
        return domain


@register_assume(d.IntegerColumn)
class IntegerAssumptions(ColumnAssumptions):

    def nonnegative(self):
        return self.ge(0)
    
    def nonpositive(self):
        return self.le(0)
    
    @A
    def gt(domain, val):
        if val >= domain.lb:
            return domain.with_new_bounds(lower_bound=floor(val + 1))
        return domain

    @A
    def ge(domain, val):
        if val > domain.lb:
            return domain.with_new_bounds(lower_bound=ceil(val))
        return domain

    @A
    def lt(domain, val):
        if domain.ub >= val:
            return domain.with_new_bounds(upper_bound=ceil(val - 1))
        return domain

    @A
    def le(domain, val):
        if domain.ub > val:
            return domain.with_new_bounds(upper_bound=floor(val))
        return domain


@register_assume(d.GenericColumn)
class GenericColumnAssumptions(ColumnAssumptions):

    @A
    def has_support(domain, values):
        return d.FiniteUnorderedColumn(values, dtype=domain.dtype)
    
    
