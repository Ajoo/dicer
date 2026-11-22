"""Fitted reconstruction tails for ordered integer columns."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import randint, rv_discrete

from dicer import domain
from dicer.stats import DiscreteGenPareto, ScipyInteger
from dicer.tracking import ColumnInferences, I, register_infer
from dicer.transform.integer import IntegerClip
from dicer.infer.continuous import _validate_tail_probability


def infer_integer_genpareto_tail(
    tail_vals, threshold, *, left=False, domain_bound=None,
    force_unbounded=True,
):
    """Fit unit-cell GPD probabilities by discrete maximum likelihood."""
    if domain_bound is None:
        domain_bound = -np.inf if left else np.inf
    distances = threshold - np.asarray(tail_vals) if left else np.asarray(tail_vals) - threshold
    if not distances.size:
        raise ValueError('GPD tail fitting requires observations in the clipped tail.')
    values, counts = np.unique(distances, return_counts=True)
    maximum = abs(domain_bound - threshold)
    preserve_endpoint = maximum if force_unbounded else values[-1]

    def objective(params):
        shape, log_scale = params
        scale = np.exp(log_scale)
        if shape < 0 and -scale / shape <= preserve_endpoint:
            return np.inf
        tail = DiscreteGenPareto(shape, scale, max_distance=maximum)
        logp = tail.logpmf(values)
        return -np.dot(counts, logp) if np.all(np.isfinite(logp)) else np.inf

    initial_scale = max(float(np.mean(distances)), 0.5)
    starts = [0.0, 0.3] if np.isinf(preserve_endpoint) else [-0.5, 0.0, 0.3]
    fits = []
    for shape in starts:
        scale = max(initial_scale, -shape * (preserve_endpoint + 1)) if shape < 0 else initial_scale
        fits.append(minimize(
            objective, [shape, np.log(scale)], method='Nelder-Mead',
            bounds=[(0, None) if np.isinf(preserve_endpoint) else (None, None), (-30, 30)],
            options={'maxiter': 2000},
        ))
    fits = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not fits:
        raise ValueError('Discrete GPD tail fit did not converge.')
    fit = min(fits, key=lambda fit: fit.fun)
    return DiscreteGenPareto(
        fit.x[0], np.exp(fit.x[1]), threshold=threshold,
        left=left, max_distance=maximum,
    )


def infer_integer_uniform_tail(
    tail_vals, threshold, *, left=False, domain_bound=None,
    force_unbounded=True,
):
    """Uniform integer tail up to a known bound or the observed extreme."""
    if domain_bound is None:
        domain_bound = -np.inf if left else np.inf
    endpoint = domain_bound
    if not np.isfinite(endpoint):
        vals = np.asarray(tail_vals)
        if not vals.size:
            raise ValueError('An unbounded uniform tail requires observations.')
        endpoint = vals.min() if left else vals.max()
    lb, ub = (endpoint, threshold) if left else (threshold, endpoint)
    tail = randint(int(lb), int(ub) + 1)
    if force_unbounded:
        lb, ub = (domain_bound, threshold) if left else (threshold, domain_bound)
    return ScipyInteger(tail, domain=domain.IntegerColumn(lb, ub))


def infer_integer_empirical_tail(
    tail_vals, threshold, *, left=False, domain_bound=None,
    force_unbounded=True, smoothing=0.5,
):
    """Add a pseudocount to every integer in the reconstruction interval."""
    if domain_bound is None:
        domain_bound = -np.inf if left else np.inf
    if smoothing < 0:
        raise ValueError('smoothing must be nonnegative.')
    vals = np.asarray(tail_vals)
    if not vals.size and not np.isfinite(domain_bound):
        raise ValueError('An unbounded empirical tail requires observations.')
    endpoint = domain_bound
    if not np.isfinite(endpoint):
        endpoint = vals.min() if left else vals.max()
    lb, ub = (endpoint, threshold) if left else (threshold, endpoint)
    values = np.arange(int(lb), int(ub) + 1)
    counts = np.bincount((vals - lb).astype(np.int64), minlength=values.size)
    probabilities = counts + smoothing
    if probabilities.sum() == 0:
        raise ValueError('An empty empirical tail requires positive smoothing.')
    probabilities = probabilities / probabilities.sum()
    tail = rv_discrete(values=(values, probabilities))()
    if force_unbounded:
        lb, ub = (domain_bound, threshold) if left else (threshold, domain_bound)
    return ScipyInteger(tail, domain=domain.IntegerColumn(lb, ub))


@register_infer(domain.IntegerColumn)
class IntegerInferences(ColumnInferences):

    @I
    def nonnegative(domain, vals):
        """Assume nonnegative when all observed integers are nonnegative."""
        if np.all(np.asarray(vals) >= 0):
            return domain.assume.nonnegative()
        return domain

    @I
    def lb(domain, vals, tail_type='min', tail_prob=0.05):
        """Assume the empirical minimum or a fitted discrete GPD endpoint."""
        if tail_type == 'min':
            return domain.assume.ge(np.min(vals))
        if tail_type != 'genpareto':
            raise ValueError("tail_type must be 'min' or 'genpareto'.")
        _validate_tail_probability(tail_prob)
        vals = np.asarray(vals)
        threshold = np.floor(np.quantile(vals, tail_prob))
        tail = infer_integer_genpareto_tail(
            vals[vals <= threshold], threshold, left=True, force_unbounded=False,
        )
        return domain.assume.ge(tail.support()[0])

    @I
    def ub(domain, vals, tail_type='max', tail_prob=0.05):
        """Assume the empirical maximum or a fitted discrete GPD endpoint."""
        if tail_type == 'max':
            return domain.assume.le(np.max(vals))
        if tail_type != 'genpareto':
            raise ValueError("tail_type must be 'max' or 'genpareto'.")
        _validate_tail_probability(tail_prob)
        vals = np.asarray(vals)
        threshold = np.ceil(np.quantile(vals, 1 - tail_prob))
        tail = infer_integer_genpareto_tail(
            vals[vals >= threshold], threshold, force_unbounded=False,
        )
        return domain.assume.le(tail.support()[1])

    @I
    def quantile_discretize(
        domain, vals, num_bins,
        min_prob_left_tail=None, min_prob_right_tail=None,
        left_tail_distribution='auto', right_tail_distribution='auto',
        assume_bounds=False, *, middle_distribution='uniform', smoothing=0.5,
    ):
        """Fit integer quantile cuts and piecewise reconstruction laws.

        Auto uses uniform on bounded sides and GPD on unbounded sides.
        Each fitted piece is conditioned separately on every bin it spans.
        Repeated cuts are deduplicated, so num_bins is an upper bound.
        """
        from dicer.infer.continuous import _validate_num_bins, _validate_tail_probability

        _validate_num_bins(num_bins)
        vals = np.asarray(vals)
        if vals.ndim != 1 or vals.size == 0:
            raise ValueError('Quantile inference requires nonempty one-dimensional data.')
        if middle_distribution not in ('uniform', 'empirical', None):
            raise ValueError("middle_distribution must be 'uniform', 'empirical', or None.")

        names, probabilities = [], []
        for bound, name, prob in (
            (domain.lb, left_tail_distribution, min_prob_left_tail),
            (domain.ub, right_tail_distribution, min_prob_right_tail),
        ):
            if name == 'auto':
                name = 'uniform' if np.isfinite(bound) else 'genpareto'
            if name not in ('uniform', 'empirical', 'genpareto'):
                raise ValueError("Tail distribution must be 'auto', 'uniform', 'empirical', or 'genpareto'.")
            if prob is None:
                prob = 0.0 if name == 'uniform' else 0.05
            _validate_tail_probability(prob)
            if not np.isfinite(bound) and name != 'uniform' and prob == 0:
                raise ValueError('An unbounded fitted tail requires probability greater than zero.')
            names.append(name)
            probabilities.append(prob)
        if sum(probabilities) >= 1:
            raise ValueError('Tail probabilities must sum to less than one.')

        raw_cuts = np.ceil(np.quantile(vals, np.arange(1, num_bins) / num_bins))
        cuts = np.unique(raw_cuts)
        cuts = cuts[(cuts > max(domain.lb, vals.min())) & (cuts <= domain.ub)]
        count = cuts.size + 1

        # A single bin has no identifiable left/middle/right split. On an
        # unbounded domain use empirical reconstruction over the observed range.
        if count == 1:
            if assume_bounds:
                domain = domain.assume.ge(vals.min()).assume.le(vals.max())
            base = None
            if not domain.is_bounded or middle_distribution == 'empirical':
                base = infer_integer_empirical_tail(
                    vals, domain.lb if domain.is_bounded_left else vals.min(),
                    domain_bound=domain.ub if domain.is_bounded_right else vals.max(),
                    smoothing=smoothing,
                )
            return domain.transform.digitize(
                cuts, piece_bins=[0, 1], piece_distributions=[base],
            )

        counts = []
        for left, prob, bound, name in (
            (True, probabilities[0], domain.lb, names[0]),
            (False, probabilities[1], domain.ub, names[1]),
        ):
            requested = int(np.ceil(prob * num_bins))
            if requested:
                threshold = raw_cuts[requested - 1 if left else -requested]
                boundary = int(np.searchsorted(cuts, threshold)) + 1
                n = boundary if left else count - boundary
            else:
                n = 0
            if not np.isfinite(bound):
                n = max(1, n)
            counts.append(min(n, count))
        left_count, right_count = counts
        right_start = count - right_count
        if left_count > right_start:
            raise ValueError('Tail pieces overlap after deduplicating quantile cuts.')

        bases = [None, None, None]
        for left, n, name, bound in (
            (True, left_count, names[0], domain.lb),
            (False, right_count, names[1], domain.ub),
        ):
            if n == 0:
                continue
            threshold = cuts[n - 1] - 1 if left else cuts[count - n - 1]
            tail_vals = vals[vals <= threshold] if left else vals[vals >= threshold]
            options = dict(left=left, domain_bound=bound, force_unbounded=not assume_bounds)
            if name == 'genpareto':
                base = infer_integer_genpareto_tail(tail_vals, threshold, **options)
            elif name == 'empirical':
                base = infer_integer_empirical_tail(tail_vals, threshold, smoothing=smoothing, **options)
            elif np.isfinite(bound):
                base = None
            else:
                base = infer_integer_uniform_tail(tail_vals, threshold, **options)
            bases[0 if left else 2] = base
            if assume_bounds and base is not None:
                domain = domain.assume.ge(base.support()[0]) if left else domain.assume.le(base.support()[1])

        if middle_distribution == 'empirical' and left_count < right_start:
            lo = cuts[left_count - 1] if left_count else domain.lb
            hi = cuts[right_start - 1] - 1 if right_start < count else domain.ub
            middle_vals = vals[(vals >= lo) & (vals <= hi)]
            bases[1] = infer_integer_empirical_tail(
                middle_vals, lo, domain_bound=hi, smoothing=smoothing,
            )

        edges, distributions = [0], []
        for start, stop, base in (
            (0, left_count, bases[0]), (left_count, right_start, bases[1]),
            (right_start, count, bases[2]),
        ):
            if start < stop:
                edges.append(stop)
                distributions.append(base)
        return domain.transform.digitize(cuts, piece_bins=edges, piece_distributions=distributions)

    @I
    def clip(
        domain, vals, lb=None, ub=None,
        left_tail_distribution='auto', right_tail_distribution='auto',
        assume_bounds=False, smoothing=0.5,
    ):
        """Clip and fit GPD, empirical, or uniform conditional integer tails.

        Auto chooses empirical on bounded sides and GPD on unbounded sides.
        Bounds are rounded inward before selecting tail observations.
        """
        vals = np.asarray(vals)
        rounded = IntegerClip(domain, lb, ub)
        lb, ub = rounded.lb, rounded.ub
        if lb is not None and ub is not None and lb >= ub:
            raise ValueError('clip lb must be strictly less than ub for reconstruction.')
        if (lb is not None and lb < domain.lb) or (ub is not None and ub > domain.ub):
            raise ValueError('clip bounds must lie within the input domain.')
        tails = []
        for left, threshold, bound, name in (
            (True, lb, domain.lb, left_tail_distribution),
            (False, ub, domain.ub, right_tail_distribution),
        ):
            active = threshold is not None and (threshold > bound if left else threshold < bound)
            if not active:
                tails.append(None)
                continue
            if name == 'auto':
                name = 'empirical' if np.isfinite(bound) else 'genpareto'
            tail_vals = vals[vals <= threshold] if left else vals[vals >= threshold]
            options = dict(left=left, domain_bound=bound, force_unbounded=not assume_bounds)
            if name == 'genpareto':
                tail = infer_integer_genpareto_tail(tail_vals, threshold, **options)
            elif name == 'empirical':
                tail = infer_integer_empirical_tail(tail_vals, threshold, smoothing=smoothing, **options)
            elif name == 'uniform':
                tail = infer_integer_uniform_tail(tail_vals, threshold, **options)
            else:
                raise ValueError("tail distribution must be 'auto', 'genpareto', 'empirical', or 'uniform'.")
            tails.append(tail)
            if assume_bounds:
                domain = domain.assume.ge(tail.support()[0]) if left else domain.assume.le(tail.support()[1])
        if all(tail is None for tail in tails):
            raise ValueError('At least one clip bound must tighten the input domain.')
        return domain.transform.clip(lb, ub, left_dist=tails[0], right_dist=tails[1])
