import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto, truncnorm, uniform

from dicer import domain
from dicer.stats import ReflectedDistribution, ScipyContinuous
from dicer.tracking import register_infer, ColumnInferences, I


MIN_PARAMETRIC_TAIL_SAMPLES = 3


@register_infer(domain.FloatingPointColumn)
class FloatingPointInferences(ColumnInferences):

    def _get_data(self, data=None):
        vals = np.asarray(super()._get_data(data))
        if vals.ndim != 1:
            raise ValueError('Floating-point inference data must be one-dimensional.')
        if vals.size == 0:
            raise ValueError('Floating-point inference requires non-empty data.')
        if not np.all(np.isfinite(vals)):
            raise ValueError('Floating-point inference data must contain only finite values.')
        return vals

    @I
    def nonnegative(domain, vals):
        """Assume nonnegative if all data values are >= 0.

        Returns
        -------
        Updated domain.
        """
        if np.all(vals >= 0.0):
            return domain.assume.nonnegative()
        return domain

    @I
    def ub(domain, vals, tail_type='max', tail_prob=0.05):
        """Infer and assume upper bound from data.

        Parameters
        ----------
        type : {'minmax', 'genpareto'}, optional
            Method for inferring the bound.
            ``'minmax'`` uses the empirical minimum.
            ``'genpareto'`` fits a generalized Pareto tail; a finite bound is
            assumed only when the fitted shape parameter ``c < 0``.
        tail_prob : float, optional
            Fraction of data treated as tail when ``type`` is ``'genpareto'``.

        Returns
        -------
        Updated domain.
        """

        if tail_type == 'max':
            return domain.assume.le(np.max(vals))
        if tail_type != 'genpareto':
            raise ValueError("tail_type must be 'max' or 'genpareto'.")
        _validate_tail_probability(tail_prob)
        
        threshold = np.quantile(vals, 1.0 - tail_prob)
        tail_vals = vals[vals >= threshold]
        tail = infer_genpareto_tail(
            tail_vals,
            threshold,
            force_unbounded=False,
        )
        return domain.assume.le(tail.support()[1])

    @I
    def lb(domain, vals, tail_type='min', tail_prob=0.05):
        """Infer and assume upper bound from data.

        Parameters
        ----------
        data : array-like, optional
            Data to inspect. If omitted, uses the tracker's tracked data.
        type : {'minmax', 'genpareto'}, optional
            Method for inferring the bound.
            ``'minmax'`` uses the empirical minimum.
            ``'genpareto'`` fits a generalized Pareto tail; a finite bound is
            assumed only when the fitted shape parameter ``c < 0``.
        tail_prob : float, optional
            Fraction of data treated as tail when ``type`` is ``'genpareto'``.

        Returns
        -------
        Updated domain.
        """

        if tail_type == 'min':
            return domain.assume.ge(np.min(vals))
        if tail_type != 'genpareto':
            raise ValueError("tail_type must be 'min' or 'genpareto'.")
        _validate_tail_probability(tail_prob)
        
        threshold = np.quantile(vals, tail_prob)
        tail_vals = vals[vals <= threshold]
        tail = infer_genpareto_tail(
            tail_vals,
            threshold,
            force_unbounded=False,
            left=True,
        )
        return domain.assume.ge(tail.support()[0])
    
    @I
    def fixed_point(domain, vals, min_digits=0, max_digits=5, rtol=1e-3, atol=1e-8):
        digits = infer_fixed_point(
            vals, 
            min_digits=min_digits, max_digits=max_digits, 
            rtol=rtol, atol=atol
        )
        if digits is None:
            return domain
        return domain.assume.fixed_point(int(digits))

    @I
    def standardize(domain, vals):
        """Affine transform ``(x - mean) / std``.

        Returns
        -------
        Updated tracker with standardized domain and data.
        """
        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0:
            std = 1.0
        return domain.transform.affine(1.0 / std, -mean / std)

    @I
    def min_max_scale(domain, vals, assume_bounds=False):
        """Affine transform ``(x - min) / (max - min)`` mapping data to [0, 1].

        Returns
        -------
        Updated tracker with min-max scaled domain and data.
        """
        lb = np.min(vals)
        ub = np.max(vals)
        if assume_bounds:
            domain = domain.assume.ge(lb).assume.le(ub)
        data_range = ub - lb
        if data_range == 0:
            data_range = 1.0
        return domain.transform.affine(
            1.0 / data_range,
            -lb / data_range,
        )

    @I
    def clip(
        domain,
        vals,
        lb=None,
        ub=None,
        left_tail_distribution='auto',
        right_tail_distribution='auto',
        assume_bounds=False,
    ):
        """Clip at explicit bounds and fit conditional reconstruction tails.

        ``auto`` uses a uniform tail on an already-bounded side and a normal
        tail on an unbounded side. Generalized-Pareto fits retain unbounded
        support unless ``assume_bounds=True``.
        """
        if lb is None and ub is None:
            raise ValueError('clip inference requires lb, ub, or both.')

        for name, value in (('lb', lb), ('ub', ub)):
            if value is not None and (
                not np.isscalar(value) or not np.isfinite(value)
            ):
                raise ValueError(f'clip {name} must be a finite scalar.')

        if lb is not None and lb < domain.lb:
            raise ValueError('clip lb must lie within the input domain.')
        if ub is not None and ub > domain.ub:
            raise ValueError('clip ub must lie within the input domain.')
        if lb is not None and ub is not None and lb >= ub:
            raise ValueError('clip lb must be strictly less than ub.')

        clips_left = lb is not None and lb > domain.lb
        clips_right = ub is not None and ub < domain.ub
        if not clips_left and not clips_right:
            raise ValueError('At least one clip bound must tighten the input domain.')

        left_tail = None
        right_tail = None
        if clips_left:
            left_tail_distribution = _check_tail_distribution(
                domain.is_bounded_left,
                left_tail_distribution,
            )
            left_tail = _fit_clip_tail(
                vals[vals <= lb],
                lb,
                left_tail_distribution,
                left=True,
                domain_bound=domain.lb,
                force_unbounded=not assume_bounds,
            )

        if clips_right:
            right_tail_distribution = _check_tail_distribution(
                domain.is_bounded_right,
                right_tail_distribution,
            )
            right_tail = _fit_clip_tail(
                vals[vals >= ub],
                ub,
                right_tail_distribution,
                left=False,
                domain_bound=domain.ub,
                force_unbounded=not assume_bounds,
            )

        if assume_bounds:
            if left_tail is not None:
                fitted_lb = left_tail.support()[0]
                if np.isfinite(fitted_lb) and fitted_lb > domain.lb:
                    domain = domain.assume.ge(fitted_lb)
            if right_tail is not None:
                fitted_ub = right_tail.support()[1]
                if np.isfinite(fitted_ub) and fitted_ub < domain.ub:
                    domain = domain.assume.le(fitted_ub)

        return domain.transform.clip(
            lb,
            ub,
            left_dist=left_tail,
            right_dist=right_tail,
        )

    @I
    def to_quantiles(domain, vals, num_quantiles=100, assume_bounds=False):
        """Piecewise affine transform mapping data to approximate Uniform[0, 1].

        Uses empirical quantiles as reference points. The inverse maps [0, 1]
        back to the original data scale (i.e., the empirical quantile function).

        Values outside the observed data range are clamped to the boundary
        output values. The inverse maps those boundaries to the empirical
        minimum and maximum.

        Parameters
        ----------
        num_quantiles : int, optional
            Number of equally-spaced quantile levels (default 100).
        assume_bounds : bool, optional
            If True assumes that min and max are domain bounds.

        Returns
        -------
        Updated domain.
        """
        yp = np.linspace(0.0, 1.0, num_quantiles + 1)
        xp = np.quantile(vals, yp)
        if assume_bounds:
            domain = domain.assume.ge(xp[0]).assume.le(xp[-1])
        return domain.transform.generalized_piecewise_affine(xp, yp)

    @I
    def quantile_discretize(domain, vals, num_bins,
        min_prob_left_tail=None,
        min_prob_right_tail=None,
        left_tail_distribution='auto',
        right_tail_distribution='auto',
        assume_bounds=False,    
        ):
        """Discretize into quantile bins using ``transform.digitize``.

        Interior bin edges are placed at empirical quantiles of the data.
        Tail distributions can be provided for unbounded domains (passed
        through to ``transform.digitize``).

        Parameters
        ----------
        num_bins : int
            Number of bins.
        min_prob_left_tail, min_prob_right_tail : float or None, optional
            Minimum probability assigned to each fitted tail. ``None`` uses
            zero for a uniform tail and 0.05 for a normal or generalized-Pareto
            tail. Zero is invalid for a non-uniform tail on an unbounded side.
        left_tail_distribution, right_tail_distribution : str, optional
            One of ``'auto'``, ``'uniform'``, ``'normal'``, or
            ``'genpareto'``. ``'auto'`` selects uniform for a bounded side and
            normal for an unbounded side.
        assume_bounds : bool, optional
            If true, tighten an unbounded input side when its fitted tail has
            finite support. Otherwise the input domain is left unchanged.
        data : array-like, optional
            Data from which to compute quantile bin edges. If omitted, uses
            tracked data.

        Returns
        -------
        Updated domain with discretized domain and data.
        """
        _validate_num_bins(num_bins)
        bounded_left = domain.is_bounded_left
        bounded_right = domain.is_bounded_right
        min_prob_left_tail, left_tail_distribution = _check_tail(
            bounded_left, left_tail_distribution, min_prob_left_tail
        )
        min_prob_right_tail, right_tail_distribution = _check_tail(
            bounded_right, right_tail_distribution, min_prob_right_tail
        )

        bins, piece_bins = quantile_discretize(vals, num_bins,
            min_prob_left_tail=min_prob_left_tail,
            min_prob_right_tail=min_prob_right_tail,
        )
        num_intervals = bins.size - 1
        num_left_tail_bins = piece_bins[0]
        num_right_tail_bins = num_intervals - piece_bins[-1]

        # A zero-mass uniform tail still supplies the reconstruction law for
        # the existing outermost quantile bin. This does not change the
        # requested minimum probability or the fitted quantile edges.
        if (
            not bounded_left
            and left_tail_distribution == 'uniform'
            and min_prob_left_tail == 0
        ):
            num_left_tail_bins = 1
        if (
            not bounded_right
            and right_tail_distribution == 'uniform'
            and min_prob_right_tail == 0
        ):
            num_right_tail_bins = 1

        tail_piece_bins = np.array([
            num_left_tail_bins,
            num_intervals - num_right_tail_bins,
        ])
        left_tail = None
        right_tail = None
        if not bounded_left or not bounded_right:
            # bins has shape [data_min, *interior, data_max].
            piece = np.digitize(
                np.digitize(vals, bins[1:-1]),
                tail_piece_bins,
            )

            if not bounded_left:
                left_vals = vals[piece == 0]
                left_threshold = bins[tail_piece_bins[0]]
                if left_tail_distribution == 'uniform':
                    left_tail = infer_uniform_tail(
                        left_vals,
                        left_threshold,
                        left=True,
                        force_unbounded=not assume_bounds,
                    )
                elif left_tail_distribution == 'normal':
                    left_tail = infer_normal_tail(
                        left_vals,
                        left_threshold,
                        left=True,
                    )
                else:
                    left_tail = infer_genpareto_tail(
                        left_vals,
                        left_threshold,
                        force_unbounded=not assume_bounds,
                        left=True,
                    )

                if assume_bounds:
                    fitted_lb = left_tail.support()[0]
                    if np.isfinite(fitted_lb) and fitted_lb > domain.lb:
                        domain = domain.assume.ge(fitted_lb)

            if not bounded_right:
                right_vals = vals[piece == 2]
                right_threshold = bins[tail_piece_bins[-1]]
                if right_tail_distribution == 'uniform':
                    right_tail = infer_uniform_tail(
                        right_vals,
                        right_threshold,
                        left=False,
                        force_unbounded=not assume_bounds,
                    )
                elif right_tail_distribution == 'normal':
                    right_tail = infer_normal_tail(
                        right_vals,
                        right_threshold,
                        left=False,
                    )
                else:
                    right_tail = infer_genpareto_tail(
                        right_vals,
                        right_threshold,
                        force_unbounded=not assume_bounds,
                    )

                if assume_bounds:
                    fitted_ub = right_tail.support()[1]
                    if np.isfinite(fitted_ub) and fitted_ub < domain.ub:
                        domain = domain.assume.le(fitted_ub)

        # Pass only interior edges; digitize prepends and appends domain bounds.
        return domain.transform.digitize(
            bins[1:-1],
            left_tail, right_tail,
            num_left_tail_bins,
            num_right_tail_bins,
        )


@register_infer(domain.FixedPointColumn)
class FixedPointInferences(FloatingPointInferences):

    @I
    def fixed_point(
        domain,
        vals,
        min_digits=0,
        max_digits=5,
        rtol=1e-3,
        atol=1e-8,
    ):
        """Keep the already-declared fixed-point precision unchanged."""
        return domain

# ---------------------------------------------------------------------------
# Standalone helper functions (used internally and exposed for advanced use)
# ---------------------------------------------------------------------------

def infer_fixed_point(vals, min_digits=0, max_digits=5, rtol=1e-3, atol=1e-8):
    digits = np.arange(min_digits, max_digits + 1)
    d = 10. ** -digits[:, None]
    r = vals % d
    r = np.minimum(r, d - r)

    m = np.max(r, axis=-1) <= atol + rtol * d[:, 0]
    dlevel = np.searchsorted(m, True)
    if dlevel > max_digits - min_digits:
        return None

    return digits[dlevel]


def infer_genpareto_tail(
    tail_vals,
    threshold,
    force_unbounded=True,
    left=False,
):
    if left:
        reflected = infer_genpareto_tail(
            -np.asarray(tail_vals),
            -threshold,
            force_unbounded=force_unbounded,
        )
        return ReflectedDistribution(reflected)

    _validate_parametric_tail_sample(tail_vals, 'genpareto')
    c, loc, scale = genpareto.fit(tail_vals, floc=threshold)

    if c < 0 and force_unbounded:
        c, loc, scale = genpareto.fit(tail_vals, f0=0, floc=threshold)

    return genpareto(c, loc=loc, scale=scale)


def infer_uniform_tail(
    tail_vals,
    threshold,
    left=False,
    force_unbounded=False,
):
    if left:
        lb, ub = np.min(tail_vals), threshold
    else:
        lb, ub = threshold, np.max(tail_vals)
    if not ub > lb:
        raise ValueError('A uniform tail requires distinct empirical endpoints.')
    tail = uniform(loc=lb, scale=ub - lb)
    if force_unbounded:
        declared_domain = domain.FloatingPointColumn(
            -np.inf if left else threshold,
            threshold if left else np.inf,
        )
        return ScipyContinuous(tail, domain=declared_domain)
    return tail


def infer_normal_tail(tail_vals, threshold, left=False):
    _validate_parametric_tail_sample(tail_vals, 'normal')
    if left:
        tail_vals, threshold = -tail_vals, -threshold

    def nll(params):
        mu, s = params
        return truncnorm.nnlf(((threshold - mu) / s, np.inf, mu, s), tail_vals)

    mu, s = minimize(nll, (np.mean(tail_vals), np.std(tail_vals))).x

    if left:
        return truncnorm(-np.inf, (mu - threshold) / s, -mu, s)
    return truncnorm((threshold - mu) / s, np.inf, mu, s)


def _fit_clip_tail(
    tail_vals,
    threshold,
    distribution,
    *,
    left,
    domain_bound,
    force_unbounded,
):
    if distribution == 'uniform':
        if np.isfinite(domain_bound):
            if left:
                return uniform(
                    loc=domain_bound,
                    scale=threshold - domain_bound,
                )
            return uniform(
                loc=threshold,
                scale=domain_bound - threshold,
            )
        return infer_uniform_tail(
            tail_vals,
            threshold,
            left=left,
            force_unbounded=force_unbounded,
        )
    if distribution == 'normal':
        return infer_normal_tail(tail_vals, threshold, left=left)
    return infer_genpareto_tail(
        tail_vals,
        threshold,
        force_unbounded=force_unbounded,
        left=left,
    )


def quantile_discretize(col, 
    num_bins, 
    min_prob_left_tail=0,
    min_prob_right_tail=0,
    ):
    _validate_num_bins(num_bins)
    discrete = np.issubdtype(col.dtype, np.integer)
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(col, quantiles, method='linear')
    
    if discrete:
        bins = np.ceil(bins).astype(np.int_)

    num_left_tail_bins = np.ceil(min_prob_left_tail*num_bins).astype(int)
    num_right_tail_bins = np.ceil(min_prob_right_tail*num_bins).astype(int)

    ubins, index = np.unique(bins[1:-1], return_inverse=True)

    start_bin = index[num_left_tail_bins]
    stop_bin = index[-num_right_tail_bins-1] + 2

    #bin_mask = np.ones(bins.size, dtype=np.bool_)
    #bin_mask[1:-1] = np.diff(bins[:-1]) > 0
    #bins = bins[bin_mask]
    bins = np.r_[bins[0], ubins, bins[-1]]
    if discrete:  # include max
        bins[-1] += 1

    return bins, np.array([start_bin, stop_bin])


def _check_tail(bounded, distribution, prob):
    distribution = _check_tail_distribution(bounded, distribution)

    if prob is None:
        prob = 0.0 if distribution == 'uniform' else 0.05
    _validate_tail_probability(prob)
    if not bounded and distribution != 'uniform' and prob == 0:
        raise ValueError(
            'min_prob must be greater than zero for a non-uniform tail on '
            'an unbounded side.'
        )
    return float(prob), distribution


def _check_tail_distribution(bounded, distribution):
    if not isinstance(distribution, str):
        raise TypeError('tail distribution must be specified by name.')
    distribution = distribution.lower()
    if distribution == 'auto':
        distribution = 'uniform' if bounded else 'normal'

    allowed = ('uniform',) if bounded else ('uniform', 'normal', 'genpareto')
    if distribution not in allowed:
        raise ValueError(
            f'{distribution!r} is not a valid tail distribution for this support.'
        )
    return distribution


def _validate_tail_probability(prob):
    if not np.isscalar(prob) or not np.isfinite(prob) or not 0 <= prob < 1:
        raise ValueError('tail probability must be finite and lie in [0, 1).')


def _validate_num_bins(num_bins):
    if (
        isinstance(num_bins, (bool, np.bool_))
        or not isinstance(num_bins, (int, np.integer))
        or num_bins < 2
    ):
        raise ValueError('num_bins must be an integer greater than or equal to 2.')


def _validate_parametric_tail_sample(tail_vals, distribution):
    tail_vals = np.asarray(tail_vals)
    if tail_vals.size < MIN_PARAMETRIC_TAIL_SAMPLES:
        raise ValueError(
            f'{distribution} tail fitting requires at least '
            f'{MIN_PARAMETRIC_TAIL_SAMPLES} observations; got {tail_vals.size}.'
        )
    if np.unique(tail_vals).size < 2:
        raise ValueError(
            f'{distribution} tail fitting requires at least two distinct '
            'observations.'
        )
