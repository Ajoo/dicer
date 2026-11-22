from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from dicer import domain
from dicer.stats import Distribution, Delta


@dataclass
class Transformation:
    input_domain: domain.Column
    inverse: Transformation | None = field(default=None, init=False, repr=False)
    is_deterministic: ClassVar[bool]
    is_lossy: ClassVar[bool]

    def __call__(self, x, random_state=None):
        return self.apply(x, random_state=random_state)

    def apply(self, x, random_state=None):
        raise NotImplementedError

    def validate(self, x) -> None:
        """Validate values against this transformation's input domain."""
        self.input_domain.validate(x)

    def restrict_output_domain(self, output_domain):
        """Return a rebuilt transformation with a narrower output domain."""
        raise NotImplementedError(
            f'{type(self).__name__} cannot propagate output domain restrictions.'
        )

    def restrict_input_domain(self, input_domain):
        """Rebuild after an upstream domain restriction."""
        raise NotImplementedError(
            f'{type(self).__name__} cannot propagate input domain restrictions.'
        )

    def conditional(self, x):
        """Returns conditional distribution p(Y|X=x).
        """
        if not self.is_deterministic:
            # Requires specific implementation by subclass
            raise NotImplementedError

        # default case for deterministic
        y = self.apply(x)
        output_domain = self.output_domain
        if isinstance(output_domain, domain.VectorColumn):
            raise NotImplementedError(
                'Distributions over grouped columns are not implemented.'
            )
        return Delta(output_domain, y)

    def pushforward_distribution(self, distribution: Distribution) -> Distribution:
        raise NotImplementedError(
            f'{type(self).__name__} does not implement distribution pushforward.'
        )

    def pullback_distribution(
        self,
        distribution: Distribution,
        **options,
    ) -> Distribution:
        raise NotImplementedError(
            f'{type(self).__name__} does not implement distribution pullback.'
        )

    def resolve(self, **options) -> Transformation:
        """Resolve configuration options to a concrete transformation."""
        return self

    @property
    def output_domain(self) -> domain.Column:
        raise NotImplementedError

    def with_inverse(self, inverse: Transformation):
        self.inverse = inverse
        inverse.inverse = self
        return self

    def __mul__(self, other):
        if isinstance(other, SeriesTransformation):
            return NotImplemented  # should call other.__rmul__

        assert other.output_domain == self.input_domain
        return SeriesTransformation(other.input_domain, [other, self])


@dataclass
class IdentityTransformation(Transformation):
    """Identity mapping used when fitting a tracker with no transformations."""

    is_deterministic: ClassVar[bool] = True
    is_lossy: ClassVar[bool] = False

    def __post_init__(self):
        self.inverse = self

    def apply(self, x, random_state=None):
        return x

    @property
    def output_domain(self):
        return self.input_domain

    def pushforward_distribution(self, distribution):
        return distribution

    def pullback_distribution(self, distribution, **options):
        return distribution


class OrderedMixin:

    def pullback_lb(self, y):
        if self.inverse is None:
            raise NotImplementedError

        if self.inverse.is_deterministic:
            return self.inverse.apply(y)
        return self.inverse.conditional(y).support()[0]

    def pullback_ub(self, y):
        if self.inverse is None:
            raise NotImplementedError

        if self.inverse.is_deterministic:
            return self.inverse.apply(y)
        return self.inverse.conditional(y).support()[1]


class PushforwardMixin:

    def pushforward_distribution(self, distribution: Distribution) -> Distribution:
        """Pushes-forward through the transformation a distribution over the input domain.

        Parameters
        ----------
        distribution : Distribution
            distribution over the input domain.

        Returns
        -------
        Distribution
            distribution over the output domain.
        """
        return type(self)._Pushforward(distribution, self)

    def pullback_distribution(
        self,
        distribution: Distribution,
        **options,
    ) -> Distribution:
        """Pulls-back through the transformation a distribution over the output domain.

        Parameters
        ----------
        distribution : Distribution
            distribution over the output domain.

        Returns
        -------
        Distribution
            distribution over the input domain.
        """
        if self.inverse is None:
            raise NotImplementedError(
                "pullback_distribution requires an inverse transformation."
            )
        inverse = self.inverse.resolve(**options)
        return inverse.pushforward_distribution(distribution)


class PullbackMixin:

    def pushforward_distribution(self, distribution: Distribution) -> Distribution:
        """Pushes-forward through the transformation a distribution over the input domain.

        Parameters
        ----------
        distribution : Distribution
            distribution over the input domain.

        Returns
        -------
        Distribution
            distribution over the output domain.
        """
        if self.inverse is None:
            raise NotImplementedError(
                "pullback_distribution requires an inverse transformation."
            )
        return self.inverse.pullback_distribution(distribution)

    def pullback_distribution(
        self,
        distribution: Distribution,
        **options,
    ) -> Distribution:
        """Pulls-back through the transformation a distribution over the output domain.

        Parameters
        ----------
        distribution : Distribution
            distribution over the output domain.

        Returns
        -------
        Distribution
            distribution over the input domain.
        """
        return type(self)._Pullback(distribution, self)


@dataclass
class SeriesTransformation(Transformation):
    transformations: list[Transformation]

    def restrict_output_domain(self, output_domain):
        steps = []
        for transformation in reversed(self.transformations):
            step = transformation.restrict_output_domain(output_domain)
            steps.append(step)
            output_domain = step.input_domain
        steps.reverse()
        return SeriesTransformation(steps[0].input_domain, steps)

    def restrict_input_domain(self, input_domain):
        steps = []
        for transformation in self.transformations:
            step = transformation.restrict_input_domain(input_domain)
            steps.append(step)
            input_domain = step.output_domain
        return SeriesTransformation(steps[0].input_domain, steps)

    @property
    def is_deterministic(self):
        return all(t.is_deterministic for t in self.transformations)

    @property
    def is_lossy(self):
        return any(t.is_lossy for t in self.transformations)

    def apply(self, x, random_state=None):
        prng = np.random.default_rng(random_state)
        for t in self.transformations:
            x = t.apply(x, random_state=prng)

        return x

    def pushforward_distribution(self, distribution):
        for t in self.transformations:
            distribution = t.pushforward_distribution(distribution)
        return distribution

    def pullback_distribution(self, distribution, **options):
        for t in reversed(self.transformations):
            distribution = t.pullback_distribution(distribution, **options)
        return distribution

    @property
    def output_domain(self):
        return self.transformations[-1].output_domain

    def __mul__(self, other):
        assert other.output_domain == self.input_domain

        if isinstance(other, SeriesTransformation):
            transformations = [*other.transformations, *self.transformations]
        else:
            transformations = [other, *self.transformations]
        return SeriesTransformation(other.input_domain, transformations)

    def __rmul__(self, other):
        assert self.output_domain == other.input_domain
        transformations = [*self.transformations, other]
        return SeriesTransformation(self.input_domain, transformations)


@dataclass(frozen=True)
class OptionalTransformationStep:
    """A series step selected by one named configuration option."""

    transformation: Transformation
    option: str
    default: bool


@dataclass
class ConfigurableSeriesTransformation(Transformation):
    """A series that resolves optional steps before it is evaluated.

    The default resolution behaves like an ordinary transformation and owns a
    single output domain. Alternative resolutions may have different output
    domains, which are available through :meth:`output_domain_for`.
    """

    transformations: list[Transformation | OptionalTransformationStep]

    def __post_init__(self):
        self.resolve()

    def _resolved_options(self, options):
        defaults = {}
        for step in self.transformations:
            if not isinstance(step, OptionalTransformationStep):
                continue
            if (
                step.option in defaults
                and defaults[step.option] != step.default
            ):
                raise ValueError(
                    f'Optional steps disagree on the default for '
                    f'{step.option!r}.'
                )
            defaults[step.option] = step.default

        allowed = set(defaults)

        unknown = set(options) - allowed
        if unknown:
            names = ', '.join(sorted(unknown))
            raise TypeError(f'Unknown transformation option(s): {names}.')

        resolved = {**defaults, **options}
        for name in allowed:
            value = resolved[name]
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f'{name} must be a boolean; got {value!r}.'
                )
            resolved[name] = bool(value)
        return resolved

    def resolve(self, **options):
        options = self._resolved_options(options)
        transformations = []
        for step in self.transformations:
            if isinstance(step, OptionalTransformationStep):
                if not options[step.option]:
                    continue
                step = step.transformation
            transformations.append(step)

        if not transformations:
            return IdentityTransformation(self.input_domain)

        current_domain = self.input_domain
        for transformation in transformations:
            if transformation.input_domain != current_domain:
                raise ValueError(
                    'Configured transformation domains do not compose: '
                    f'{current_domain!r} != '
                    f'{transformation.input_domain!r}.'
                )
            current_domain = transformation.output_domain

        if len(transformations) == 1:
            return transformations[0]
        return SeriesTransformation(self.input_domain, transformations)

    def __call__(self, x, random_state=None, **options):
        return self.apply(x, random_state=random_state, **options)

    def apply(self, x, random_state=None, **options):
        return self.resolve(**options).apply(x, random_state=random_state)

    def pushforward_distribution(self, distribution, **options):
        return self.resolve(**options).pushforward_distribution(distribution)

    def pullback_distribution(self, distribution, **options):
        return self.resolve(**options).pullback_distribution(
            distribution,
            **options,
        )

    def output_domain_for(self, **options):
        return self.resolve(**options).output_domain

    @property
    def output_domain(self):
        return self.output_domain_for()

    @property
    def is_deterministic(self):
        return self.resolve().is_deterministic

    @property
    def is_lossy(self):
        return self.resolve().is_lossy
