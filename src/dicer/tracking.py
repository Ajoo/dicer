from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import reduce, wraps
from operator import mul
from typing import TYPE_CHECKING, NamedTuple, Any

import numpy as np

from dicer.domain import Column, Frame

if TYPE_CHECKING:
    from dicer.transform._base import Transformation
    from dicer.transform.frame import FrameTransformation


class FitResult(NamedTuple):
    data: Any | None
    transformation: Transformation | FrameTransformation


@dataclass(frozen=True)
class ObservedColumn:
    """A name-free domain and aligned values supplied to an inference."""

    domain: Column
    values: Any
    
# -----------------------------------------------------------------
#                           DESCRIPTORS
# -----------------------------------------------------------------

class Descriptor:
    def __init__(self, domain_or_tracker):
        if isinstance(domain_or_tracker, Tracker):
            self.domain = domain_or_tracker.unpack('domain')
            self._tracker = domain_or_tracker
        else:
            self.domain = domain_or_tracker
            self._tracker = None

    @property
    def _domain_or_tracker(self):
        return self._tracker or self.domain

    def _finalize_assume(self, domain):
        if self._tracker:
            data = self._tracker._tracked_data
            if data is not None:
                domain.validate(data)
            return self._tracker._with_domain(domain)
        return domain
    
    def _finalize_transform(self, t):
        if self._tracker:
            return self._tracker._apply_transform(t)
        return t


class ColumnAssumptions(Descriptor):
    pass


class ColumnInferences(Descriptor):
    def _get_data(self, data=None):
        if data is not None:
            return data
        if self._tracker is not None and self._tracker._tracked_data is not None:
            return np.asarray(self._tracker._tracked_data)
        raise ValueError(
            "No data provided. Pass `data=` or use a tracker with tracked data."
        )


class ColumnTransformations(Descriptor):
    def _inverse_projection(self, other_domain):
        assert self.domain == other_domain
        return None


# ----------  Decorators for methods of Descriptor Classes ------------


def A(assumption):

   @wraps(assumption)
   def _assumption(self, *args, **kwargs):
       new_domain = assumption(self.domain, *args, **kwargs)
       return self._finalize_assume(new_domain)

   return _assumption


def T(transformation_method):
    @wraps(transformation_method)
    def transformation(self: ColumnTransformations, *args, **kwargs):
        t = transformation_method(self.domain, *args, **kwargs)
        if isinstance(t, tuple):
            assert len(t) == 2
            t, t_inv = t
            projection = self._inverse_projection(t_inv.output_domain)
            if projection is not None:
                from dicer.transform._base import ConfigurableSeriesTransformation

                t_inv.inverse = t
                t_inv = ConfigurableSeriesTransformation(
                    t_inv.input_domain,
                    [t_inv, projection],
                )
            t = t.with_inverse(t_inv)
        return self._finalize_transform(t)

    return transformation


def I(infer_method):
    @wraps(infer_method)
    def inference(self: ColumnInferences, *args, data=None, **kwargs):
        data = self._get_data(data)
        return infer_method(self._domain_or_tracker, data, *args, **kwargs)

    return inference

# ----------  Register new Descriptor Classes ------------


def register_transform(domain_cls):
    if issubclass(domain_cls, Frame):
        Tracker = FrameTracker
    elif issubclass(domain_cls, Column):
        Tracker = ColumnTracker
    else:
        raise TypeError(
            "domain_class should be a subclass of Column or Frame."
            f"Got {domain_cls}."
            )
    def decorator(descriptor_cls):
        def transform(self):
            return descriptor_cls(Tracker(self))
        domain_cls._Transformations = descriptor_cls
        domain_cls.transform = property(transform)
        return descriptor_cls

    return decorator


def register_assume(domain_cls):
    def decorator(descriptor_cls):
        domain_cls._Assumptions = descriptor_cls
        domain_cls.assume = property(descriptor_cls)
        return descriptor_cls

    return decorator


def register_infer(domain_cls):
    if issubclass(domain_cls, Frame):
        Tracker = FrameTracker
    elif issubclass(domain_cls, Column):
        Tracker = ColumnTracker
    else:
        raise TypeError(
            "domain_cls should be a subclass of Column or Frame. "
            f"Got {domain_cls}."
        )
    def decorator(descriptor_cls):
        def infer(self):
            return descriptor_cls(Tracker(self))
        domain_cls._Inferences = descriptor_cls
        domain_cls.infer = property(infer)
        return descriptor_cls
    return decorator

# -----------------------------------------------------------------
#                           TRACKERS
# -----------------------------------------------------------------

# TODO: name mangle attribute names to avoid collisions with _domain attrs
# Tracker is a transparent proxy to a domain
class Tracker:

    def __init__(self, domain, data=None, transformations=None, random_state=None):
        if transformations is None:
            transformations = self._init_transformations()

        self._tracked_domain = domain
        self._tracked_transformations = transformations
        self._tracked_data = data
        self._tracked_prng = np.random.default_rng(random_state)

    def __getattr__(self, name):
        return getattr(self._tracked_domain, name)
    
    @property
    def transformations(self): # to be implemented by subclass
        raise NotImplementedError
    
    def unpack(self, name=None):
        if name:
            if name=='transformations':
                return self.transformations
            return object.__getattribute__(self, f'_tracked_{name}')
        if self._tracked_data is None:
            return self._tracked_domain, self.transformations
        
        return self._tracked_domain, self.transformations, self._tracked_data  
    
    @property
    def assume(self):
        return self._tracked_domain._Assumptions(self)
    
    @property
    def transform(self):
        return self._tracked_domain._Transformations(self)

    @property
    def infer(self):
        return self._tracked_domain._Inferences(self)
    
    def _with_domain(self, domain):
        self = self.copy()
        self._tracked_domain = domain
        return self
    
    def copy(self):
        return type(self)(
            self._tracked_domain,
            data=self._tracked_data, # should make a copy()?
            transformations=copy(self._tracked_transformations),
            random_state=deepcopy(self._tracked_prng),
        )
    
    def __repr__(self):
        return f'Tracker({repr(self._tracked_domain)})'
    
    @staticmethod
    def from_dtype(dtype):
        return ColumnTracker(Column.from_dtype(dtype))

    @staticmethod
    def from_series(series):
        try:
            vals = series.values
        except AttributeError:
            vals = np.asarray(series)

        domain = Column.from_dtype(vals.dtype)
        return ColumnTracker(domain, vals)
    
    @staticmethod
    def from_dtypes(dtypes):
        return FrameTracker(Frame.from_dtypes(dtypes))
    
    @staticmethod
    def from_dataframe(df):
        domain = Frame.from_dtypes(df.dtypes)
        return FrameTracker(domain, df.copy(deep=True))
    

def _merge_transformations(transformations: list):
    if not transformations:
        return None
    return reduce(mul, reversed(transformations))


class ColumnTracker(Tracker):
    _init_transformations = staticmethod(list)

    def _with_domain(self, domain):
        if domain == self._tracked_domain:
            return self.copy()
        steps = []
        restricted = domain
        for transformation in reversed(self._tracked_transformations):
            step = transformation.restrict_output_domain(restricted)
            steps.append(step)
            restricted = step.input_domain
        tracker = self.copy()
        tracker._tracked_domain = domain
        tracker._tracked_transformations = list(reversed(steps))
        return tracker

    @property
    def transformations(self):
        return _merge_transformations(self._tracked_transformations)
    
    def _apply_transform(self, t):
        self = self.copy()
        self._tracked_transformations.append(t)
        self._tracked_domain = t.output_domain
        if self._tracked_data is not None:
            self._tracked_data = t.apply(self._tracked_data, self._tracked_prng)
        return self

    def finish(self):
        """Return fitted column data and transformation without consuming this tracker."""
        from dicer.transform._base import IdentityTransformation

        transformation = self.transformations
        if transformation is None:
            transformation = IdentityTransformation(self._tracked_domain)
        return FitResult(self._tracked_data, transformation)


class FrameTracker(Tracker):
    def __init__(
        self,
        *args,
        input_domain=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tracked_input_domain = (
            self._tracked_domain if input_domain is None else input_domain
        )
        self._consumed = False

    def _ensure_active(self):
        if self._consumed:
            raise RuntimeError(
                'FrameTracker has already been consumed by an operation. '
                'Continue with the tracker returned by that operation; branching '
                'a FrameTracker is not supported.'
            )

    def __getattr__(self, name):
        self._ensure_active()
        return super().__getattr__(name)

    def unpack(self, name=None):
        self._ensure_active()
        return super().unpack(name)

    @staticmethod
    def _init_transformations():
        return []

    def copy(self):
        self._ensure_active()
        raise RuntimeError(
            'FrameTracker cannot be copied because branching is not supported. '
            'Continue with the tracker returned by each operation.'
        )

    def _advance(self, domain=None):
        """Transfer owned state to a new tracker and consume this tracker."""
        self._ensure_active()
        next_tracker = type(self)(
            self._tracked_domain if domain is None else domain,
            data=self._tracked_data,
            transformations=self._tracked_transformations,
            random_state=self._tracked_prng,
            input_domain=self._tracked_input_domain,
        )
        self._consumed = True
        return next_tracker

    def _with_domain(self, domain):
        if not self._tracked_transformations:
            next_tracker = self._advance(domain)
            next_tracker._tracked_input_domain = domain
            return next_tracker

        pending = {
            name: domain[name] for name in domain.column_names
            if domain[name] != self._tracked_domain[name]
        }
        operations = list(self._tracked_transformations)
        # Walk field versions backward; a derived field leaves its source alive.
        for index in range(len(operations) - 1, -1, -1):
            operation = operations[index]
            output = pending.pop(operation.target, None)
            if output is None:
                continue
            transformation = operation.transformation
            if output != transformation.output_domain:
                transformation = transformation.restrict_output_domain(output)
                operations[index] = replace(operation, transformation=transformation)
            source = transformation.input_domain
            if operation.source in pending:
                other = pending[operation.source]
                source = source.with_new_bounds(
                    lower_bound=max(source.lb, other.lb),
                    upper_bound=min(source.ub, other.ub),
                )
            pending[operation.source] = source

        input_domain = self._tracked_input_domain.with_domains(pending)
        current = input_domain
        # Reconcile all consumers, including siblings of a restricted field.
        for index, operation in enumerate(operations):
            transformation = operation.transformation
            if transformation.input_domain != current[operation.source]:
                transformation = transformation.restrict_input_domain(current[operation.source])
                operations[index] = replace(operation, transformation=transformation)
            current = current.with_field(
                operation.source, operation.target, transformation.output_domain,
                operation.output_components,
            )
        if self._tracked_data is not None:
            current.validate(self._tracked_data)
        next_tracker = self._advance(current)
        next_tracker._tracked_input_domain = input_domain
        next_tracker._tracked_transformations = operations
        return next_tracker

    @property
    def assume(self):
        self._ensure_active()
        return self._tracked_domain._Assumptions(self)

    @property
    def infer(self):
        self._ensure_active()
        return self._tracked_domain._Inferences(self)
    
    @property
    def transformations(self):
        self._ensure_active()
        return tuple(self._tracked_transformations)
    
    @property
    def transform(self):
        self._ensure_active()
        return self._tracked_domain._Transformations(self)
    
    def _apply_transform(self, operations):
        from dicer.transform.frame import assign_field

        self._ensure_active()
        operations = list(operations)
        current_domain = self._tracked_domain
        staged = {}

        for operation in operations:
            if current_domain[operation.source] != operation.transformation.input_domain:
                raise ValueError(
                    f'Transformation input domain for {operation.source!r} does '
                    'not match the current frame domain.'
                )
            if self._tracked_data is not None:
                values = staged.get(operation.source)
                if values is None:
                    values = current_domain.extract(
                        self._tracked_data,
                        operation.source,
                    )
                transformed = operation.transformation.apply(
                    values,
                    self._tracked_prng,
                )
            next_domain = current_domain.with_field(
                operation.source,
                operation.target,
                operation.transformation.output_domain,
                operation.output_components,
            )
            if self._tracked_data is not None:
                staged[operation.target] = transformed
            current_domain = next_domain

        output = self._tracked_data
        if output is not None:
            owned = self._tracked_data
            passthrough = {
                component: owned[component].array
                for name in current_domain.column_names
                if name not in staged
                for component in current_domain.components(name)
            }
            for column in list(owned.columns):
                del owned[column]
            for name in current_domain.column_names:
                components = current_domain.components(name)
                if name in staged:
                    assign_field(owned, components, staged[name])
                else:
                    for component in components:
                        owned[component] = passthrough[component]
            output = owned

        next_tracker = self._advance(current_domain)
        next_tracker._tracked_data = output
        next_tracker._tracked_transformations.extend(operations)
        return next_tracker

    def _refine_column_domain(self, name, column_domain):
        """Refine a logical column without changing its representation."""
        values = (
            None
            if self._tracked_data is None
            else self._tracked_domain.extract(self._tracked_data, name)
        )
        if values is not None:
            column_domain.validate(values)
        updated = self._tracked_domain.with_domains({name: column_domain})
        return self._with_domain(updated)

    def _apply_column_inference(self, source, target, result, components=None):
        from dicer.transform.frame import FrameOperation, normalize_components

        transformations = list(result._tracked_transformations)
        if not transformations:
            if target != source:
                raise ValueError(
                    'An inference without a transformation cannot derive a new column.'
                )
            return self._refine_column_domain(source, result.unpack('domain'))

        tracker = self
        first_input = transformations[0].input_domain
        if tracker._tracked_domain[source] != first_input:
            tracker = tracker._refine_column_domain(source, first_input)

        operations = []
        current_domain = tracker._tracked_domain
        for index, transformation in enumerate(transformations):
            operation_source = source if index == 0 else target
            operation_target = target
            is_last = index == len(transformations) - 1
            output_components = normalize_components(
                operation_target,
                transformation.output_domain,
                components if is_last else None,
                input_components=current_domain.components(operation_source),
                replacing=operation_source == operation_target,
            )
            operation = FrameOperation(
                operation_source,
                operation_target,
                transformation,
                output_components,
            )
            operations.append(operation)
            current_domain = current_domain.with_field(
                operation.source,
                operation.target,
                transformation.output_domain,
                output_components,
            )
        return tracker._apply_transform(operations)

    def finish(self):
        """Consume this tracker and return fitted data plus replayable transform."""
        from dicer.transform.frame import FrameTransformation

        self._ensure_active()
        transformation = FrameTransformation(
            input_domain=self._tracked_input_domain,
            output_domain=self._tracked_domain,
            transformations=tuple(self._tracked_transformations),
        )
        result = FitResult(self._tracked_data, transformation)
        self._consumed = True
        return result
    
    def __repr__(self):
        return f'Tracker: \n{repr(self._tracked_domain)}'
    
# -----------------------------------------------------------------
#                     DESCRIPTORS FOR FRAMES
# -----------------------------------------------------------------


class FrameDescriptor(Descriptor):
    def __init__(
        self,
        domain_or_tracker,
        filters=None,
        *,
        into=None,
        components=None,
        using=None,
    ):
        super().__init__(domain_or_tracker)
        self._filters = filters or []
        self._into = into
        self._components = components
        self._using = {} if using is None else dict(using)

    def _clone(self, *, filters=None, into=None, components=None, using=None):
        domain_or_tracker = self._tracker or self.domain
        return type(self)(
            domain_or_tracker,
            self._filters if filters is None else filters,
            into=self._into if into is None else into,
            components=self._components if components is None else components,
            using=self._using if using is None else using,
        )

    def _items(self):
        """Yield ``(name, Column)`` pairs for filter evaluation."""
        raise NotImplementedError

    def _matching(self):
        """Return list of ``(name, Column)`` pairs satisfying all filters."""
        items = list(self._items())
        if not self._filters:
            return items
        matching = [
            (name, col)
            for name, col in items
            if all(f.matches(name, col) for f in self._filters)
        ]
        if not matching:
            name_filters = [
                filter_
                for filter_ in self._filters
                if isinstance(filter_, NameFilter)
            ]
            named_matching = [
                (name, col)
                for name, col in items
                if all(
                    filter_.matches(name, col)
                    for filter_ in name_filters
                )
            ]
            if name_filters and not named_matching:
                selections = [
                    name
                    for filter_ in name_filters
                    for name in filter_.names
                ]
                raise KeyError(
                    'No logical columns match the explicit name or pattern '
                    f'selection {selections!r}.'
                )
        return matching

    def __getitem__(self, name) -> FrameDescriptor:
        """Return a new descriptor filtered to *name*.

        If *name* is a string containing ``*``, ``?``, or ``[`` it is treated
        as a shell-style wildcard pattern; otherwise it is an exact match (or
        a list/set of exact names).
        """
        f = NameFilter(name)
        return self._clone(filters=[*self._filters, f])

    def __call__(self, name=None, *,
                 pattern=None,
                 continuous=None, ordered=None, bounded=None,
                 bounded_left=None, bounded_right=None, finite=None,
                 min_cardinality=None, max_cardinality=None,
                 min_decimals=None, max_decimals=None,
                 into=None, components=None, using=None) -> FrameDescriptor:
        """Return a new descriptor with additional filter(s).

        Parameters
        ----------
        name : str or list of str, optional
            Exact column name(s). A string containing wildcards is treated as
            a shell-style wildcard pattern.
        pattern : str, optional
            Explicit shell-style wildcard pattern (always treated as a pattern
            regardless of content).
        continuous : bool, optional
            If given, select only columns where ``is_continuous`` matches.
        ordered : bool, optional
            If given, select only columns where ``is_ordered`` matches.
        bounded : bool, optional
            If given, select only columns where ``is_bounded`` matches.
        bounded_left : bool, optional
            If given, select only columns where ``is_bounded_left`` matches.
        bounded_right : bool, optional
            If given, select only columns where ``is_bounded_right`` matches.
        finite : bool, optional
            If given, select only columns where ``is_finite`` matches.
        min_cardinality : int, optional
            Select only columns with ``cardinality >= min_cardinality`` (and
            positive, finite cardinality).
        max_cardinality : int, optional
            Select only columns with ``cardinality <= max_cardinality`` (and
            positive, finite cardinality).
        min_decimals : int, optional
            Select only columns with ``decimals >= min_decimals``.
        max_decimals : int, optional
            Select only columns with ``decimals <= max_decimals``.
        into : hashable, optional
            Logical destination name. Transformations retain the source when
            this differs from the selected name.
        components : sequence of hashable, optional
            Physical names for a structured destination. Defaults to
            ``<into>.0``, ``<into>.1``, and so on.
        using : mapping, optional
            Named fitting-only observations supplied to an inference.
        """
        new_filters = []
        if name is not None:
            new_filters.append(NameFilter(name))
        if pattern is not None:
            new_filters.append(NameFilter(pattern))
        for prop, val in [
            ('continuous', continuous),
            ('ordered', ordered),
            ('bounded', bounded),
            ('bounded_left', bounded_left),
            ('bounded_right', bounded_right),
            ('finite', finite),
        ]:
            if val is not None:
                new_filters.append(BoolPropertyFilter(prop, val))
        if min_cardinality is not None or max_cardinality is not None:
            new_filters.append(NumericRangeFilter('cardinality', min_cardinality, max_cardinality))
        if min_decimals is not None or max_decimals is not None:
            new_filters.append(NumericRangeFilter('decimals', min_decimals, max_decimals))
        if using is not None and not hasattr(using, 'items'):
            raise TypeError('using must be a mapping from argument names to columns.')
        return self._clone(
            filters=[*self._filters, *new_filters],
            into=into,
            components=components,
            using=using,
        )
    
    def _items(self):
        return self.domain.columns.items()
    

@register_assume(Frame)
class FrameAssumptions(FrameDescriptor):

    def _group(self, make_domain):
        if self._into is None:
            raise ValueError('Grouped assumptions require into=<logical name>.')
        if self._components is not None:
            raise ValueError(
                'Grouped assumptions reuse the selected physical components; '
                'components= is not applicable.'
            )
        if self._using:
            raise ValueError('using= is only supported by inferences.')
        matching = self._matching()
        names = [name for name, _ in matching]
        if not names:
            raise ValueError('Grouped assumption did not select any columns.')
        physical = tuple(
            component
            for name in names
            for component in self.domain.components(name)
        )
        dtypes = []
        for name in names:
            column = self.domain[name]
            dtype = getattr(column, 'dtype', None)
            if dtype is None:
                raise TypeError(f'Logical column {name!r} has no physical dtype.')
            dtypes.extend([np.dtype(dtype)] * column.event_size)
        if any(dtype != dtypes[0] for dtype in dtypes[1:]):
            raise TypeError('Grouped physical columns must have the same dtype.')
        grouped_domain = make_domain(physical, dtypes[0])
        return self._finalize_assume(
            self.domain.grouped(names, self._into, grouped_domain)
        )

    def one_hot(self, categories=None):
        from dicer import domain as d

        def make_domain(physical, dtype):
            labels = physical if categories is None else categories
            if len(labels) != len(physical):
                raise ValueError('categories must match the number of components.')
            return d.OneHotColumn(labels, dtype=dtype)

        return self._group(make_domain)

    def simplex(self, labels=None, rtol=1e-7, atol=1e-9):
        from dicer import domain as d

        return self._group(lambda physical, dtype: d.SimplexColumn(
            len(physical),
            dtype=dtype,
            labels=labels,
            rtol=rtol,
            atol=atol,
        ))

    def vector(self, labels=None):
        from dicer import domain as d

        return self._group(lambda physical, dtype: d.VectorColumn(
            len(physical),
            dtype=dtype,
            labels=physical if labels is None else labels,
        ))

    # methods deferred to individual columns
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def method(*args, **kwargs) -> Frame:
            updates = {}
            for col_name, col in self._matching():
                assume_method = getattr(col.assume, name, None)
                if assume_method is None:
                    raise AttributeError(f'{col} has no {name} assume method.')
                updates[col_name] = assume_method(*args, **kwargs)
            return self._finalize_assume(self.domain.with_domains(updates))

        return method

@register_infer(Frame)
class FrameInferences(FrameDescriptor):

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def method(*args, **kwargs):
            if self._tracker is None:
                raise TypeError(
                    "FrameInferences requires a FrameTracker with data. "
                    "Use FrameTracker.from_dataframe() or pass data to the tracker."
                )
            matching = self._matching()
            if self._into is not None and len(matching) != 1:
                raise ValueError('into= requires exactly one selected logical column.')
            if self._components is not None and self._into is None:
                raise ValueError('components= requires into= for an inference.')

            observations = {}
            for argument, logical_name in self._using.items():
                if logical_name not in self.domain.columns:
                    raise KeyError(f'Unknown using= column {logical_name!r}.')
                if argument in kwargs:
                    raise TypeError(
                        f'Inference argument {argument!r} was supplied both '
                        'directly and through using=.'
                    )
                observations[argument] = ObservedColumn(
                    self.domain[logical_name],
                    self.domain.extract(self._tracker._tracked_data, logical_name),
                )

            results = []
            for col_name, col in matching:
                data = self.domain.extract(self._tracker._tracked_data, col_name)
                col_tracker = ColumnTracker(
                    col,
                    data=data,
                    random_state=deepcopy(self._tracker._tracked_prng),
                )
                infer_method = getattr(col_tracker._Inferences(col_tracker), name, None)
                if infer_method is None:
                    raise AttributeError(f'{col} has no {name} infer method.')
                result = infer_method(*args, **kwargs, **observations)
                if not isinstance(result, ColumnTracker):
                    raise TypeError(
                        f'Column inference {name!r} returned {type(result).__name__}; '
                        'expected a ColumnTracker.'
                    )
                results.append((col_name, result))

            if all(not result._tracked_transformations for _, result in results):
                if self._into is not None and any(name != self._into for name, _ in results):
                    raise ValueError('An inference without a transformation cannot derive a new column.')
                return self._finalize_assume(self.domain.with_domains({
                    name: result.unpack('domain') for name, result in results
                }))

            tracker = self._tracker
            for col_name, result in results:
                target = self._into if self._into is not None else col_name
                tracker = tracker._apply_column_inference(
                    col_name,
                    target,
                    result,
                    components=self._components,
                )
            return tracker

        return method


@register_transform(Frame)
class FrameTransformations(FrameDescriptor):
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def method(*args, **kwargs) -> Frame:
            from dicer.transform.frame import FrameOperation, normalize_components

            if self._using:
                raise ValueError('using= is only supported by inferences.')
            matching = self._matching()
            if self._into is not None and len(matching) != 1:
                raise ValueError('into= requires exactly one selected logical column.')
            if self._components is not None and self._into is None:
                raise ValueError('components= requires into= for a transformation.')

            operations = []
            current_domain = self.domain
            for col_name, col in matching:
                # By not calling col.transform we bypass wrapping in a Tracker object
                transform_method = getattr(col._Transformations(col), name, None)
                if transform_method is None:
                    raise AttributeError(f'{col} has no {name} transform method.')
                transformation = transform_method(*args, **kwargs)
                target = self._into if self._into is not None else col_name
                output_components = normalize_components(
                    target,
                    transformation.output_domain,
                    self._components,
                    input_components=current_domain.components(col_name),
                    replacing=target == col_name,
                )
                operation = FrameOperation(
                    col_name,
                    target,
                    transformation,
                    output_components,
                )
                operations.append(operation)
                current_domain = current_domain.with_field(
                    operation.source,
                    operation.target,
                    transformation.output_domain,
                    output_components,
                )
            return self._finalize_transform(operations)

        return method


# --------------------- Filters ----------------------------
from abc import ABC, abstractmethod
from fnmatch import fnmatch

class ColumnFilter(ABC):
    """Predicate that selects a subset of columns by name and/or domain properties."""

    @abstractmethod
    def matches(self, name: str, col: Column) -> bool: ...


class NameFilter(ColumnFilter):
    """Selects columns by exact name or a set of names."""

    def __init__(self, names):
        if isinstance(names, str):
            self._names = frozenset([names])
        else:
            self._names = frozenset(names)

    @property
    def names(self):
        return self._names

    def matches(self, name: str, col: Column) -> bool:
        return any(fnmatch(name, pattern) for pattern in self._names)


class BoolPropertyFilter(ColumnFilter):
    """Selects columns where ``is_<prop>`` equals *value*.

    If a column does not have the ``is_<prop>`` attribute the filter does not
    match, regardless of *value*.
    """

    def __init__(self, prop: str, value: bool):
        self._prop = prop
        self._value = value

    def matches(self, name: str, col: Column) -> bool:
        attr = getattr(col, f'is_{self._prop}', None)
        if attr is None:
            return False
        return bool(attr) == self._value


class NumericRangeFilter(ColumnFilter):
    """Selects columns where a numeric property falls within [min_val, max_val].

    The property must exist on the column and have a nonnegative, finite value for
    the filter to match.  Continuous columns (``cardinality == -1``) are
    therefore excluded by a cardinality range filter.
    """

    def __init__(self, prop: str, min_val=None, max_val=None):
        self._prop = prop
        self._min_val = min_val
        self._max_val = max_val

    def matches(self, name: str, col: Column) -> bool:
        if not hasattr(col, self._prop):
            return False
        val = getattr(col, self._prop)
        if not (np.isfinite(val) and val >= 0):
            return False
        if self._min_val is not None and val < self._min_val:
            return False
        if self._max_val is not None and val > self._max_val:
            return False
        return True
