from __future__ import annotations

from copy import copy
from math import ceil, floor
from numbers import Real
from typing import ClassVar, Hashable, Protocol

import numpy as np
import pandas as pd

class Column:
    cardinality: int # -1 if continuous
    ordered: ClassVar[bool]
    dtype = None
    event_shape: ClassVar[tuple[int, ...]] = ()

    @property
    def event_size(self):
        return int(np.prod(self.event_shape, dtype=np.intp)) if self.event_shape else 1

    def check_dtype(self, dtype):
        return np.issubdtype(dtype, self._check_dtype)

    def validate(self, values) -> None:
        """Validate values against the assumptions represented by this domain."""

    @property
    def finite(self):
        return self.cardinality >= 0 and np.isfinite(self.cardinality)

    @property
    def is_finite(self):
        return self.finite

    @staticmethod
    def from_dtype(dtype: np.dtype):
        if isinstance(dtype, pd.CategoricalDtype):
            return PandasCategoricalColumn.from_dtype(dtype)
        elif np.issubdtype(dtype, np.integer):
            return IntegerColumn.from_dtype(dtype)
        elif np.issubdtype(dtype, np.floating):
            return ContinuousColumn.from_dtype(dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            raise NotImplementedError
        else:
            return GenericColumn.from_dtype(dtype)


class MaybeFinite(Protocol):

    @property
    def finite(self):
        return not np.isinf(self.cardinality)

    @property
    def is_finite(self):
        return self.finite

    def index(self, x, last=False): ...

    def at(self, index): ...


class OrderedColumn(Column): # i.e., numerical
    lb: Real
    ub: Real
    ordered: ClassVar[bool] = True
    is_ordered: ClassVar[bool] = True

    @staticmethod
    def _validate_bounds(lower_bound, upper_bound) -> None:
        for name, value in (
            ('Lower', lower_bound),
            ('Upper', upper_bound),
        ):
            if not np.isscalar(value):
                raise ValueError(f'{name} bound must be a scalar.')
            if value != value:
                raise ValueError(f'{name} bound must not be NaN.')
        if lower_bound > upper_bound:
            raise ValueError(
                f'Lower bound {lower_bound!r} exceeds upper bound '
                f'{upper_bound!r}.'
            )

    @property
    def is_bounded_left(self):
        return np.isfinite(self.lb)
    
    @property
    def is_bounded_right(self):
        return np.isfinite(self.ub)
    
    @property
    def is_bounded(self):
        return self.is_bounded_left and self.is_bounded_right

    def with_new_bounds(self, lower_bound=None, upper_bound=None):
        self = copy(self)
        if lower_bound is not None:
            self.lb = lower_bound
        if upper_bound is not None:
            self.ub = upper_bound
        self._validate_bounds(self.lb, self.ub)
        return self

    def validate(self, values) -> None:
        values_array = np.asarray(values)
        valid = (values_array >= self.lb) & (values_array <= self.ub)
        if not np.all(valid):
            raise ValueError(
                f'Values must lie within [{self.lb!r}, {self.ub!r}].'
            )
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        
        return (
            (self.lb == other.lb)
            and (self.ub == other.ub)
        )


class FinitePrecisionMixin: #(Protocol):
    decimals: int

    @property
    def precision(self):
        return 10 ** -self.decimals


class IntegerColumn(FinitePrecisionMixin, OrderedColumn, MaybeFinite):
    decimals: ClassVar[type] = 0
    _check_dtype: ClassVar[type] = np.integer
    is_continuous: ClassVar[bool] = False

    @property
    def cardinality(self):
        return 1 + self.ub - self.lb
    
    @property
    def finite(self):
        return np.isfinite(self.cardinality)

    @property
    def is_finite(self):
        return self.finite

    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        self._validate_bounds(lower_bound, upper_bound)
        for name, value in (
            ('Lower', lower_bound),
            ('Upper', upper_bound),
        ):
            if np.isfinite(value) and np.mod(value, 1) != 0:
                raise ValueError(f'{name} bound must be an integer.')
        self.lb = lower_bound
        self.ub = upper_bound

    @staticmethod
    def from_dtype(dtype, assume_bounds=False):
        info = np.iinfo(dtype)
        if assume_bounds:
            self = IntegerColumn(lower_bound=info.min, upper_bound=info.max)
        elif info.min == 0:
            self = IntegerColumn(lower_bound=0)
        else:
            self = IntegerColumn()

        self.dtype = dtype
        return self
    
    def index(self, x, last=False):
        """ Maps from domain to [0, cardinality)
        """
        if np.isinf(self.lb):
            raise ValueError(
                "Cannot encode IntegerColumn with infinite lower bound."
                )

        x = np.asarray(x)
        y = x - self.lb
        if last:
            y = np.floor(y)
            y = np.where(np.isnan(y), -1, y)
            y = np.maximum(y, -1)
            if self.finite:
                y = np.minimum(y, self.cardinality - 1)
            if not np.all(np.isfinite(y)):
                raise ValueError('Cannot encode an infinite value as an index.')
            return y.astype(np.intp)
        
        valid = (np.mod(y, 1) == 0) & (y >= 0) & (x <= self.ub)
        return np.where(valid, y, -1).astype(np.intp)

    def validate(self, values) -> None:
        super().validate(values)
        values_array = np.asarray(values)
        if not np.all(np.mod(values_array, 1) == 0):
            raise ValueError('Values must be integers.')
    
    def at(self, index):
        if np.isinf(self.lb):
            raise ValueError(
                "Cannot encode IntegerColumn with infinite lower bound."
                )
        
        assert np.all(index >= 0)
        return self.lb + index
    
    @property
    def values(self):
        assert self.finite
        return np.arange(self.lb, self.ub+1)


    def __repr__(self):
        return f'int: [{self.lb}..{self.ub}]'


class ContinuousColumn(OrderedColumn):
    cardinality: ClassVar[int] = -1
    _check_dtype: ClassVar[type] = np.floating
    is_continuous: ClassVar[bool] = True
    
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        self._validate_bounds(lower_bound, upper_bound)
        self.lb = lower_bound
        self.ub = upper_bound

    @staticmethod
    def from_dtype(dtype):
        self = FloatingPointColumn()
        self.dtype = dtype
        return self
    
    
class FloatingPointColumn(ContinuousColumn):

    def __repr__(self):
        return f'float: ({self.lb}, {self.ub})'


class FixedPointColumn(FinitePrecisionMixin, ContinuousColumn):
    decimals: int
    rtol: float = 1e-3
    atol: float = 1e-8

    def __init__(self, digits, lower_bound=-np.inf, upper_bound=np.inf):
        if digits % 1 != 0:
            raise ValueError('digits must be an integer.')
        if digits < 0:
            raise ValueError('digits must be nonnegative.')
        super().__init__(lower_bound, upper_bound)
        self.decimals = int(digits)

    def __eq__(self, other):
        matching_bounds = super().__eq__(other)
        if matching_bounds is NotImplemented:
            return matching_bounds
        
        return matching_bounds and self.decimals == other.decimals
    
    def check_decimals(self, values):
        precision = 10 ** -self.decimals
        remainder = np.remainder(values, precision)
        distance = np.minimum(remainder, precision - remainder)
        return np.all(distance <= self.rtol * precision + self.atol)

    def validate(self, values) -> None:
        super().validate(values)
        if not self.check_decimals(np.asarray(values)):
            raise ValueError(
                f'Values must have at most {self.decimals} decimal places.'
            )

    def __repr__(self):
        return f'fixed-point ({self.decimals} digits): ({self.lb}, {self.ub})'

    @staticmethod
    def from_floating(domain, decimals):
        h = 10**decimals
        lb = ceil(domain.lb * h)/h if np.isfinite(domain.lb) else domain.lb
        ub = floor(domain.ub * h)/h if np.isfinite(domain.ub) else domain.ub
        return FixedPointColumn(decimals, lb, ub)
    

class GenericColumn(Column):

    def __init__(self, dtype):
        self.dtype = dtype

    @staticmethod
    def from_dtype(dtype):
        return GenericColumn(dtype)

    def validate(self, values) -> None:
        pass


class FiniteColumn(Column, MaybeFinite):
    _check_dtype: ClassVar[type] = np.generic
    is_continuous: ClassVar[bool] = False
    is_bounded: ClassVar[bool] = True
    is_bounded_left: ClassVar[bool] = True
    is_bounded_right: ClassVar[bool] = True
    
    def __init__(self, values):
        """Create a finite domain with a stable enumeration of its values.

        Parameters
        ----------
        values : sequence
            Non-empty sequence of unique, hashable values. The sequence is
            retained as the domain's stable enumeration.

        Raises
        ------
        ValueError
            If `values` is empty or cannot be coerced to a one-dimensional array.
        """
        if isinstance(values, (set, frozenset)):
            raise TypeError(
                'values must be an ordered sequence, not a set or frozenset.'
            )

        try:
            values_index = pd.Index(values, tupleize_cols=False)
        except (TypeError, ValueError) as error:
            raise ValueError('values must be a one-dimensional sequence.') from error

        if values_index.size == 0:
            raise ValueError('values must contain at least one element.')
        if values_index.hasnans:
            raise ValueError('values must not contain missing values.')
        try:
            is_unique = values_index.is_unique
        except TypeError as error:
            raise TypeError('values must contain only hashable values.') from error
        if not is_unique:
            raise ValueError('values must contain only unique values.')

        self.values = values_index.to_numpy(copy=True)
        
    @property
    def cardinality(self):
        return self.values.size
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self.values, other.values)
    
    def index(self, x, last=False):
        raise NotImplementedError()
    
    def at(self, index):
        raise NotImplementedError()

    def validate(self, values) -> None:
        if not np.all(np.isin(np.asarray(values), self.values)):
            raise ValueError('Values must belong to the finite domain.')


class PandasCategoricalColumn(FiniteColumn):
    _check_dtype: ClassVar[type] = pd.Categorical

    def __init__(self, dtype: pd.CategoricalDtype):
        self.dtype = dtype

    @property
    def values(self):
        return self.dtype.categories

    # TODO: Handle last for an ordered categorical
    def index(self, x, last=False):
        if isinstance(x, pd.Series) and x.dtype == self.dtype:
            return x.cat.codes.values

        values = np.asarray(x)
        y = pd.Categorical(values.reshape(-1), dtype=self.dtype).codes
        y = y.reshape(values.shape)
        if np.isscalar(x):
            return int(y)

        return y
    
    # TODO: test this
    def at(self, index):
        return pd.Categorical.from_codes(index, dtype=self.dtype)
    
    @property
    def ordered(self):
        return self.dtype.ordered

    @property
    def is_ordered(self):
        return self.dtype.ordered

    @property
    def lb(self):
        if not self.ordered:
            raise AttributeError('An unordered categorical has no lower bound.')
        return self.values[0]

    @property
    def ub(self):
        if not self.ordered:
            raise AttributeError('An unordered categorical has no upper bound.')
        return self.values[-1]

    @property
    def is_finite(self):
        return True

    @staticmethod
    def from_dtype(dtype: pd.CategoricalDtype):
        return PandasCategoricalColumn(dtype)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.ordered == other.ordered
            and np.array_equal(self.values, other.values)
        )
    
    def __repr__(self):
        return repr(self.dtype)


class FiniteOrderedColumn(FiniteColumn):
    _check_dtype: ClassVar[type] = np.number
    ordered: ClassVar[bool] = True
    is_ordered: ClassVar[bool] = True

    def __init__(self, values):
        super().__init__(np.sort(values))

    @property
    def dtype(self):
        return self.values.dtype

    def index(self, x, last=False):
        if last:
            return np.searchsorted(self.values, x, side='right') - 1
        y = np.minimum(np.searchsorted(self.values, x), self.cardinality-1)
        return np.where(self.values[y] == x, y, -1)
    
    def at(self, index):
        return self.values[index]

    @property
    def lb(self):
        return self.values[0]
    
    @property
    def ub(self):
        return self.values[-1]

    def __repr__(self):
        values_preview = ', '.join(map(str, self.values.tolist()))
        return f'ordered finite: {{{values_preview}}}'


class FiniteUnorderedColumn(FiniteColumn):
    _check_dtype: ClassVar[type] = np.generic
    ordered: ClassVar[bool] = False
    is_ordered: ClassVar[bool] = False

    def __init__(self, values, dtype=None):
        super().__init__(values)
        self.dtype = self.values.dtype if dtype is None else dtype
        self._index = pd.Index(self.values, tupleize_cols=False)

    def __eq__(self, other):
        matching_values = super().__eq__(other)
        if matching_values is NotImplemented:
            return matching_values
        return matching_values and self.dtype == other.dtype

    def index(self, x, last=False):
        if last:
            raise ValueError('An unordered domain has no predecessor index.')

        is_scalar = np.isscalar(x)
        if isinstance(x, tuple):
            is_scalar = x in self._index

        if is_scalar:
            targets = np.empty(1, dtype=object)
            targets[0] = x
        else:
            targets = np.asarray(x)
        indexes = self._index.get_indexer(targets.reshape(-1))
        indexes = indexes.reshape(targets.shape)
        return int(indexes[0]) if is_scalar else indexes

    def at(self, index):
        is_scalar = np.isscalar(index)
        indexes = np.asarray([index] if is_scalar else index)
        if not np.issubdtype(indexes.dtype, np.integer):
            raise TypeError('Indexes must be integers.')
        if np.any((indexes < 0) | (indexes >= self.cardinality)):
            raise ValueError('Indexes must lie in [0, cardinality).')
        values = self.values[indexes]
        return values[0] if is_scalar else values

    def validate(self, values) -> None:
        if np.any(np.asarray(self.index(values)) < 0):
            raise ValueError('Values must belong to the finite domain.')

    def __repr__(self):
        values_preview = ', '.join(map(repr, self.values.tolist()))
        return f'unordered finite: {{{values_preview}}}'


class UnorderedIntegerColumn(FiniteUnorderedColumn):
    """The unordered integer index set ``[K] = {0, ..., K - 1}``."""

    _check_dtype: ClassVar[type] = np.integer

    def __init__(self, cardinality, dtype=np.dtype(np.int64)):
        if not isinstance(cardinality, (int, np.integer)):
            raise TypeError('cardinality must be an integer.')
        if cardinality <= 0:
            raise ValueError('cardinality must be positive.')
        self._cardinality = int(cardinality)
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError('dtype must be an integer dtype.')

    @property
    def cardinality(self):
        return self._cardinality

    @property
    def values(self):
        return np.arange(self.cardinality, dtype=self.dtype)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.cardinality == other.cardinality
            and self.dtype == other.dtype
        )

    def index(self, x, last=False):
        if last:
            raise ValueError('An unordered domain has no predecessor index.')

        values = np.asarray(x)
        with np.errstate(invalid='ignore'):
            valid = (
                np.isfinite(values)
                & (np.mod(values, 1) == 0)
                & (values >= 0)
                & (values < self.cardinality)
            )
        indexes = np.where(valid, values, -1).astype(np.intp)
        return int(indexes) if np.isscalar(x) else indexes

    def at(self, index):
        indexes = np.asarray(index)
        if not np.issubdtype(indexes.dtype, np.integer):
            raise TypeError('Indexes must be integers.')
        if np.any((indexes < 0) | (indexes >= self.cardinality)):
            raise ValueError('Indexes must lie in [0, cardinality).')
        values = indexes.astype(self.dtype, copy=False)
        return values.item() if np.isscalar(index) else values

    def validate(self, values) -> None:
        if np.any(np.asarray(self.index(values)) < 0):
            raise ValueError(
                f'Values must be integers in [0, {self.cardinality}).'
            )

    def __repr__(self):
        return f'unordered int: [0..{self.cardinality})'


class VectorColumn(Column):
    """A fixed-width vector interpreted as one logical column."""

    cardinality: ClassVar[int] = -1
    ordered: ClassVar[bool] = False
    is_ordered: ClassVar[bool] = False
    is_continuous: ClassVar[bool] = True
    is_bounded: ClassVar[bool] = False
    is_bounded_left: ClassVar[bool] = False
    is_bounded_right: ClassVar[bool] = False

    def __init__(self, dimension, dtype=np.dtype(np.float64), labels=None):
        if not isinstance(dimension, (int, np.integer)):
            raise TypeError('dimension must be an integer.')
        if dimension <= 0:
            raise ValueError('dimension must be positive.')
        self.dimension = int(dimension)
        self.event_shape = (self.dimension,)
        self.dtype = np.dtype(dtype)
        if not (
            np.issubdtype(self.dtype, np.number)
            or np.issubdtype(self.dtype, np.bool_)
        ):
            raise TypeError('VectorColumn dtype must be numeric.')
        if labels is None:
            labels = tuple(range(self.dimension))
        if len(labels) != self.dimension:
            raise ValueError('labels must match the vector dimension.')
        self.labels = tuple(labels)

    def validate(self, values) -> None:
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[-1] != self.dimension:
            raise ValueError(
                f'Vector values must have final dimension {self.dimension}.'
            )
        if not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TypeError('Vector values must be numeric.')

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.dtype == other.dtype
            and self.labels == other.labels
        )

    def __repr__(self):
        return f'vector[{self.dimension}]'


class SimplexColumn(VectorColumn):
    """A probability vector: finite, nonnegative, and summing to one."""

    is_bounded: ClassVar[bool] = True
    is_bounded_left: ClassVar[bool] = True
    is_bounded_right: ClassVar[bool] = True

    def __init__(
        self,
        dimension,
        dtype=np.dtype(np.float64),
        labels=None,
        rtol=1e-7,
        atol=1e-9,
    ):
        labels_provided = labels is not None
        super().__init__(dimension, dtype=dtype, labels=labels)
        if not labels_provided:
            self.labels = None
        self.rtol = rtol
        self.atol = atol

    def validate(self, values) -> None:
        super().validate(values)
        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            raise ValueError('Simplex values must be finite.')
        if np.any(values < -self.atol):
            raise ValueError('Simplex values must be nonnegative.')
        if not np.allclose(
            np.sum(values, axis=-1),
            1.0,
            rtol=self.rtol,
            atol=self.atol,
        ):
            raise ValueError('Simplex values must sum to one.')

    def __eq__(self, other):
        matching = super().__eq__(other)
        if matching is NotImplemented:
            return matching
        return matching and self.rtol == other.rtol and self.atol == other.atol

    def __repr__(self):
        return f'simplex[{self.dimension}]'


class OneHotColumn(SimplexColumn):
    """A categorical value represented by one active component."""

    is_continuous: ClassVar[bool] = False

    def __init__(self, categories, dtype=np.dtype(np.int64)):
        categories = pd.Index(categories, tupleize_cols=False)
        if categories.empty:
            raise ValueError('categories must contain at least one value.')
        if categories.hasnans or not categories.is_unique:
            raise ValueError('categories must be unique and non-missing.')
        self.categories = categories
        super().__init__(
            len(categories),
            dtype=dtype,
            labels=tuple(categories),
            rtol=0.0,
            atol=0.0,
        )

    @property
    def cardinality(self):
        return self.dimension

    @property
    def finite(self):
        return True

    @property
    def is_finite(self):
        return True

    def validate(self, values) -> None:
        VectorColumn.validate(self, values)
        values = np.asarray(values)
        if not np.all((values == 0) | (values == 1)):
            raise ValueError('One-hot values must contain only zeros and ones.')
        if not np.all(np.sum(values, axis=-1) == 1):
            raise ValueError('One-hot values must contain exactly one active component.')

    def index(self, values, last=False):
        if last:
            raise ValueError('An unordered domain has no predecessor index.')
        self.validate(values)
        indexes = np.argmax(np.asarray(values), axis=-1)
        return int(indexes) if indexes.ndim == 0 else indexes

    def at(self, indexes):
        indexes = np.asarray(indexes)
        if not np.issubdtype(indexes.dtype, np.integer):
            raise TypeError('Indexes must be integers.')
        if np.any((indexes < 0) | (indexes >= self.cardinality)):
            raise ValueError('Indexes must lie in [0, cardinality).')
        return np.eye(self.cardinality, dtype=self.dtype)[indexes]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.dtype == other.dtype and self.categories.equals(other.categories)

    def __repr__(self):
        return f'one-hot[{self.cardinality}]'


class Frame:
    """Cartesian product of logical columns and their physical layout.

    Parameters
    ----------
    columns : dict
        Mapping from column name (str) to ``Column`` domain.

    Examples
    --------
    >>> import numpy as np
    >>> d = Frame.from_dtypes({'price': np.float64, 'count': np.int32})
    >>> d2 = d.assume['price'].ge(0.0)
    >>> d3 = d.assume(pattern='price').ge(0.0)
    """

    def __init__(
        self,
        columns: dict[Hashable, Column],
        layout: dict[Hashable, tuple[Hashable, ...]] | None = None,
    ):
        self.columns = dict(columns)
        if layout is None:
            layout = {name: (name,) for name in self.columns}
        if set(layout) != set(self.columns):
            raise ValueError('Frame layout must contain every logical column exactly once.')
        self.layout = {name: tuple(layout[name]) for name in self.columns}
        for name, column in self.columns.items():
            components = self.layout[name]
            if len(components) != column.event_size:
                raise ValueError(
                    f'Logical column {name!r} requires {column.event_size} physical '
                    f'components, got {len(components)}.'
                )
        physical = self.physical_columns
        if len(physical) != len(set(physical)):
            raise ValueError('Physical DataFrame columns may belong to only one logical column.')

    @property
    def column_names(self):
        return self.columns.keys()

    @property
    def physical_columns(self):
        return tuple(
            component
            for name in self.column_names
            for component in self.layout[name]
        )

    def components(self, name):
        return self.layout[name]

    def extract(self, frame, name):
        components = self.components(name)
        if len(components) == 1:
            return frame[components[0]]
        return frame.loc[:, list(components)].to_numpy(copy=False)

    @staticmethod
    def from_dtypes(dtypes) -> Frame:
        """Create a ``Frame`` from named numpy/pandas dtypes.

        Parameters
        ----------
        dtypes : mapping
            A mapping from column names to numpy dtypes. Accepts
            ``df.dtypes`` (a pandas ``Series``) or any ``{name: dtype}``
            dict-like object.

        Returns
        -------
        Frame
        """
        return Frame({
            name: Column.from_dtype(dtype)
            for name, dtype in dtypes.items()
        })

    def __getitem__(self, name: Hashable) -> Column:
        return self.columns[name]

    def with_domains(self, updates: dict[Hashable, Column]) -> Frame:
        """Replace domains of existing logical columns without changing layout."""
        unknown = [name for name in updates if name not in self.columns]
        if unknown:
            raise KeyError(f'Unknown logical columns: {unknown!r}.')
        return Frame({**self.columns, **updates}, self.layout)

    def grouped(self, names, into, grouped_domain) -> Frame:
        names = list(names)
        if not names:
            raise ValueError('Grouping requires at least one logical column.')
        if len(set(names)) != len(names):
            raise ValueError('Grouping column names must be unique.')
        unknown = [name for name in names if name not in self.columns]
        if unknown:
            raise KeyError(f'Unknown logical columns: {unknown!r}.')
        if into in self.columns and into not in names:
            raise ValueError(f'Logical column {into!r} already exists.')

        selected = set(names)
        components = tuple(
            component
            for name in names
            for component in self.layout[name]
        )
        columns = {}
        layout = {}
        inserted = False
        for name, column in self.columns.items():
            if name in selected:
                if not inserted:
                    columns[into] = grouped_domain
                    layout[into] = components
                    inserted = True
                continue
            columns[name] = column
            layout[name] = self.layout[name]
        return Frame(columns, layout)

    def with_field(self, source, target, output_domain, components) -> Frame:
        if source not in self.columns:
            raise KeyError(f'Unknown logical column {source!r}.')
        replacing = source == target
        if not replacing and target in self.columns:
            raise ValueError(f'Logical column {target!r} already exists.')
        components = tuple(components)

        retained_physical = {
            component
            for name in self.column_names
            if not (replacing and name == source)
            for component in self.layout[name]
        }
        collisions = [component for component in components if component in retained_physical]
        if collisions:
            raise ValueError(
                f'Output components already exist in the frame: {collisions!r}.'
            )

        columns = {}
        layout = {}
        for name, column in self.columns.items():
            if replacing and name == source:
                columns[target] = output_domain
                layout[target] = components
            else:
                columns[name] = column
                layout[name] = self.layout[name]
        if not replacing:
            columns[target] = output_domain
            layout[target] = components
        return Frame(columns, layout)

    def validate(self, frame) -> None:
        """Validate a DataFrame against its column domains and assumptions."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f'Expected a pandas DataFrame, got {type(frame).__name__}.'
            )
        if not frame.columns.is_unique:
            raise ValueError('DataFrame columns must be unique.')

        expected = list(self.physical_columns)
        actual = list(frame.columns)
        missing = [name for name in expected if name not in frame.columns]
        extra = [name for name in actual if name not in expected]
        if missing or extra:
            details = []
            if missing:
                details.append(f'missing columns: {missing!r}')
            if extra:
                details.append(f'unexpected columns: {extra!r}')
            raise ValueError('DataFrame does not match Frame; ' + '; '.join(details))

        for name, column_domain in self.columns.items():
            components = self.components(name)
            expected_dtype = getattr(column_domain, 'dtype', None)
            if expected_dtype is not None:
                mismatched = [
                    component
                    for component in components
                    if frame[component].dtype != expected_dtype
                ]
                if mismatched:
                    actual_dtypes = [frame[component].dtype for component in mismatched]
                    raise TypeError(
                        f'Logical column {name!r} has component dtypes '
                        f'{actual_dtypes!r}; expected {expected_dtype!r}.'
                    )
            try:
                column_domain.validate(self.extract(frame, name))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'Column {name!r} does not satisfy {column_domain!r}: {error}'
                ) from error

    def validate_dataframe(self, frame) -> None:
        """Backward-compatible alias for :meth:`validate`."""
        self.validate(frame)

    def __repr__(self) -> str:
        import pandas as pd
        return repr(pd.Series(self.columns))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return self.columns == other.columns and self.layout == other.layout
    
    def copy(self):
        return Frame(copy(self.columns), copy(self.layout))
