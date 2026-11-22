from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd

from dicer.domain import Column, Frame
from dicer.transform._base import IdentityTransformation, Transformation


def default_components(
    target: Hashable,
    output_domain: Column,
    *,
    input_components=(),
    replacing=False,
):
    """Choose physical names for a logical transformation output."""
    width = output_domain.event_size
    input_components = tuple(input_components)
    if replacing and len(input_components) == width:
        return input_components
    if width == 1:
        return (target,)
    return tuple(f'{target}.{index}' for index in range(width))


def normalize_components(
    target: Hashable,
    output_domain: Column,
    components=None,
    *,
    input_components=(),
    replacing=False,
):
    if components is None:
        components = default_components(
            target,
            output_domain,
            input_components=input_components,
            replacing=replacing,
        )
    elif isinstance(components, (str, bytes)):
        components = (components,)
    else:
        components = tuple(components)
    if len(components) != output_domain.event_size:
        raise ValueError(
            f'Logical column {target!r} requires {output_domain.event_size} '
            f'physical components, got {len(components)}.'
        )
    if len(set(components)) != len(components):
        raise ValueError('Output component names must be unique.')
    return tuple(components)


def assign_field(frame, components, values):
    """Assign one logical scalar or structured value to physical columns."""
    components = tuple(components)
    n_rows = len(frame)
    if len(components) == 1:
        if not isinstance(values, pd.api.extensions.ExtensionArray):
            values = np.asarray(values)
        if np.shape(values) != (n_rows,):
            raise ValueError(
                f'Scalar transformation output has shape {np.shape(values)!r}; '
                f'expected {(n_rows,)!r}.'
            )
        frame[components[0]] = values
        return

    values = np.asarray(values)
    expected_shape = (n_rows, len(components))
    if values.shape != expected_shape:
        raise ValueError(
            f'Structured transformation output has shape {values.shape!r}; '
            f'expected {expected_shape!r}.'
        )
    for index, component in enumerate(components):
        frame[component] = values[:, index]


@dataclass(frozen=True)
class FrameOperation:
    """One name-aware assignment around a name-agnostic transformation."""

    source: Hashable
    target: Hashable
    transformation: Transformation
    output_components: tuple[Hashable, ...]


@dataclass(frozen=True)
class FrameTransformation:
    """A fitted, replayable sequence of logical-column operations."""

    input_domain: Frame
    output_domain: Frame
    transformations: tuple[FrameOperation, ...]

    def __post_init__(self):
        if self.transformations and not isinstance(
            self.transformations[0],
            FrameOperation,
        ):
            current = self.input_domain
            operations = []
            for name, transformation in self.transformations:
                components = normalize_components(
                    name,
                    transformation.output_domain,
                    input_components=current.components(name),
                    replacing=True,
                )
                operation = FrameOperation(
                    name,
                    name,
                    transformation,
                    components,
                )
                operations.append(operation)
                current = current.with_field(
                    name,
                    name,
                    transformation.output_domain,
                    components,
                )
            object.__setattr__(self, 'transformations', tuple(operations))

        current = self.input_domain
        for operation in self.transformations:
            if operation.source not in current.columns:
                raise ValueError(
                    f'Operation refers to unknown source {operation.source!r}.'
                )
            if current[operation.source] != operation.transformation.input_domain:
                raise ValueError(
                    f'Operation input domain for {operation.source!r} does not '
                    'match the current frame domain.'
                )

            current = current.with_field(
                operation.source,
                operation.target,
                operation.transformation.output_domain,
                operation.output_components,
            )
        if current != self.output_domain:
            raise ValueError(
                'Frame operations do not produce the declared output domain.'
            )

    def __getitem__(self, name: Hashable) -> Transformation:
        """Return the fitted lineage from an original source to one output."""
        if name not in self.output_domain.columns:
            raise KeyError(f'Unknown output column {name!r}.') from None

        source = name
        lineage = None
        for operation in reversed(self.transformations):
            if operation.target != source:
                continue
            if lineage is None:
                lineage = operation.transformation
            else:
                lineage = lineage * operation.transformation
            source = operation.source

        if lineage is None:
            return IdentityTransformation(self.input_domain[name])

        if source not in self.input_domain.columns:
            raise RuntimeError(
                f'Column lineage for {name!r} does not end at an input column.'
            )
        return lineage

    @property
    def operations(self):
        return self.transformations

    @property
    def is_deterministic(self):
        return all(
            operation.transformation.is_deterministic
            for operation in self.transformations
        )

    @property
    def is_lossy(self):
        return any(
            operation.source == operation.target
            and operation.transformation.is_lossy
            for operation in self.transformations
        )

    def __call__(self, frame: pd.DataFrame, random_state=None) -> pd.DataFrame:
        return self.apply(frame, random_state=random_state)

    def validate(self, frame: pd.DataFrame) -> None:
        """Validate a DataFrame against this transformation's input domain."""
        self.input_domain.validate(frame)

    def apply(self, frame: pd.DataFrame, random_state=None) -> pd.DataFrame:
        """Replay logical-column operations without mutating the input frame."""
        output = frame.loc[:, list(self.input_domain.physical_columns)].copy(deep=True)
        current = self.input_domain
        prng = np.random.default_rng(random_state)

        for operation in self.transformations:
            values = current.extract(output, operation.source)
            transformed = operation.transformation.apply(values, random_state=prng)
            next_domain = current.with_field(
                operation.source,
                operation.target,
                operation.transformation.output_domain,
                operation.output_components,
            )
            assign_field(output, operation.output_components, transformed)
            output = output.loc[:, list(next_domain.physical_columns)]
            current = next_domain

        return output.loc[:, list(self.output_domain.physical_columns)]
