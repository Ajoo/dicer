from dicer.infer import continuous  # registers continuous inference descriptors
from dicer.infer import categorical  # registers categorical inference descriptors
from dicer.infer.integer import (
    IntegerInferences,
    infer_integer_genpareto_tail,
    infer_integer_empirical_tail,
    infer_integer_uniform_tail,
)
from dicer.infer.categorical import (
    FiniteColumnInferences,
    GenericColumnInferences,
    UnorderedIntegerInferences,
)
from dicer.infer.continuous import (
    FixedPointInferences,
    FloatingPointInferences,
    infer_fixed_point,
    infer_genpareto_tail,
    infer_normal_tail,
    infer_uniform_tail,
    quantile_discretize,
)


__all__ = [
    'IntegerInferences',
    'infer_integer_genpareto_tail',
    'infer_integer_empirical_tail',
    'infer_integer_uniform_tail',
    'FixedPointInferences',
    'FloatingPointInferences',
    'FiniteColumnInferences',
    'GenericColumnInferences',
    'UnorderedIntegerInferences',
    'infer_fixed_point',
    'infer_genpareto_tail',
    'infer_normal_tail',
    'infer_uniform_tail',
    'quantile_discretize',
]
