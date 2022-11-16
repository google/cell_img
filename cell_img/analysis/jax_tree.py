"""Decorator for making a dataclass into a PyTree with constant fields.

Fields that are declared constant are treated as metadata and are not involved
in JAX gradients. Fields can be declared constant either at class declaration
time via the decorator's constant_fields parameter or at class construction
time, again via a constant_fields parameter.

constant_fields should contain an optional iterable of strings.

Sample usage:

@jax_tree.dataclass_with_constants(constant_fields=['c'])
@dataclasses.dataclass(frozen=True)
class MyClass:
  a: int
  b: float
  c: float
  constant_fields: Optional[Iterable[str]] = None

myclass = MyClass(a=1, b=2, c=3, constant_fields=['b'])

Inspired by https://github.com/google/jax/issues/2371
"""

import dataclasses
from typing import Any, Iterable, Optional, Tuple, TypeVar
import jax


T = TypeVar('T')

_CONSTANT_FIELDS = 'constant_fields'
_DATA_FIELDS_KEY = '__data_fields'


def dataclass_with_constants(
    constant_fields: Optional[Iterable[str]] = None,
    ) -> T:
  """Make a dataclass a PyTree with specified constant fields.

  T is assumed to be a dataclass.

  Args:
    constant_fields: fields that will always be treated as constant.

  Returns:
    The decorated class.
  """
  always_constant_fields = {_CONSTANT_FIELDS}
  if constant_fields:
    always_constant_fields.update(constant_fields)
  del constant_fields

  def decorate(cls: T) -> T:

    def flatten_fn(
        t: T,
        ) -> Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]:
      """PyTree flatten method for T."""
      constants = set()
      fields = getattr(t, _CONSTANT_FIELDS, None)
      if fields:
        constants.update(fields)
      constants |= always_constant_fields

      metadata_items = []
      data_fields = []
      data_values = []
      for field in dataclasses.fields(t):
        value = getattr(t, field.name)
        if field.name in constants:
          metadata_items.append((field.name, value))
        else:
          data_fields.append(field.name)
          data_values.append(value)
      metadata_items.append((_DATA_FIELDS_KEY, data_fields))
      return tuple(data_values), tuple(metadata_items)

    def unflatten_fn(
        metadata: Tuple[Tuple[str, Any], ...],
        data: Tuple[Any, ...]
    ) -> T:
      """PyTree unflatten method for T."""
      metadata = dict(metadata)
      data = dict(zip(metadata.pop(_DATA_FIELDS_KEY), data))
      return cls(**metadata, **data)

    jax.tree_util.register_pytree_node(cls, flatten_fn, unflatten_fn)

    return cls

  return decorate
