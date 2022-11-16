"""Tests for jax_tree."""
import dataclasses
from typing import Optional, Set

from absl.testing import absltest
from cell_img.analysis import jax_tree
import jax.tree_util


@dataclasses.dataclass(frozen=True)
class ABCConstants:
  a: int
  b: int
  c: int
  constant_fields: Optional[Set[str]] = None


@dataclasses.dataclass(frozen=True)
class ABCNoConstants:
  a: int
  b: int
  c: int


class JaxTreeTest(absltest.TestCase):

  def test_flatten_unflatten(self):
    cls = dataclasses.make_dataclass(
        'ABC', fields=[], bases=(ABCConstants,), frozen=True)
    cls = jax_tree.dataclass_with_constants()(cls)

    ab = cls(a=1, b=2, c=3)
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([1, 2, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

  def test_dynamic_constants(self):
    cls = dataclasses.make_dataclass(
        'ABC', fields=[], bases=(ABCConstants,), frozen=True)
    cls = jax_tree.dataclass_with_constants()(cls)

    ab = cls(a=1, b=2, c=3)
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([1, 2, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    ab = cls(a=1, b=2, c=3, constant_fields=['a'])
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([2, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    ab = cls(a=1, b=2, c=3, constant_fields=['a', 'b', 'c'])
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

  def test_always_constant(self):
    cls = dataclasses.make_dataclass(
        'ABC', fields=[], bases=(ABCConstants,), frozen=True)
    cls = jax_tree.dataclass_with_constants(constant_fields=['a'])(cls)

    ab = cls(a=1, b=2, c=3)
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([2, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    ab = cls(a=1, b=2, c=3, constant_fields=['b'])
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    ab = cls(a=1, b=2, c=3, constant_fields=['a', 'b'])
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    ab = cls(a=1, b=2, c=3, constant_fields=['b', 'c'])
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

  def test_without_constants(self):
    cls = dataclasses.make_dataclass(
        'AB', fields=[], bases=(ABCConstants,), frozen=True)
    cls = jax_tree.dataclass_with_constants()(cls)

    ab = cls(a=1, b=2, c=3)
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([1, 2, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)

    cls = dataclasses.make_dataclass(
        'AB', fields=[], bases=(ABCNoConstants,), frozen=True)
    cls = jax_tree.dataclass_with_constants(constant_fields=['b'])(cls)

    ab = cls(a=1, b=2, c=3)
    data, metadata = jax.tree_util.tree_flatten(ab)
    self.assertEqual([1, 3], data)
    ab2 = jax.tree_util.tree_unflatten(metadata, data)
    self.assertEqual(ab, ab2)


if __name__ == '__main__':
  absltest.main()
