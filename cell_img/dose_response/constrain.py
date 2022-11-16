"""Bijectors for constraining unconstrained variables."""

import jax
import jax.numpy as jnp


def to_positive(
    unconstrained: jnp.array,
) -> jnp.array:
  """Constrain an unconstrained array to being positive."""
  return jnp.exp(unconstrained)


def to_positive_inverse(
    constrained: jnp.array,
) -> jnp.array:
  """Invert to_positive."""
  return jnp.log(constrained)


def to_bounded(
    unconstrained: jnp.array,
    minimum: float,
    maximum: float,
    ) -> jnp.array:
  """Constrain an unconstrained array to lie between minimum and maximum."""
  return (maximum - minimum) * jax.scipy.special.expit(unconstrained) + minimum


def to_bounded_inverse(
    constrained: jnp.array,
    minimum: float,
    maximum: float,
    ) -> jnp.array:
  """Invert to_bounded."""
  return jax.scipy.special.logit((constrained - minimum) / (maximum - minimum))
