"""Bijectors for constraining unconstrained variables."""

import jax
import jax.numpy as jnp


def to_positive(
    unconstrained: jnp.ndarray,
) -> jnp.ndarray:
  """Constrain an unconstrained array to being positive."""
  return jnp.exp(unconstrained)


def to_positive_inverse(
    constrained: jnp.ndarray,
) -> jnp.ndarray:
  """Invert to_positive."""
  return jnp.log(constrained)


def to_bounded(
    unconstrained: jnp.ndarray,
    minimum: float,
    maximum: float,
) -> jnp.ndarray:
  """Constrain an unconstrained array to lie between minimum and maximum."""
  return (maximum - minimum) * jax.scipy.special.expit(unconstrained) + minimum


def to_bounded_inverse(
    constrained: jnp.ndarray,
    minimum: float,
    maximum: float,
) -> jnp.ndarray:
  """Invert to_bounded."""
  return jax.scipy.special.logit((constrained - minimum) / (maximum - minimum))
