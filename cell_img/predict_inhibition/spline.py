"""JAX implementation of M-splines and I-splines."""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def n_basis_fns(k: int, knots: Tuple[float, ...]) -> int:
  """Get the number of spline basis functions.

  Args:
    k: Order of the spline.
    knots: Tuple of knots.

  Returns:
    knot tuple with outermost knots having multiplicity k.
  """
  return len(knots) - k


def set_knot_multiplicity(
    k: int, knots: Tuple[float, ...]) -> Tuple[float, ...]:
  """Set the multiplicity of outermost knots to ensure uniform smoothness.

  Splines of order k need to have outermost knots with multiplicity k to
  preserve smoothness at the boundaries. This utility method increases the
  multiplicity of the outermost knots to k.

  Args:
    k: Order of the spline.
    knots: Tuple of knots with multiplicity 1.

  Returns:
    Tuple of knots with outermost knots having multiplicity k.
  """
  if knots != tuple(sorted(knots)):
    raise ValueError('knots must be non-decreasing')
  return (knots[0],) * k + knots[1:-1] + (knots[-1],) * k


@functools.partial(jax.jit, static_argnames=('k', 'knots'))
def m_spline(x: jnp.array,
             i: int,  # spline i in the basis
             k: int,  # spline order k
             knots: Tuple[float, ...]) -> jnp.array:
  """Evaluate a kth order M-spline.

  M-splines:
  * are non-negative
  * spline i is 0 outside the interval [knot[i], knot[i+k])
  * have k-2 continuous derivatives at the interior knots
  * are normalized to integrate to 1

  Knots should be non-decreasing.
  We should have
    knot[0] = knot[1] = ... = knot[k-1]
    knot[n] = knot[n+1] = ... = knot[n+k-1]
  and
    knot[k-1] < knot[k] < ... < knot[n]
  When we have n+k knots, we will have n basis functions

  See https://en.wikipedia.org/wiki/M-spline

  Args:
    x: Points at which to evaluate the spline
    i: Index in the spline basis
    k: Order of the spline. Order 1 = piecewise constant,
      order 2 = piecewise linear, etc.
    knots: Tuple of knots. For an order k spline, the first
      and last k knots will typically all be equal.

  Returns:
    Value of the given spline at x.
  """
  if knots != tuple(sorted(knots)):
    raise ValueError('knots must be non-decreasing')
  knots_array = jnp.array(knots)

  # Knots at the boundaries have multiplicity > 1, so delta_knot may be 0.
  # The recurrence relation involves division by delta_knot, and although
  # we use a jnp.where to prevent division by 0, we can still end up with
  # NaNs in the gradient because one branch of the where function has infinite
  # values. Using delta_knot_safe doesn't change the return value, but it does
  # prevent NaNs from showing up in the gradient because of the infinity in
  # the non-returned branch. See go/tf-where-nan for details.
  delta_knot = knots_array[i + k] - knots_array[i]
  delta_knot_safe = jnp.where(delta_knot == 0., 1., delta_knot)

  if k == 1:
    return jnp.where(
        (x >= knots_array[i]) & (x < knots_array[i + 1]),
        1. / delta_knot_safe,
        0.)

  return jnp.where(
      delta_knot == 0.,
      0.,
      (k *
       ((x - knots_array[i]) * m_spline(x, i, k - 1, knots) +
        (knots_array[i + k] - x) * m_spline(x, i + 1, k - 1, knots)) /
       ((k - 1) * delta_knot_safe)))


@functools.partial(jax.jit, static_argnames=['k', 'knots'])
def i_spline(x: jnp.array,
             i: int,  # spline i in the basis
             k: int,  # spline order k
             knots: Tuple[float, ...]) -> jnp.array:
  """Evaluate a kth order I-spline.

  I-splines are the integrals of M-splines. Sums of I-splines with positive
  weights are monotonically increasing functions of x.

  Args:
    x: Points at which to evaluate the spline
    i: Index in the spline basis
    k: Order of the spline. Order 1 = piecewise constant,
      order 2 = piecewise linear, etc.
    knots: Array of knots. For an order k spline, the first
      and last k knots will typically all be equal.

  Returns:
    Value of the given spline at x.
  """
  if knots != tuple(sorted(knots)):
    raise ValueError('knots must be non-decreasing')
  knots_array = jnp.array(knots)
  j = jnp.digitize(x, knots_array, right=False) - 1

  def i_spline_term(
      total: jnp.array,
      m_minus_i: int,
  ) -> jnp.array:
    """Single term in the sum used to compute an I-spline."""
    m = m_minus_i + i
    term = jnp.where(
        m > j,
        0.,
        (knots_array[m+k+1] - knots_array[m]) * m_spline(x, m, k+1, knots) /
        (k + 1))
    return total + term, term

  return jnp.where(
      i > j,
      0.,
      jnp.where(
          j - k + 1 > i,
          1.,
          jax.lax.scan(
              i_spline_term,
              jnp.zeros_like(x),
              jnp.arange(0, k, 1))[0]
      )
  )


@functools.partial(jax.jit, static_argnames=['spline_fn', 'k', 'knots'])
def _spline_approx(
    spline_fn: Callable[[jnp.array, int, int, Tuple[float, ...]], jnp.array],
    x: jnp.array,
    k: int,
    knots: Tuple[float, ...],
    coeffs: jnp.array) -> jnp.array:
  """Compute a weighted sum of splines.

  Args:
    spline_fn: function for evaluating a spline (either m_spline or i_spline)
    x: array of points at which to evaluate the weighted sum
    k: order of the splines
    knots: knots for the splines
    coeffs: coefficients for the splines in the weighted sum

  Returns:
    The weighted sum of splines evaluated at x.
  """

  def multiply_and_add(total, coeff_and_i):
    coeff, i = coeff_and_i
    prod = coeff * spline_fn(x, i, k, knots)
    return total + prod, prod

  return jax.lax.scan(
      multiply_and_add,
      jnp.zeros_like(x),
      (jnp.swapaxes(coeffs, axis1=0, axis2=-1),
       jnp.arange(0, len(knots) - k, 1)))[0]


def m_spline_approx(
    x: jnp.array,
    k: int,
    knots: Tuple[float, ...],
    coeffs: jnp.array) -> jnp.array:
  """Compute a weighted sum of M-splines.

  Args:
    x: points at which to evaluate the sum of M-splines
    k: M-spline order
    knots: M-spline knots
    coeffs: coefficients of the M-spline basis functions

  Returns:
    The weighted sum of M-splines, evaluated at x.
  """
  return _spline_approx(
      spline_fn=m_spline,
      x=x,
      k=k,
      knots=knots,
      coeffs=coeffs)


def i_spline_approx(
    x: jnp.array,
    k: int,
    knots: Tuple[float, ...],
    coeffs: jnp.array) -> jnp.array:
  """Compute a weighted sum of I-splines.

  Args:
    x: points at which to evaluate the sum of I-splines
    k: I-spline order
    knots: I-spline knots
    coeffs: coefficients of the I-spline basis functions

  Returns:
    The weighted sum of I-splines, evaluated at x.
  """
  return _spline_approx(
      spline_fn=i_spline,
      x=x,
      k=k,
      knots=knots,
      coeffs=coeffs)
