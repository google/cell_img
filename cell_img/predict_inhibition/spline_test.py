"""Tests for spline."""
from absl.testing import absltest
from cell_img.predict_inhibition import spline
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy.testing


class SplineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    # knots with multiplicity 1
    self.knots_mult_1 = (0., 0.2, 0.4, 0.6, 0.8, 1.)

  def test_m_spline_is_non_negative(self):
    x = jax.random.uniform(key=self.key, shape=(100,))
    for k in range(1, 4):
      knots = spline.set_knot_multiplicity(k, self.knots_mult_1)
      for i in range(spline.n_basis_fns(k=k, knots=knots)):
        values = spline.m_spline(x, i, k, knots)
        self.assertTrue(jnp.all(values >= 0.))

  def test_m_spline_is_0_outside_interval(self):
    # spline i is 0 outside the interval [knot[i], knot[i+k])
    x = jax.random.uniform(key=self.key, shape=(100,))
    for k in range(1, 4):
      knots = spline.set_knot_multiplicity(k, self.knots_mult_1)
      for i in range(spline.n_basis_fns(k=k, knots=knots)):
        values = spline.m_spline(x, i, k, knots)
        self.assertTrue(
            jnp.all(
                values[x < knots[i]] == 0.))
        self.assertTrue(
            jnp.all(
                values[x >= knots[i + k]] == 0.))

  def test_m_spline_integrates_to_1(self):
    x = jnp.linspace(0., 1., num=1000)
    for k in range(1, 4):
      knots = spline.set_knot_multiplicity(k, self.knots_mult_1)
      for i in range(spline.n_basis_fns(k=k, knots=knots)):
        y = spline.m_spline(x, i, k, knots)
        integral = jsp.integrate.trapezoid(y, x)
        self.assertAlmostEqual(integral, 1., delta=0.01)

  def test_m_spline_fails_with_nonincreasing_knots(self):
    knots = (0., 0.4, 0.2, 0.6)
    x = jnp.linspace(0., 1., num=10)
    with self.assertRaises(ValueError):
      spline.m_spline(x, 0, 1, knots)

  def test_i_spline_is_integral_of_m_spline(self):
    endpoints = jnp.array([0.3, 0.8])
    x = jnp.linspace(0., 1., num=1000)
    for k in range(1, 3):
      knots = spline.set_knot_multiplicity(k, self.knots_mult_1)
      for i in range(spline.n_basis_fns(k=k, knots=knots)):
        y = spline.m_spline(x, i, k, knots)
        i_spline_values = spline.i_spline(endpoints, i, k, knots)
        for endpoint, value in zip(endpoints, i_spline_values):
          # integrate corresponding m_spline from 0 to endpoint
          integral = jsp.integrate.trapezoid(y[x <= endpoint], x[x <= endpoint])
          self.assertAlmostEqual(integral, value, delta=0.01)

  def test_i_spline_fails_with_nonincreasing_knots(self):
    knots = (0., 0.4, 0.2, 0.6)
    x = jnp.linspace(0., 1., num=10)
    with self.assertRaises(ValueError):
      spline.i_spline(x, 0, 1, knots)

  def test_m_spline_approx(self):
    key, subkey = jax.random.split(self.key)
    # points at which to compare
    x = jax.random.uniform(key=subkey, shape=(5,))

    for k in range(1, 3):
      knots = spline.set_knot_multiplicity(k=k, knots=self.knots_mult_1)
      n = spline.n_basis_fns(k=k, knots=knots)
      key, subkey = jax.random.split(key)
      coeffs = jax.random.normal(key=subkey, shape=(n,))

      actual = spline.m_spline_approx(x, k=k, knots=knots, coeffs=coeffs)
      expected = jnp.zeros_like(x)
      for i in range(n):
        expected += coeffs[i] * spline.m_spline(x, i, k, knots)
      numpy.testing.assert_allclose(actual, expected, rtol=1.e-6)

  def test_i_spline_approx(self):
    key, subkey = jax.random.split(self.key)
    # points at which to compare
    x = jax.random.uniform(key=subkey, shape=(5,))

    for k in range(1, 3):
      knots = spline.set_knot_multiplicity(k=k, knots=self.knots_mult_1)
      n = spline.n_basis_fns(k=k, knots=knots)
      key, subkey = jax.random.split(key)
      coeffs = jax.random.normal(key=subkey, shape=(n,))

      actual = spline.i_spline_approx(x, k=k, knots=knots, coeffs=coeffs)
      expected = jnp.zeros_like(x)
      for i in range(n):
        expected += coeffs[i] * spline.i_spline(x, i, k, knots)
      numpy.testing.assert_allclose(actual, expected, rtol=1.e-6)

if __name__ == '__main__':
  absltest.main()
