"""Tests for joint_model."""
from absl.testing import absltest
from cell_img.predict_inhibition import joint_model
import jax
import jax.numpy as jnp
import numpy.testing


class JointModelTest(absltest.TestCase):

  def test_get_normal_log_prob_from_summary_stats(self):
    key = jax.random.PRNGKey(0)
    mu = 3.
    sigma = 2.
    sigma_sq = sigma ** 2.
    n = 20
    x = mu + sigma * jax.random.normal(key=key, shape=(n,))
    expected = jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=mu, scale=sigma))

    x_sum = jnp.sum(x)
    x_sum_sq = jnp.sum(x**2.)
    actual = joint_model.get_normal_log_prob_from_summary_stats(
        n=n, x_sum=x_sum, x_sum_sq=x_sum_sq, mu=mu, sigma_sq=sigma_sq)
    numpy.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
  absltest.main()
