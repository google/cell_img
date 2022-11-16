"""Tests for negbin."""

from absl.testing import absltest
from cell_img.analysis import negbin
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


class NegbinTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    key = jax.random.PRNGKey(0)
    key, subkey0, subkey1 = jax.random.split(key, num=3)
    self.lams = jnp.exp(jax.random.normal(key=subkey0, shape=(3,)) + 1)
    self.kappas = jnp.exp(jax.random.normal(key=subkey1, shape=(3,)) + 0.5)
    self.key = key

  def test_probs_should_sum_to_1(self):
    for lam in self.lams:
      for kappa in self.kappas:
        nb = negbin.NegBin(lam=lam, kappa=kappa)
        log_probs = []
        for i in range(200):
          log_probs.append(nb.log_prob(i))
        lse = jax.scipy.special.logsumexp(jnp.array(log_probs))
        np.testing.assert_allclose(lse, 0., atol=1.e-5)

  def test_mean(self):
    for lam in self.lams:
      for kappa in self.kappas:
        nb = negbin.NegBin(lam=lam, kappa=kappa)
        expected = 0.
        for i in range(200):
          expected += i * jnp.exp(nb.log_prob(i))
        np.testing.assert_allclose(expected, nb.mean(), atol=1.e-2)

  def test_var(self):
    for lam in self.lams:
      for kappa in self.kappas:
        nb = negbin.NegBin(lam=lam, kappa=kappa)
        expected = 0.
        for i in range(200):
          expected += ((i - nb.mean()) ** 2.) * jnp.exp(nb.log_prob(i))
        np.testing.assert_allclose(expected, nb.var(), rtol=1.e-2)

  def test_log_prob_agrees_with_scipy_nb2(self):
    # Sanity check the implementation by comparing against scipy's NB2.
    # scipy.stats.nbinom is parameterized in terms of n and p:
    # n = mean^2/(var - mean)
    # p = mean/var
    for lam in self.lams:
      for kappa in self.kappas:
        nb = negbin.NegBin(lam=lam, kappa=kappa)
        n = nb.mean() ** 2. / (nb.var() - nb.mean())
        p = nb.mean() / nb.var()
        for y in range(5):
          np.testing.assert_allclose(
              nb.log_prob(y),
              scipy.stats.nbinom.logpmf(k=y, n=n, p=p, loc=0.),
              atol=1.e-5)

  def test_sample(self):
    key = self.key
    for kappa in self.kappas:
      key, subkey1, subkey2 = jax.random.split(key, num=3)
      nb1 = negbin.NegBin(lam=jnp.ones(10000), kappa=kappa)
      s1 = nb1.sample(key=subkey1)
      nb2 = negbin.NegBin(lam=2.*jnp.ones(10000), kappa=kappa)
      s2 = nb2.sample(key=subkey2)
      np.testing.assert_allclose(jnp.mean(s1), nb1.mean()[0], rtol=5.e-2)
      np.testing.assert_allclose(jnp.mean(s2), nb2.mean()[0], rtol=5.e-2)
      np.testing.assert_allclose(jnp.var(s1), nb1.var()[0], rtol=1.e-1)
      np.testing.assert_allclose(jnp.var(s2), nb2.var()[0], rtol=1.e-1)

  def test_sample_with_zeros(self):
    lam = jnp.concatenate(
        [jnp.zeros((10,)), jnp.ones(10000), jnp.zeros((10,))])
    nb = negbin.NegBin(lam=lam, kappa=3.)
    s = nb.sample(key=self.key)
    self.assertTrue(jnp.all(s[:10] == 0.))
    self.assertTrue(jnp.all(s[-10:] == 0.))
    np.testing.assert_allclose(jnp.mean(s[10:-10]), nb.mean()[10], rtol=5.e-2)

if __name__ == '__main__':
  absltest.main()
