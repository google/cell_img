r"""Methods related to the negative binomial distribution.

We use the negative binomial in count regressions because it gives us greater
control in the variance than a Poisson regression.

* The Poisson distribution has mean $\lambda$ and variance $\lambda$.
* The standard negative binomial distribution (NB2) adds an overdispersion
term to the variance. We have mean $\lambda$ and variance $\lambda + \kappa
\lambda^2$.
"""

import jax
import jax.numpy as jnp
import numpy as np


class NegBin:
  """Implementation of the negative binomial distribution."""

  def __init__(self, lam: jnp.ndarray, kappa: jnp.ndarray):
    """Constructor.

    Args:
      lam: the conditional mean for the distribution
      kappa: the amount of overdispersion - the conditional variance of the
        distribution is lam + kappa lam^2

    Returns:
      a NegBin object.
    """
    kappa = jnp.broadcast_to(kappa, np.shape(lam))
    self.lam = lam
    self.kappa = kappa

  def mean(self) -> jnp.ndarray:
    """Get the mean of the distribution."""
    return self.lam

  def var(self) -> jnp.ndarray:
    """Get the variance of the distribution."""
    return self.lam + self.kappa * (self.lam ** 2.)

  def log_prob(self, y: jnp.ndarray) -> jnp.ndarray:
    """Get the log probability of observing a count of y.

    See Cameron & Trivedi, eq (7) with r=nu_i and lam=phi_i for details.
    https://www.jstor.org/stable/2096536?seq=5#metadata_info_tab_contents

    Args:
      y: array of observed counts

    Returns:
      The log likelihood of y.
    """
    r = 1. / self.kappa
    p = r / (self.lam + r)
    return (
        jax.scipy.special.gammaln(y + r) +
        -jax.scipy.special.gammaln(r) +
        -jax.scipy.special.gammaln(1. + y) +
        r * jnp.log(p) + y * jnp.log(1. - p))

  def sample(self, key: jnp.ndarray) -> jnp.ndarray:
    """Generate negative binomial samples."""
    # Following Cameron & Trivedi, we add dispersion to a Poisson random
    # variable by adding randomness to the rate parameter. Instead of using
    # rate lambda, we use a gamma distributed variable with mean lambda.

    # filter out zero means to avoid infinities
    idx_nonzero = jnp.nonzero(self.lam)  # get the indices of nonzero values
    lam = self.lam[idx_nonzero]  # filter to nonzeros
    kappa = self.kappa[idx_nonzero]
    subkey1, subkey2 = jax.random.split(key)
    shape = lam.shape
    g = jax.random.gamma(key=subkey1, a=1./kappa, shape=shape) * kappa * lam
    rnd = jax.random.poisson(key=subkey2, lam=g, shape=shape)
    output = jnp.zeros_like(self.lam)
    return output.at[idx_nonzero].set(rnd)  # fill in values with lambda != 0
