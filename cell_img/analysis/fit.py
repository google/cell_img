"""Methods for maximumizing likelihood functions."""

from typing import Callable, Optional, Tuple, TypeVar

from epi_forecast_stat_mech import optim_lib
import jax
import jax.numpy as jnp


T = TypeVar('T')


def maximize_likelihood(
    loglik_fn: Callable[[T], jnp.ndarray],
    params: T,
    learning_rate: float = 1.e-2,
    steps: int = 50000,
    verbose: bool = True
) -> T:
  """Fit parameters by maximizing a likelihood function.

  Args:
    loglik_fn: likelihood function to maximize
    params: initial parameters to refine
    learning_rate: learning rate for the Adam optimizer
    steps: number of training steps
    verbose: if True, print diagnostics

  Returns:
    Fitted ModelParams
  """
  init_vector, vector_to_pytree = jax.flatten_util.ravel_pytree(params)

  @jax.jit
  def loglik_vector(params_vector):
    return -loglik_fn(vector_to_pytree(params_vector))

  # Do an initial fit using BFGS
  xhat, opt_status, _ = optim_lib.lbfgs_optimize(
      loglik_vector,
      init_vector,
      maxcor=100,
      maxfun=20000,
      maxiter=20000,
      maxls=20000,)
  # Deal with BFGS not converging
  if not opt_status[0]:
    if verbose:
      print('BFGS failed to converge', opt_status)
    loglik_init = loglik_vector(init_vector)
    loglik_xhat = loglik_vector(xhat)
    # if BFGS estimate is not finite or worse than starting point,
    # go back to starting point
    if not jnp.isfinite(loglik_xhat) or loglik_xhat > loglik_init:
      xhat = init_vector
  # Refine BFGS estimate
  xhat2 = optim_lib.adam_optimize(
      loglik_vector,
      xhat,
      learning_rate=learning_rate,
      train_steps=steps,
      verbose=int(verbose))

  return vector_to_pytree(xhat2)


def eigval_eigvec_prod(
    eigval: jnp.ndarray,
    q: jnp.ndarray,
) -> jnp.ndarray:
  """Compute q eigval q^T."""
  qt = jnp.transpose(q)
  return jnp.matmul(q, jnp.matmul(jnp.diag(eigval), qt))


def get_mle_stderr(
    loglik_fn: Callable[[T], jnp.ndarray],
    mle_pytree: T,
    epsilon: Optional[float] = None
) -> Tuple[T, jnp.ndarray]:
  """Get an approximate standard error for a maximum likelihood estimate.

  Args:
    loglik_fn: log likelihood function that takes a pytree argument
    mle_pytree: pytree containing the maximum of the likelihood function
    epsilon: weight to use when regularizing the Hessian to ensure it is
      positive definite. If None, we use max(-2 x the smallest eigenvalue, 0)

  Returns:
    A tuple containing an estimate of the standard error for the MLE based on
    the observed Fisher information and the eigenvalues of the -Hessian of the
    log likelihood function evaluated at the MLE. Ideally the smallest
    eigenvalue should be positive (see discussion below); if it is 0 or
    negative, we will need some kind of regularization to obtain reasonable
    variance estimates. The eigenvalues can be a useful diagnostic.
  """
  # flatten the MLE to a vector and get a function for inverting the flattening
  mle_vector, vector_to_pytree = jax.flatten_util.ravel_pytree(mle_pytree)

  def loglik_vector(params_vector):
    """Wrapped loglik_fn that takes a vector argument rather than a pytree."""
    return loglik_fn(vector_to_pytree(params_vector))

  # If we use jacfwd / jacrev to compute the Hessian of a function with a pytree
  # argument, we get a Hessian as a pytree of pytrees. Flattening this object
  # is tricky! Instead we wrap the function so that it takes a vector-valued
  # argument so that when we compute the Hessian, we get a square matrix.

  # Get the Hessian of the wrapped log likelihood function evaluated at the MLE
  hessian = jax.jacfwd(jax.jacrev(loglik_vector))(mle_vector)

  # Invert the negative Hessian to get the Fisher information
  eigvals, q = jnp.linalg.eigh(-hessian)
  # At a local maximum, the Hessian should be strictly negative definite, so the
  # eigenvalues of the -Hessian should all be positive. In practice, they won't
  # be because of a combination of numerical errors and the fact that we only
  # have an approximate maximum. The result can be pathologies such as negative
  # variance estimates.
  #
  # Our workaround is to regularize the Hessian by adding a weighted identity
  # matrix to ensure that it is positive definite. This may result in
  # underestimates of the variance. Resampling approaches may provide a better
  # (but computationally more expensive) alternative.
  if epsilon is None:
    epsilon = max(-2. * eigvals[0], 0.)
  elif epsilon < -eigvals[0]:
    raise ValueError(
        f'Insufficient regularization; use epsilon > {-eigvals[0]}')

  # compute regularize eigenvalues
  eigvals_reg = eigvals + epsilon
  # The observed Fisher information is the inverse of the Hessian and is an
  # asymptotic approximation to the covariance of the MLE.
  observed_fisher_information = eigval_eigvec_prod(1./eigvals_reg, q)
  # Use the square root of the variance as the standard error.
  return (vector_to_pytree(jnp.sqrt(jnp.diag(observed_fisher_information))),
          eigvals)
