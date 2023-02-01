"""Construct Adam and BFGS and training loops for a generic objective f.
"""

import functools
import logging
import jax
from jax import flatten_util
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import scipy


def make_repeated_adam(value_and_grad_f,
                       learning_rate=1E-3,
                       b1=0.9,
                       b2=0.999,
                       eps=1E-6):
  opt_init, opt_update, get_params = optimizers.adam(
      learning_rate, b1=b1, b2=b2, eps=eps)

  @jax.jit
  def train_step(step, opt_state):
    params = get_params(opt_state)
    loss_value, grad = value_and_grad_f(params)
    opt_state = opt_update(step, grad, opt_state)
    return opt_state, loss_value

  # For some of these models (especially on accelerators), a single training
  # step runs very quickly. Fusing steps together considerably improves
  # performance.
  @functools.partial(jax.jit, static_argnums=(1,))
  def repeated_train_step(step, repeats, opt_state):

    def step_helper(carray, _):
      step, opt_state, _ = carray
      opt_state, loss_value = train_step(step, opt_state)
      return (step + 1, opt_state, loss_value), None

    (_, opt_state, loss_value), _ = jax.lax.scan(
        step_helper, (step, opt_state, 0.0), xs=None, length=repeats)
    return opt_state, loss_value

  return opt_init, repeated_train_step, get_params


def contains_nans(tree):
  return jax.tree_util.tree_reduce(
      lambda accum, val: accum or jnp.isnan(val).any(),
      tree,
      initializer=False)


def adam_optimize(f,
                  x0,
                  train_steps=10000,
                  learning_rate=1E-3,
                  b1=0.9,
                  b2=0.999,
                  eps=1E-6,
                  fused_train_steps=100,
                  verbose=1):
  """Run an adam training loop to minimize f."""

  value_and_grad_f = jax.jit(jax.value_and_grad(f))
  opt_init, repeated_train_step, get_params = make_repeated_adam(
      value_and_grad_f, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
  if verbose >= 2:
    print(f'x0: {x0}')
    print(f'f(x0): {f(x0)}')
  opt_state = opt_init(x0)
  old_opt_state = opt_state
  for step in range(0, train_steps, fused_train_steps):
    opt_state, loss_value = repeated_train_step(step, fused_train_steps,
                                                opt_state)
    if step % 1000 == 0:
      if verbose >= 1:
        print(f'Loss at step {step} is: {loss_value}.')
        logging.info(f'Loss at step {step} is: {loss_value}.')  # pylint: disable=logging-format-interpolation
    if contains_nans(value_and_grad_f(get_params(opt_state))):
      learning_rate /= 2.
      _, repeated_train_step, get_params = make_repeated_adam(
          value_and_grad_f, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
      print(f'nan encountered. adjusted learning_rate: {learning_rate}')
      opt_state = old_opt_state
    else:
      old_opt_state = opt_state

  x = get_params(opt_state)
  return x


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def jnp_float(x):
  return jnp.asarray(x, dtype=jnp.float32)


def jnp_float_star(val):
  if isinstance(val, tuple):
    return tuple(jnp_float_star(u) for u in val)
  if isinstance(val, list):
    return [jnp_float_star(u) for u in val]
  return jnp_float(val)


def np_float_star(val):
  if isinstance(val, tuple):
    return tuple(np_float_star(u) for u in val)
  if isinstance(val, list):
    return [np_float_star(u) for u in val]
  return np_float(val)


def jnp_to_np_wrap_val_grad(jnp_val_grad_fun, unravel):

  def wrapped(*pargs):
    pargs2 = jnp_float_star(pargs)
    val, grad = np_float_star(
        jnp_val_grad_fun(*((unravel(pargs2[0]),) + pargs2[1:])))
    flat_grad, _ = flatten_util.ravel_pytree(grad)
    return val, np_float(flat_grad)

  return wrapped


def _wrap_minimize(jnp_fun, x0_in, **kwargs):
  x0, unravel = flatten_util.ravel_pytree(x0_in)
  fun = jnp_to_np_wrap_val_grad(jnp_fun, unravel)
  opt1 = scipy.optimize.minimize(fun=fun, x0=np_float(x0), **kwargs)
  opt_status = (opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  x = opt1.x
  x_out = unravel(x)
  return x_out, opt_status, opt1


def lbfgs_optimize(f, x0, max_iter=10000, **kwargs):
  options = {'maxiter': max_iter}
  options.update(kwargs)
  return _wrap_minimize(
      jax.jit(jax.value_and_grad(f)),
      x0,
      jac=True,
      method='L-BFGS-B',  # sometimes line-search failure.
      options=options)
