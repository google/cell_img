"""Model for joint distribution of appearance scores and counts."""

import dataclasses
from typing import Callable, Iterable, List, Optional, Tuple

from cell_img.analysis import jax_tree
from cell_img.dose_response import constrain
from cell_img.predict_inhibition import spline
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


# default maximum and minimum values of logit inhbition
MIN_LOGIT_INHIBITION = -6.5
MAX_LOGIT_INHIBITION = 6.5
# default maximum spline coefficients for mean and variance models
MAX_MEAN_COEFF = 10.
MAX_VAR_COEFF = 10.


@jax_tree.dataclass_with_constants(
    constant_fields=['spline_order_mean', 'knots_mean',
                     'spline_order_var', 'knots_var',
                     'min_logit_inhibition', 'max_logit_inhibition',
                     'max_mean_coeff', 'max_var_coeff'])
@dataclasses.dataclass(frozen=True)
class InhibitionParams():
  """Parameters for a model of inhibition's affect on appearance and counts."""
  mean_items_unconstrained: jnp.array  # (n_plates)
  overdispersion_unconstrained: jnp.array  # (n_plates)
  inhibition_unconstrained: jnp.array  # (n_treatments-1)
  mean_offset_unconstrained: jnp.array  # (n_plates,)
  mean_coeffs_unconstrained: jnp.array  # (n_hep_lots, n_knots)
  var_offset_unconstrained: jnp.array  # (n_plates,)
  var_coeffs_unconstrained: jnp.array  # (n_hep_lots, n_knots)

  spline_order_mean: int  # order of spline approximating mean(inhibition)
  knots_mean: List[float]  # knots for spline approximating mean(inhibition)
  spline_order_var: int  # order of spline approximating var(inhibition)
  knots_var: List[float]  # knots for spline approximating var(inhibition)
  min_logit_inhibition: float = MIN_LOGIT_INHIBITION
  max_logit_inhibition: float = MAX_LOGIT_INHIBITION
  max_mean_coeff: float = MAX_MEAN_COEFF
  max_var_coeff: float = MAX_VAR_COEFF
  constant_fields: Optional[Iterable[str]] = None

  @property
  def mean_items(self) -> jnp.ndarray:
    return constrain.to_positive(self.mean_items_unconstrained)

  @property
  def overdispersion(self) -> jnp.ndarray:
    return constrain.to_positive(self.overdispersion_unconstrained)

  @property
  def logit_inhibition(self) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.array([self.min_logit_inhibition]),
         constrain.to_bounded(
             unconstrained=self.inhibition_unconstrained,
             minimum=self.min_logit_inhibition,
             maximum=self.max_logit_inhibition)], axis=0)

  @property
  def inhibition(self) -> jnp.ndarray:
    return jax.scipy.special.expit(self.logit_inhibition)

  # We want a monotonic map from true inhibition to mean(score). We use
  # I-splines plus non-negative spline coefficients to obtain monotonicity.
  # I-spline models have a minimum value of 0, but we'll need more flexibility
  # so we add an offset that doesn't need be positive.
  #
  # Another benefit of the division of the model into an offset + coefficients
  # is that we get more flexibility in model fitting. Some small experiments
  # suggested fitting plate level spline models leads to overfitting, but
  # fitting hep lot level spline models may underfit. We can compromise by
  # fitting hep lot level spline coefficients and plate level offsets, and this
  # appears to work better than a strictly plate level or hep lot level model.

  @property
  def mean_offset(self) -> jnp.ndarray:
    # no constraints on the offset
    return self.mean_offset_unconstrained

  @property
  def mean_coeffs(self) -> jnp.ndarray:
    # positive mean coefficients + an i-spline model ensure that the inhibition
    # to mean function will be monotonically increasing
    return constrain.to_bounded(
        self.mean_coeffs_unconstrained, minimum=0., maximum=self.max_mean_coeff)

  @property
  def var_offset(self) -> jnp.ndarray:
    # we want positive variances, so we use a positive offset
    return constrain.to_positive(self.var_offset_unconstrained)

  @property
  def var_coeffs(self) -> jnp.ndarray:
    # we want positive variances so we use positive variances + an m-spline
    # model
    return constrain.to_bounded(
        self.var_coeffs_unconstrained, minimum=0., maximum=self.max_var_coeff)

  def pred_mean(
      self,
      logit_inhibition: jnp.ndarray,
      spline_index: jnp.ndarray,
      offset_index: jnp.ndarray,
      k: int,
      knots: Tuple[float, ...]) -> jnp.ndarray:
    """Model for the mean of the predicted scores as a function of inhibition.

    Args:
      logit_inhibition: values of true inhibition (as logits) at which to
        evaluate the spline model
      spline_index: index for the set of spline coefficients to use
        (in practice we'll use one set of spline coefficients per hep lot)
      offset_index: index for the offset to use (in practice we fit a different
        offset for each plate)
      k: order of the spline model
      knots: knots for the spline model

    Returns:
      The spline model evaluated at the points specified in logit_inhibition.
    """
    return self.mean_offset[offset_index] + spline.i_spline_approx(
        x=logit_inhibition,
        k=k,
        knots=knots,
        coeffs=self.mean_coeffs[spline_index])

  def pred_var(
      self,
      logit_inhibition: jnp.ndarray,
      spline_index: jnp.ndarray,
      offset_index: jnp.ndarray,
      k: int,
      knots: Tuple[float, ...]) -> jnp.ndarray:
    """Model for the variance of predicted scores as a function of inhibition.

    Args:
      logit_inhibition: values of true inhibition (as logits) at which to
        evaluate the spline model
      spline_index: index for the set of spline coefficients to use
        (in practice we'll use one set of spline coefficients per hep lot)
      offset_index: index for the offset to use (in practice we fit a different
        offset for each plate)
      k: order of the spline model
      knots: knots for the spline model

    Returns:
      The spline model evaluated at the points specified in logit_inhibition.
    """
    return self.var_offset[offset_index] + spline.m_spline_approx(
        x=logit_inhibition,
        k=k,
        knots=knots,
        coeffs=self.var_coeffs[spline_index])


def get_log_prob_count(
    df: pd.DataFrame,
    count_column: str,
) -> jnp.ndarray:
  """Get a function returning log probability of a set of counts.

  Args:
    df: a DataFrame with columns 'plate_index', 'treatment_index', and
      a column of counts specified in count_column
    count_column: name of a column of counts

  Returns:
    A function that takes an InhibitionParams and returns the log likelihood
    of the observed counts.
  """
  plate_index = jnp.array(df.plate_index.to_numpy())
  treatment_index = jnp.array(df.treatment_index.to_numpy())
  n_items = jnp.array(df[count_column].to_numpy())

  def log_prob_count(params: InhibitionParams) -> jnp.array:
    """Get the log probability of a set of counts given a set of parameters.

    We assume counts of items (hypnozoites / hepatocytes / etc) have a negative
    binomial (NB1 form) distribution with mean given by params.mean_items
    inhibition given by params.inhibition.

    Args:
      params: an InhibitionParams object with model parameters

    Returns:
      Log probability of df[count_column] for the parameters in params.
    """

    lmbda = (params.mean_items[plate_index] *
             (1. - params.inhibition[treatment_index]))
    kappa = params.overdispersion[plate_index]
    y = n_items

    r = 1. / kappa
    p = r / (lmbda + r)
    return (
        jax.scipy.special.gammaln(y + r) +
        -jax.scipy.special.gammaln(r) +
        -jax.scipy.special.gammaln(1. + y) +
        r * jnp.log(p) + y * jnp.log(1. - p))

  return log_prob_count  # pytype: disable=bad-return-type  # jax-ndarray


def get_normal_log_prob_from_summary_stats(
    n: jnp.ndarray,
    x_sum: jnp.ndarray,
    x_sum_sq: jnp.ndarray,
    mu: jnp.ndarray,
    sigma_sq: jnp.ndarray,
) -> jnp.ndarray:
  return jnp.where(
      n == 0,
      0.,
      -0.5 * ((n * jnp.log(2. * np.pi * sigma_sq) +
               (x_sum_sq - 2. * mu * x_sum + n * (mu**2.)) / sigma_sq)))


def get_log_prob_pred(df: pd.DataFrame,) -> jnp.ndarray:
  """Get a function returning log probability of a set of LightGBM model scores.

  Args:
    df: a DataFrame with columns 'plate_index', 'hep_lot_index',
      'treatment_index' plus summary statistics for a set of predictions
      for a well: 'n_x' (the number of scores), 'sum_x' (the sum of the
      scores), and 'sum_x2' (the sum of the scores squared)

  Returns:
    A function that takes an InhibitionParams and returns the log likelihood
    of the scores for a set of treatments.
  """
  treatment_index = jnp.array(df.treatment_index.to_numpy())
  plate_index = jnp.array(df.plate_index.to_numpy())
  hep_lot_index = jnp.array(df.hep_lot_index.to_numpy())
  spline_index = hep_lot_index
  offset_index = plate_index

  n = jnp.array(df.n_x.fillna(0.).to_numpy())
  x_sum = jnp.array(df.sum_x.fillna(0.).to_numpy())
  x_sum_sq = jnp.array(df.sum_x2.fillna(0.).to_numpy())

  def log_prob_pred(params: InhibitionParams) -> jnp.ndarray:
    """Get the log probability of sets of scores given a set of parameters.

    We assume that the scores for each treatment are normally distributed with
    a mean and variance that are given by functions (modeled with splines) of
    the treatment's "true" inhibition. The mean for a set of scores is given by
    params.pred_mean and the variance by params.pred_var.

    Args:
      params: an InhibitionParams object with model parameters

    Returns:
      Log probability of the scores with summary statistics given by n_x,
      sum_x, and sum_x2 above.
    """
    logit_inhibition = params.logit_inhibition[treatment_index]

    mu = params.pred_mean(
        logit_inhibition,
        spline_index=spline_index,
        offset_index=offset_index,
        k=params.spline_order_mean,
        knots=params.knots_mean,
    )
    sigma_sq = params.pred_var(
        logit_inhibition,
        spline_index=spline_index,
        offset_index=offset_index,
        k=params.spline_order_var,
        knots=params.knots_var,
    )

    return get_normal_log_prob_from_summary_stats(
        n=n,
        x_sum=x_sum,
        x_sum_sq=x_sum_sq,
        mu=mu,
        sigma_sq=sigma_sq,
    )
  return log_prob_pred  # pytype: disable=bad-return-type  # jax-ndarray


def get_objective_count(
    df: pd.DataFrame,
    count_column: str,
) -> Callable[[InhibitionParams], jnp.ndarray]:
  """Get the objective function to maximize for the count model."""
  log_prob_count = get_log_prob_count(df=df, count_column=count_column)

  def _objective_count(params: InhibitionParams) -> jnp.ndarray:
    return jnp.mean(log_prob_count(params))  # pytype: disable=not-callable  # jax-ndarray
  return _objective_count


def get_objective_appearance(
    df: pd.DataFrame,
) -> Callable[[InhibitionParams], jnp.ndarray]:
  """Get the objective function to maximize for the appearance model."""
  log_prob_pred = get_log_prob_pred(df=df)

  def _objective_appearance(params: InhibitionParams) -> jnp.ndarray:
    return jnp.mean(log_prob_pred(params))  # pytype: disable=not-callable  # jax-ndarray
  return _objective_appearance


def get_objective_joint(
    df: pd.DataFrame,
    count_column: str
) -> Callable[[InhibitionParams], jnp.ndarray]:
  """Get the objective function to maximize for the count + appearance model."""
  log_prob_count = get_log_prob_count(df=df, count_column=count_column)
  log_prob_pred = get_log_prob_pred(df=df)

  def _objective_joint(params: InhibitionParams) -> jnp.ndarray:
    return jnp.mean(log_prob_count(params) + log_prob_pred(params))  # pytype: disable=not-callable  # jax-ndarray
  return _objective_joint
