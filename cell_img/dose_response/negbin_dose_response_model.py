"""PyTree for negative binomial dose response model."""

import dataclasses
from typing import Any, Dict, Optional, Set

from cell_img.analysis import jax_tree
from cell_img.analysis import negbin
from cell_img.dose_response import base_count_model
from cell_img.dose_response import constrain
from cell_img.dose_response import observations
import jax
from jax import numpy as jnp


@jax_tree.dataclass_with_constants(
    constant_fields={
        'compound_slope_max',
        'log_midpoint_min',
        'log_midpoint_max',}.union(base_count_model.CONSTANT_FIELDS))
@dataclasses.dataclass(frozen=True)
class NegBinDoseResponseModel(base_count_model.BaseCountModel):
  """Model parameters + likelihood methods."""

  overdispersion_unconstrained: float
  # Note that we don't fit midpoints / slopes for the 3 types of control
  # (uninfected, active, or infected), and the controls are counted in
  # n_compounds, so the shapes below are n_compounds - 3
  log_compound_midpoint_unconstrained: jnp.ndarray  # shape (n_compounds - 3,)
  compound_slope_unconstrained: jnp.ndarray  # shape (n_compounds - 3)

  # fields not optimized
  compound_slope_max: float
  log_midpoint_min: float
  log_midpoint_max: float
  # set of fields that will remain constant during model fitting
  constant_fields: Optional[Set[str]] = None

  def log_compound_midpoint_unconstrained_to_constrained(
      self,
      log_compound_midpoint_unconstrained: jnp.ndarray
  ) -> jnp.ndarray:
    """Map unconstrined midpoint to constrained."""
    return jnp.concatenate(
        [jnp.zeros((3,)),  # dummy values for controls
         constrain.to_bounded(
             log_compound_midpoint_unconstrained,
             minimum=self.log_midpoint_min,
             maximum=self.log_midpoint_max)],
        axis=0)

  @property
  def log_compound_midpoint(self) -> jnp.ndarray:
    """Log compound midpoint (constrained)."""
    return self.log_compound_midpoint_unconstrained_to_constrained(
        self.log_compound_midpoint_unconstrained)

  def compound_slope_unconstrained_to_constrained(
      self,
      compound_slope_unconstrained: jnp.ndarray,
  ) -> jnp.ndarray:
    """Map unconstrined slope to constrained."""
    return jnp.concatenate(
        [jnp.ones((3,)),   # dummy values for controls
         constrain.to_bounded(
             compound_slope_unconstrained,
             minimum=0.,
             maximum=self.compound_slope_max)], axis=0)

  @property
  def compound_slope(self) -> jnp.ndarray:
    """-Slope of the dose response curve at the midpoint (always positive)."""
    return self.compound_slope_unconstrained_to_constrained(
        self.compound_slope_unconstrained)

  def get_inhibition(
      self,
      log_concentration: jnp.ndarray,
      compound_index: jnp.ndarray,
      ) -> jnp.ndarray:
    """Get the fraction of cells cleared by a treatment.

    Args:
      log_concentration: log concentration of the compound in the treatment
      compound_index: index for the compound in question

    Returns:
      An array containing the fraction of cells cleared by a treatment.
      The negative control is assumed to have an inhibition of 0.
    """
    return jnp.where(
        # infected controls all have the same inhibition (0)
        compound_index == observations.Actives.INFECTED_CONTROL,
        self.infected_control_inhibition,
        jnp.where(
            # active controls all have the same inhibition (fitted)
            compound_index == observations.Actives.ACTIVE_CONTROL,
            self.active_control_inhibition,
            jnp.where(
                # uninfected controls all have the same inhibition (set at
                # class construction time)
                compound_index == observations.Actives.UNINFECTED_CONTROL,
                self.uninfected_control_inhibition,
                # sample inhibition is assumed to follow a dose-response
                # curve. As the concentration increases, expit -> 1
                jax.scipy.special.expit(
                    self.compound_slope[compound_index] *
                    (log_concentration -
                     self.log_compound_midpoint[compound_index])))))

  @property
  def overdispersion(self) -> float:
    return constrain.to_positive(self.overdispersion_unconstrained)

  def get_conditional_mean(self, obs: observations.Observations) -> jnp.ndarray:
    """Predict the mean count for each well."""
    return (
        jnp.exp(self.get_nontreatment_effect(obs)) *
        (1. - self.get_inhibition(
            compound_index=obs.compound_index,
            log_concentration=obs.log_concentration)))

  def get_negbin(self, obs: observations.Observations) -> negbin.NegBin:
    """Get a NegBin for simulating data or computing likelihoods."""
    lam = self.get_conditional_mean(obs)
    # linear overdispersion
    kappa = self.overdispersion / lam
    return negbin.NegBin(lam=lam, kappa=kappa)

  def get_log_prob(
      self,
      obs: observations.Observations,
  ) -> jnp.ndarray:
    """Get the log probability for individual wells."""
    nb = self.get_negbin(obs=obs)
    y = getattr(obs, self.count_field)
    return nb.log_prob(y)

  def get_log_prob_sum(
      self,
      obs: observations.Observations,
  ) -> jnp.ndarray:
    """Get the sum of the log probabilities for all wells."""
    return jnp.sum(self.get_log_prob(obs=obs))


def from_random(
    key: jnp.ndarray,
    obs: observations.Observations,
    count_field: str,
    exposure_field: Optional[str],
    uninfected_control_inhibition: float,
    constant: Optional[float] = None,
    constant_params: Optional[Dict[str, Any]] = None,
) -> NegBinDoseResponseModel:
  """Generate model with random initial parameters.

  Args:
    key: random key
    obs: Observations for which to build a model
    count_field: field in obs that contains counts being modeled
    exposure_field: optional field in obs containing an exposure variable.
      If None, no exposure term will be used
    uninfected_control_inhibition: inhibition value for uninfected wells
    constant: initial value for the model's constant. Good values speed
      convergence considerably. Useful starting points are log(mean(count))
      for models with no exposure term or log(mean(count/exposure)) for models
      with exposure terms.
    constant_params: Dict containing names of constant fields as keys and
      constant values as values.

  Returns:
    A random CountModel.
  """

  key, subkey = jax.random.split(key)
  params = base_count_model.params_from_random(
      key=subkey,
      obs=obs,
      exposure_field=exposure_field,
      uninfected_control_inhibition=uninfected_control_inhibition,
      constant=constant)

  key, subkey0, subkey1 = jax.random.split(key, 3)
  n_compounds = len(obs.sorted_compounds)
  # shape (n_compounds - 3 types of control,)
  log_compound_midpoint_unconstrained = 0.1 * jax.random.normal(
      key=subkey0, shape=(n_compounds - 3,))
  compound_slope_unconstrained = 0.1 * jax.random.normal(
      key=subkey1, shape=(n_compounds - 3,))

  # linear overdispersion term coefficient
  key, subkey = jax.random.split(key)
  overdispersion_unconstrained = jax.random.normal(
      key=subkey, shape=()) - 1.

  params.update({
      'count_field': count_field,
      'overdispersion_unconstrained':
          overdispersion_unconstrained,
      'log_compound_midpoint_unconstrained':
          log_compound_midpoint_unconstrained,
      'compound_slope_unconstrained':
          compound_slope_unconstrained,
  })
  if constant_params is not None:
    params.update(constant_params)
    params['constant_fields'] = set(constant_params.keys())
  return NegBinDoseResponseModel(**params)
