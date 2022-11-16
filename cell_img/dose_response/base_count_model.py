"""PyTree for model parameters + methods for computing model components."""

import dataclasses
from typing import Any, Dict, Optional

from cell_img.dose_response import constrain
from cell_img.dose_response import observations
import jax
from jax import numpy as jnp


CONSTANT_FIELDS = frozenset([
    'count_field',
    'well_row_max', 'well_row_sq_max',
    'well_col_max', 'well_col_sq_max',
    'exposure_field', 'uninfected_control_inhibition'
])


@dataclasses.dataclass(frozen=True)
class BaseCountModel:
  """Model parameters."""

  constant: float

  plate_effect_unconstrained: jnp.ndarray  # shape (n_plates - 1,)

  well_row_unconstrained: jnp.array
  well_row_sq_unconstrained: jnp.array
  well_col_unconstrained: jnp.array
  well_col_sq_unconstrained: jnp.array

  # We're assuming that controls get special handling. In particular, all
  # active control wells are pooled to give a single estimate for active
  # control inhibition. 'sample' wells in model subclasses may be treated
  # differently, e.g. for a dose-response plate, we may estimate parameters of
  # a dose-response curve that we can't do for the active controls because
  # we only have the active controls at a single concentration.
  active_control_inhibition_unconstrained: float

  # static fields
  count_field: str  # Observations field used for counts
  well_row_max: float
  well_row_sq_max: float
  well_col_max: float
  well_col_sq_max: float
  # As with the active controls, we treat uninfected controls separately.
  # Inhibition will typically be 1 for hypnozoites and 0 for hepatocytes.
  uninfected_control_inhibition: float
  exposure_field: Optional[str]  # Observations field used as exposure term

  @property
  def plate_effect(self) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.zeros((1,)), self.plate_effect_unconstrained], axis=0)

  @property
  def infected_control_inhibition(self) -> float:
    """Inhibition for infected controls."""
    return 0.

  @property
  def active_control_inhibition(self) -> float:
    """Inhibition for active controls."""
    return jax.scipy.special.expit(self.active_control_inhibition_unconstrained)

  @property
  def well_row(self) -> jnp.ndarray:
    return constrain.to_bounded(
        self.well_row_unconstrained,
        minimum=-self.well_row_max,
        maximum=self.well_row_max)

  @property
  def well_row_sq(self) -> jnp.ndarray:
    return constrain.to_bounded(
        self.well_row_sq_unconstrained,
        minimum=-self.well_row_sq_max,
        maximum=self.well_row_sq_max)

  @property
  def well_col(self) -> jnp.ndarray:
    return constrain.to_bounded(
        self.well_col_unconstrained,
        minimum=-self.well_col_max,
        maximum=self.well_col_max)

  @property
  def well_col_sq(self) -> jnp.ndarray:
    return constrain.to_bounded(
        self.well_col_sq_unconstrained,
        minimum=-self.well_col_sq_max,
        maximum=self.well_col_sq_max)

  def get_location_effect(
      self,
      obs: observations.Observations,
  ) -> jnp.ndarray:
    """Get the location effect in a count model.

    Args:
      obs: observations

    Returns:
      An array of location effects.
    """
    return (
        self.plate_effect[obs.plate_index] +
        self.well_row[obs.plate_index] * obs.well_row +
        self.well_row_sq[obs.plate_index] * obs.well_row ** 2. +
        self.well_col[obs.plate_index] * obs.well_col +
        self.well_col_sq[obs.plate_index] * obs.well_col ** 2.)

  def get_exposure(
      self,
      obs: observations.Observations,
  ) -> jnp.ndarray:
    """Get the exposure effect (if used)."""
    if self.exposure_field:
      return jnp.log(getattr(obs, self.exposure_field))
    return 0.

  def get_nontreatment_effect(
      self,
      obs: observations.Observations
  ) -> jnp.ndarray:
    """Get the conditional mean for each well."""
    return (self.constant +
            self.get_location_effect(obs) +
            self.get_exposure(obs))


def params_from_random(
    key: jnp.ndarray,
    obs: observations.Observations,
    exposure_field: Optional[str],
    uninfected_control_inhibition: float,
    constant: Optional[float] = None,
) -> Dict[str, Any]:
  """Generate model with random initial parameters.

  Args:
    key: random key
    obs: Observations for which to build a model
    exposure_field: optional field in obs containing an exposure variable.
      If None, no exposure term will be used
    uninfected_control_inhibition: inhibition value for uninfected wells
    constant: initial value for the model's constant. Good values speed
      convergence considerably. Useful starting points are log(mean(count))
      for models with no exposure term or log(mean(count/exposure)) for models
      with exposure terms.

  Returns:
    A dict containing parameters for a BaseCountModel.
  """
  if constant is None:
    key, subkey = jax.random.split(key)
    constant = jax.random.normal(key=subkey)

  # start with small effects to keep things from blowing up
  n_plates = jnp.max(obs.plate_index) + 1
  key, subkey = jax.random.split(key)
  plate_effect_unconstrained = 0.1 * jax.random.normal(
      key=subkey, shape=(n_plates - 1,))

  key, subkey = jax.random.split(key)
  (well_row_unconstrained, well_row_sq_unconstrained,
   well_col_unconstrained, well_col_sq_unconstrained) = 0.1 * (
       jax.random.normal(key=subkey, shape=(4, n_plates)))

  key, subkey = jax.random.split(key)
  active_control_inhibition_unconstrained = jax.random.normal(key=subkey)

  params = {
      'constant': constant,
      'plate_effect_unconstrained': plate_effect_unconstrained,
      'well_row_unconstrained': well_row_unconstrained,
      'well_col_unconstrained': well_col_unconstrained,
      'well_row_sq_unconstrained': well_row_sq_unconstrained,
      'well_col_sq_unconstrained': well_col_sq_unconstrained,
      'active_control_inhibition_unconstrained':
          active_control_inhibition_unconstrained,
      'exposure_field': exposure_field,
      'uninfected_control_inhibition': uninfected_control_inhibition,
  }
  return params
