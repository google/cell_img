"""DataClass that holds a single batch of counts and covariates.

A few assumptions:
* The data set contains infected controls, active controls, uninfected controls,
  and samples. These are indicated by an "actives" column. The values of the
  actives column are encoded in the Actives enum.
* Controls get special handling: we order compounds and treatments so that
  controls are first, and we assume that all active controls have the same
  compound and concentration.
* A row flagged as a sample is treated differently than a control even if the
  sample row has the same compound / concentration as a control.
"""

import dataclasses
import enum
import string
from typing import Any, Dict, List, Optional, Tuple

from cell_img.common import plate_utils
from jax import numpy as jnp
import numpy as np
import pandas as pd


class Actives(enum.IntEnum):
  INFECTED_CONTROL = 0
  ACTIVE_CONTROL = 1
  UNINFECTED_CONTROL = 2
  SAMPLE = 3


def actives_to_str(actives: Actives) -> str:
  return actives.name.lower()


def str_to_actives(s: str) -> Actives:
  return Actives[s.upper()]


@dataclasses.dataclass(frozen=True)
class Observations:
  """Dataclass containing counts and covariates."""
  # covariates
  plate_index: jnp.ndarray
  well_row: jnp.ndarray
  well_col: jnp.ndarray
  compound_index: jnp.ndarray
  treatment_index: jnp.ndarray
  log_concentration: jnp.ndarray
  # counts
  n_artifact: Optional[jnp.ndarray]
  n_hypnozoite: jnp.ndarray
  n_schizont: Optional[jnp.ndarray]
  n_hepatocyte: jnp.ndarray

  # values for converting Observations back to a dataframe
  sorted_plates: List[str]
  well_row_min: int
  well_row_max: int
  well_col_min: int
  well_col_max: int
  sorted_compounds: List[str]
  sorted_treatments: List[Tuple[str, float]]
  # preserve the compound / concentrations for controls
  control_compound_map: Dict[Actives, str]
  # Any allows for special concentrations (e.g. 0, -1, np.nan, etc)
  control_concentration_map: Dict[Actives, Any]


# default column names
PLATE_COL = 'plate'
WELL_COL = 'well'
ACTIVES_COL = 'actives'
COMPOUND_COL = 'blinded_concept'
CONCENTRATION_COL = 'concentration'


def from_dataframe(
    df: pd.DataFrame,
    well_row_min: int,
    well_row_max: int,
    well_col_min: int,
    well_col_max: int,
    hypnozoite_col: str,
    hepatocyte_col: str,
    artifact_col: Optional[str] = None,
    schizont_col: Optional[str] = None,
    plate_col: str = PLATE_COL,
    well_col: str = WELL_COL,
    actives_col: str = ACTIVES_COL,
    compound_col: str = COMPOUND_COL,
    concentration_col: str = CONCENTRATION_COL,
    ) -> Observations:
  """Create an Observations from a DataFrame.

  Args:
    df: DataFrame from which to obtain values
    well_row_min: minimum well row
    well_row_max: maximum well row
    well_col_min: minimum well column
    well_col_max: maximum well column
    hypnozoite_col: column containing hypnozoite counts
    hepatocyte_col: column containing hepatocyte counts
    artifact_col: column containing artifact counts
    schizont_col: column containing schizont counts
    plate_col: column containing plate names
    well_col: column containing wells
    actives_col: column containing actives information
    compound_col: column containing compound
    concentration_col: column containing concentration

  Returns:
    An Observations object.
  """

  # Map plates to integers
  sorted_plates = sorted(set(df[plate_col]))
  plate_index = []
  for plate in df[plate_col]:
    plate_index.append(sorted_plates.index(plate))
  plate_index = jnp.array(plate_index)

  # Map well row and column to floats between -1 and 1
  well_row_float = []
  well_col_float = []
  for well in df[well_col]:
    row_int = plate_utils.get_row_int(well)
    col_int = plate_utils.get_column_int(well)
    well_row_float.append(
        2. * (row_int - well_row_min) / (well_row_max - well_row_min) - 1.)
    well_col_float.append(
        2. * (col_int - well_col_min) / (well_col_max - well_col_min) - 1.)
  well_row_float = jnp.array(well_row_float)
  well_col_float = jnp.array(well_col_float)

  # Map compounds and treatments to integers
  control_compound = {}
  control_concentration = {}
  treatments = []
  for actives, compound, concentration in zip(
      df[actives_col],
      df[compound_col],
      df[concentration_col]):
    actives_enum = str_to_actives(actives)
    if actives_enum == Actives.SAMPLE:
      treatments.append((compound, concentration))
    else:
      if actives_enum in control_compound:
        if compound != control_compound[actives_enum]:
          raise ValueError(
              f'Multiple compounds for {actives_enum.name}:'
              f'{compound} != {control_compound[actives_enum]}')
      control_compound[actives_enum] = compound
      if actives_enum in control_concentration:
        if concentration != control_concentration[actives_enum]:
          raise ValueError(
              f'Multiple concentrations for {actives_enum.name}:'
              f'{concentration} != {control_concentration[actives_enum]}')
      control_concentration[actives_enum] = concentration
      actives_str = actives_to_str(actives_enum)
      treatments.append((actives_str, 1.))  # use dummy conc to prevent NaNs
  log_concentrations = jnp.log(jnp.array([t[1] for t in treatments]))

  compounds = [t[0] for t in treatments]
  sorted_compounds = sorted(set(compounds))
  sorted_treatments = sorted(set(treatments))

  # the initial compounds will *always* be the controls (even if they aren't
  # present in the input data)
  for actives_enum in reversed(list(Actives)):
    if actives_enum == Actives.SAMPLE:
      continue

    actives_str = actives_to_str(actives_enum)
    if actives_str in sorted_compounds:
      sorted_compounds.remove(actives_str)
    sorted_compounds = [actives_str] + sorted_compounds

    treatment = (actives_str, 0.)
    if treatment in sorted_treatments:
      sorted_treatments.remove(treatment)
    sorted_treatments = [treatment] + sorted_treatments

  # map compound -> compound_index
  compound_to_index = {c: i for i, c in enumerate(sorted_compounds)}
  compound_index = []
  for compound in compounds:
    compound_index.append(compound_to_index[compound])
  compound_index = jnp.array(compound_index)

  # map treatment -> treatment_index
  treatment_to_index = {t: i for i, t in enumerate(sorted_treatments)}
  treatment_index = []
  for treatment in treatments:
    treatment_index.append(treatment_to_index[treatment])
  treatment_index = jnp.array(treatment_index)

  n_hypnozoite = jnp.array(df[hypnozoite_col].to_numpy().astype(np.float32))
  n_hepatocyte = jnp.array(df[hepatocyte_col].to_numpy().astype(np.float32))
  if artifact_col:
    n_artifact = jnp.array(df[artifact_col].to_numpy().astype(np.float32))
  else:
    n_artifact = None
  if schizont_col:
    n_schizont = jnp.array(df[schizont_col].to_numpy().astype(np.float32))
  else:
    n_schizont = None

  return Observations(
      plate_index=plate_index,
      well_row=well_row_float,
      well_col=well_col_float,
      compound_index=compound_index,
      treatment_index=treatment_index,
      log_concentration=log_concentrations,
      n_hypnozoite=n_hypnozoite,
      n_hepatocyte=n_hepatocyte,
      n_artifact=n_artifact,
      n_schizont=n_schizont,
      sorted_plates=sorted_plates,
      well_row_min=well_row_min,
      well_row_max=well_row_max,
      well_col_min=well_col_min,
      well_col_max=well_col_max,
      control_compound_map=control_compound,
      control_concentration_map=control_concentration,
      sorted_compounds=sorted_compounds,
      sorted_treatments=sorted_treatments,)


def row_int_to_str(row_int: int) -> str:
  """Convert a row integer to a well row string."""
  row = []
  while row_int > 0:
    row_int -= 1
    row.append(string.ascii_uppercase[row_int % 26])
    row_int //= 26
  row.reverse()
  return ''.join(row)


def to_dataframe(
    obs: Observations,
    hypnozoite_col: str,
    hepatocyte_col: str,
    artifact_col: Optional[str] = None,
    schizont_col: Optional[str] = None,
    plate_col: str = PLATE_COL,
    well_col: str = WELL_COL,
    actives_col: str = ACTIVES_COL,
    compound_col: str = COMPOUND_COL,
    concentration_col: str = CONCENTRATION_COL) -> pd.DataFrame:
  """Create a DataFrame from an Observations.

  Args:
    obs: Observations from which to obtain values
    hypnozoite_col: column containing hypnozoite counts
    hepatocyte_col: column containing hepatocyte counts
    artifact_col: column containing artifact counts
    schizont_col: column containing schizont counts
    plate_col: column containing plate names
    well_col: column containing wells
    actives_col: column containing actives information
    compound_col: column containing compound
    concentration_col: column containing concentration

  Returns:
    A DataFrame.
  """
  n_hypnozoite = np.array(obs.n_hypnozoite)
  n_hepatocyte = np.array(obs.n_hepatocyte)

  plate = [obs.sorted_plates[pi] for pi in obs.plate_index]

  well = []
  for row, col in zip(obs.well_row, obs.well_col):
    row_int = int(round(
        obs.well_row_min +
        0.5 * (row + 1.) * (obs.well_row_max - obs.well_row_min)))
    row_str = row_int_to_str(row_int)
    col_int = int(round(
        obs.well_col_min +
        0.5 * (col + 1.) * (obs.well_col_max - obs.well_col_min)))
    col_str = str(col_int)
    well.append(row_str + col_str)

  compound = []
  actives = []
  concentration = np.exp(np.array(obs.log_concentration))
  for i, ci in enumerate(obs.compound_index.tolist()):
    if ci < len(Actives) and Actives(ci) != Actives.SAMPLE:
      actives_enum = Actives(ci)
      compound.append(obs.control_compound_map[actives_enum])
      # overwrite concentration for the controls (handles None / NA values)
      concentration[i] = obs.control_concentration_map[actives_enum]
      actives.append(actives_to_str(actives_enum))
    else:
      compound.append(obs.sorted_compounds[ci])
      actives.append(actives_to_str(Actives.SAMPLE))

  cols = {
      hypnozoite_col: n_hypnozoite,
      hepatocyte_col: n_hepatocyte,
      plate_col: plate,
      well_col: well,
      actives_col: actives,
      compound_col: compound,
      concentration_col: concentration,
  }
  if artifact_col:
    cols[artifact_col] = np.array(obs.n_artifact)
  if schizont_col:
    cols[schizont_col] = np.array(obs.n_schizont)
  return pd.DataFrame(cols)
