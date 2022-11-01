"""Functions for using Hotelling's T-squared test to compare cell embeddings.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from cell_img import data_utils

CONTROL_UNINFECTED = 'uninfected_control'
CONTROL_INACTIVE = 'infected_control'
CONTROL_ACTIVE = 'active_control'
SAMPLE = 'sample'


@functools.partial(jax.jit, static_argnames=('shrinkage'))
def get_regularized_covariance(input_matrix: jnp.array,
                               shrinkage: Optional[float] = 0) -> jnp.array:
  """Get the regularized covariance for a matrix given the amount of shrinkage.

  Args:
    input_matrix: the matrix to compute covariance on
    shrinkage: amount of shrinkage used to regularize the covariance

  Returns:
    The regularized covariance matrix
  """
  raw_cov = jnp.cov(input_matrix, rowvar=False)
  return shrink_covariance_matrix(raw_cov, shrinkage)


@functools.partial(jax.jit, static_argnames=('shrinkage'))
def shrink_covariance_matrix(input_covariance_matrix: jnp.array,
                             shrinkage: Optional[float] = 0) -> jnp.array:
  """Shrink a covariance matrix given the amount of shrinkage.

  This follows scikit-learn's implementation from
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html

  Args:
    input_covariance_matrix: the raw covariance matrix to shrink
    shrinkage: amount of shrinkage used to regularize the covariance

  Returns:
    The regularized covariance matrix
  """
  if shrinkage == 0 or not input_covariance_matrix.shape:
    return input_covariance_matrix
  return (1 - shrinkage) * input_covariance_matrix + shrinkage * jnp.trace(
      input_covariance_matrix) / input_covariance_matrix.shape[0] * jnp.eye(
          input_covariance_matrix.shape[0])


@functools.partial(jax.jit, static_argnames=('scale_to_f', 'shrinkage'))
def get_hotelling_t2(emb0: jnp.array,
                     emb1: jnp.array,
                     scale_to_f: bool,
                     shrinkage: Optional[float] = 0) -> jnp.array:
  """Get Hotelling's T-squared statistic for 2 sets of embeddings."""
  n0 = emb0.shape[0]
  mean0 = jnp.mean(emb0, axis=0)
  cov0 = jnp.cov(emb0, rowvar=False)

  n1 = emb1.shape[0]
  mean1 = jnp.mean(emb1, axis=0)
  cov1 = jnp.cov(emb1, rowvar=False)

  delta_mean = mean1 - mean0
  pooled_cov = ((n0 - 1.) * cov0 + (n1 - 1.) * cov1) / (n0 + n1 - 2.)
  if shrinkage:
    pooled_cov = shrink_covariance_matrix(pooled_cov, shrinkage)

  t2 = n0 * n1 / (n0 + n1) * jnp.matmul(
      delta_mean.T, jnp.linalg.solve(pooled_cov, delta_mean))
  if scale_to_f:
    p = emb0.shape[-1]
    return (n0 + n1 - p - 1) / (p * (n0 + n1 - 2)) * t2
  else:
    return t2


@functools.partial(jax.jit, static_argnames=('scale_to_f', 'shrinkage'))
def get_hotelling_t2_wrt_control(n_control_samples: int,
                                 control_mean: jnp.array,
                                 control_eigenvalues: jnp.array,
                                 control_eigenvectors: jnp.array,
                                 emb1: jnp.array,
                                 scale_to_f: bool,
                                 shrinkage: Optional[float] = 0) -> jnp.array:
  """Get Hotelling's T-squared statistic, using precomputed control covariance.

  Args:
    n_control_samples: the number of control embeddings
    control_mean: mean of control wells
    control_eigenvalues: eigenvalues for control covariance matrix
    control_eigenvectors: eigenvectors for control covariance matrix
    emb1: embeddings for the sample well
    scale_to_f: scale the t2 values to match the F distribution
    shrinkage: amount of shrinkage used to regularize the covariance

  Returns:
    Hotelling's t-squared statistic
  """
  n0 = n_control_samples
  n1 = emb1.shape[0]
  mean1 = jnp.mean(emb1, axis=0)
  # cov1 should be diagonal from change of basis
  cov1 = jnp.var(emb1, axis=0)
  cov1 = shrink_covariance_matrix(jnp.diag(cov1), shrinkage)
  delta_mean = jnp.matmul(control_eigenvectors.T, mean1 - control_mean)
  pooled_cov = ((n0 - 1.) * jnp.diag(control_eigenvalues) +
                (n1 - 1.) * cov1) / (n0 + n1 - 2.)
  pooled_cov_inv = jnp.diag(1. / jnp.diag(pooled_cov))

  t2 = n0 * n1 / (n0 + n1) * jnp.matmul(delta_mean.T,
                                        jnp.matmul(pooled_cov_inv, delta_mean))
  if scale_to_f:
    p = emb1.shape[-1]
    return (n0 + n1 - p - 1) / (p * (n0 + n1 - 2)) * t2
  else:
    return t2


def get_controls(emb_df: pd.DataFrame,
                 bad_well_column: Optional[str] = None) -> pd.DataFrame:
  """Get a set of control embeddings to use for the Hotelling test.

  This is a reference population to which we will compare other wells.
  Here we'll want to filter out wells where things went wrong for various
  reasons. The flags for wells to drop are given by bad_well_column.

  Args:
    emb_df: dataframe of embeddings
    bad_well_column: if not None, the wells with bad_well_column values set to
      True are dropped

  Returns:
    A DataFrame of control embeddings.
  """
  control_df = emb_df[emb_df.index.get_level_values('actives') ==
                      CONTROL_INACTIVE].copy()
  if bad_well_column is not None:
    control_df = control_df.query(f'{bad_well_column}==False')
  return control_df


def normalize_wrt_controls(
    emb_df: pd.DataFrame,
    bad_well_column: Optional[str] = None) -> pd.DataFrame:
  """Remove plate effects from embeddings using the plate control means.

  Args:
    emb_df: dataframe of embeddings
    bad_well_column: if not None, the wells with bad_well_column values set to
      True are dropped

  Returns:
    A DataFrame of embeddings.
  """
  control_emb_df = get_controls(emb_df, bad_well_column=bad_well_column)
  emb_df = emb_df.sub(control_emb_df.groupby('plate').mean(), level='plate')
  return emb_df


def get_t2_statistic(
    emb_df: pd.DataFrame,
    bad_well_column: Optional[str] = None,
    shrinkage: Optional[float] = None,
    scale_to_control_emb: Optional[bool] = False,
    t2_threshold_percentile: Optional[float] = 0.95) -> pd.DataFrame:
  """Get Hotelling's T-squared statistic for all wells in the embedding dataframe.

  Args:
    emb_df: dataframe of embeddings
    bad_well_column: if not None, the wells with bad_well_column values set to
      True are dropped
    shrinkage: Amount of regularization used for covariance calculation. If
      None, OAS is used to determine the amount of shrinkage
    scale_to_control_emb: scale the embedding vectors by dividing them by the
      standard deviation of the control embeddings
    t2_threshold_percentile: Cutoff for flagging wells as different from
      controls

  Returns:
    A DataFrame with each well as entries, annotated with t2 values for the well
    and if it's different from control wells
  """
  emb_df = normalize_wrt_controls(emb_df, bad_well_column)
  control_emb_df = get_controls(emb_df, bad_well_column)

  # Use OAS to determine the shrinkage
  if shrinkage is None:
    oas = OAS().fit(control_emb_df.to_numpy())
    shrinkage = oas.shrinkage_
  else:
    oas = None

  n_control_samples = len(control_emb_df)
  if oas:
    control_cov = jnp.array(oas.covariance_)
  else:
    control_cov = get_regularized_covariance(control_emb_df.to_numpy(),
                                             shrinkage)
  control_eigenvalues, control_eigenvectors = jnp.linalg.eigh(control_cov)

  # Change of basis to control eigenvectors. Numpy preserves pandas indices, yay
  emb_df = np.matmul(emb_df, np.array(control_eigenvectors))
  control_emb_df = get_controls(emb_df, bad_well_column)
  if scale_to_control_emb:
    control_std = control_emb_df.groupby('plate').std()
    emb_df = emb_df.div(
        control_std, level='plate')
    control_emb_df = get_controls(emb_df, bad_well_column)
    control_eigenvalues = np.var(control_emb_df.to_numpy(), axis=0)

  control_mean = jnp.mean(control_emb_df.to_numpy(), axis=0)
  other_emb_df = emb_df.query(f'actives!="{CONTROL_INACTIVE}"')

  t2_df = []
  # control wells
  for (compound, concentration, actives, batch, plate, well, n_hepatocytes,
       n_hypnozoites), df_well in control_emb_df.groupby([
           'compound', 'concentration', 'actives', 'batch', 'plate', 'well',
           'n_hepatocytes', 'n_hypnozoites'
       ]):
    # We're going to use the Hotelling T2 statistic for the
    # infected control wells to get a threshold to use for the treatment wells.
    # If df_well represents a control well, exclude it from the
    # control embeddings.
    is_control = ((control_emb_df.index.get_level_values('batch') == batch) &
                  (control_emb_df.index.get_level_values('plate') == plate) &
                  (control_emb_df.index.get_level_values('well') == well))
    control_emb = control_emb_df[~is_control].to_numpy()
    t2 = float(
        get_hotelling_t2(
            jnp.array(control_emb),
            jnp.array(df_well.to_numpy()),
            scale_to_f=True,
            shrinkage=shrinkage))
    t2_df.append({
        'compound': compound,
        'concentration': concentration,
        'actives': actives,
        'is_control': True,
        'batch': batch,
        'plate': plate,
        'well': well,
        'n_hepatocytes': n_hepatocytes,
        'n_hypnozoites': n_hypnozoites,
        't2': t2
    })
  # other wells
  for (compound, concentration, actives, batch, plate, well, n_hepatocytes,
       n_hypnozoites), df_well in other_emb_df.groupby([
           'compound', 'concentration', 'actives', 'batch', 'plate', 'well',
           'n_hepatocytes', 'n_hypnozoites'
       ]):
    t2 = float(
        get_hotelling_t2_wrt_control(
            n_control_samples,
            control_mean,
            control_eigenvalues,
            control_eigenvectors,
            jnp.array(df_well.to_numpy()),
            scale_to_f=True,
            shrinkage=shrinkage))
    t2_df.append({
        'compound': compound,
        'concentration': concentration,
        'actives': actives,
        'is_control': False,
        'batch': batch,
        'plate': plate,
        'well': well,
        'n_hepatocytes': n_hepatocytes,
        'n_hypnozoites': n_hypnozoites,
        't2': t2
    })
  t2_df = pd.DataFrame(t2_df)

  # Flag wells that are different
  t2_control = t2_df['t2'][t2_df['is_control']].to_numpy()
  t2_threshold = np.quantile(t2_control, t2_threshold_percentile)
  t2_df['is_different'] = t2_df.t2.apply(lambda x: x > t2_threshold)

  return t2_df


def get_t2_summary(t2_df: pd.DataFrame, is_low: pd.Series) -> pd.DataFrame:
  """Get a summary of Hotelling's test aggregated by compound and concentration.

  Args:
    t2_df: Dataframe of Hotelling's test results, output of get_t2_statistic
    is_low: whether a well has a low number of cells - when there are only a few
      cells, the test is unreliable.

  Returns:
    pd.Dataframe of results aggregated by treatment
  """
  t2_df = t2_df.copy()
  t2_df['is_low'] = is_low
  t2_summary = []
  for (compound, concentration
      ), df_treatment in t2_df[t2_df['actives'] != CONTROL_INACTIVE].groupby(
          ['compound', 'concentration']):
    n_wells = df_treatment.shape[0]
    mean_hypnozoites = df_treatment['n_hypnozoites'].mean()
    mean_hepatocytes = df_treatment['n_hepatocytes'].mean()
    mean_t2 = df_treatment['t2'].mean()
    n_different = np.sum(df_treatment['is_different'])
    fraction_different = n_different / n_wells
    n_low = np.sum(df_treatment['is_low'])
    fraction_low = n_low / n_wells
    n_different_or_low = np.sum(df_treatment['is_different']
                                | df_treatment['is_low'])
    fraction_different_or_low = n_different_or_low / n_wells
    t2_summary.append({
        'compound': compound,
        'concentration': concentration,
        'mean_hypnozoites': mean_hypnozoites,
        'mean_hepatocytes': mean_hepatocytes,
        'mean_t2': mean_t2,
        'n_wells': n_wells,
        'n_different': n_different,
        'fraction_different': fraction_different,
        'n_low': n_low,
        'fraction_low': fraction_low,
        'n_different_or_low': n_different_or_low,
        'fraction_different_or_low': fraction_different_or_low,
    })
  return pd.DataFrame(t2_summary)


def load_emb_df_from_cloud(filepath,
                           filetype='parquet',
                           cell_type='hypnozoite'):
  """Loads embedding dataframe on cloud and formats it for hotelling's test.

  Args:
    filepath: file path of the embedding dataframe, can include * for glob
    filetype: file type of the saved dataframe on cloud
    cell_type: hypnozoite or hepatocyte

  Returns:
    embedding dataframe as pd.DataFrame
  """
  emb_df = data_utils.read_file_df_from_cloud(filepath, filetype)

  # filter for hypnozoites
  if cell_type == 'hypnozoite':
    if 'parasite_stage_names' not in emb_df.columns or 'parasite_stage_infer' not in emb_df.columns:
      raise ValueError(
          'parasite stage prediction are not in this dataframe. Make sure parasite_stage_names and parasite_stage_infer are in the columns.'
      )
    emb_df['stage'] = [
        stage[index]
        for stage, index in zip(emb_df.parasite_stage_names,
                                emb_df.parasite_stage_infer.apply(np.argmax))
    ]
    emb_df = emb_df.query('stage=="%s"' % cell_type)

  # add counts
  counts = emb_df.groupby(['batch', 'plate', 'well']).count()['site']
  counts.rename(f'ml_{cell_type}', inplace=True)
  emb_df = pd.merge(emb_df, counts, how='outer', on=['batch', 'plate', 'well'])
  emb_df[np.arange(192)] = np.array(
      [list(item) for item in emb_df.embedding.values])
  emb_df[np.arange(192)] = np.array(
      [list(item) for item in emb_df.embedding.values])
  emb_df.drop(
      columns=['parasite_stage_infer', 'parasite_stage_names', 'embedding'],
      inplace=True)
  emb_df.set_index([c for c in emb_df.columns if c not in np.arange(192)],
                   inplace=True)
  return emb_df


# TODO(gowoon): Add the patch visualization code that uses image grid,
# current version uses volumestore
