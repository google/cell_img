"""pinv_regression and residualization.

These functions build blocks (i.e. DataFrame's embodying one logical component
of an overall design matrix) and string them together into lists to
embody full linear regression designs. The structuring helps later, if one
wants to inspect the fitted coefficients group by group or if one wants to
residualize some data w.r.t. some blocks and not others. Helpers to build
blocks from the MultiIndex's of dataframes.


The actual core of the regression analysis is done by compute_beta_hat_blocks,
which uses pinv, i.e. Moore-Penrose inverse type regression, i.e. as if it was a
Ridge regression with an infinitesimal regularization.
"""

import functools
import itertools

import numpy as np
import pandas as pd


def series_of_tuples_for_levels(index, levels):
  values = zip(*[index.get_level_values(level) for level in levels])
  return pd.Series(values, index=index)


def dataframe_of_levels(index, levels):
  values = zip(*[index.get_level_values(level) for level in levels])
  return pd.DataFrame(values, index=index, columns=levels)


def dummies_for_levels(index, levels):
  series = series_of_tuples_for_levels(index, levels)
  return pd.get_dummies(series)


def compute_metadata_blocks(index, levels_list_list, include_intercept=True):
  metadata_blocks = [
      dummies_for_levels(index, levels) for levels in levels_list_list
  ]
  if include_intercept:
    intercept_blocks = [pd.DataFrame(1, index=index, columns=['intercept'])]
    blocks = intercept_blocks + metadata_blocks
    return blocks
  else:
    return metadata_blocks


def make_positional_block(index):
  """Compute a positional block with interactions."""
  pos_df = dataframe_of_levels(index, [
      'well_row', 'well_col', 'site_row', 'site_col'
  ])
  pos_df['well_row'] = [(ord(x[0]) - ord('A')) for x in pos_df['well_row']]
  for key in [
      'well_row', 'well_col', 'site_row', 'site_col'
  ]:
    pos_df[key] = pos_df[key].astype(float)
  pos_df['well_row__site_row'] = pos_df['well_row'] * pos_df['site_row']
  pos_df['well_col__site_col'] = pos_df['well_col'] * pos_df['site_col']
  return pos_df


def interact_blocks(*blocks):
  accum = []
  for column_pair_tuple in itertools.product(
      *(df.iteritems() for df in blocks)):
    column_names, columns = zip(*column_pair_tuple)
    combination_name = tuple(column_names)
    column_product = functools.reduce(lambda x, y: x * y, columns)
    accum.append((combination_name, column_product))
  return pd.DataFrame(dict(accum))


def compute_beta_hat_blocks(blocks, y_df, rcond):
  x_df = pd.concat(blocks, axis=1)
  x_mat_pinv = np.linalg.pinv(x_df.values, rcond=rcond)
  beta_hat_mat = x_mat_pinv.dot(y_df)
  beta_hat_df = pd.DataFrame(
      beta_hat_mat, columns=y_df.columns, index=x_df.columns)
  beta_hat_blocks = [beta_hat_df.loc[b.columns, :] for b in blocks]
  return beta_hat_blocks, x_df, x_mat_pinv, beta_hat_df


def progressive_residualize(y_df, blocks, beta_hat_blocks, do_resid_for_block):
  """Progressively residualize y_df by the predictions of each block."""
  resid_df = y_df.copy()
  if (len(blocks) != len(beta_hat_blocks) or
      len(blocks) != len(do_resid_for_block)):
    raise ValueError(
        'Arguments must have same length: blocks, beta_hat_blocks, do_resid_for_block'
    )
  for block, b, do_resid in zip(blocks, beta_hat_blocks, do_resid_for_block):
    if do_resid:
      missing_from_b_set = set(block.columns).difference(set(b.index))
      if missing_from_b_set:
        raise ValueError(f"b doesn't contain {missing_from_b_set}.")
      y_hat_df_block = block.dot(b)
      resid_df = resid_df - y_hat_df_block
  return resid_df


def basic_residualize(y_df, block_maker, do_resid_for_block, rcond):
  blocks = block_maker(y_df.index)
  beta_hat_blocks, _, _, _ = compute_beta_hat_blocks(blocks, y_df, rcond)
  resid_df = progressive_residualize(y_df, blocks, beta_hat_blocks,
                                     do_resid_for_block)
  return resid_df, beta_hat_blocks, blocks


def residualize_first_using_second(first_df, second_df, block_maker,
                                   do_resid_for_block, rcond):
  second_blocks = block_maker(second_df.index)
  beta_hat_blocks, _, _, _ = compute_beta_hat_blocks(second_blocks, second_df,
                                                     rcond)
  first_blocks = block_maker(first_df.index)
  first_resid_df = progressive_residualize(first_df, first_blocks,
                                           beta_hat_blocks, do_resid_for_block)
  return first_resid_df, beta_hat_blocks, first_blocks, second_blocks
