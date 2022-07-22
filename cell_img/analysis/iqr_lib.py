"""Compute iqr_z values that standardize values within a null_agg_keys grouping."""

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model

iqr = scipy.stats.iqr


def compute_null_values(df, null_drugs, null_agg_keys, iqr_median_agg_keys):
  df = df.query('drug in @null_drugs')
  df = df.groupby(null_agg_keys).agg(['mean', 'count', 'median', iqr])
  df.columns = [f'null_{n}_{m}' for n, m in df.columns.values]
  iqr_cols = [key for key in df.columns if key.endswith('_iqr')]
  df_iqr = df[iqr_cols]
  df_iqr_median = df_iqr.groupby(iqr_median_agg_keys).median()
  df_iqr_median.columns = [name + '_median' for name in df_iqr_median.columns]
  df = df.join(df_iqr_median)
  return df


def adjoin_null_and_iqr_z(df, null_drugs, null_agg_keys, iqr_median_agg_keys):
  df_columns = list(df.columns)
  null_df = compute_null_values(df, null_drugs, null_agg_keys,
                                iqr_median_agg_keys)
  df = df.join(null_df)

  z_iqr = 1.34896
  for key in df_columns:
    df[key + '_iqr_z'] = (df[key] - df['null_' + key + '_median']
                         ) / df['null_' + key + '_iqr_median'] * z_iqr
  df.loc[:, 'null_drug'] = 'False'
  df.loc[df.query('drug in @null_drugs').index, 'null_drug'] = 'True'
  return df, null_df


def compute_null_expectation_for_column_using_sklearn(orig_df, make_blocks,
                                                      filter_to_null, y_column,
                                                      make_model):
  orig_df_null = filter_to_null(orig_df)
  blocks = make_blocks(orig_df_null.index)
  x_df = pd.concat(blocks, axis=1)
  y = orig_df_null[y_column]

  # Making a design for the full, instead of just the null, may invoke
  # columns in the blocks that aren't present in the null -- depending on the
  # design this may be a fatal flaw or ignorable. We just dump it here.
  blocks_full = make_blocks(orig_df.index)
  x_df_full = pd.concat(blocks_full, axis=1)
  a = set(x_df.columns)
  b = set(x_df_full.columns)
  print(f'unused coef: {a.difference(b)}')
  print(f'unestimated coef: {b.difference(a)}')
  # Re-index to eliminate any discrepancy.
  x_df_full2 = x_df_full.T.reindex(x_df.T.index).fillna(0.).T

  lasso1 = make_model()
  lasso1.fit(x_df, y)
  beta = pd.DataFrame(
      lasso1.coef_[:, np.newaxis], index=x_df.columns, columns=[y.name])
  beta.loc['intercept', y.name] = lasso1.intercept_
  y_hat = pd.Series(lasso1.predict(x_df), index=y.index, name=y.name)
  y_hat_full = pd.Series(
      lasso1.predict(x_df_full2), index=x_df_full2.index, name=y.name)
  y_full = orig_df[y_column]
  return y_hat_full, y_full, y_hat, y, beta, lasso1, x_df, x_df_full, x_df_full2


def compute_null_expectations_using_sklearn(
    orig_df,
    make_blocks,
    filter_to_null,
    make_model=lambda: sklearn.linear_model.LassoCV(max_iter=1000000, tol=1E-6)
):
  accum = {}
  for y_column in orig_df.columns:
    y_hat_full, _, _, _, _, _, _, _, _ = compute_null_expectation_for_column_using_sklearn(
        orig_df, make_blocks, filter_to_null, y_column, make_model)
    accum[y_column] = y_hat_full
  null_expectation_df = pd.DataFrame(accum)
  return null_expectation_df


def adjoin_null_and_iqr_z_using_null_expectation(df, null_drugs,
                                                 null_expectation_df,
                                                 median_abs_dev_agg_keys):
  df_columns = list(df.columns)
  df = df.copy()
  null_expectation_df = null_expectation_df.copy()
  null_expectation_df.columns = [
      f'null_{x}' for x in null_expectation_df.columns
  ]
  df = df.join(null_expectation_df)
  resid_columns = []
  for key in df_columns:
    resid_key = key + '_resid'
    df[resid_key] = df[key] - df['null_' + key]
    resid_columns.append(resid_key)
  abs_dev = np.abs(df[resid_columns])
  df_median_abs_dev = abs_dev.groupby(median_abs_dev_agg_keys).median()
  df_median_abs_dev.columns = [name + '_mad' for name in df_columns]
  df = df.join(df_median_abs_dev)

  z_iqr = 1.34896
  for key in df_columns:
    df[key + '_iqr_z'] = df[key + '_resid'] / (df[key + '_mad'] * 2.) * z_iqr
  df.loc[:, 'null_drug'] = 'False'
  df.loc[df.query('drug in @null_drugs').index, 'null_drug'] = 'True'
  return df


def drug_group_augment(drug_effects, grouping_keys):
  # Compute mean, min, and max for each drug and join it back.
  # The result is still a well-level df, but using the drug-aggregated columns
  # makes it easy to filter at the drug level.
  de_mean2 = drug_effects.groupby(
      grouping_keys).agg(['mean', 'min', 'max'])
  de_mean2.columns = [f'{x}_drug{y}' for x, y in de_mean2.columns.values]
  de_ct = drug_effects.groupby(grouping_keys).count().iloc[:, [0]]
  de_ct.columns = ['num_wells_total']
  drug_effects2 = drug_effects.join(de_mean2).join(de_ct)
  return drug_effects2
