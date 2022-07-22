"""Compute iqr_z values that standardize values within a null_agg_keys grouping."""

import scipy.stats

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


def drug_group_augment(drug_effects, grouping_keys):
  # Compute mean, min, and max for each drug and join it back.
  # The result is still a well-level df, but using the drug-aggregated columns
  # makes it easy to filter at the drug level.
  de_mean2 = drug_effects.groupby(
      grouping_keys).agg(['mean', 'min', 'max'])
  de_mean2.columns = [f'{x}_drug{y}' for x, y in de_mean2.columns.values]
  de_ct = drug_effects.groupby(grouping_keys).count()[['num_wells']].rename(
      columns={'num_wells': 'num_wells_total'})
  drug_effects2 = drug_effects.join(de_mean2).join(de_ct)
  return drug_effects2
