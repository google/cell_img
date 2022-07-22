"""Compute null-drug-whitened data and use this to assess activity."""

import dataclasses
import typing

from IPython import display
import numpy as np
import pandas as pd

from cell_img.analysis import anova


# This code does not drop bad patches/wells -- drop those before calling.


def compute_centering_anova(emb_df_null, anova_formula, do_print=True):
  as1 = anova.do_anova(
      anova_formula, emb_df_null, verbose=True, compute_beta=True)
  if do_print:
    display.display(as1.summary)
  return as1


def residualize_by_anova_fit(emb_df, as1, undo=False):
  # Subtract the anova prediction to get resid -- remove line effects too.
  emb_df_hat_with_nan = anova.predict_from_anova(
      as1, emb_df, allow_nan_predictions=True)
  if undo:
    resid_df_with_nan = emb_df + emb_df_hat_with_nan
  else:
    resid_df_with_nan = emb_df - emb_df_hat_with_nan
  resid_df = resid_df_with_nan.loc[resid_df_with_nan.isna().sum(axis=1) == 0, :]
  assert resid_df_with_nan.shape == resid_df.shape
  return resid_df


def compute_inv_root(resid_df, keep_k):
  values, vectors = np.linalg.eigh(np.cov(resid_df.T))
  if keep_k is None:
    # None will result in the complete whitening. This is not recommended
    # unless you have many many replicas of the null. A modest number
    # like 50 (or smaller if your sample size is small) is generally preferred.
    keep_k = len(values)
  top_k = sorted(values)[::-1][:keep_k]
  values2 = np.maximum(values, top_k[-1])
  root = np.diag(np.sqrt(values2)).dot(vectors.T)
  inv_root = vectors.dot(np.diag(1. / np.sqrt(values2)))
  return root, inv_root, values, values2, vectors


@dataclasses.dataclass
class WhiteningFacts:
  """Track adjustments necessary to whiten a DataFrame."""
  centering_anova: anova.AnovaSummary
  root: np.ndarray
  inv_root: np.ndarray
  original_null_e_values: np.ndarray
  clipped_null_e_values: np.ndarray
  preprocess_before_anova: typing.Callable[[pd.DataFrame], pd.DataFrame]
  preprocess_before_covariance: typing.Callable[[pd.DataFrame], pd.DataFrame]

  def compute_semi_whitened_resid_df(self, emb_df):
    """Has anova's effect removed and "centered" away."""
    resid_df = residualize_by_anova_fit(emb_df, self.centering_anova)
    return resid_df.dot(self.inv_root)

  def undo_compute_semi_whitened_resid_df(self, semi_whitened_resid_df):
    resid_df = semi_whitened_resid_df.dot(self.root)
    emb_df = residualize_by_anova_fit(resid_df, self.centering_anova, undo=True)
    return emb_df

  def semi_whiten(self, emb_df):
    return emb_df.dot(self.inv_root)

  def un_semi_whiten(self, semi_whitened_df):
    return semi_whitened_df.dot(self.root)

  def expected_sum_squares(self):
    return np.sum(self.original_null_e_values / self.clipped_null_e_values)

  def expected_variance_of_sum_squares(self):
    # error?
    return np.sum((self.original_null_e_values / self.clipped_null_e_values)*2)

  def gamma_a_estimate(self):
    return self.expected_sum_squares(
    )**2 / self.expected_variance_of_sum_squares()

  def gamma_scale_estimate(self):
    return self.expected_variance_of_sum_squares() / self.expected_sum_squares()

  def log_ss(self, emb_df):
    semi_whitened_resid_df = self.compute_semi_whitened_resid_df(emb_df)
    return np.log((semi_whitened_resid_df ** 2).sum(axis=1)).to_frame('log_ss')


def make_whitening_facts(emb_df,
                         keep_k,
                         preprocess_before_anova,
                         preprocess_before_covariance,
                         anova_formula,
                         do_print=True):
  emb_df_null = preprocess_before_anova(emb_df)
  centering_anova = compute_centering_anova(
      emb_df_null, anova_formula, do_print=do_print)
  resid_df_null = residualize_by_anova_fit(emb_df_null, centering_anova)
  resid_df_null_final = preprocess_before_covariance(resid_df_null)
  root, inv_root, values, values2, unused_vectors = compute_inv_root(
      resid_df_null_final, keep_k)
  return WhiteningFacts(centering_anova, root, inv_root, values, values2,
                        preprocess_before_anova, preprocess_before_covariance)
