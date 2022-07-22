"""The do_anova function computes a summary of coordinate-wise ANOVA's.

Basic usage:

  as1 = do_anova(model_formula_string, emb_df)
  as1.summary
See, for example:
  https://colab.corp.google.com/drive/1HhQrsU0-gTERHeOxJwQsUoMyy7pmg_uJ#scrollTo=bF2KSvlS_uLm

Where emb_df is a (subset of) and embedding DataFrame. The index names
as well as column names from emb_df are available for model formula terms. For
example, the model_formula_string might be any valid patsy formula such as:
  ('(C(batch) + C(plate) + age + '
   'C(batch) * C(plate) * C(disease_condition) + '
   'C(batch) * C(plate) * C(cell_line_id)')
Note, here, that age is used as a numeric variable, and the others variables
are coded as factors.

There is an undesirable feature of this implementation, which is that
terms get sorted according to the number of factors. E.g., in the above,
disease_conditions levels will come out before the interaction terms of, say
batch with plate. This removes the ability to control which terms are
"controlled" for, when interpreting the sum of squares attributed to the later
terms. This is all well intentioned (e.g. "include main effects before their
interactions"), but a bit heavy handed. It is a deeply buried part of patsy.
The call stack to the issue looks something like:
 dmatrix -> ... -> build.design_matrix_builders -> build._make_subterm_infos
The last function seems to do all the real work regarding ordering.
"""

import collections


import numpy as np
import pandas as pd
import patsy
import scipy.stats


ANOVA_SUMMARY_ORDER = ('summary', 'design_info', 'ss_under_q', 'q', 'r',
                       'dmatrix', 'r_diag_series', 'beta',
                       'model_formula_string')


R_DIAG_SMALL = 1E-8


def extend_patsy_syntax_with_ord_float():
  # It's possible to extend the "syntax" of the patsy formulas by putting
  # functions into the np module. For example, here's how to add a function
  # that converts letters into floats:

  def ord_float(x):
    return np.float32([ord(u) for u in x])

  # Emplace the function into numpy so that patsy sees it as "ord_float".
  np.ord_float = ord_float


class AnovaSummary(collections.namedtuple('AnovaSummary', ANOVA_SUMMARY_ORDER)):
  """AnovaSummary."""
  __slots__ = ()


def do_anova(model_formula_string,
             emb_df_sub,
             verbose=True,
             compute_beta=False):
  """Computes a coordinate-wise summary of the variance components.

  Args:
    model_formula_string: A model formula string using features from the
      multi-index of an embedding DataFrame. The syntax is dictated by
      the patsy package, and resembles that of R. C.f.
        https://patsy.readthedocs.io/en/latest/formulas.html#the-formula-language
    emb_df_sub: An embedding DataFrame to analyze. The index will be
      reset before model building.
    verbose: Whether to print diagnostics as the computation proceeds.
    compute_beta: Whether to compute beta and store it on the result.
  Returns:
    An AnovaSummary instance, which has fields:
      'summary', 'design_info', 'ss_under_q', 'q', 'r',
      'dmatrix', 'r_diag_series'
  """
  if verbose: print('Computing design matrix.')
  dmatrix = patsy.dmatrix(model_formula_string, data=emb_df_sub.reset_index())

  if verbose: print('Model has %d columns' % dmatrix.shape[1])
  if dmatrix.shape[1] > 2000:
    print('Warning!!!: Big model.')

  di = dmatrix.design_info
  if verbose: print('Expansion:')
  terms = list(di.term_name_slices.keys())
  if terms[0] == 'Intercept':
    has_intercept = True
    expansion = '1 + '
    terms = terms[1:]
  else:
    has_intercept = False
    expansion = '0 + '
  expansion += ' + '.join(terms)
  if verbose: print(expansion)
  assert dmatrix.shape[0] >= dmatrix.shape[1], ('Error: there are not more rows'
                                                ' of data than regression '
                                                'columns. Try a simpler model.')
  if verbose: print('Computing QR decomposition.')
  q, r = np.linalg.qr(dmatrix, mode='reduced')
  emb_under_q = q.T.dot(emb_df_sub.values)
  r_diag_series = pd.Series(np.diag(r), di.column_names)
  r_diag_small_indicator = np.abs(r_diag_series) < R_DIAG_SMALL
  r_diag_small = r_diag_series[r_diag_small_indicator]
  # Remark: Due to collinearity, the beta for the regression is not always
  # defined in cases where the ANOVA is still defined. If we want beta,
  # one way to address the problem is to drop the r_diag_small_indicator
  # predictor columns, update the QR as needed, and let:
  #   beta = np.linalg.solve(r, emb_under_q).
  #
  # Here's is a derivation (geoffd@):
  #  You have a design matrix that might be singular, or very close to singular.
  #  You do a QR decomposition, then project out the singular bit.
  #   y = X beta
  #  Left-multiply by Q^T, where X=QR:
  #   Q^T y = R beta
  #  If R has 0's on the diagonal, you drop those rows and columns from R
  #  and those columns from Q to get a smaller, non-singular system.
  #   Q'^T y = R' beta'
  #  You solve that and get a projected beta, beta'
  # Then you fill in the original beta, setting the projected out part to 0.

  if compute_beta:
    emb_under_q_keep = emb_under_q[~r_diag_small_indicator, :]
    r_keep = r[~r_diag_small_indicator, :][:, ~r_diag_small_indicator]
    beta_keep = np.linalg.solve(r_keep, emb_under_q_keep)
    beta = np.zeros((r.shape[1], beta_keep.shape[1]))
    index_kept = np.where(~r_diag_small_indicator)[0]
    beta[index_kept, :] = beta_keep
    beta = pd.DataFrame(beta, index=di.column_names, columns=emb_df_sub.columns)
  else:
    beta = None
  ss_under_q = pd.Series(np.sum(emb_under_q ** 2., axis=1), di.column_names)
  tss = np.sum(emb_df_sub.values ** 2.)
  rss = tss - np.sum(ss_under_q)
  accum = []
  if len(r_diag_small) and verbose:
    print('Dropping model term components corresponding to:')
    print(r_diag_small)
  model_df = 0
  for term_name, term_slice in di.term_name_slices.items():
    term_df = np.sum(~r_diag_small_indicator[term_slice])
    model_df += term_df
    term_ss = float(np.sum((ss_under_q * ~r_diag_small_indicator)[term_slice]))
    entry = (term_name, term_ss, term_df)
    if term_df > 0:
      accum.append(entry)
  resid_df = emb_df_sub.shape[0] - model_df
  entry = ('residual', rss, resid_df)
  accum.append(entry)
  # Normally one doesn't consider Intercept as a fully fledged part of the R^2
  # calculation.
  if has_intercept:
    tss_excluding_intercept = tss - accum[0][1]
  else:
    tss_excluding_intercept = tss

  row_accum = []
  for entry in accum:
    term_name, term_ss, term_df = entry
    term_f = term_ss / term_df / (rss / resid_df)
    term_p_value = scipy.stats.f.sf(term_f, term_df, resid_df)
    term_nominal_significance = term_p_value < 0.05
    unused_term_r2_including_intercept = term_ss / tss
    term_r2_excluding_intercept = term_ss / tss_excluding_intercept
    row_accum.append((term_name, term_r2_excluding_intercept * 100., term_df,
                      term_f, term_ss, term_p_value, term_nominal_significance))
  model_summary = pd.DataFrame.from_records(
      row_accum,
      columns=('term', 'r2_percent', 'df', 'f_stat', 'ss', 'p_value',
               'p_less_05')).set_index('term')
  if has_intercept:
    model_summary.loc['Intercept', 'r2_percent'] = 0.
  model_summary.loc['residual', 'p_value'] = 1.
  model_summary['cumulative_r2_percent'] = np.cumsum(
      model_summary['r2_percent'])
  return AnovaSummary(model_summary, di, ss_under_q, q, r, dmatrix,
                      r_diag_series, beta, model_formula_string)


def predict_from_anova(anova1, emb_df, allow_nan_predictions=False):
  """Predict effects from anova1 appropriate to emb_df's design."""
  if anova1.beta is None:
    raise ValueError('anova argument must have beta; use compute_beta=True')
  dmatrix = patsy.dmatrix(
      anova1.model_formula_string, data=emb_df.reset_index())
  dmatrix_df = pd.DataFrame(
      np.asarray(dmatrix),
      index=emb_df.index,
      columns=dmatrix.design_info.column_names)
  if allow_nan_predictions:
    beta_nan = anova1.beta.reindex(index=dmatrix_df.columns)
    # The complexity here is basically because 0 * np.nan = np.nan instead of 0.
    # If we had a version of dot that respected the latter convention, it
    # could replace most of what follows (and more precisely handle partial
    # missing values -- at the moment I nan-out an entire rows of the
    # prediction.
    beta_nan_index_flag = beta_nan.isna().sum(axis=1) > 0
    eventually_nan_locations = np.abs(
        dmatrix_df.loc[:, beta_nan_index_flag]).sum(axis=1) > 0.
    beta = anova1.beta.reindex(index=dmatrix_df.columns, fill_value=0.)
    emb_df_hat = dmatrix_df.dot(beta)
    emb_df_hat.loc[eventually_nan_locations, :] = np.nan
  else:
    beta = anova1.beta
    emb_df_hat = dmatrix_df.dot(beta)
  pd.testing.assert_index_equal(emb_df_hat.columns, emb_df.columns)
  return emb_df_hat
