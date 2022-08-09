"""Relaxed Lasso regression.

We follow the simplified relaxed Lasso conventions from:
  Hastie, Trevor, Robert Tibshirani, and Ryan Tibshirani.
  "Best subset, forward stepwise or lasso? Analysis and recommendations based on
  extensive comparisons."
  Statistical Science 35, no. 4 (2020): 579-592.
  https://www.stat.cmu.edu/~ryantibs/papers/bestsubset-sts.pdf

That work is based on the original definition from:
  Meinhausen, N. (2007). "Relaxed Lasso." Comput. Statist. Data
  Anal. 52 374–393. MR2409990
  https://doi.org/10.1016/j.csda.2006.12.019  
  https://stat.ethz.ch/~nicolai/relaxo.pdf
"""

import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing


class RelaxedLassoPath(object):

  def __init__(self,
               fit_intercept=True,
               gamma_grid=None,
               eps=1E-6,
               **lasso_kwargs):
    self.fit_intercept = fit_intercept
    self.gamma_grid = gamma_grid if gamma_grid is not None else np.linspace(
        0, 1, num=5)
    assert len(self.gamma_grid.shape) == 1
    self.eps = eps
    self.lasso_kwargs = lasso_kwargs
    self.fitted_ = False

  def fit(self, X, y, alphas=None):
    X = np.asarray(X)
    y = np.asarray(y)
    self.fitted_ = True
    self.y_shape = y.shape
    y1 = y
    if len(y1.shape) == 1:
      y1 = y1[:, np.newaxis]
    if self.fit_intercept:
      self.X_scaler = sklearn.preprocessing.StandardScaler(with_std=False)
      self.y_scaler = sklearn.preprocessing.StandardScaler(with_std=False)
      Xc = self.X_scaler.fit_transform(X)
      yc = self.y_scaler.fit_transform(y1)
    else:
      Xc = X
      yc = y1
    self.lp1 = sklearn.linear_model.lasso_path(
        Xc, yc, alphas=alphas, **self.lasso_kwargs)
    # TODO(): Use dual_gaps.
    alphas, coefs_lasso, unused_dual_gaps = self.lp1
    # Xc.shape, yc.shape: (n, p), (p, q)
    # coefs_lasso.shape: (q, p, m)
    # [m is the number of lasso steps == len(alphas)]
    coefs2_lasso = np.transpose(coefs_lasso, (2, 0, 1))
    # coefs2_lasso.shape: (m, q, p)
    yc2 = np.transpose(yc, (1, 0))  #: (q, n)
    coefs2_lstsq = do_lstsq_vec(Xc, yc2, coefs2_lasso, self.eps)
    coefs_lstsq = np.transpose(coefs2_lstsq, (1, 2, 0))  #: (q, p, m)

    gamma_grid = self.gamma_grid.__getitem__((slice(None),) + (np.newaxis,) *
                                             len(coefs_lstsq.shape))
    gamma_coefs = gamma_grid * coefs_lasso + ((1. - gamma_grid) * coefs_lstsq)
    self.alphas = alphas
    self.coefs_lasso = coefs_lasso
    self.coefs_lstsq = coefs_lstsq
    self.coefs = gamma_coefs

  def predict(self, X, gamma_ix=None, alphas_ix=None):
    X = np.asarray(X)
    assert self.fitted_
    if self.fit_intercept:
      Xc = self.X_scaler.transform(X)
    else:
      Xc = X
    if gamma_ix is None or alphas_ix is None:
      # It's faster to not Xc.dot(self.coefs), but expand here.
      y_hat_c_lstsq = Xc.dot(self.coefs_lstsq)
      y_hat_c_lasso = Xc.dot(self.coefs_lasso)
      gamma_grid = self.gamma_grid.__getitem__((slice(None),) + (np.newaxis,) *
                                               len(y_hat_c_lasso.shape))
      y_hat_c = gamma_grid * y_hat_c_lasso + (1. - gamma_grid) * y_hat_c_lstsq

      if self.fit_intercept:
        # Be careful to choose the intercept to make this true,
        # when we get to that. :)
        y_hat = y_hat_c + self.y_scaler.mean_[np.newaxis, np.newaxis, :,
                                              np.newaxis]
    else:
      y_hat_c_lstsq = Xc.dot(self.coefs_lstsq[:, :, alphas_ix].T)
      y_hat_c_lasso = Xc.dot(self.coefs_lasso[:, :, alphas_ix].T)
      gamma = self.gamma_grid[gamma_ix]
      y_hat_c = gamma * y_hat_c_lasso + (1. - gamma) * y_hat_c_lstsq
      if self.fit_intercept:
        # Be careful to choose the intercept to make this true, when we get
        # to that. :)
        y_hat = y_hat_c + self.y_scaler.mean_[np.newaxis, :]
    if len(self.y_shape) == 1:
      y_hat = y_hat.flatten()
    return y_hat


def do_lstsq(Xc, yc, c, eps):
  c_lstsq = np.zeros_like(c)
  w = np.where(np.abs(c) > eps)[0]
  b, unused_resid, unused_rank, unused_s = np.linalg.lstsq(
      Xc[:, w], yc, rcond=None)
  c_lstsq[w] = b
  return c_lstsq


do_lstsq_vec = np.vectorize(do_lstsq, signature='(n,p),(n),(p),()->(p)')


class RelaxedLassoCV(object):

  def __init__(self, fit_intercept=True, cv=None, gamma_grid=None, eps=1E-6,
               **lasso_kwargs):
    self.fit_intercept = fit_intercept
    self.cv = sklearn.model_selection.check_cv(cv)
    self.gamma_grid = gamma_grid if gamma_grid is not None else np.linspace(
        0, 1, num=5)
    assert len(self.gamma_grid.shape) == 1
    self.eps = eps
    self.lasso_kwargs = lasso_kwargs
    self.fitted_ = False

  def fit(self, X, y, alphas=None):
    X = np.asarray(X)
    y = np.asarray(y)
    self.fitted_ = True
    self.y_shape = y.shape
    y1 = y
    if len(y1.shape) == 1:
      y1 = y1[:, np.newaxis]
    self.rlp = RelaxedLassoPath(
        fit_intercept=self.fit_intercept,
        gamma_grid=self.gamma_grid,
        eps=self.eps, **self.lasso_kwargs)
    self.rlp.fit(X, y1, alphas=alphas)
    accum = []
    for tr_ix, tst_ix in self.cv.split(X, y1):
      # print(len(tr_ix), len(tst_ix))
      Xtr = X[tr_ix, :]
      ytr = y1[tr_ix, :]
      Xtst = X[tst_ix, :]
      ytst = y1[tst_ix, :]
      rlp1 = RelaxedLassoPath(
          fit_intercept=self.fit_intercept,
          gamma_grid=self.gamma_grid,
          eps=self.eps, **self.lasso_kwargs)
      rlp1.fit(Xtr, ytr, alphas=self.rlp.alphas)
      ytst_hat = rlp1.predict(Xtst)
      mse = np.mean((ytst_hat - ytst[..., np.newaxis])**2, axis=(1, 2))
      accum.append(mse)
    mmse = np.stack(accum).mean(axis=0)
    self.mmse = mmse
    self.j_min = np.argmin(np.min(mmse, axis=0))
    self.i_min = np.argmin(np.min(mmse, axis=1))
    self.gamma = self.gamma_grid[self.i_min]
    self.alpha = self.rlp.alphas[self.j_min]
    self.coef_ = self.rlp.coefs[self.i_min, :, :, self.j_min]
    if self.fit_intercept:
      self.intercept_ = self.rlp.y_scaler.mean_ - self.coef_.dot(
          self.rlp.X_scaler.mean_)

  def predict(self, X):
    X = np.asarray(X)
    assert self.fitted_
    y_hat = self.rlp.predict(X, gamma_ix=self.i_min, alphas_ix=self.j_min)
    # This should be equivalent:
    #   y_hat = X.dot(self.coef_.T) + self.intercept_
    if len(self.y_shape) == 1:
      y_hat = y_hat.flatten()
    return y_hat
