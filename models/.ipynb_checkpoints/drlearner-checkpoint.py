# Copyright (c) 2023, Ahmed Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Global options for baselearners (see class attributes below)

base_learners_dict = dict({"GBM": GradientBoostingRegressor, "RF": RandomForestRegressor})


class conformalMetalearner:

  """

    Model class for conformal pseudo-outcome regression. Given an observational dataset (X_i, W_i, Y_i)_i and a prespecified coverage level 1-\alpha, 
    instances of this class conduct predictive inference on individual treatment effects (ITEs) through the following steps:

    (1) Constructing a dataset of covariates and pseudo-outcomes (X_i, \phi_i). The class support two options for the pseudo-outcomes:

        - Inverse propensity weighted outcomes: \phi_IPW = (p(X) - W)/p(X)(1-p(X)) * Y :: here, p is the propensity score at X
        - Doubly robust transformed pseudo-outcomes: \phi_DR = (p(X) - W)/p(X)(1-p(X)) * (Y - \hat{\mu}_W(X)) + (\hat{\mu}_1(X) - \hat{\mu}_0(X)) :: here, \hat{\mu}_W is a plug-in estimate of \mu_W

    (2) Cross-fitting a DR learner on training data as described in [1]. The DR learner can be a regression model for (X_i, \phi_i) or a quantile regression model
        with the prespecified coverage level 1-\alpha

    (3) Apply the standard conformal procedure in [2] (in the case of quantile regression) to obtain predictive intervals for ITE

    Note: This model assumes that the propensity score p(x) is known.
    ----
    References:
    -----------
    
    [1] E. Kennedy. "Towards optimal doubly robust estimation of heterogeneous causal effects", 2020.
    [2] Y. Romano and E. Candes. "Conformalized Quantile Regression", 2019.

  """
  
  def __init__(self, n_folds=5, alpha=0.1, base_learner="GBM", quantile_regression=True, metalearner="DR"):

    """
        :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
        :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
        :param base_learner: the underlying regression model
                             - current options: ["GBM": gradient boosting machines, "RF": random forest]
        :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                    or a point estimate of the CATE function. 

    """

    # set base learner
    self.base_learner        = base_learner
    self.quantile_regression = quantile_regression
    n_estimators_nuisance    = 100
    n_estimators_target      = 100
    alpha_ = alpha #0.3 #
    
    # set meta learner type
    self.metalearner  = metalearner
 
    # set conformal correction term to 0
    self.offset       = 0

    # set cross-fitting parameters and plug-in models for \mu_0 and \mu_1
    self.n_folds      = n_folds
    self.models_0     = [base_learners_dict[self.base_learner](n_estimators=n_estimators_nuisance) for _ in range(self.n_folds)] 
    self.models_1     = [base_learners_dict[self.base_learner](n_estimators=n_estimators_nuisance) for _ in range(self.n_folds)]

    # set the meta-learner and cross-fitting parameters
    self.skf          = StratifiedKFold(n_splits=self.n_folds)  

    if self.quantile_regression:

      base_args_u    = dict({"loss": "quantile", "alpha":1 - (alpha_/2), "n_estimators": n_estimators_target}) 
      base_args_l    = dict({"loss": "quantile", "alpha":alpha_/2, "n_estimators": n_estimators_target}) 

      self.models_u  = [base_learners_dict[self.base_learner](**base_args_u) for _ in range(self.n_folds)] 
      self.models_l  = [base_learners_dict[self.base_learner](**base_args_l) for _ in range(self.n_folds)]
    
    else:

      base_args_m    = dict({"n_estimators": n_estimators}) 
      self.models_m  = [base_learners_dict[self.base_learner](**base_args_m) for _ in range(self.n_folds)] 



  def get_pseudo_outcomes(self, W, pscores, Y, Y_hat_0, Y_hat_1, metalearner="DR"):

    """
    Function for constructing the DR pseudo-outcomes

    :param W: treatment assignment indicator
    :param pscores: true propensity scores
    :param Y: observed factual outcomes
    :param Y_hat_0: plug-in estimates for untreated outcomes
    :param Y_hat_1: plug-in estimates for treated outcomes

    outputs >> pseudo-outcomes corresponding to the input samples

    """
    if metalearner=="DR":

      Y_hat     = Y_hat_1 * W + Y_hat_0 * (1 - W)
      Y_pseudo  = ((W.reshape((-1, 1)) - pscores.reshape((-1, 1))) / (pscores.reshape((-1, 1)) * (1 - pscores.reshape((-1, 1))))) * (Y.reshape((-1, 1)) - Y_hat.reshape((-1, 1))) + (Y_hat_1.reshape((-1, 1)) - Y_hat_0.reshape((-1, 1)))  

    elif metalearner=="IPW":
      
      Y_pseudo  = ((W.reshape((-1, 1)) - pscores.reshape((-1, 1))) / (pscores.reshape((-1, 1)) * (1 - pscores.reshape((-1, 1))))) * Y.reshape((-1, 1)) 

    elif metalearner=="X":

      Y_pseudo  = Y_hat_1.reshape((-1, 1)) - Y_hat_0.reshape((-1, 1))

    self.Y_pseudo = Y_pseudo

    return Y_pseudo


  def fit(self, X, W, Y, pscores):

    """
    Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores

    :param W: treatment assignment indicator
    :param pscores: true propensity scores
    :param Y: observed factual outcomes
    :param X: covariates

    """

    # loop over the cross-fitting folds

    for i, (train_index, test_index) in enumerate(self.skf.split(W, W)):
      
      X_1, W_1, Y_1, pscores_1 = X[train_index, :], W[train_index], Y[train_index], pscores[train_index]
      X_2, W_2, Y_2, pscores_2 = X[test_index, :], W[test_index], Y[test_index], pscores[test_index]

      # fit the plug-in models \hat{\mu}_0 and \hat{\mu}_1

      self.models_0[i].fit(X_1[W_1==0, :], Y_1[W_1==0])
      self.models_1[i].fit(X_1[W_1==1, :], Y_1[W_1==1])

      Y_hat_0  = self.models_0[i].predict(X_2)
      Y_hat_1  = self.models_1[i].predict(X_2)

      # construct the pseudo-outcomes 

      Y_pseudo = self.get_pseudo_outcomes(W_2, pscores_2, Y_2, Y_hat_0, Y_hat_1, self.metalearner).reshape((-1, )) 

      if self.quantile_regression:

        self.models_u[i].fit(X_2, Y_pseudo)
        self.models_l[i].fit(X_2, Y_pseudo)
      
      else:

        self.models_m[i].fit(X_2, Y_pseudo)


  def predict(self, X):

    """
    Interval-valued prediction of ITEs

    :param X: covariates of the test point

    outputs >> point estimate, lower bound and upper bound

    """
    if self.quantile_regression:

      y_preds_u = [] 
      y_preds_l = [] 

      for j in range(len(self.models_u)):

        y_preds_u.append(self.models_u[j].predict(X))
        y_preds_l.append(self.models_l[j].predict(X))
    
      T_hat_DR_l = np.mean(np.array(y_preds_l), axis=0).reshape((-1,))
      T_hat_DR_u = np.mean(np.array(y_preds_u), axis=0).reshape((-1,))
      T_hat_DR   = (T_hat_DR_l + T_hat_DR_u) / 2

      # add conformal offset

      T_hat_DR_l = T_hat_DR_l - self.offset
      T_hat_DR_u = T_hat_DR_u + self.offset
    
    else:

      y_preds_m = [] 

      for j in range(len(self.models_m)):

        y_preds_m.append(self.models_m[j].predict(X))

      T_hat_DR   = np.mean(np.array(y_preds_m), axis=0).reshape((-1,))

      # add conformal offset

      T_hat_DR_l = T_hat_DR - self.offset
      T_hat_DR_u = T_hat_DR+ self.offset

    return T_hat_DR, T_hat_DR_l, T_hat_DR_u 


  def conformalize(self, alpha, X_calib, W_calib, Y_calib, pscores_calib, oracle=None):

    """
    Calibrate the predictions of the meta-learner using standard conformal prediction

    """

    self.offset     = 0
    
    T_hat_DR_calib, T_hat_DR_calib_l, T_hat_DR_calib_u = self.predict(X_calib)

    y_hat_0_calibs  = []
    y_hat_1_calibs  = []

    for i in range(len(self.models_0)):

      Y_hat_0_calib = self.models_0[i].predict(X_calib)
      Y_hat_1_calib = self.models_1[i].predict(X_calib)

      y_hat_0_calibs.append(Y_hat_0_calib)
      y_hat_1_calibs.append(Y_hat_1_calib)

    y_hat_0_calibs  = np.mean(np.array(y_hat_0_calibs), axis=0).reshape((-1,))
    y_hat_1_calibs  = np.mean(np.array(y_hat_1_calibs), axis=0).reshape((-1,))

    Y_DR_calib      = self.get_pseudo_outcomes(W_calib, pscores_calib, Y_calib, y_hat_0_calibs, y_hat_1_calibs, self.metalearner)

    if self.quantile_regression:

      self.residuals  = np.maximum(T_hat_DR_calib_l.reshape((-1,1)) - Y_DR_calib, Y_DR_calib - T_hat_DR_calib_u.reshape((-1,1)))
    
    else:

      self.residuals  = np.abs(T_hat_DR_calib.reshape((-1,1)) - Y_DR_calib)

    if oracle is not None:

      if self.quantile_regression:

        self.oracle_residuals = np.maximum(T_hat_DR_calib_l.reshape((-1,1)) - oracle.reshape((-1,1)), oracle.reshape((-1,1)) - T_hat_DR_calib_u.reshape((-1,1)))

      else:

        self.oracle_residuals = np.abs(T_hat_DR_calib.reshape((-1,1)) - oracle.reshape((-1,1)),)

    self.offset     = np.sort(self.residuals, axis=0)[int(np.ceil((1-alpha) * (1 + 1/len(W_calib)) * len(W_calib)))]
    self.Y_DR_calib = Y_DR_calib



