import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from models.drlearner import *



def conformal_metalearner_experiment(df, metalearner="DR", quantile_regression=True, alpha=0.1, test_frac=0.1):
    
    if len(df)==2:
        
        train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, 
                                    metalearner=metalearner) 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores


def dr_cqr_random_forests(df, alpha):

    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, metalearner="DR") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores


def ipw_cqr_random_forests(df, alpha):
    
    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", quantile_regression=True, metalearner="IPW") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores




def x_cqr_random_forests(df, alpha):
    
    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", quantile_regression=True, metalearner="X") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores



def run(data, func, **kwargs): # alpha):

    results = []
  
    if type(data)==tuple:
        
        for df_train, df_test in zip(data[0], data[1]):
            
            result = func((df_train, df_test), **kwargs)# alpha)
            results.append(result)

    else:

        for df in data:
            
            result = func(df, **kwargs)# alpha)
            results.append(result)
  
    return results

