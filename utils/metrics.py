import numpy as np
import scipy.stats

from utils.plotting import *


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return np.min(a), np.max(a) #m-h, m+h

def mean_confidence_interval(data, confidence=0.95, min_max=False):

    a     = 1.0 * np.array(data).reshape((-1,))
    n     = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h     = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    if min_max:
        
        out_l, out_h = np.min(a), np.max(a)
    
    else:

        out_l, out_h = m, h
    
    return out_l, out_h

def coverage(experiment):
    
    # Extract the first element of each tuple in the list
    coverages = [t[0] for t in experiment]
    
    return coverages

def intervals(experiment):
    
    # Extract the first element of each tuple in the list
    interval_widths = [t[1] for t in experiment]
    
    return interval_widths

def PEHE(experiment):
    
    # Extract the first element of each tuple in the list
    PEHEs = [t[2] for t in experiment]
    
    return PEHEs

def conformity_scores(experiment):

    if len(experiment) > 3:
        
        conformity_scores_ = [t[3] for t in experiment]

    else:
        
        conformity_scores_ = None

    return conformity_scores_


def compute_cdf(sample_data, conformity_cdf_x):
    
    _cdf = np.array([np.mean(sample_data < conformity_cdf_x[u]) for u in range(len(conformity_cdf_x))])

    return _cdf

def get_cdf_conformity_scores(conformity_data, cdf_x):

    conformity_meta   = [] 
    conformity_oracle = [] 
    graph             = []

    for k in range(len(conformity_data)): 
        
        conformity_meta_       = compute_cdf(conformity_data[k][0], cdf_x)
        conformity_oracle_     = compute_cdf(conformity_data[k][1], cdf_x)

        conformity_meta.append(conformity_meta_) 
        conformity_oracle.append(conformity_oracle_)

    conformity_meta          = np.array(conformity_meta)
    conformity_oracle        = np.array(conformity_oracle)

    conformity_meta_CI       = np.array([mean_confidence_interval(conformity_meta[:, k], confidence=0.95) for k in range(conformity_meta.shape[1])]) 
    conformity_oracle_CI     = np.array([mean_confidence_interval(conformity_oracle[:, k], confidence=0.95) for k in range(conformity_oracle.shape[1])]) 

    conformity_meta_mean     = np.mean(conformity_meta, axis=0)
    conformity_oracle_mean   = np.mean(conformity_oracle, axis=0)

    return (conformity_meta_mean, conformity_meta_CI), (conformity_oracle_mean, conformity_oracle_CI)


def evaluate_metrics(experiment_list):
    
    coverage_levels = list(map(coverage, experiment_list))
    interval_widths = list(map(intervals, experiment_list))
    RMSE            = list(map(PEHE, experiment_list))
    
    return coverage_levels, interval_widths, RMSE


def evaluate_stochastic_orders(experiments, path=None, save=True, experiment_name="synthetic"):
    
    for baseline in experiments.keys():
        
        conformity_scrs  = conformity_scores(experiments[baseline])
        cdf_x_           = np.linspace(np.min(conformity_scrs), np.max(conformity_scrs), 500)
        
        pseudo_scores, oracle_scores = get_cdf_conformity_scores(conformity_scrs, cdf_x_)
        
        plot_stochastic_order(cdf_x_, 
                              pseudo_scores, 
                              oracle_scores, 
                              save=save, 
                              path=path,
                              filename= experiment_name + "_" + baseline)




