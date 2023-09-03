# coding: utf-8
# Copyright (c) 2023, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import argparse
import warnings
import logging 
import os
from datetime import date, datetime
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import numpy as np
import pandas as pd

from data.datasets import *
from models.metalearners import *
from utils.plotting import *
from utils.metrics import *

# Global variables and parameters

synthetic_setups    = dict({"A": 1, 
                            "B": 0})

dr_plot_params      = dict({"color":"b", "marker":"o", "markeredgecolor":"white", 
                            "markersize":5, "markeredgewidth":1})

ipw_plot_params     = dict({"color":"r", "marker":"v", "markeredgecolor":"white", 
                            "markersize":6, "markeredgewidth":1})

x_plot_params       = dict({"color":"g", "marker":"+", "markersize":5, "alpha":.7})

plot_params         = [dr_plot_params, ipw_plot_params, x_plot_params]

if not os.path.exists("figures"):

    os.makedirs("figures")
    
if not os.path.exists("logs"):

    os.makedirs("logs")

# Functions for running experiments

def run_experiment(alpha=0.1, n=1000, d=10, nexps=10,
                   quantile_regression=True, test_frac=0.1, 
                   baselines=["DR", "IPW", "X"],
                   experiment_name="Synthetic",
                   setup="A",
                   path=None,
                   save=True, 
                   plot=True, 
                   logger=None):
    
    print             = logger.info
    print("Loading '% s' dataset" % experiment_name)
    
    if experiment_name=="Synthetic":
        
        dataset       = generate_data(n=n, d=d, 
                                      gamma=synthetic_setups[setup], 
                                      alpha=alpha, 
                                      nexps=nexps) 
    
        oracle_width  = dataset[0]["width"].loc[0]
    
    elif experiment_name=="IHDP":
        
        dataset       = IHDP_data()
        oracle_width  = None
    
    experiments   = dict.fromkeys(baselines)
    

    for baseline in baselines:
        
        print("Running the '% s' learner baseline" % baseline)
        
        experiments[baseline] = run(dataset, 
                                    conformal_metalearner_experiment, 
                                    metalearner=baseline, 
                                    quantile_regression=quantile_regression, 
                                    alpha=alpha, 
                                    test_frac=test_frac)
    
    
    experiment_list                        = [experiments[key] for key in list(experiments.keys())]
    coverage_levels, interval_widths, RMSE = evaluate_metrics(experiment_list)

    Results_data                           = [coverage_levels, 
                                              interval_widths, 
                                              RMSE]
    
    if plot:
    
        evaluate_stochastic_orders(experiments, path=path, save=save, experiment_name=experiment_name)
    
        plot_results(baseline_names=baselines, 
                     results_data=Results_data, 
                     oracle_width=oracle_width, 
                     alpha=alpha, 
                     save=save, 
                     path=path,
                     filename=experiment_name + "_results")
    
    return Results_data, experiments



def run_coverage_sweeps(alphas=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                        n=1000, d=10, nexps=10, quantile_regression=True, test_frac=0.1, 
                        baselines=["DR", "IPW", "X"], experiment_name="Synthetic", path=None,
                        filenames=["Coverage_sweep", "AveLength_sweep", "RMSE_sweep"], 
                        setup="A", save=True, logger=None):
    
    print    = logger.info
    Results_ = []

    for alpha in alphas:
        
        Results_data, _ = run_experiment(alpha=alpha, n=n, d=d, nexps=nexps,
                                         quantile_regression=quantile_regression, 
                                         test_frac=test_frac, 
                                         baselines=baselines,
                                         experiment_name=experiment_name, 
                                         setup=setup,
                                         path=path,
                                         save=False,
                                         plot=False)
    
        Results_.append(Results_data)

        Results_avg = np.mean(Results_, axis=3)

    plot_sweeps(alphas, Results_avg, plot_params, path=path, 
                filename="Coverage", calibration=True, save=True, 
                perf_metric="Coverage", alpha=alpha)       
    
    plot_sweeps(alphas, Results_avg, plot_params, path=path,
                filename="RMSE", calibration=True, save=True, 
                perf_metric="RMSE", alpha=alpha) 

    plot_sweeps(alphas, Results_avg, plot_params, path=path,
                filename="Average Length", calibration=True, save=True, 
                perf_metric="Average Length", alpha=alpha) 
    
    

# Main script

def main(args):

    exp_log_time  = str(datetime.now())
    
    logging.basicConfig(filename="logs/conformal_metalearners " + exp_log_time + ".log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 

    logger        = logging.getLogger() 
    logger.setLevel(logging.DEBUG) 
    print         = logger.info
    
    results_PATH  = "figures/" + exp_log_time
    os.mkdir(results_PATH)
    print("Directory '% s' created" % results_PATH)
    
    test_frac     = args.test_frac
    baselines     = args.baselines
    synth_setup   = args.setup
    exp_type      = args.exp_type
    num_samples   = args.num_samples
    num_features  = args.num_features
    quantile_reg  = args.quantile_reg
    save_fig      = args.save_fig 
    num_exp       = args.num_experiments
    alpha         = args.target_coverage
    sweep_exp     = args.sweep_experiments
    plot_flag     = args.plot
    
    logger.info("Starting the experiments...")
    
    Results_data, experiments = run_experiment(alpha=alpha, 
                                               n=num_samples, 
                                               d=num_features, 
                                               nexps=num_exp,
                                               quantile_regression=quantile_reg, 
                                               test_frac=test_frac, 
                                               baselines=baselines,
                                               experiment_name=exp_type,
                                               path=exp_log_time, 
                                               setup=synth_setup,
                                               save=save_fig,
                                               plot=plot_flag, 
                                               logger=logger)
    
    logger.info("Experiment complete!")
    logger.info("Summary of results...")
    
    result_summary            = np.mean(np.array(Results_data), axis=2)
    
    for u in range(len(baselines)):
        
        print("%s -> Coverage: %.3f | Avg. Interval Length: %.3f | RMSE: %.3f  " % (baselines[u], 
                                                                                    result_summary[0, u],
                                                                                    result_summary[1, u],
                                                                                    result_summary[2, u]))
    
    if sweep_exp:
        logger.info("Sweeping values of target coverage...")
        
        run_coverage_sweeps(alphas=[0.05, 0.15, 0.25, 0.35, 0.45, 
                                    0.55, 0.65, 0.75, 0.85, 0.95],
                            n=num_samples, d=num_features, nexps=num_exp, 
                            quantile_regression=quantile_reg, 
                            test_frac=test_frac, 
                            baselines=baselines, 
                            experiment_name=exp_type, 
                            setup=synth_setup, save=save_fig,
                            path=exp_log_time, 
                            filenames=["Coverage_sweep", 
                                       "AveLength_sweep", 
                                       "RMSE_sweep"], 
                            logger=logger)
    
    logger.info("Experiments completed!")            

    
    
if __name__ == "__main__":

    default_setup        = "A"
    deafult_exp_name     = "Synthetic"
    default_baselines    = ["DR", "IPW", "X"]
    parser               = argparse.ArgumentParser(description="Conformal Meta-learner Experiments")
    
    parser.add_argument("-t", "--test-frac", default=.1, type=float)
    parser.add_argument("-b", "--baselines", nargs="+", default=default_baselines)
    parser.add_argument("-s", "--setup", default=default_setup, type=str)
    parser.add_argument("-e", "--exp-type", default="Synthetic", type=str)
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("-d", "--num-features", default=10, type=int)
    parser.add_argument("-q", "--quantile-reg", default=True, type=bool)
    parser.add_argument("-v", "--save-fig", default=True, type=bool)
    parser.add_argument("-x", "--num-experiments", default=10, type=int)
    parser.add_argument("-c", "--target-coverage", default=.1, type=float)
    parser.add_argument("-w", "--sweep-experiments", default=True, type=bool)
    parser.add_argument("-p", "--plot", default=True, type=bool)
    
    args = parser.parse_args()

    main(args)