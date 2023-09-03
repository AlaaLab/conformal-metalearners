import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from scipy.special import erfinv
from sklearn.linear_model import LogisticRegression
import os

TRAIN_DATASET = "./data/IHDP/ihdp_npci_1-100.train.npz"
TEST_DATASET  = "./data/IHDP/ihdp_npci_1-100.test.npz"
TRAIN_URL     = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL      = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
PATH_dir      = "./data/NLSM/data"

from pathlib import Path
from typing import Any, Tuple



def convert(npz_file, scale=None):

    npz_data = np.load(npz_file)
    scales   = []
    
    x   = npz_data['x']
    t   = npz_data['t']
    yf  = npz_data['yf']
    ycf = npz_data['ycf']
    mu0 = npz_data['mu0']
    mu1 = npz_data['mu1']
    
    num_realizations = x.shape[2]
    
    dataframes = []
    
    for i in range(num_realizations):
        
        x_realization   = x[:, :, i]
        t_realization   = t[:, i]
        yf_realization  = yf[:, i]
        ycf_realization = ycf[:, i]
        mu1_realization = mu1[:, i]
        mu0_realization = mu0[:, i]

        model           = LogisticRegression()
        model.fit(x_realization, t_realization) 
        
        df         = pd.DataFrame(x_realization, columns=[f'X{j + 1}' for j in range(x_realization.shape[1])])
        df['T']    = t_realization
        df['Y']    = yf_realization
        df['Y_cf'] = ycf_realization
        df['Y1']   = yf_realization * t_realization + ycf_realization * (1 - t_realization)  #mu1_realization
        df['Y0']   = ycf_realization * t_realization + yf_realization * (1 - t_realization)#mu0_realization
        df['ITE']  = df['Y1'] - df['Y0']
        df["ps"]   = model.predict_proba(x_realization)[:, 1]

        df["CATE"] = mu1_realization - mu0_realization

        sd_cate = np.sqrt((np.array(df["CATE"])).var())

        if scale is None:
            
            if sd_cate > 1:

                error_0  = np.array(df['Y0']) - mu0_realization 
                error_1  = np.array(df['Y1']) - mu1_realization

                mu0_     = mu0_realization / sd_cate
                mu1_     = mu1_realization / sd_cate

                scales.append(sd_cate) 
              
                df['Y0']   = mu0_ + error_0
                df['Y1']   = mu1_ + error_1
                df['ITE']  = df['Y1'] - df['Y0']
                df["CATE"] = mu1_ - mu0_ 
          
            else:

                scales.append(1)

        elif scale is not None:
            
            # test data
            error_0  = np.array(df['Y0']) - mu0_realization 
            error_1  = np.array(df['Y1']) - mu1_realization

            mu0_   = mu0_realization / scale[i]
            mu1_   = mu1_realization / scale[i]

            df['Y0']  = mu0_ + error_0
            df['Y1']  = mu1_ + error_1
            df['ITE'] = df['Y1'] - df['Y0']
            df["CATE"] = mu1_ - mu0_ 
        
        dataframes.append(df)
    
    return dataframes, scales


def IHDP_data():
    
    train      = './data/IHDP/ihdp_npci_1-100.train.npz'
    test       = './data/IHDP/ihdp_npci_1-100.test.npz'
    
    train_data, scale = convert(train)
    test_data, _      = convert(test, scale)
    
    return train_data, test_data


def NLSM_data():

    NLSM_files = os.listdir(PATH_dir) 
    
    dataset    = []

    for nlsm_file in NLSM_files: 
        
        df = pd.read_csv(PATH_dir + "/" + nlsm_file)
        df["CATE"] = df["Etau"]

        dataset.append(df)

    return dataset


def generate_data(n, d, gamma, alpha, nexps, correlated=False, heteroscedastic=True):
    
    def correlated_covariates(n, d):
        
        rho = 0.9
        X   = np.random.randn(n, d)
        fac = np.random.randn(n, d)
        X   = X * np.sqrt(1 - rho) + fac * np.sqrt(rho)
      
        return norm.cdf(X)
    
    datasets = []

    for _ in range(nexps):
        
        if correlated == False and heteroscedastic == False:
            
            X       = np.random.uniform(0, 1, (n, d))
            tau     = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))) 
            tau     = tau.reshape((-1,))
            tau_0   = gamma * tau 

            std     = np.ones(X.shape[0])
            ps      = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)

            err_0   = np.random.normal(0, 1, n)

            Y0      = tau_0 + std * err_0  #np.zeros(n)
            Y1      = tau + std * errdist
            T       = np.random.uniform(size=n) < ps
            Y       = Y0.copy()
            Y[T]    = Y1[T]
            
            #Pseudolabel calculation
            A       = T
            pi      = ps
            xi      = (A - pi) * Y / (pi * (1 - pi))
            
            data    = np.column_stack((X, T, Y))
      
            column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
            df           = pd.DataFrame(data, columns=column_names)
            df['xi']     = xi
            df["ps"]     = np.array(ps).reshape((-1,))
            df["Y1"]     = Y1.reshape((-1,))
            df["Y0"]     = Y0.reshape((-1,))
            df["CATE"]   = tau - tau_0
            df["width"]  = np.mean(np.sqrt(2)*(np.sqrt(2)*std)*erfinv(2*(1-(alpha/2))-1) * 2) 

            datasets.append(df)
    
        elif correlated == False and heteroscedastic == True:
            
            # Generate dataset with heteroscedastic errors and independent covariates
            X       = np.random.uniform(0, 1, (n, d))
        
            tau     = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))) 
            tau     = tau.reshape((-1,))

            tau_0   = gamma * tau 
            std     = -np.log(X[:, 0] + 1e-9)
            ps      = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

            errdist = np.random.normal(0, 1, n)
            err_0   = np.random.normal(0, 1, n)

            Y0      = tau_0 + 1 * err_0  
            Y1      = tau + np.sqrt(std) * errdist
            T       = np.random.uniform(size=n) < ps
            Y       = Y0.copy()
            Y[T]    = Y1[T]
            
            #Pseudolabel calculation
            A       = T
            pi      = ps
            xi      = (A - pi) * Y / (pi * (1 - pi))
            
            #Stratify by conditional variance, CATE
            n_percentiles    = 100
            cate             = Y1 - Y0
            conditional_var  = std**2 * (1 - pi) + pi * std**2 * (1 + tau)**2
            cate_percentiles = np.zeros(n)
            var_percentiles  = np.zeros(n)

            for j in range(n):
                
                cate_percentiles[j] = np.searchsorted(np.percentile(cate, np.linspace(0, 100, n_percentiles+1)), cate[j])
                var_percentiles[j] = np.searchsorted(np.percentile(conditional_var, np.linspace(0, 100, n_percentiles+1)), conditional_var[j])
            
            
            data         = np.column_stack((X, T, Y))
            column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
            df           = pd.DataFrame(data, columns=column_names)
            df['xi']     = xi
            df['cate_percentile'] = cate_percentiles / n_percentiles
            df['var_percentile']  = var_percentiles / n_percentiles

            df["ps"]     = np.array(ps).reshape((-1,))
            df["Y1"]     = Y1.reshape((-1,))
            df["Y0"]     = Y0.reshape((-1,))
            df["CATE"]   = tau - tau_0
            df["width"]  = np.mean(np.sqrt(2)*(np.sqrt(2)*std)*erfinv(2*(1-(alpha/2))-1) * 2) 
      
            datasets.append(df)
      
        elif correlated == True and heteroscedastic == False:
            
            # Generate dataset with homoscedastic errors and correlated covariates
            X       = correlated_covariates(n, d)
            tau     = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5))))
            std     = np.ones(X.shape[0])
            ps      = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)
            Y0      = np.zeros(n)
            Y1      = tau + std * errdist
            T       = np.random.uniform(size=n) < ps
            Y       = Y0.copy()
            Y[T]    = Y1[T]
            
            #Pseudolabel calculation
            A       = T
            pi      = ps
            xi      = (A - pi) * Y / (pi * (1 - pi))
            
            data         = np.column_stack((X, T, Y))
            column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
            df           = pd.DataFrame(data, columns=column_names)
            df['xi']     = xi
            df["ps"]     = np.array(ps).reshape((-1,))
            df["Y1"]     = Y1.reshape((-1,))
            df["Y0"]     = Y0.reshape((-1,))
            
            datasets.append(df)

        elif correlated == True and heteroscedastic == True:
            
            # Generate dataset with heteroscedastic errors and correlated covariates
            X       = correlated_covariates(n, d)
            tau     = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5))))

            tau_0   = (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5))))
            
            std     = -np.log(X[:, 0] + 1e-9)
            ps      = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
            errdist = np.random.normal(0, 1, n)

            Y0      = tau_0 #np.zeros(n)
            Y1      = tau + std * errdist
            T       = np.random.uniform(size=n) < ps
            Y       = Y0.copy()
            Y[T]    = Y1[T]
            
            #Pseudolabel calculation
            A       = T
            pi      = ps
            xi      = (A - pi) * Y / (pi * (1 - pi))
            
            #Stratify by conditional variance, CATE
            n_percentiles    = 100
            cate             = Y1 - Y0
            conditional_var  = std**2 * (1 - pi) + pi * std**2 * (1 + tau)**2
            cate_percentiles = np.zeros(n)
            var_percentiles  = np.zeros(n)

            for j in range(n):
                
                cate_percentiles[j] = np.searchsorted(np.percentile(cate, np.linspace(0, 100, n_percentiles+1)), cate[j])
                var_percentiles[j]  = np.searchsorted(np.percentile(conditional_var, np.linspace(0, 100, n_percentiles+1)), conditional_var[j])

            data                  = np.column_stack((X, T, Y))
            column_names          = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
            df                    = pd.DataFrame(data, columns=column_names)
            df['xi']              = xi
            df['cate_percentile'] = cate_percentiles / n_percentiles
            df['var_percentile']  = var_percentiles / n_percentiles 
            df["ps"]              = np.array(ps).reshape((-1,))
            df["Y1"]              = Y1.reshape((-1,))
            df["Y0"]              = Y0.reshape((-1,))
            df["CATE"]            = tau - tau_0

            datasets.append(df)
    
    return datasets
