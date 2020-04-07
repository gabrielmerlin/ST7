import pandas as pd
import numpy as np
import datetime

def optimisation_ERB(mu_sigma_dic):
    w_dic = {}
    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]
        sigma=sigma.to_numpy()
        n = len(sigma)
        Lambda=np.zeros((n,n))
        for i in range(n):
            Lambda[i][i]=sigma[i][i]
            vect = np.ones(n)
        try:
            s_inv_mu = np.linalg.solve(Lambda, vect)
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1=date
            dateprime2=dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month==dateprime2.month):
                date_str=dateprime1.strftime('%Y-%m-%d')
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1=dateprime2
                dateprime2=dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(Lambda, vect, rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1=date
            dateprime2=dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month==dateprime2.month):
                date_str=dateprime1.strftime('%Y-%m-%d')
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1=dateprime2
                dateprime2=dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            #w_dic[date] = pd.Series(w, index=mu.index)
            continue
    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)

def optimisation_MV(mu_sigma_dic):
    w_dic = {}
    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]
        n = len(sigma)
        vect = np.ones(n)
        try:
            s_inv_mu = np.linalg.solve(sigma.to_numpy(), vect)
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1=date
            dateprime2=dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month==dateprime2.month):
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(sigma.to_numpy(), vect, rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1=date
            dateprime2=dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month == dateprime2.month):
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)

def optimisation_EW(mu_sigma_dic):
    w_dic = {}
    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]
        n = len(sigma)
        Lambda=np.identity(n)
        vect = np.ones(n)
        try:
            s_inv_mu = np.linalg.solve(Lambda, vect)
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1 = date
            dateprime2 = dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month == dateprime2.month):
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(Lambda, vect, rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1 = date
            dateprime2 = dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month == dateprime2.month):
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1+datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)
