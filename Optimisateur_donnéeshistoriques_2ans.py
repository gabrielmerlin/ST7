import numpy as np
import cvxpy as cp
import pandas as pd
import json
import datetime

data_file = open("data_ST7MDS.json")
data = json.load(data_file)
market_cap = data['MarketCap']
date_debut= datetime.datetime(2005,1,1)
print(date_debut)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
#print(market)

def mean_covariance_matrix_over_time(market):
    """
    Cette fonction estime pour chaque date mu et sigma

    :param market: Dataframe Panda contenant les données de marché
    :return: Dictionnaire associant à chaque date le couple mu, sigma
    """
    rendements = (market['Close'] - market['Close'].shift()) / (market['Close'].shift())
    rendements = rendements.unstack(0).sort_index()

    date_debut = datetime.datetime(2005, 1, 1)
    date_fin = datetime.datetime(2020, 2, 1)

    mu_sigma_dic = {}

    while date_debut < date_fin:
        # On parcourt chaque début de mois pour trouver une stratégie correspondante
        yeard = date_debut.year
        monthd = date_debut.month
        dayd = date_debut.day

        #determination des bornes des données historiques
        debut = datetime.datetime(yeard - 2, monthd, dayd)
        fin = date_debut - datetime.timedelta(days=1)

        rendements_periode = rendements.loc[slice(str(debut), str(fin))]

        # calcul du vecteur des rendements à l'aide des données historiques
        mu = rendements_periode.mean()

        #création de la matrice de covariance à l'aide des données historiques
        sigma = rendements_periode.cov()

        mu_sigma_dic[date_debut] = mu, sigma

        if monthd<12:
            date_debut = datetime.datetime(yeard, monthd + 1, dayd)
        else:
            date_debut = datetime.datetime(yeard + 1, 1, 1)

    return mu_sigma_dic

def optimisateur(mu_sigma_dic):
    """
    Cette fonction détermine un portefeuille en utilisant la méthode MVO

    :param mu_sigma_dic: Dictionnaire associant à chaque date le couple mu, sigma
    :return: Dictionnaire associant à chaque date un tableau numpy contenant les poids de chaque actif
    """
    d = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        w = cp.Variable(mu.size())
        muprime = np.transpose(mu.to_numpy())

        objective = cp.Maximize(muprime * w)

        risk = cp.quad_form(w, sigma.to_numpy())
        constraints = [risk <= 0.1 ** 2]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        d[date] = w.value

    return(d)

m_s_d = mean_covariance_matrix_over_time(market)
w_d = optimisateur(m_s_d)
print(w_d)
#reconstitution du nouvel indice
def valeur_new_indice(market,d):
    value_new= d['Poigts']*market['Close']
    value_new=value_new.reset_index()
    value_new=value_new.groupby(['Date']).sum()
    return(value_new)










