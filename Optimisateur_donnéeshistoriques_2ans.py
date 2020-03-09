import numpy as np
import cvxpy as cp
import pandas as pd
import json
import datetime

from date_formatting import date_formate

data_file = open("data_ST7MDS.json")
data = json.load(data_file)

unordonned_market_cap_evol = data['MarketCap']

dic = {}

for i in range(len(unordonned_market_cap_evol)):
    date_i = date_formate(unordonned_market_cap_evol[i]['Date'][0])

    unordonned_market_caps = unordonned_market_cap_evol[i]['MarketCap']

    market_caps = {}

    for j in range(len(unordonned_market_caps)):
        sedol, market_cap = unordonned_market_caps[j]['Sedol'], unordonned_market_caps[j].get('MarketCap', 0)
        market_caps[sedol] = market_cap

    dic[date_i] = market_caps

market_caps = pd.DataFrame(dic)


date_debut= datetime.datetime(2005,1,1)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()

def mean_covariance_matrix_over_time(market):
    """
    Cette fonction estime pour chaque date mu et sigma

    :param market: Dataframe Panda contenant les données de marché
    :return: Dictionnaire associant à chaque date le couple mu, sigma
    """
    rendements = (market['Close'] - market['Close'].shift()) / (market['Close'].shift())
    rendements = rendements.unstack(0).sort_index()

    date_debut = datetime.datetime(2005, 1, 1)
    date_fin = datetime.datetime(2005, 2, 1)

    mu_sigma_dic = {}

    while date_debut < date_fin:
        # On parcourt chaque début de mois pour trouver une stratégie correspondante
        yeard = date_debut.year
        monthd = date_debut.month
        dayd = date_debut.day

        #determination des bornes des données historiques
        debut = datetime.datetime(yeard - 3, monthd, dayd)
        fin = date_debut - datetime.timedelta(days=1)

        print(debut, fin)

        rendements_periode = rendements.loc[slice(str(debut), str(fin))].dropna(axis=1,how='all')
        print(rendements_periode)

        # calcul du vecteur des rendements à l'aide des données historiques
        mu = rendements_periode.mean()

        #création de la matrice de covariance à l'aide des données historiques
        sigma = rendements_periode.cov().fillna(0)

        print(sigma)

        mu_sigma_dic[date_debut] = mu, sigma

        if monthd<12:
            date_debut = datetime.datetime(yeard, monthd + 1, dayd)
        else:
            date_debut = datetime.datetime(yeard + 1, 1, 1)

    return mu_sigma_dic

def optimisateur(mu_sigma_dic, market_caps):
    """
    Cette fonction détermine un portefeuille en utilisant la méthode MVO

    :param mu_sigma_dic: Dictionnaire associant à chaque date le couple mu, sigma
    :param market_caps:
    :return: Dictionnaire associant à chaque date un tableau numpy contenant les poids de chaque actif
    """
    d = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        w = cp.Variable(mu.size)
        muprime = np.transpose(mu.to_numpy())

        objective = cp.Maximize(muprime * w)

        print(sigma.to_numpy().shape, mu.size)
        print(np.linalg.matrix_rank(sigma))

        risk = cp.quad_form(w, sigma.to_numpy())
        constraints = [risk <= 0.2]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        d[date] = pd.Series(w.value, index=mu.index)
        print(d[date])

    return pd.DataFrame(d)

m_s_d = mean_covariance_matrix_over_time(market)
print("Estimation finie.")
w_d = optimisateur(m_s_d, market_caps)
print(w_d)

#def valeur_new_indice(d):









