import numpy as np
import cvxpy as cp
import pandas as pd
import datetime

def mean_covariance_matrix_over_time(market):
    """
    Cette fonction estime pour chaque date mu et sigma.

    :param market: Dataframe Panda contenant les données de marché
    :return: Dictionnaire associant à chaque date le couple mu, sigma
    """
    prices = market['Close'].unstack(0).sort_index()
    rendements = (prices - prices.shift(fill_value=np.nan)) / (prices.shift(fill_value=np.nan))
    rendements = rendements.iloc[1:]

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

        rendements_periode = rendements.loc[slice(str(debut), str(fin))].dropna(axis=1, how='all').fillna(0)

        # calcul du vecteur des rendements à l'aide des données historiques
        mu = rendements_periode.mean()

        #création de la matrice de covariance à l'aide des données historiques
        sigma = rendements_periode.cov().fillna(0)

        mu_sigma_dic[date_debut] = mu, sigma

        if monthd < 12:
            date_debut = datetime.datetime(yeard, monthd + 1, dayd)
        else:
            date_debut = datetime.datetime(yeard + 1, 1, 1)

    return mu_sigma_dic

def optimisateur(mu_sigma_dic):
    """
    Cette fonction détermine un portefeuille en utilisant la méthode MVO

    :param mu_sigma_dic: Dictionnaire associant à chaque date le couple mu, sigma
    :return: Serie Panda associant à chaque multi-indice (sedol, date) le poids convenable
    """
    d = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        w = cp.Variable(mu.size)
        muprime = np.transpose(mu.to_numpy())

        objective = cp.Maximize(sum(muprime * w))

        risk = cp.quad_form(w, sigma.to_numpy())
        constraints = [risk <= 0.2, w >= 0., sum(w) == 1.]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        d[date] = pd.Series(w.value, index=mu.index)

    return pd.DataFrame(d).stack().rename('Poids', axis='column')


#reconstitution du nouvel indice
def valeur_new_indice(market,d):
    value_new= d['Poigts']*market['Close']
    value_new=value_new.reset_index()
    value_new=value_new.groupby(['Date']).sum()
    return(value_new)

if __name__ == "__main__":
    market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
    m_s_d = mean_covariance_matrix_over_time(market)
    print("Estimation finie.")
    w_d = optimisateur(m_s_d)
    print(w_d)









