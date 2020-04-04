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
        # Parcourir chaque mois pour estimer les paramètres
        yeard = date_debut.year
        monthd = date_debut.month
        dayd = date_debut.day

        # Détermination des bornes des données historiques
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
    w_dic = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        w = cp.Variable(mu.size)
        muprime = np.transpose(mu.to_numpy())

        objective = cp.Maximize(sum(muprime * w))

        risk = cp.quad_form(w, sigma.to_numpy())
        constraints = [risk <= 0.2, w >= 0., sum(w) == 1.]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        w_dic[date] = pd.Series(w.value, index=mu.index)

    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights

def optimisation_MVO(mu_sigma_dic):
    """
    Cette fonction détermine un portefeuille en utilisant la méthode MVO implémentée avec les formules exactes.

    :param mu_sigma_dic: Dictionnaire associant à chaque date le couple mu, sigma
    :param risk_max : Valeur maximale pour le risque
    :return: Serie Panda associant à chaque multi-indice (sedol, date) le poids convenable
    """
    w_dic = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        try:
            s_inv_mu = np.linalg.solve(sigma.to_numpy(), mu.to_numpy())
            w = s_inv_mu / np.sum(s_inv_mu)
            w_dic[date] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(sigma.to_numpy(), mu.to_numpy(), rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            w_dic[date] = pd.Series(w, index=mu.index)
            continue

    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights

def optimisation_MV(mu_sigma_dic):
    w_dic = {}
    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]
        n=len(sigma)
        vect=np.ones(n)
        try:
            s_inv_mu = np.linalg.solve(sigma.to_numpy(), vect)
            w = s_inv_mu / np.sum(s_inv_mu)
            w_dic[date] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(sigma.to_numpy(), vect, rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            w_dic[date] = pd.Series(w, index=mu.index)
            continue

    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights

def optimisation_rob(mu_sigma_dict,lan,k):
    w_d={}
    for date in mu_sigma_dict:
        mu, sigma = mu_sigma_dict[date]

        sigma = sigma.to_numpy()
        n = len(sigma)

        omega = np.zeros((n, n))
        omega_sqrt = np.zeros((n, n))   # omega étant diagonale, omega_sqrt est sa racine carrée

        i = 0
        while i < n:
            omega[i][i] = sigma[i][i]
            omega_sqrt[i][i] = np.sqrt(sigma[i][i])
            if sigma[i][i] == 0 and mu[i] == 0:
                # il faut éliminer la variable
                i -= 1
                n -= 1
            else:
                i += 1

        w = cp.Variable(n)

        risk = cp.quad_form(w, omega)
        risk = cp.multiply(lan/2, risk)
        error = cp.norm(omega_sqrt * w, 2)   # √ w.t * omega * w
        error = cp.multiply(k, error)
        objective = cp.Maximize((mu.to_numpy() * w) - risk - error)
        constraints = [w >= 0, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True)
        w_d[date] = pd.Series(w.value, index=mu.index)

    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights


#reconstitution du nouvel indice
def valeur_new_indice(market, d):
    price = pd.DataFrame(market['Close'].loc[(slice(None), slice('2005-01-01', '2020-01-01'))]).fillna(method='pad').fillna(method='backfill')
    weight_price = price.join(d, how='inner')
    value_new = weight_price.prod(axis=1)
    return value_new.sum(level=1)

lan = 4
k = 0.2

if __name__ == "__main__":
    market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
    print(market['Close'].loc[(slice(None), slice('2005-01-01', '2020-01-01'))])
    m_s_d = mean_covariance_matrix_over_time(market)
    print("Estimation finie.")
    #print(m_s_d)
    w_d = optimisateur(m_s_d)
    print(w_d)

    print(valeur_new_indice(market, w_d))


