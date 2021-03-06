import numpy as np
import cvxpy as cp
import pandas as pd
import datetime
from Mesure_de_risque import *
import matplotlib.pyplot as plt

from S_and_P_250 import valid_SNP_250_sedols, SnP_250
from import_market_caps import import_data_from_json

def date_intiale():
    return (datetime.datetime(2004,12,31))

def mean_covariance_matrix_over_time(market, selected_sedols):
    """
    Cette fonction estime pour chaque date mu et sigma.

    :param market: Dataframe Panda contenant les données de marché
    :param selected_sedols: Dictionnaire associant à chaque date la liste des sedols à considérer
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
        rendements_periode = rendements_periode.transpose().reindex(selected_sedols[date_debut].to_list()).transpose()

        # calcul du vecteur des rendements à l'aide des données historiques
        mu = rendements_periode.mean().replace([np.nan, np.inf, - np.inf], 0)

        #création de la matrice de covariance à l'aide des données historiques
        sigma = rendements_periode.cov().replace([np.nan, np.inf, - np.inf], 0)

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

    weights = pd.DataFrame(d).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)

def optimisation_MVO(mu_sigma_dic):
    """
    Cette fonction détermine un portefeuille en utilisant la méthode MVO implémentée avec les formules exactes.

    :param mu_sigma_dic: Dictionnaire associant à chaque date le couple mu, sigma
    :return: Serie Panda associant à chaque multi-indice (sedol, date) le poids convenable
    """
    w_dic = {}

    for date in mu_sigma_dic:
        mu, sigma = mu_sigma_dic[date]

        try:
            s_inv_mu = np.linalg.solve(sigma.to_numpy(), mu.to_numpy())
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1 = date
            dateprime2 = dateprime1+datetime.timedelta(days=1)
            while (dateprime1.month == dateprime2.month):
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1 + datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue
        except np.linalg.linalg.LinAlgError:
            s_inv_mu = np.linalg.lstsq(sigma.to_numpy(), mu.to_numpy(), rcond=None)[0]
            w = s_inv_mu / np.sum(s_inv_mu)
            dateprime1=date
            dateprime2=dateprime1+datetime.timedelta(days=1)
            while dateprime1.month==dateprime2.month:
                w_dic[dateprime1] = pd.Series(w, index=mu.index)
                dateprime1 = dateprime2
                dateprime2 = dateprime1 + datetime.timedelta(days=1)
            w_dic[dateprime1] = pd.Series(w, index=mu.index)
            continue

    weights = pd.DataFrame(w_dic).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)

def rendement(market, selected_sedol):
    dic = {}
    prices = market['Close'].unstack(0).sort_index()

    rendements = (prices - prices.shift(fill_value=np.nan)) / (prices.shift(fill_value=np.nan))
    rendements = rendements.iloc[1:]

    for date in selected_sedol:
        end_month_date = pd.Timestamp(date.year, date.month, date.days_in_month)

        rendements_monthly = rendements.loc[slice(date, end_month_date)]
        rendements_monthly = rendements_monthly.transpose().reindex(selected_sedol[date]).transpose()

        dic[date] = rendements_monthly.to_dict(orient='series')

    return dic

def rend_total(weights,rend):
    dic={}
    for date in rend:
        h=weights[date]*rend[date]
        dic[date]=h

def optimisation_rob(mu_sigma_dict,lan,k):
    w_d = {}
    for date in mu_sigma_dict:
        mu, sigma = mu_sigma_dict[date]
        w = cp.Variable(mu.size)
        sigma = sigma.to_numpy()
        n = mu.size
        omega = np.zeros((n, n))
        omega_sqrt = np.zeros((n, n))

        zero_indices = []

        for i in range(n):
            if mu[i] <= 0 or sigma[i][i] < 0:
                zero_indices.append(i)
            else:
                omega[i][i] = sigma[i][i]
                omega_sqrt[i][i] = np.sqrt(sigma[i][i])

        risk = cp.quad_form(w, omega)
        risk = cp.multiply(lan/2, risk)
        error = cp.norm(omega_sqrt * w, 2)   # √ w.t * omega * w
        error = cp.multiply(k, error)
        objective = cp.Maximize((mu.to_numpy() * w) - risk - error)
        constraints = [w >= 0, cp.sum(w) == 1] + [w[i] == 0 for i in zero_indices]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        w_d[date] = pd.Series(w.value, index=mu.index)

    weights = pd.DataFrame(w_d).stack().rename('Poids', axis='column')
    indices = weights.index
    indices.set_names('Date', level=1, inplace=True)

    return weights.fillna(0)


#reconstitution du nouvel indice
def valeur_new_indice(market, weights, value0):
    """
    Cette fonction détermine les valeurs associées au nouvel indice donné en paramètre.

    :param market: DataFrame contenant les prix à la fermeture en fonction du sedol et de la date
    :param weights: Serie contenant les poids alloués à chaque actif mensuellement
    :param value0: Valeur initiale du portefeuille
    :return: Serie associant à chaque date la valeur du portefeuille
    """

    # Formatage avec les dates en lignes et les sedols en colonne
    u_weights = weights.unstack(0)
    begin_date = u_weights.first_valid_index()
    end_date = u_weights.last_valid_index()

    prices = market['Close'].sort_index().loc[(slice(None), slice(begin_date, end_date))].unstack(0).fillna(method='pad').fillna(method='backfill')

    value = value0
    current_month = begin_date.month
    cap_quantity = u_weights.loc[begin_date] / prices.loc[prices.first_valid_index()] * value
    values = {begin_date: value0}

    for date, prices_today in prices.iterrows():
        if date.month != current_month:
            # Actualiser les quantités de chaque actif
            current_month = date.month
            begin_month_date = pd.Timestamp(year=date.year, month=current_month, day=1)
            cap_quantity = u_weights.loc[begin_month_date] / prices_today * value
        prod = cap_quantity * prices_today
        values[date] = prod.sum()

    return pd.Series(values)

def rendements_portfolio(market, weights):
    # Formatage avec les dates en lignes et les sedols en colonne
    u_weights = weights.unstack(0)
    begin_date = u_weights.first_valid_index()
    end_date = u_weights.last_valid_index()

    prices = market['Close'].sort_index().loc[(slice(None), slice(begin_date, end_date))].unstack(0).fillna(
        method='pad').fillna(method='backfill')

    market_rend = prices.diff()/prices
    market_rend = market_rend.iloc[1:]
    market_rend = market_rend.clip(-5, 5)

    prod = market_rend * u_weights

    return prod.sum(axis=1)

def rendement_portfolio2(weights, rendements):
    """
    Cette fonction calcule le rendement d'un portefeuille
    :param weights:
    :param rendements:
    :return:
    """
    rend_pf = {}

    for date in rendements:
        begin_month_date = date
        begin_month_date.day = 1
        prod = rendements[date] * weights[date]
        rend_pf[date] = prod.sum()

    return pd.DataFrame(rend_pf)

def prices_by_weights(market, weights):
    u_weights = weights.unstack(0)
    begin_date = u_weights.first_valid_index()
    end_date = u_weights.last_valid_index()

    prices = market['Close'].sort_index().loc[(slice(None), slice(begin_date, end_date))].unstack(0).fillna(
        method='pad').fillna(method='backfill')

    prod = prices * u_weights

    return prod.sum(axis=1)

def pick_clip(values):
    """
    Cette fonction supprime les variations trop importantes dans values.
    :param values:
    :return: Les
    """
    variations = values.diff() / values
    variations = variations.iloc[1:].abs()

    last_values = values[0]

    for date, variations in variations.items():
        if np.max(variations) > 1.5:
            values[date] = last_values
            print(last_values, values)
        else:
            last_values = values.loc[date]


lan = 4
k = 0.2

if __name__ == "__main__":
    market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
    market_caps, Rf = import_data_from_json()
    Rf = Rf - 1
    print(Rf)

    m_s_d = mean_covariance_matrix_over_time(market, valid_SNP_250_sedols(market_caps))
    print("Estimation finie.")
    SnP = SnP_250(market_caps, market)
    print(SnP)

    w_d = optimisation_ERB(m_s_d)
    print("ERB")
    rend = rendements_portfolio(market, w_d)
    print("Vol:", volatility(rend))
    print("Mean rend", rendement_moyen(rend))
    print("SR", shape_ration(rend, Rf))
    print("Var", VAR(rend, 0.95))
    print("CVaR", CVAR(rend, 0.95))
    print("Tracking", tracking_error(rend, SnP))

    w_d = optimisation_EW(m_s_d)
    print("EW")
    rend = rendements_portfolio(market, w_d)
    print("Vol:", volatility(rend))
    print("Mean rend", rendement_moyen(rend))
    print("SR", shape_ration(rend, Rf))
    print("Var", VAR(rend, 0.95))
    print("CVaR", CVAR(rend, 0.95))
    print("Tracking", tracking_error(rend, SnP))

    w_d = optimisation_MVO(m_s_d)
    print("MVO")
    rend = rendements_portfolio(market, w_d)
    print("Vol:", volatility(rend))
    print("Mean rend", rendement_moyen(rend))
    print("SR", shape_ration(rend, Rf))
    print("Var", VAR(rend, 0.95))
    print("CVaR", CVAR(rend, 0.95))
    print("Tracking", tracking_error(rend, SnP))

    w_d = optimisation_MV(m_s_d)
    print("MV")
    rend = rendements_portfolio(market, w_d)
    print("Vol:", volatility(rend))
    print("Mean rend", rendement_moyen(rend))
    print("SR", shape_ration(rend, Rf))
    print("Var", VAR(rend, 0.95))
    print("CVaR", CVAR(rend, 0.95))
    print("Tracking", tracking_error(rend, SnP))

    w_d = optimisation_rob(m_s_d, lan, k)
    print("rob")
    rend = rendements_portfolio(market, w_d)
    print("Vol:", volatility(rend))
    print("Mean rend", rendement_moyen(rend))
    print("SR", shape_ration(rend, Rf))
    print("Var", VAR(rend, 0.95))
    print("CVaR", CVAR(rend, 0.95))
    print("Tracking", tracking_error(rend, SnP))

    d = valeur_new_indice(market, w_d, 8450064)
    pick_clip(d)
    d.plot()
    plt.title('Optimisation robuste')
    plt.ylabel("Valeur")
    plt.xlabel("Date")

    plt.figure()
    rend = rendements_portfolio(market, w_d)
    pick_clip(rend)
    plt.title('Optimisation robuste')
    plt.ylabel("Rendement")
    plt.xlabel("Date")
    rend.plot()
    plt.show()

