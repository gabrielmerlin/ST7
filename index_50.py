
import pandas as pd
from datetime import datetime

def index_50_biggest(market_caps, market):
    """
    Cette fonction calcule l'indice sur les 50 plus grosses capitalisations.

    :param market_caps: Tableau Panda avec en ligne les dates et en colonne la capitalisation par sedol
    :param market:      Tableau Panda avec en ligne les multi-indices (sedol, date) et en colonne des données
    :return:            Série Panda associant à chaque date l'indice correspondant
    """
    index_per_month = []
    for end_month_date, market_caps_at_month in market_caps.items():
        # Itération sur tous les mois
        begin_month_date = end_month_date.replace(day=1)

        # Sélection des 50 plus grosses capitalisations
        selected_market_caps_at_month = market_caps_at_month.nlargest(50)
        selected_sedols = selected_market_caps_at_month.index.to_list()

        # Extraction des données de prix sur le mois et suppression des NaN
        prices_during_month = market.loc[(selected_sedols, slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0).sort_index()
        prices_during_month.fillna(method='ffill', inplace=True)
        prices_during_month.fillna(method='bfill', inplace=True)

        if datetime.combine(end_month_date, datetime.min.time()) in prices_during_month.index.to_pydatetime() :
            price_end_month = prices_during_month.loc[str(end_month_date)]
            N_share_month = selected_market_caps_at_month / price_end_month

            ind_month_unreduced = N_share_month * prices_during_month
            ind_month = ind_month_unreduced.agg('sum', axis="columns").rename('Big caps 250')
            index_per_month.append(ind_month)

    return pd.concat(index_per_month)

def index_50_smallest(market_caps, market):
    """
    Cette fonction calcule l'indice sur les 50 plus petites capitalisations.

    :param market_caps: Tableau Panda avec en ligne les dates et en colonne la capitalisation par sedol
    :param market:      Tableau Panda avec en ligne les multi-indices (sedol, date) et en colonne des données
    :return:            Série Panda associant à chaque date l'indice correspondant
    """
    index_per_month = []
    for end_month_date, market_caps_at_month in market_caps.items():
        # Itération sur tous les mois
        begin_month_date = end_month_date.replace(day=1)

        # Sélection des 50 plus petites capitalisations
        selected_market_caps_at_month = market_caps_at_month.nsmallest(50)
        selected_sedols = selected_market_caps_at_month.index.to_list()

        # Extraction des données de prix sur le mois et suppression des NaN
        prices_during_month = market.loc[(selected_sedols, slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0).sort_index()
        prices_during_month.fillna(method='ffill', inplace=True)
        prices_during_month.fillna(method='bfill', inplace=True)

        if datetime.combine(end_month_date, datetime.min.time()) in prices_during_month.index.to_pydatetime() :
            price_end_month = prices_during_month.loc[str(end_month_date)]
            N_share_month = selected_market_caps_at_month / price_end_month

            ind_month_unreduced = N_share_month * prices_during_month
            ind_month = ind_month_unreduced.agg('sum', axis="columns").rename('Small caps 50')
            index_per_month.append(ind_month)

    return pd.concat(index_per_month)
