
import pandas as pd
from datetime import datetime

def valid_SNP_250_sedols(market_caps):
    """
    Cette fonction extrait les sedols du S&P250 pour chaque mois
    :param market_caps: Tableau Panda avec en ligne les dates et en colonne la capitalisation (S&P 250) par sedol
    :return:            Série Panda associant à chaque date les sedols correspondant
    """

    selected_sedols = {}

    for end_month_date, market_caps_at_month in market_caps.items():
        begin_month_date = end_month_date + pd.Timedelta(weeks=2)
        begin_month_date = pd.Timestamp(year=begin_month_date.year, month=begin_month_date.month, day=1)
        selected_sedols[begin_month_date] = market_caps_at_month.index[market_caps_at_month.isnull().to_numpy()]

    return selected_sedols


def SnP_250(market_caps, market):
    """
    Cette fonction calcule le S&P 250.

    :param market_caps: Tableau Panda avec en ligne les dates et en colonne la capitalisation par sedol
    :param market:      Tableau Panda avec en ligne les multi-indices (sedol, date) et en colonne des données
    :return:            Série Panda associant à chaque date le S&P 250 correspondant
    """
    SnP_per_month = []
    for end_month_date, market_caps_at_month in market_caps.items():
        # Itération sur tous les mois
        begin_month_date = end_month_date.replace(day=1)

        # Extraction des données de prix sur le mois et suppression des NaN
        prices_during_month = market.loc[(slice(None), slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0).sort_index()
        prices_during_month.fillna(method='ffill', inplace=True)
        prices_during_month.fillna(method='bfill', inplace=True)

        if datetime.combine(end_month_date, datetime.min.time()) in prices_during_month.index.to_pydatetime():
            price_end_month = prices_during_month.loc[str(end_month_date)]
            N_share_month = market_caps_at_month / price_end_month

            SnP_month_unreduced = N_share_month * prices_during_month
            SnP_month = SnP_month_unreduced.agg('sum', axis="columns").rename('S&P250')
            SnP_per_month.append(SnP_month)

    return pd.concat(SnP_per_month)
