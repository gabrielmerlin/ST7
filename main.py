# Import des données

import json
from numpy import nan
import pandas as pd
import date_formatting
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt

# Chargement des données json

data_file = open("data_ST7MDS.json")
data = json.load(data_file)

unordonned_market_cap_evol = data['MarketCap']

dic = {}

for i in range(len(unordonned_market_cap_evol)):
    date_i = date_formatting.date_formate(unordonned_market_cap_evol[i]['Date'][0])

    unordonned_market_caps = unordonned_market_cap_evol[i]['MarketCap']

    market_caps = {}

    for j in range(len(unordonned_market_caps)):
        sedol, market_cap = unordonned_market_caps[j]['Sedol'], unordonned_market_caps[j].get('MarketCap', nan)
        market_caps[sedol] = market_cap

    dic[date_i] = market_caps

market_caps = pd.DataFrame(dic).fillna(0)   # Cette table (ligne: sedol, colonne: date) panda contient les marketcaps

# Chargement des données yfinance
# La table panda market possède un multi-indice (sedol, date)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()

# S&P 250
SnP_per_month = []
for end_month_date, market_caps_at_month in market_caps.items():
    # Itération sur tous les mois
    begin_month_date = end_month_date.replace(day=1)

    prices_during_month = market.loc[(slice(None), slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0).sort_index()
    prices_during_month.fillna(method='ffill', inplace=True)
    prices_during_month.fillna(method='bfill', inplace=True)

    if datetime.combine(end_month_date, datetime.min.time()) in prices_during_month.index.to_pydatetime() :
        price_end_month = prices_during_month.loc[str(end_month_date)]
        N_share_month = market_caps_at_month / price_end_month

        SnP_month_unreduced = N_share_month * prices_during_month
        SnP_month = SnP_month_unreduced.agg('sum', axis="columns")
        SnP_per_month.append(SnP_month)

SnP = pd.concat(SnP_per_month)
print(SnP)
SnP.plot()
plt.show()


# # 50 plus grosses capitalisations par mois
# biggest_per_month = []
# for end_month_date, market_caps_at_month in market_caps.items():
#     # Itération sur tous les mois
#     # Déterminer les 50 plus grosses capitalisations
#     rank_month = market_caps_at_month.rank(method='max')
#
#     begin_month_date = end_month_date.replace(day=1)
#     prices_during_month = market.loc[(slice(None), slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0)
#
#     if datetime.combine(end_month_date, datetime.min.time()) in prices_during_month.index.to_pydatetime() :
#         price_end_month = prices_during_month.loc[str(end_month_date)]
#
#
#
#         N_share_month = market_caps_at_month / price_end_month
#         SnP_month_unreduced = N_share_month * prices_during_month
#         SnP_month = SnP_month_unreduced.agg('sum', axis="columns")
#         SnP_per_month.append(SnP_month)
#
# SnP = pd.concat(SnP_per_month)
# print(SnP)
