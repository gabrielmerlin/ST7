# Import des données

import json
import yfinance as yf
import pandas as pd
import date_formatting

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
        sedol, market_cap = unordonned_market_caps[j]['Sedol'], unordonned_market_caps[j].get('MarketCap', 0)
        market_caps[sedol] = market_cap

    dic[date_i] = market_caps

market_caps = pd.DataFrame(dic).T   # Cette table (ligne: date, colonne: sedol) panda contient les marketcaps

# Chargement des données yfinance
market_list = {}
sedol_list = []
i = 0
for value in data['Mapping']:
    if i < 2:
        price = yf.Ticker(value['Ticker'])
        market_list[value["Sedol"]] = price.history(start="2002-12-31", end="2020-02-14")
        #print(type(market_dic[value["Sedol"]]))
        sedol_list.append(value["Sedol"])
        i += 1
market = pd.concat(market_list, keys=sedol_list).sort_index()
print(market)   # Cette table panda possède un multi-indice (sedol, date)

# S&P 250

for end_month_date, market_caps_at_month in market_caps.iterrows():
    begin_month_date = end_month_date.replace(day=1)
    prices_during_month = market.loc[(slice(None), slice(str(begin_month_date), str(end_month_date))), 'Close'].unstack(0)


