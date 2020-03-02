# Import des données

import json
import pandas as pd
import date_formatting
from S_and_P_250 import SnP_250
from index_50 import index_50_biggest
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
        sedol, market_cap = unordonned_market_caps[j]['Sedol'], unordonned_market_caps[j].get('MarketCap', 0)
        market_caps[sedol] = market_cap

    dic[date_i] = market_caps

market_caps = pd.DataFrame(dic)  # Cette table (ligne: sedol, colonne: date) panda contient les marketcaps

# Chargement des données yfinance
# La table panda market possède un multi-indice (sedol, date)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()

# S&P 250
#SnP = SnP_250(market_caps, market)
SnP = index_50_biggest(market_caps, market)
print(SnP)
SnP.plot()
plt.show()


