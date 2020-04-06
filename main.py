# Import des données

import pandas as pd
from S_and_P_250 import SnP_250
from index_50 import index_50_biggest, index_50_smallest
from import_market_caps import import_data_from_json
import matplotlib.pyplot as plt
from Optimisateur_donnéeshistoriques_2ans import *

# Chargement des données json

market_caps, rend_sans_risque = import_data_from_json()
print(market_caps)

# Chargement des données yfinance
# La table panda market possède un multi-indice (sedol, date)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()


# S&P 250
SnP = SnP_250(market_caps, market)
print(SnP)
SnP.plot()



