# Import des données

import pandas as pd
from S_and_P_250 import SnP_250
from index_50 import index_50_biggest, index_50_smallest
from import_market_caps import import_data_from_json
import matplotlib.pyplot as plt

# Chargement des données json

market_caps, rend_sans_risque = import_data_from_json()
print(market_caps)

# Chargement des données yfinance
# La table panda market possède un multi-indice (sedol, date)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
#print(market)
#market2=market.reset_index()
rendement= (market['Close'] - (market['Close']).shift() )/(market['Close']).shift()
print(rendement.unstack(0))

# S&P 250
SnP = SnP_250(market_caps, market)
print(SnP)
SnP.plot()

from Optimisateur_donnéeshistoriques_2ans import *

m_s_d = mean_covariance_matrix_over_time(market)
print("Estimation finie.")
w_d = optimisation_MV(m_s_d)
d = valeur_new_indice(market, w_d) * 1000
print(d)
d.plot()

plt.show()

# Autres indices
biggest_ind = index_50_biggest(market_caps, market)
print(biggest_ind)
biggest_ind.plot()

smallest_ind = index_50_smallest(market_caps, market)
print(smallest_ind)
smallest_ind.plot()

plt.show()


