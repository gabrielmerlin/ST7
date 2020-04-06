# Import des données

import pandas as pd
from S_and_P_250 import SnP_250
from index_50 import index_50_biggest, index_50_smallest
from import_market_caps import import_data_from_json
import matplotlib.pyplot as plt
import Mesure_de_risque as mr
from Optimisateur_donnéeshistoriques_2ans import *

# Chargement des données json

market_caps, rend_sans_risque = import_data_from_json()
rend_sans_risque=rend_sans_risque-1

# Chargement des données yfinance
# La table panda market possède un multi-indice (sedol, date)

market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
print('fin calcul')

# S&P 250
SnP = SnP_250(market_caps, market)
rendemen_SnP=mr.rendement_moyen(SnP)
 #détermination de la valeur initiale à allouer
date_initiale=date_intiale()
valeur_initiale=SnP.loc[date_initiale]

# creation de la matrice historique de variance,covariance
m_s_d = mean_covariance_matrix_over_time(market)
print('estimation finie')
#optimisation MVO
w_d = optimisateur(m_s_d)
d = valeur_new_indice(market, w_d, valeur_initiale)
d.plot()
plt.title('optimisation MVO avec contrainte')
plt.xlabel('days')
plt.ylabel('value')
plt.show()







