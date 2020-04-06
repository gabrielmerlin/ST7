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
#market=pd.read_csv('prices.csv')
#print(market)
market = pd.read_pickle("data_yfinance.pkl.gz", compression="gzip").reindex()
#print('fin calcul')

SnP = SnP_250(market_caps, market)

big50 = index_50_biggest(market_caps, market)
big50.plot()
plt.title("Portefeuille avec les 50 plus grosses capitalisations")
plt.xlabel("Date")
plt.ylabel("Valeur")

rendmax = big50.diff()/big50
plt.figure()
rendmax.plot()
plt.title("Portefeuille avec les 50 plus grosses capitalisations")
plt.xlabel("Date")
plt.ylabel("Rendement")

min50 = index_50_smallest(market_caps, market)
plt.figure()
min50.plot()
plt.title("Portefeuille avec les 50 plus petites capitalisations")
plt.xlabel("Date")
plt.ylabel("Valeur")

rendmin = min50.diff()/min50
plt.figure()
rendmin.plot()
plt.title("Portefeuille avec les 50 plus petites capitalisations")
plt.xlabel("Date")
plt.ylabel("Rendement")

plt.figure()
SnP = SnP_250(market_caps, market)
SnP.plot()
big50.plot()
min50.plot()
plt.title("Comparaisons des portefeuilles")
plt.xlabel("Date")
plt.ylabel("Valeur")
plt.legend(('S&P 250', '50 plus grosses capitalisations', '50 plus petites capitalisations'))
plt.show()

# S&P 250
#SnP = SnP_250(market_caps, market)
#rendemen_SnP=mr.rendement_moyen(SnP)
 #détermination de la valeur initiale à allouer
#date_initiale=date_intiale()
#valeur_initiale=SnP.loc[date_initiale]

# creation de la matrice historique de variance,covariance
#date_debut = datetime.datetime(2005, 1, 1)
#m_s_d = mean_covariance_matrix_over_time(market)
#print('estimation finie')
#optimisation MVO
#w_d = optimisation_MV(m_s_d)
#d = valeur_new_indice(market, w_d, valeur_initiale)
#print('valeur')
#rend=mr.rendement_moyen(d)
#p=mr.VAR(rend,0.95)
#print(p)
#d.plot()
#plt.title('optimisation minimum variance')
#plt.xlabel('days')
#plt.ylabel('value')
#plt.show()

