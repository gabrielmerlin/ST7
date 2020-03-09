import numpy
import cvxpy as cp
import pandas as pd
import json
import datetime
data_file = open("data_ST7MDS.json")
data = json.load(data_file)
market_cap = data['MarketCap']
date_debut= datetime.datetime(2005,1,1)
print(date_debut)

def optimasateur(market,data,sedol):
    n=len (data)
    d={}
    date_debut=datetime.datetime(2005,1,1)
    print(date_debut)
    date_fin= datetime.datetime(2020,2,1)
    while date_debut <= date_fin:#on parcours chaque début de mois pour trouver une stratégie correspondante
        yeard=date_debut.year-2
        monthd=date_debut.month
        dayd=date_debut.day
        #determination des bornes des données historiques
        debut=datetime.datetime(yeard,monthd,dayd)
        fin= date_debut-datetime.timedelta(days=1)
        #creation a partir du tableau existant d'un tableau dans lequel les sedol sont en colonne
        df=market.copy()
        df=df.reset_index()
        #table rendement à partir du tableau market original qui donne pour chaque date les rendements de chaque entreprise
        rendement = (market['Close'] - (market['Close']).shift()) / (market['Close']).shift()
        matrix=rendement.unstack(0)
        #création de la matrice de covariance à l'aide des données historiques
        matrix=matrix.loc[slice(debut,fin)]
        sigma=matrix.cov()
        #création d'un tableau regroupant la moyenne des rendements historiques par entreprise et sur deux ans
        df['rendement'] = (df['Close'] - (df['Close']).shift()) / (df['Close']).shift()
        colonnes = ['level_0','rendement']
        df=df.loc[(df['Date']>=debut) &(df['Date']<=fin),colonnes]
        df2= df.groupby(['level_0']).mean()
        w = cp.Variable(len(df2))
        mu_historique=df2['rendement'].to_numpy()
        muprime=numpy.transpose(mu_historique)
        objective = cp.Maximize(muprime * w)
        risk = cp.quad_form(w, sigma.to_numpy())
        constraints = [risk <= 0.1 ** 2]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        d[date_debut]= w.value
        if monthd<12:
            date_debut=datetime.datetime(yeard,monthd+1,dayd)
        else:
            date_debut=datetime.datetime(yeard+1,1,1)
    return(d)

#def valeur_new_indice(d):









