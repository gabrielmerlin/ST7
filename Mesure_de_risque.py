import pandas as pd
import numpy as np

def volatility(indice):
    return(indice.std())
def rendement_moyen(indice):
    #rendements = (indice - indice.shift(fill_value=np.nan)) / (indice.shift(fill_value=np.nan))
    rendements=indice.diff()/indice
    rendements = rendements.iloc[1:]
    rendements=rendements.dropna()
    return(rendements)
def shape_ration(indice,Rf):
    SR=(rendement_moyen(indice).mean()-Rf.mean())/volatility(indice)
    return(SR)

def VAR(indice,alpha):
   Var=indice.quantile(alpha)
   #Var=Var.to_numpy()
   return(Var)

def CVAR(indice, alpha):# ici on utilise plûtot en entrée les rendements des indice
    var=VAR(indice,alpha)
    cvar= indice[indice > var].mean()
    return(cvar)

def maximum_drawdown(indice):
    return (indice.max()-indice.min()/indice.max())

def tracking_error(indice,benchmarck) :#ici le benchmark est le rendement du portefeuilleSNP
    Diff=indice-benchmarck
    return(Diff.std())




