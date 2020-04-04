import pandas as pd
import numpy as np
def volatility(indice):
    return(indice.std())
def rendement_moyen(indice):
    rendements = (indice - indice.shift(fill_value=np.nan)) / (indice.shift(fill_value=np.nan))
    rendements = rendements.iloc[1:]
    return(rendements.mean())
def shape_ration(indice,Rf):
    SR=(rendement_moyen(indice)-Rf)/volatility(indice)
    return(SR)

def VAR(indice,alpha):
   Var=indice.quantile(alpha)
   #Var=Var.to_numpy()
   return(Var)

def CVAR(indice,alpha):# ici on utilise plûtot en entrée les rendements des indice
    var=VAR(indice,alpha)
    cvar=[indice[indice[0]>var]].mean()
    return(cvar)




