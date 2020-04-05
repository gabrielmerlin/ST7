import json
import pandas as pd

def import_data_from_json():
    """
    Cette fonction importe le contenu du fichier json
    :return: couple formé d'un DataFrame contenant les marketcaps (ligne : sedol, colonne : date) et Serie contenant
                les rendements sans risques
    """
    data_file = open("data_ST7MDS.json")
    data = json.load(data_file)

    # Import des parts de marchés

    unordonned_market_cap_evol = data['MarketCap']

    dic = {}

    for i in range(len(unordonned_market_cap_evol)):
        date_i = pd.to_datetime(unordonned_market_cap_evol[i]['Date'][0])

        unordonned_market_caps = unordonned_market_cap_evol[i]['MarketCap']

        market_caps = {}

        for j in range(len(unordonned_market_caps)):
            sedol, market_cap = unordonned_market_caps[j]['Sedol'], unordonned_market_caps[j].get('MarketCap', 0)
            market_caps[sedol] = market_cap

        dic[date_i] = market_caps

    market_caps = pd.DataFrame(dic)   # Cette table (ligne: sedol, colonne: date) panda contient les marketcaps

    # Import des rendements sans risques

    rendements_sans_risque = pd.DataFrame(data['Cash'])
    rendements_sans_risque['Date'] = pd.to_datetime(rendements_sans_risque['Date'])
    rendements_sans_risque.set_index('Date', inplace=True)

    return market_caps, rendements_sans_risque