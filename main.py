# Import des données

import json
import yfinance as yf

# Chargement des données json

data_file = open("data_ST7MDS.json")
data = json.load(data_file)
market = {}

# Chargement des données yfinance
for value in data['Mapping']:
    price = yf.Ticker(value['Ticker'])
    market[value["Sedol"]] = price.history(start="2002-12-31", end="2020-02-14")


# S&P 250

#for t in range()
