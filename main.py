# Import des données

import json
import yfinance as yf

# Chargement des données

data_file = open("data_ST7MDS.json")
data = json.load(data_file)
market = {}
for value in data['Mapping']:
    price = yf.Ticker(value['Ticker'])
    market[value["Sedol"]] = price.history(period="max")

# S&P 250

#for t in range()
