
import yfinance as yf
import pandas as pd
import json

data_file = open("data_ST7MDS.json")
data = json.load(data_file)

# Chargement des données yfinance
market_list = {}
sedol_list = []

for value in data['Mapping']:
    price = yf.Ticker(value['Ticker'])
    market_list[value["Sedol"]] = price.history(start="2002-12-01", end="2020-02-14")
    #print(type(market_dic[value["Sedol"]]))
    sedol_list.append(value["Sedol"])
market = pd.concat(market_list, keys=sedol_list).sort_index()
#print(market)   # Cette table panda possède un multi-indice (sedol, date)

market.to_json("data_yfinance.json")