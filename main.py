# Import des données

import json
import yfinance as yf

data_file = open("data_ST7MDS.json")
data = json.load(data_file)
i=1
for value in data['Mapping']:
    price = yf.Ticker(value['Ticker'])
    price_history = price.history(period="max")
    if i==1:
        market = price_history
        market["Sedol"] = value["Sedol"]
        print(market)
    else:
        price_history["Sedol"]=value["Sedol"]
        market=market.merge(price_history)
    i+=1
