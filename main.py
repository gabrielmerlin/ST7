# Import des données

import json
import yfinance as yf

data_file = open("data_ST7MDS.json")
data = json.load(data_file)
i=1
for value in data['Mapping']:
    price= yf.Ticker(value['Ticker'])
    if i==1:
        market=price.history(period= "max")
    else:
        market=market.merge(price.history(period="max"))
    i+=1

