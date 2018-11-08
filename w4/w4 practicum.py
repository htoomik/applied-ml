import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from dateutil.parser import parse

raw = pd.read_csv("AirPassengers2.csv")
raw['ds'] = raw['Month'].apply(lambda d: parse(d + "-01"))
raw['y'] = raw['Passengers']
raw.set_index('ds')

split_index = int(len(raw) * 2/3)

training = raw.iloc[:split_index]
test = raw.iloc[split_index+1:]

model = Prophet()
model.fit(training)

# 'MS' = "month start"
future = model.make_future_dataframe(periods=len(test)+1, freq='MS')

forecast = model.predict(future)
forecast.set_index('ds')

model.plot(forecast).savefig("forecast.png")

comp = test.merge(forecast, on='ds')

comp.set_index('ds')[['y', 'yhat']].plot().figure.savefig('plot.png')
