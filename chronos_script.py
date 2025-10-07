import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings("ignore")

def autocorr(ts, lag):
    x = np.asarray(ts)
    x_mean = np.mean(x)
    return np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / np.sum(np.power(x - x_mean, 2))


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.float32,
)

# Load the data
df = pd.read_csv("ecg_1d_timeseries_prediction.csv", sep=';')

context = torch.tensor(df["ecg_value"])
ts=pd.Series(df["ecg_value"])
prediction_length = 120
forecast = pipeline.predict(context, prediction_length)

math_expect = np.mean(ts)
dispersion = np.sum(np.power(ts - math_expect, 2)) / len(ts)
auto_corr = autocorr(ts, 10)

forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

print("Матожидание: ", math_expect, "\nДисперсия: ",
      dispersion, "\nАвтокорреляция: ", auto_corr)

plt.figure(figsize=(30, 10))
plt.plot(df["ecg_value"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()