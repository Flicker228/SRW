import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def autocorr(ts, lag):
    x = np.asarray(ts)
    x_mean = np.mean(x)
    return np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / np.sum(np.power(x - x_mean, 2))


n = 200
np.random.seed(0)
t = np.arange(n)
signal = np.sin(0.1 * t) + np.random.normal(0, 0.5, n)
ts = pd.Series(signal, index=t)

math_expect = np.mean(ts)
dispersion = np.sum(np.power(ts - math_expect, 2)) / n
auto_corr = autocorr(ts, 1)

print("Матожидание: ", math_expect, "\nДисперсия: ",
      dispersion, "\nАвтокорреляция: ", auto_corr)

df = pd.DataFrame(signal, columns=["Значения"])
df.to_excel("my_list.xlsx", index=False)

ts.plot()
plt.xlabel("Секунды")
plt.ylabel("Значение сигнала")
plt.show()