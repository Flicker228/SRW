import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.diagnostic
import statsmodels.api as sm


def normality_test(data):
    n = len(data)

    lillie_stat, lillie_p = statsmodels.stats.diagnostic.lilliefors(data)

    anderson_result = stats.anderson(data, dist='norm')
    critical_value_5per = anderson_result.critical_values[2]

    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    print(f"Лиллиефорс: p-value = {lillie_p:.10f}")

    if anderson_result.statistic < critical_value_5per:
        print("Андерсон-Дарлинг: распределение нормальное.")
    else:
        print("Андерсон-Дарлинг: распределение НЕ нормальное.")

    skew_ok = -0.5 <= skewness <= 0.5
    kurt_ok = -1.0 <= kurtosis <= 1.0
    lillie_ok = lillie_p > 0.01

    print(f"\nПрактическая оценка:")
    print(f"Асимметрия в норме: {skew_ok}")
    print(f"Эксцесс в норме: {kurt_ok}")
    print(f"Лиллиефорс в норме: {lillie_ok}")

    if skew_ok and kurt_ok:
        print("Эксцесс и асимметрия в пределах нормы")
    else:
        print("Заметные отклонения по эксцессу/асимметрии")
    return 0


# def norm_check(data, length):
#     section_num = max(1, round(len(data) / length))
#     data = np.array_split(data, section_num)
#     shapiro_probs = np.array([stats.shapiro(segment) for segment in data])
#
#     index = np.where(shapiro_probs[:, 1] < 0.01)[0]
#     print(index)
#
#     for i in index:
#         sm.qqplot(data[i], line='s')
#         plt.show()
#     return shapiro_probs


np.set_printoptions(
    precision=8,
    suppress=True
)

df = pd.read_csv("test_data/ecg_1d_timeseries_prediction.csv", sep=';')
t = pd.Series(df["time"]).values
ts = pd.Series(df["ecg_value"]).values

dt = 0.001
fd = 1 / dt
n = len(ts)
nyq = 0.5 * fd

ts = ts - np.mean(ts)
fur = np.fft.rfft(ts)
fur_abs = np.abs(fur) / n
freq = np.fft.rfftfreq(n, d=dt)

noise_mask = (freq > 100)
fur_noise = fur.copy()
fur_noise[~noise_mask] = 0

harm_arr = np.arange(0, nyq + 1, 50)

for x in harm_arr:
    indices = np.where((np.abs(freq) >= x - 0.06) &
                       (np.abs(freq) <= x + 0.06))[0]

    fur_noise[indices] = 0

fur_noise_abs = np.abs(fur_noise) / n

ns = np.fft.irfft(fur_noise, n=n)[n // 100: - n // 100]

print(normality_test(ns))

if n % 2 == 0:
    fur_abs[1:-1] = 2 * fur_abs[1:-1]
    fur_noise_abs[1:-1] = 2 * fur_noise_abs[1:-1]
else:
    fur_abs[1:] = 2 * fur_abs[1:]
    fur_noise_abs[1:] = 2 * fur_noise_abs[1:]

fig, axs = plt.subplots(3, 1, figsize=(25, 20))

axs[0].plot(freq, fur_abs)
axs[0].grid()

axs[1].plot(t, ts)
axs[1].grid()

axs[2].plot(t[n // 100:-n // 100], ns)
axs[2].grid()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(ns, bins=50, density=True)
plt.title('Гистограмма всего сигнала')

plt.subplot(1, 2, 2)
stats.probplot(ns, dist="norm", plot=plt)
plt.title('Q-Q plot')
plt.show()

