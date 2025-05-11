import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# inputs ... 
n = 9
intervals = np.array([(n+i, n+i+1) for i in range(8)])
frequencies = np.array([10, 8, 7, 10, 22, 18, 14, 11])
alpha = 0.05


# Обсяг вибірки
sample_size = sum(frequencies)

# Інтервальний розподіл у точковий  
midpoints = np.mean(intervals, axis=1) # середини інтервалів
 
# Вибіркове середнє x^в (середнє усіх елементів вибірки)
sample_mean = np.sum(midpoints * frequencies) / sample_size

# Вибіркова дисперсія (наскільки данні розкидані)
sample_var = np.sum(frequencies * (midpoints - sample_mean)**2) / sample_size 

# Середньоквадратичне відхилення
sample_std = np.sqrt(sample_var)

# Теоретичні ймовірності для нормального розподілу
# (Тобто, ми беремо функцію яка генерує нормальний розподіл використовуючи вибіркове середнє та середньоквадратичне відхилення)
# І звідси у нас "імовнірності" потрапляння в інтервал. 
probs = []
for a, b in intervals:
    prob = stats.norm.cdf(b, sample_mean, sample_std) - stats.norm.cdf(a, sample_mean, sample_std)
    probs.append(prob)
probs = np.array(probs)

# Теоретичні частоти 
# Оскільки імовірності в ренджі 0-1, ми можемо домножити обсяг вибірки, щоб отримати теоретичний нормальний розподіл з нашими данними
expected_freq = probs * sample_size

# Об'єднання інтервалів з малими теоретичними частотами (<5)
small_groups = np.where(expected_freq < 5)[0]
print(small_groups)
if len(small_groups) > 0:
    # (Це можна закодить нормально, але я подивився які інтервали мають маленькі частоти і об'єднав їх (перший має малі частоти тому об'єдную з другим))
    observed_merged = [frequencies[0] + frequencies[1]] + list(frequencies[2:])
    expected_merged = [expected_freq[0] + expected_freq[1]] + list(expected_freq[2:])
else:
    observed_merged = frequencies
    expected_merged = expected_freq

observed_merged = np.array(observed_merged)
expected_merged = np.array(expected_merged)

# Обчислення статистики x^2
# критерій пірсона?
# тобто ми отримуємо таку собі метрику наскільки наш розподіл відрізняється від теоретичного нормального
# оскільки віднімаємо реальні частоти від "теоретичних нормального розподілу" і нормалізуємо відносно очікуваних
chi2_stat = np.sum((observed_merged - expected_merged)**2 / expected_merged)

# Ступені свободи
groups_count = len(observed_merged)  # 
params_count = 2  # два параметри для нормального розподілу
df = groups_count - params_count - 1  # 4

# Критичне значення x^2
chi2_critical = stats.chi2.ppf(1 - alpha, df)

# alpha - рівень значущості визначає % (5) у нашому випадку - ризик помилки, що якщо ми відхилимо гіпотезу - вона вірна
# КРитичне значення це критичне значення різниці теоретичного нормального розподілу та нашого розподілу 
# тобто якщо наш розподіл дуже відрізняєтья від теоретичного нормального (тобто X^2 спостережене > критичне) - гіпотезу про нормальний розподіл треба відхилити. 
# і тут ми маємо alpha% (5%) шанс помилки. Якщо неправі - розподіл нормальний.  

# Перевірка гіпотези
if chi2_stat > chi2_critical:
    pearson_result = "Відхиляємо гіпотезу про нормальний розподіл"
else:
    pearson_result = "Не відхиляємо гіпотезу про нормальний розподіл"


# ЧАСТИНА 2: Перевірка гіпотези критерієм Колмогорова

# Створимо вибіркову кумулятивну функцію розподілу
cumulative_freq = np.cumsum(frequencies)
empirical_cdf = cumulative_freq / sample_size

# Створимо теоретичну кумулятивну функцію розподілу для нормального розподілу
theoretical_cdf = np.array([stats.norm.cdf(x[1], sample_mean, sample_std) for x in intervals])

# Відхилення між емпіричною та теоретичною функціями розподілу
D_plus = np.max(empirical_cdf - theoretical_cdf)  # максимальне додатне відхилення
D_minus = np.max(theoretical_cdf - np.append(0, empirical_cdf[:-1]))  # максимальне від'ємне відхилення
D_n = max(D_plus, D_minus)  # максимальне абсолютне відхилення

# Обчислення статистики Колмогорова
kolmogorov_stat = D_n * np.sqrt(sample_size)

# Критичне значення для рівня значущості alpha=0.05
# Використовуємо формулу для критичного значення при великих вибірках
kolmogorov_critical = 1.36  # приблизне значення для alpha=0.05

# Перевірка гіпотези
if kolmogorov_stat > kolmogorov_critical:
    kolmogorov_result = "Відхиляємо гіпотезу про нормальний розподіл"
else:
    kolmogorov_result = "Не відхиляємо гіпотезу про нормальний розподіл"

# Виведення результатів
print(f"Вибіркове середнє: {sample_mean:.2f}, СКВ: {sample_std:.2f}")
print("\nКритерій Пірсона:")
print(f"χ² спостережене: {chi2_stat:.2f}, χ² критичне: {chi2_critical:.2f}")
print(f"Висновок за критерієм Пірсона: {pearson_result}")

print("\nКритерій Колмогорова:")
print(f"D_n: {D_n:.4f}, λ спостережене: {kolmogorov_stat:.2f}, λ критичне: {kolmogorov_critical:.2f}")
print(f"Висновок за критерієм Колмогорова: {kolmogorov_result}")

# Візуалізація даних та функцій розподілу
plt.figure(figsize=(12, 8))

# Гістограма з частотами
plt.subplot(2, 1, 1)
plt.bar(midpoints, frequencies, width=0.8, edgecolor='black', color='skyblue', alpha=0.7, label='Спостережені частоти')
plt.plot(midpoints, expected_freq, 'ro-', linewidth=1.5, label='Теоретичні частоти (нормальний розподіл)')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.title('Гістограма та теоретичні частоти')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Емпірична та теоретична функції розподілу
plt.subplot(2, 1, 2)
plt.step(np.append(intervals[0][0], intervals[:, 1]), np.append(0, empirical_cdf), 'b-', where='post', label='Емпірична функція розподілу')

# Графік теоретичної функції розподілу
x_range = np.linspace(intervals[0][0], intervals[-1][1], 100)
plt.plot(x_range, stats.norm.cdf(x_range, sample_mean, sample_std), 'r-', label='Теоретична функція розподілу')

plt.xlabel('Значення')
plt.ylabel('Ймовірність')
plt.title('Емпірична та теоретична функції розподілу')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('./hist_with_kolmogorov.png')
