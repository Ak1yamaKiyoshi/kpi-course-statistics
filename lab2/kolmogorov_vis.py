import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Вхідні дані
intervals = [[21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29]]
frequencies = [10, 8, 7, 10, 22, 18, 14, 11]
sample_size = sum(frequencies)
alpha = 0.05

# Обчислення середнього та стандартного відхилення
midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]
weighted_sum = sum(mid * freq for mid, freq in zip(midpoints, frequencies))
sample_mean = weighted_sum / sample_size

weighted_squared_diff = sum(freq * (mid - sample_mean) ** 2 for mid, freq in zip(midpoints, frequencies))
sample_std = np.sqrt(weighted_squared_diff / (sample_size - 1))

print(f"Середнє значення вибірки: {sample_mean:.4f}")
print(f"Стандартне відхилення вибірки: {sample_std:.4f}")

# Створення емпіричної функції розподілу (ECDF)
cumulative_freq = np.cumsum(frequencies)
empirical_cdf = cumulative_freq / sample_size

# Створення x-координат для точок ECDF (праві межі інтервалів)
ecdf_x = [interval[1] for interval in intervals]

# Повна емпірична функція розподілу (з початковою точкою)
full_ecdf_x = [intervals[0][0]] + ecdf_x
full_ecdf_y = [0] + list(empirical_cdf)

# Створення теоретичної функції розподілу
# Генеруємо 1000 точок для плавної кривої
x_smooth = np.linspace(intervals[0][0] - 1, intervals[-1][1] + 1, 1000)
theoretical_cdf_smooth = stats.norm.cdf(x_smooth, loc=sample_mean, scale=sample_std)

# Теоретичні значення в точках ECDF
theoretical_cdf = [stats.norm.cdf(x, loc=sample_mean, scale=sample_std) for x in full_ecdf_x]

# Обчислення різниць між емпіричною та теоретичною функціями розподілу
differences = np.abs(np.array(empirical_cdf) - np.array(theoretical_cdf[1:]))
max_diff_index = np.argmax(differences)
max_diff = differences[max_diff_index]
max_diff_x = ecdf_x[max_diff_index]

print(f"Максимальна різниця D_n: {max_diff:.4f} в точці x = {max_diff_x}")

# Обчислення статистики Колмогорова
kolmogorov_stat = max_diff * np.sqrt(sample_size)
kolmogorov_critical = 1.36  # Приблизне значення для alpha=0.05

print(f"Статистика Колмогорова λ = D_n * √n: {kolmogorov_stat:.4f}")
print(f"Критичне значення для α = {alpha}: {kolmogorov_critical}")

if kolmogorov_stat > kolmogorov_critical:
    print("Відхиляємо гіпотезу про нормальний розподіл")
else:
    print("Не відхиляємо гіпотезу про нормальний розподіл")

# Створення графіка
plt.figure(figsize=(12, 8))

# Малюємо плавну теоретичну функцію розподілу
plt.plot(x_smooth, theoretical_cdf_smooth, 'g-', linewidth=2, 
         label='Теоретична функція розподілу F(x)')

# Малюємо емпіричну функцію розподілу як ступінчасту функцію
plt.step(full_ecdf_x, full_ecdf_y, 'b-', where='post', linewidth=2, 
         label='Емпірична функція розподілу Fn(x)')

# Виділяємо точки на емпіричній функції
plt.plot(ecdf_x, empirical_cdf, 'bo', markersize=8)

# Виділяємо точки на теоретичній функції (для тих самих x-координат)
theoretical_at_steps = [stats.norm.cdf(x, loc=sample_mean, scale=sample_std) for x in ecdf_x]
plt.plot(ecdf_x, theoretical_at_steps, 'go', markersize=8)

# Малюємо вертикальні лінії для всіх різниць
for i, (x, emp, theo) in enumerate(zip(ecdf_x, empirical_cdf, theoretical_at_steps)):
    plt.plot([x, x], [emp, theo], 'r--', linewidth=1, alpha=0.6)
    # Додаємо анотацію з значенням різниці
    diff = abs(emp - theo)
    plt.text(x+0.1, (emp + theo)/2, f"{diff:.4f}", fontsize=8)

# Виділяємо максимальну різницю
plt.plot([max_diff_x, max_diff_x], 
         [empirical_cdf[max_diff_index], theoretical_at_steps[max_diff_index]], 
         'r-', linewidth=3, label=f'Максимальна різниця D_n = {max_diff:.4f}')

# Додаємо анотацію про максимальну різницю
plt.annotate(f'D_n = {max_diff:.4f}',
             xy=(max_diff_x, (empirical_cdf[max_diff_index] + theoretical_at_steps[max_diff_index])/2),
             xytext=(max_diff_x+1, (empirical_cdf[max_diff_index] + theoretical_at_steps[max_diff_index])/2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

# Покращуємо оформлення графіка
plt.title('Візуалізація критерію Колмогорова', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Функція розподілу F(x)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)

# Додаємо текстовий блок з результатами
textstr = '\n'.join((
    f'Статистика Колмогорова λ = {kolmogorov_stat:.4f}',
    f'Критичне значення = {kolmogorov_critical}',
    f'Рівень значущості α = {alpha}',
    f'Розмір вибірки n = {sample_size}',
    'Висновок: ' + ('Відхиляємо H₀' if kolmogorov_stat > kolmogorov_critical else 'Не відхиляємо H₀')
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.show()