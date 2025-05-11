import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

y = np.array([240, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 65, 60, 55, 50, 45])
x = np.array([170, 180, 200, 230, 240, 250, 280, 300, 310, 320, 330, 350, 380, 400, 410, 420, 430, 440, 450, 460])

# Кількість спостережень
n = len(x)

# 1. Обчислення точкових оцінок параметрів лінійної регресії
# Обчислення сум для МНК
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x*y)

# Обчислення коефіцієнтів регресії
beta1 = (n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2)
beta0 = (sum_y - beta1*sum_x)/n

print(f"Коефіцієнти регресії:")
print(f"β₀ = {beta0:.4f}")
print(f"β₁ = {beta1:.4f}")

# Рівняння регресії
print(f"Рівняння регресії: y = {beta0:.4f} + ({beta1:.4f}) × x")

# Розрахунок прогнозних значень
y_pred = beta0 + beta1 * x

# 2. Обчислення залишкової дисперсії
residuals = y - y_pred
RSS = np.sum(residuals**2)  # Залишкова сума квадратів
s2 = RSS / (n - 2)  # Незміщена оцінка дисперсії
s = np.sqrt(s2)  # Стандартне відхилення

# Обчислення стандартних помилок для коефіцієнтів
se_beta1 = s / np.sqrt(np.sum((x - np.mean(x))**2))
se_beta0 = s * np.sqrt(sum_x2 / (n * (n*sum_x2 - sum_x**2)))

print(f"\nСтандартні помилки коефіцієнтів:")
print(f"SE(β₀) = {se_beta0:.4f}")
print(f"SE(β₁) = {se_beta1:.4f}")

# 3. Довірчі інтервали для коефіцієнтів (γ = 0.99)
alpha = 0.01  # рівень значущості для довірчого інтервалу 99%
t_crit = stats.t.ppf(1 - alpha/2, n-2)  # критичне значення t
print(f"\nКритичне значення t (γ = 0.99): {t_crit:.4f}")

# Довірчі інтервали для β₀
CI_beta0_lower = beta0 - t_crit * se_beta0
CI_beta0_upper = beta0 + t_crit * se_beta0

# Довірчі інтервали для β₁
CI_beta1_lower = beta1 - t_crit * se_beta1
CI_beta1_upper = beta1 + t_crit * se_beta1

print(f"\nДовірчі інтервали (γ = 0.99):")
print(f"β₀: [{CI_beta0_lower:.4f}; {CI_beta0_upper:.4f}]")
print(f"β₁: [{CI_beta1_lower:.4f}; {CI_beta1_upper:.4f}]")

# 4. Перевірка значущості коефіцієнта β₁ (α = 0.01)
t_stat = beta1 / se_beta1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))  # двосторонній тест

print(f"\nПеревірка значущості β₁:")
print(f"t-статистика = {t_stat:.4f}")
print(f"p-значення = {p_value:.8f}")

if abs(t_stat) > t_crit:
    print(f"Коефіцієнт β₁ статистично значущий на рівні α = 0.01")
else:
    print(f"Коефіцієнт β₁ статистично незначущий на рівні α = 0.01")

# 5. Довірчий інтервал для функції регресії (γ = 0.99)
# Довірчі смуги для лінії регресії
x_mean = np.mean(x)
CI_width = t_crit * s * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
CI_regression_lower = y_pred - CI_width
CI_regression_upper = y_pred + CI_width

# 6. Обчислення коефіцієнта кореляції
r_xy = (n*sum_xy - sum_x*sum_y) / (np.sqrt(n*sum_x2 - sum_x**2) * np.sqrt(n*np.sum(y**2) - sum_y**2))

print(f"\nКоефіцієнт кореляції r = {r_xy:.4f}")
print(f"Коефіцієнт детермінації R² = {r_xy**2:.4f}")

# 7. Довірчий інтервал для прогнозованих значень (γ = 0.99)
CI_pred_width = t_crit * s * np.sqrt(1 + 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
CI_pred_lower = y_pred - CI_pred_width
CI_pred_upper = y_pred + CI_pred_width

# Створення таблиці результатів
results = pd.DataFrame({
    'x': x,
    'y': y,
    'y_pred': y_pred,
    'Довірчі інтервали для регресії (нижня межа)': CI_regression_lower,
    'Довірчі інтервали для регресії (верхня межа)': CI_regression_upper,
    'Довірчі інтервали для прогнозу (нижня межа)': CI_pred_lower,
    'Довірчі інтервали для прогнозу (верхня межа)': CI_pred_upper
})

print("\nРезультати обчислень:")
print(results)

# Побудова графіків
plt.figure(figsize=(12, 8))

# Кореляційне поле та лінія регресії
plt.scatter(x, y, color='blue', label='Спостереження')
plt.plot(x, y_pred, color='red', label=f'Лінія регресії: y = {beta0:.2f} + ({beta1:.4f}) × x')

# Довірчі інтервали для лінії регресії
plt.fill_between(x, CI_regression_lower, CI_regression_upper, color='red', alpha=0.2, label='99% довірчий інтервал для регресії')

# Довірчі інтервали для прогнозованих значень
plt.fill_between(x, CI_pred_lower, CI_pred_upper, color='green', alpha=0.1, label='99% довірчий інтервал для прогнозу')

plt.xlabel('Звуковий канал (X)')
plt.ylabel('Відеоканал (Y)')
plt.title('Залежність чутливості відеоканалу від звукового каналу')
plt.legend()
plt.grid(True)
plt.savefig('regression_analysis.png')
plt.close()

# Детальний графік залишків
plt.figure(figsize=(12, 6))
plt.scatter(x, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='-')
plt.xlabel('Звуковий канал (X)')
plt.ylabel('Залишки (Y - Ŷ)')
plt.title('Графік залишків')
plt.grid(True)
plt.savefig('residuals_plot.png')
plt.close()

print("\nГрафіки збережено як 'regression_analysis.png' та 'residuals_plot.png'")

# Додаткові статистики
print(f"\nДодаткові статистики:")
print(f"Середнє X: {np.mean(x):.4f}")
print(f"Середнє Y: {np.mean(y):.4f}")
print(f"Стандартне відхилення X: {np.std(x, ddof=1):.4f}")
print(f"Стандартне відхилення Y: {np.std(y, ddof=1):.4f}")
print(f"Коваріація: {np.cov(x, y, ddof=1)[0, 1]:.4f}")