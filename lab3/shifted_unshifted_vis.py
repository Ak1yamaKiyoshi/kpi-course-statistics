import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib.patches as patches

# Вхідні дані (варіант 8)
y = np.array([240, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 65, 60, 55, 50, 45])
x = np.array([170, 180, 200, 230, 240, 250, 280, 300, 310, 320, 330, 350, 380, 400, 410, 420, 430, 440, 450, 460])

# Кількість спостережень
n = len(x)

# Обчислення коефіцієнтів регресії
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x*y)

beta1 = (n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2)
beta0 = (sum_y - beta1*sum_x)/n

# Розрахунок прогнозних значень
y_pred = beta0 + beta1 * x

# Залишки (відхилення фактичних значень від прогнозованих)
residuals = y - y_pred

# Сума квадратів залишків
RSS = np.sum(residuals**2)

# Зміщена оцінка дисперсії (ділення на n)
biased_variance = RSS / n

# Незміщена оцінка дисперсії (ділення на n-2)
unbiased_variance = RSS / (n-2)

print(f"Сума квадратів залишків (RSS): {RSS:.4f}")
print(f"Зміщена оцінка дисперсії (RSS/n): {biased_variance:.4f}")
print(f"Незміщена оцінка дисперсії (RSS/(n-2)): {unbiased_variance:.4f}")
print(f"Різниця: {unbiased_variance - biased_variance:.4f}")

# Створення графіка
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Графік 1: Лінійна регресія з відображенням залишків
axs[0].scatter(x, y, color='blue', label='Фактичні значення')
axs[0].plot(x, y_pred, color='red', label='Лінія регресії')

# Додавання вертикальних ліній для відображення залишків
for i in range(n):
    axs[0].plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.3)

axs[0].set_title('Лінійна регресія з відображенням залишків')
axs[0].set_xlabel('Звуковий канал (X)')
axs[0].set_ylabel('Відеоканал (Y)')
axs[0].legend()
axs[0].grid(True)

# Додаємо формули на графік
formula_text = f"y = {beta0:.2f} + ({beta1:.4f})x"
axs[0].text(170, 70, formula_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Графік 2: Візуалізація зміщеної та незміщеної оцінок дисперсії
axs[1].scatter(range(1, n+1), residuals, color='green', label='Залишки (y - ŷ)')
axs[1].axhline(y=0, color='red', linestyle='-')

# Додаємо зони для візуалізації дисперсії
std_biased = np.sqrt(biased_variance)
std_unbiased = np.sqrt(unbiased_variance)

# Створюємо прямокутники для зміщеної та незміщеної оцінок
rect_biased = patches.Rectangle((0, -std_biased), n+1, 2*std_biased, 
                                linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2)
rect_unbiased = patches.Rectangle((0, -std_unbiased), n+1, 2*std_unbiased, 
                                 linewidth=1, edgecolor='red', facecolor='red', alpha=0.2)

# Додаємо прямокутники на графік
axs[1].add_patch(rect_biased)
axs[1].add_patch(rect_unbiased)

# Підписи для графіка
axs[1].set_title('Залишки регресії та діапазони стандартних відхилень')
axs[1].set_xlabel('Номер спостереження')
axs[1].set_ylabel('Значення залишку')
axs[1].legend(['Середня лінія (0)', 'Залишки'])
axs[1].grid(True)

# Додаємо текстові блоки для пояснення
legend_blue = patches.Patch(color='blue', alpha=0.2, label=f'Зміщена оцінка: ±{std_biased:.2f} (RSS/n)')
legend_red = patches.Patch(color='red', alpha=0.2, label=f'Незміщена оцінка: ±{std_unbiased:.2f} (RSS/(n-2))')
axs[1].legend(handles=[legend_blue, legend_red], loc='lower right')

# Додаємо текстове пояснення
explanation = (
    f"Зміщена оцінка дисперсії: {biased_variance:.2f} (RSS/n)\n"
    f"Незміщена оцінка дисперсії: {unbiased_variance:.2f} (RSS/(n-2))\n"
    f"Різниця: {unbiased_variance - biased_variance:.2f}\n"
    f"Фактор корекції: n/(n-2) = {n/(n-2):.3f}"
)
axs[1].text(1, max(residuals) - 0.2 * (max(residuals) - min(residuals)), explanation,
          fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

# Бонус: Імітація для демонстрації незміщеності
np.random.seed(42)

# Кількість імітацій
n_simulations = 1000
biased_vars = np.zeros(n_simulations)
unbiased_vars = np.zeros(n_simulations)

# Справжні параметри моделі
true_beta0 = 325
true_beta1 = -0.68
true_sigma = 8  # Справжнє стандартне відхилення помилок

for i in range(n_simulations):
    # Генеруємо нові y з тими ж x, але різними випадковими помилками
    errors = np.random.normal(0, true_sigma, n)
    y_sim = true_beta0 + true_beta1 * x + errors
    
    # Оцінюємо параметри регресії
    sum_y_sim = np.sum(y_sim)
    sum_xy_sim = np.sum(x * y_sim)
    
    beta1_sim = (n*sum_xy_sim - sum_x*sum_y_sim)/(n*sum_x2 - sum_x**2)
    beta0_sim = (sum_y_sim - beta1_sim*sum_x)/n
    
    # Прогноз та залишки
    y_pred_sim = beta0_sim + beta1_sim * x
    residuals_sim = y_sim - y_pred_sim
    
    # Обчислення зміщеної та незміщеної оцінок
    RSS_sim = np.sum(residuals_sim**2)
    biased_vars[i] = RSS_sim / n
    unbiased_vars[i] = RSS_sim / (n-2)

# Створення графіка для демонстрації незміщеності
plt.figure(figsize=(10, 6))
plt.axvline(x=true_sigma**2, color='black', linestyle='-', label='Справжня дисперсія')
plt.hist(biased_vars, bins=30, alpha=0.5, color='blue', label='Зміщена оцінка')
plt.hist(unbiased_vars, bins=30, alpha=0.5, color='red', label='Незміщена оцінка')

# Середні значення оцінок
plt.axvline(x=np.mean(biased_vars), color='blue', linestyle='--', 
           label=f'Середнє зміщеної: {np.mean(biased_vars):.2f}')
plt.axvline(x=np.mean(unbiased_vars), color='red', linestyle='--', 
           label=f'Середнє незміщеної: {np.mean(unbiased_vars):.2f}')

plt.title('Розподіл оцінок дисперсії в 1000 імітаціях')
plt.xlabel('Оцінка дисперсії')
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.show()

print(f"Справжня дисперсія: {true_sigma**2}")
print(f"Середнє зміщених оцінок: {np.mean(biased_vars):.4f}")
print(f"Середнє незміщених оцінок: {np.mean(unbiased_vars):.4f}")
print(f"Відхилення зміщеної оцінки від справжньої: {np.mean(biased_vars) - true_sigma**2:.4f}")
print(f"Відхилення незміщеної оцінки від справжньої: {np.mean(unbiased_vars) - true_sigma**2:.4f}")