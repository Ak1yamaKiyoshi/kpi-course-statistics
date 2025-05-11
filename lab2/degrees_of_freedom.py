import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Задаємо ступені свободи
degrees_of_freedom = [1, 2, 3, 4, 5, 10, 15]

# Створюємо фігуру з декількома підграфіками
fig, axes = plt.subplots(len(degrees_of_freedom), 1, figsize=(10, 15))
fig.suptitle('Розподіли χ² з різними ступенями свободи', fontsize=16)

# Генеруємо дані для кожного розподілу і будуємо гістограму
for i, df in enumerate(degrees_of_freedom):
    # Генеруємо випадкові величини з розподілу χ² з відповідними ступенями свободи
    data = np.random.chisquare(df, size=10000)
    
    # Знаходимо критичне значення для α=0.05
    critical_value = stats.chi2.ppf(0.95, df)
    
    # Обмежуємо дані для кращої візуалізації
    data = data[data <= 30]
    
    # Будуємо гістограму
    axes[i].hist(data, bins=50, density=True, alpha=0.7, color='skyblue', 
                 edgecolor='black', label=f'Симуляція')
    
    # Додаємо теоретичну криву щільності
    x = np.linspace(0, 30, 1000)
    y = stats.chi2.pdf(x, df)
    axes[i].plot(x, y, 'r-', linewidth=2, label='Теоретична крива')
    
    # Додаємо вертикальну лінію для критичного значення
    axes[i].axvline(x=critical_value, color='green', linestyle='--', 
                   label=f'Критичне значення (α=0.05): {critical_value:.2f}')
    
    # Підписи і легенда
    axes[i].set_title(f'df = {df}')
    axes[i].set_xlim(0, 30)
    axes[i].set_ylabel('Щільність')
    axes[i].legend()

# Підпис осі x тільки для нижнього графіка
axes[-1].set_xlabel('Значення χ²')

# Додаємо трохи простору між графіками
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Зберігаємо і показуємо результат
plt.savefig('chi_square_distributions.png', dpi=300)
plt.show()

# Виведення критичних значень для кожного ступеня свободи
print("Критичні значення для α=0.05:")
for df in degrees_of_freedom:
    critical_value = stats.chi2.ppf(0.95, df)
    print(f"df = {df}, критичне значення = {critical_value:.4f}")