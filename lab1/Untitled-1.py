import numpy as np
import matplotlib.pyplot as plt
from math import log, ceil

A = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
B = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
C = [52, 44, 40, 57, 45, 37, 53, 30, 33, 59, 50, 40, 39, 38, 45, 55, 57, 77, 44, 40, 46, 40, 28, 46, 43, 52, 50, 68, 48, 47, 60, 43, 34, 51, 40, 45, 38, 37, 47, 47, 33, 51, 58, 56, 25, 67, 47, 64, 45, 54, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69]
D = [36, 64, 50, 67, 37, 48, 51, 54, 55, 28, 54, 47, 45, 57, 51, 46, 57, 50, 45, 54, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 59, 72, 47, 39, 39, 54, 57, 39, 57, 49, 57, 59, 39, 45, 33, 70, 64, 49, 48, 62, 35, 54, 42, 34, 49, 42, 48, 34, 54, 51, 70, 39, 44, 41, 41, 50, 62, 43, 47, 49]
E = [48, 29, 52, 55, 63, 67, 64, 44, 53, 69, 32, 53, 44, 36, 45, 49, 63, 53, 40, 45, 66, 41, 43, 60, 61, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69, 72, 49, 59, 55, 52, 51, 53, 50, 46, 63, 50, 27, 61, 48, 51, 59, 37, 54, 46, 59, 41, 47, 44, 75, 48]
x = A + B + C + D + E
n = len(x)

# 1. Ранжування даних (Лекція 1, 1.5)
x_sorted = sorted(x)

# 2. Інтервальний варіаційний ряд (Лекція 2, 2.2)
xmax, xmin = max(x), min(x)
# Формула Стерджеса для кількості інтервалів (Лекція 2, 2.2)
k = ceil(1 + 3.322 * log(n, 10))
# Ширина інтервалу
h = (xmax - xmin) / k

# Створення інтервалів
intervals = []
current = xmin
while current < xmax + h:
    intervals.append((current, current + h))
    current += h

# Підрахунок частот
frequencies = []
for interval in intervals:
    count = sum(1 for value in x if interval[0] <= value < interval[1] or 
               (value == interval[1] and interval[1] == xmax))
    frequencies.append(count)

# Виведення інтервального ряду
print("Інтервальний варіаційний ряд (Лекція 2, 2.2):")
print(f"    a  - b      | частота |")
frequency_sum = 0
for i, interval in enumerate(intervals):
    frequency_sum += frequencies[i]
    print(f"({str(interval[0])[:4]:<5} - {str(interval[1])[:4]:<5}) | {str(frequencies[i]):<7}")
print(f"Сума частот: {frequency_sum}, Розмір вибірки: {len(x)}")

results_x_interval = {"frequencies": frequencies, "intervals": intervals}

# 3. Дискретний варіаційний ряд (Лекція 2, 2.1)
unique_values = sorted(set(x))
discrete_freq = [x.count(val) for val in unique_values]

print("\nДискретний варіаційний ряд (Лекція 2, 2.1):")
print("Значення | Частота | Відносна частота")
for i, val in enumerate(unique_values):
    rel_freq = discrete_freq[i] / n
    print(f"{val:<8} | {discrete_freq[i]:<7} | {rel_freq:.4f}")

results_x_discrete = {"frequencies": discrete_freq, "values": unique_values}

# 4. Числові характеристики для дискретного ряду (Лекція 3, 3.1)
def discrete_stats(values, frequencies):
    # Розмах (Лекція 3, формула 3.3)
    R = max(values) - min(values)
    
    # Вибіркова середня (Лекція 3, формула 3.1)
    xB = sum(values[i] * frequencies[i] for i in range(len(values))) / sum(frequencies)
    
    # Дисперсія (Лекція 3, формула 3.5)
    D = sum(values[i]**2 * frequencies[i] for i in range(len(values))) / sum(frequencies) - xB**2
    
    # Середнє квадратичне відхилення (Лекція 3, формула 3.6)
    sigma = np.sqrt(D)
    
    # Коефіцієнт варіації (Лекція 3, формула 3.7)
    V = sigma / xB * 100
    
    # Мода - значення з найбільшою частотою (Лекція 3, 3.1)
    mode_index = np.argmax(frequencies)
    Mo = values[mode_index]
    
    # Медіана (Лекція 3, 3.1)
    expanded_data = np.repeat(values, frequencies)
    Me = np.median(expanded_data)
    
    return {"R": R, "xB": xB, "D": D, "sigma": sigma, "V": V, "Me": Me, "Mo": Mo}

# 5. Числові характеристики для інтервального ряду (Лекція 3, 3.2)
def interval_stats(intervals, frequencies):
    # Розмах (Лекція 3, формула 3.3)
    R = intervals[-1][1] - intervals[0][0]
    
    # Середини інтервалів
    midpoints = [(interval[0] + interval[1])/2 for interval in intervals]
    
    # Вибіркова середня (Лекція 3, 3.2)
    xB = sum(midpoints[i] * frequencies[i] for i in range(len(midpoints))) / sum(frequencies)
    
    # Дисперсія (Лекція 3, 3.2)
    D = sum(midpoints[i]**2 * frequencies[i] for i in range(len(midpoints))) / sum(frequencies) - xB**2
    
    # Середнє квадратичне відхилення
    sigma = np.sqrt(D)
    
    # Коефіцієнт варіації (Лекція 3, формула 3.7)
    V = sigma / xB * 100
    
    # Модальний інтервал (Лекція 3, 3.2)
    mode_interval_index = np.argmax(frequencies)
    mode_interval_lower = intervals[mode_interval_index][0]
    mode_interval_width = intervals[mode_interval_index][1] - intervals[mode_interval_index][0]
    
    # Частоти до і після модального інтервалу
    freq_before_mode = 0 if mode_interval_index == 0 else frequencies[mode_interval_index - 1]
    freq_after_mode = 0 if mode_interval_index == len(frequencies) - 1 else frequencies[mode_interval_index + 1]
    
    # Мода для інтервального ряду (Лекція 3, формула 3.9)
    Mo = mode_interval_lower + mode_interval_width * ((frequencies[mode_interval_index] - freq_before_mode) / 
              ((frequencies[mode_interval_index] - freq_before_mode) + (frequencies[mode_interval_index] - freq_after_mode)))
    
    # Медіанний інтервал
    total_freq = sum(frequencies)
    cumulative_freq = 0
    
    for i, freq in enumerate(frequencies):
        prev_cumulative_freq = cumulative_freq
        cumulative_freq += freq
        if cumulative_freq >= total_freq / 2:
            median_interval_lower = intervals[i][0]
            median_interval_width = intervals[i][1] - intervals[i][0]
            median_interval_freq = freq
            break
    
    # Медіана для інтервального ряду (Лекція 3, формула 3.8)
    Me = median_interval_lower + median_interval_width * ((total_freq / 2 - prev_cumulative_freq) / median_interval_freq)
    
    return {"R": R, "xB": xB, "D": D, "sigma": sigma, "V": V, "Me": Me, "Mo": Mo}

# Розрахунок статистик
disc_stats = discrete_stats(results_x_discrete["values"], results_x_discrete["frequencies"])
int_stats = interval_stats(results_x_interval["intervals"], results_x_interval["frequencies"])

print("\nСтатистичні показники для дискретного ряду (Лекція 3, 3.1):")
print(f"R = {disc_stats['R']:.4f}, Me* = {disc_stats['Me']:.4f}, Mo* = {disc_stats['Mo']}")
print(f"xB = {disc_stats['xB']:.4f}, D = {disc_stats['D']:.4f}, V = {disc_stats['V']:.2f}%")

print("\nСтатистичні показники для інтервального ряду (Лекція 3, 3.2):")
print(f"R = {int_stats['R']:.4f}, Me* = {int_stats['Me']:.4f}, Mo* = {int_stats['Mo']:.4f}")
print(f"xB = {int_stats['xB']:.4f}, D = {int_stats['D']:.4f}, V = {int_stats['V']:.2f}%")

# 6. Графічні зображення (Лекція 2, 2.3)
def plot_charts():
    plt.figure(figsize=(12, 10))
    
    # Полігон частот для дискретного ряду
    plt.subplot(2, 2, 1)
    plt.plot(results_x_discrete["values"], results_x_discrete["frequencies"], 'b-o', linewidth=2, markersize=4)
    plt.grid(True, alpha=0.3)
    plt.title('Полігон частот (дискретний ряд)')
    plt.xlabel('Значення')
    plt.ylabel('Частота')
    
    # Гістограма для дискретного ряду
    plt.subplot(2, 2, 2)
    plt.bar(results_x_discrete["values"], results_x_discrete["frequencies"], width=0.8, alpha=0.7, color='skyblue', edgecolor='blue')
    plt.grid(True, alpha=0.3)
    plt.title('Гістограма (дискретний ряд)')
    plt.xlabel('Значення')
    plt.ylabel('Частота')
    
    # Полігон частот для інтервального ряду
    midpoints = [(interval[0] + interval[1])/2 for interval in results_x_interval["intervals"]]
    plt.subplot(2, 2, 3)
    plt.plot(midpoints, results_x_interval["frequencies"], 'r-o', linewidth=2, markersize=4)
    plt.grid(True, alpha=0.3)
    plt.title('Полігон частот (інтервальний ряд)')
    plt.xlabel('Середина інтервалу')
    plt.ylabel('Частота')
    
    # Гістограма для інтервального ряду
    plt.subplot(2, 2, 4)
    widths = [interval[1] - interval[0] for interval in results_x_interval["intervals"]]
    plt.bar([interval[0] for interval in results_x_interval["intervals"]], 
            results_x_interval["frequencies"], 
            width=widths, 
            alpha=0.7, 
            color='lightgreen', 
            edgecolor='green')
    plt.grid(True, alpha=0.3)
    plt.title('Гістограма (інтервальний ряд)')
    plt.xlabel('Інтервали')
    plt.ylabel('Частота')
    
    plt.tight_layout()
    plt.show()

# 7. Контрольна карта середніх (Лекція 4, 4.2)
def build_control_chart(data, frequencies):
    # Розгортання даних з урахуванням частот
    expanded_data = np.repeat(data, frequencies)
    
    # Розподіл на підгрупи по 5 елементів
    subgroup_size = 5
    n_total = len(expanded_data)
    n_subgroups = n_total // subgroup_size
    
    # Обрізання даних до кратного розміру підгрупи
    data_reshaped = expanded_data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
    
    # Обчислення середніх для кожної підгрупи
    subgroup_means = np.mean(data_reshaped, axis=1)
    
    # Загальне середнє
    grand_mean = np.mean(subgroup_means)
    
    # Середнє стандартне відхилення підгруп
    subgroup_std = np.std(data_reshaped, axis=1, ddof=1)
    average_std = np.mean(subgroup_std)
    
    # Контрольні межі (Лекція 4, 4.2)
    A2 = 0.577  # Константа для підгрупи розміром 5
    UCL = grand_mean + A2 * average_std  # Верхня контрольна межа
    LCL = grand_mean - A2 * average_std  # Нижня контрольна межа
    
    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(subgroup_means, 'bo-', label='Середні підгруп')
    plt.axhline(y=grand_mean, color='g', linestyle='-', label='Центральна лінія')
    plt.axhline(y=UCL, color='r', linestyle='--', label='UCL')
    plt.axhline(y=LCL, color='r', linestyle='--', label='LCL')
    
    # Виділення точок за межами контролю
    out_of_control = np.where((subgroup_means > UCL) | (subgroup_means < LCL))[0]
    if len(out_of_control) > 0:
        plt.plot(out_of_control, subgroup_means[out_of_control], 'ro', markersize=10)
    
    plt.title('Контрольна карта середніх арифметичних (Лекція 4, 4.2)')
    plt.xlabel('Номер підгрупи')
    plt.ylabel('Середнє значення')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nКонтрольна карта (Лекція 4, 4.2):")
    print(f"Центральна лінія: {grand_mean:.4f}")
    print(f"Верхня контрольна межа (UCL): {UCL:.4f}")
    print(f"Нижня контрольна межа (LCL): {LCL:.4f}")
    print(f"Кількість точок за межами контролю: {len(out_of_control)}")
    
    if len(out_of_control) > 0:
        print(f"Підгрупи за межами контролю: {out_of_control}")

# Виконання візуалізації
plot_charts()
build_control_chart(results_x_discrete["values"], results_x_discrete["frequencies"])