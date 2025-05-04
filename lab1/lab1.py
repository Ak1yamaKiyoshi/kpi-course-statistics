from math import *
from statistics import *

import matplotlib.pyplot as plt

A = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
B = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
C = [52, 44, 40, 57, 45, 37, 53, 30, 33, 59, 50, 40, 39, 38, 45, 55, 57, 77, 44, 40, 46, 40, 28, 46, 43, 52, 50, 68, 48, 47, 60, 43, 34, 51, 40, 45, 38, 37, 47, 47, 33, 51, 58, 56, 25, 67, 47, 64, 45, 54, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69]
D = [36, 64, 50, 67, 37, 48, 51, 54, 55, 28, 54, 47, 45, 57, 51, 46, 57, 50, 45, 54, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 59, 72, 47, 39, 39, 54, 57, 39, 57, 49, 57, 59, 39, 45, 33, 70, 64, 49, 48, 62, 35, 54, 42, 34, 49, 42, 48, 34, 54, 51, 70, 39, 44, 41, 41, 50, 62, 43, 47, 49]
E = [48, 29, 52, 55, 63, 67, 64, 44, 53, 69, 32, 53, 44, 36, 45, 49, 63, 53, 40, 45, 66, 41, 43, 60, 61, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69, 72, 49, 59, 55, 52, 51, 53, 50, 46, 63, 50, 27, 61, 48, 51, 59, 37, 54, 46, 59, 41, 47, 44, 75, 48]


VARIANTS = "Варіанти"
ABS_FREQUENCY = "Абсолютна частота"
ABS_INTERVAL_FREQUENCY = "Абсолютна частота інтервалу"
REL_FREQUENCY = "Відносна частота"
DENSITY = "Щільність"
INTERVAL_END = "Кінець інтервалу"
INTERVAL_BEGIN = "Початок інтервалу"
x = A  


# DISPLAY
def pretty_print_table(data, title=None, precision=4, column_widths=None, ):
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if not data:  # Empty list
            print("No data to display")
            return    
        columns = set()
        for item in data:
            columns.update(item.keys())
        columns = sorted(columns)
        column_data = {col: [] for col in columns}
        for row in data:
            for col in columns:
                column_data[col].append(row.get(col, ""))
        data = column_data

    columns = list(data.keys())
    if not columns:  # No columns
        print("No data to display")
        return
        
    num_rows = len(data[columns[0]])
    
    if column_widths is None:
        column_widths = {}
        for col in columns:
            max_len = max(len(str(data[col][i])) for i in range(num_rows))
            max_len = max(max_len, len(str(col)))
            column_widths[col] = max_len + 4

    if title:
        print(title)
        print('=' * sum(column_widths.values()))

    for col in columns:
        print(f"{col:{column_widths[col]}}", end="")
    print()
    print('-' * sum(column_widths.values()))

    for i in range(num_rows):
        for col in columns:
            value = data[col][i]
            if isinstance(value, float):
                formatted_value = f"{value:.{precision}f}"
            else:
                formatted_value = str(value)
            print(f"{formatted_value:{column_widths[col]}}", end="")
        print()

#  UTILS 
def count_a_in_x(a, x):
    count = 0
    for i in x:
        if i == a:
            count += 1
    return count


def ceil_even(number):
    c = ceil(number)
    f = floor(number)
    return c if c % 2 != 0 else f 
 

def in_range(x: list, min: float, max: float) -> list:
    result = [value for value in x if min <= value <= max]   
    return result


def in_range(x: list, min: float, max: float, return_indices: bool = False) -> list:    
    if return_indices:
        return [i for i, value in enumerate(x) if min <= value <= max]
    else:
        return [value for value in x if min <= value <= max]
    
def get_by_idx(x:list, idx:list):
    filtered_by_idx = []
    for i, value in enumerate(x):
        if i in idx:
            filtered_by_idx.append(value)
    return filtered_by_idx


def get_range(x):
    return max(x) - min(x)

def mode_interval(interval_width, interval_row_data):
    maximum_abs_freq, submaximum_abs_freq = None, None
    begin_of_mode_interval, begin_of_mode_submaxinterval = None, None
    maximum_abs_freq_idx = None
    for i, interval in enumerate(interval_row_data):
        if maximum_abs_freq is None or interval[ABS_INTERVAL_FREQUENCY] > maximum_abs_freq:
            submaximum_abs_freq = maximum_abs_freq 
            maximum_abs_freq = interval[ABS_INTERVAL_FREQUENCY]
            begin_of_mode_submaxinterval = begin_of_mode_interval
            begin_of_mode_interval = interval[INTERVAL_BEGIN]
            maximum_abs_freq_idx = i


    interval_mode = \
        begin_of_mode_submaxinterval + \
            (maximum_abs_freq - interval_row_data[maximum_abs_freq_idx - 1][ABS_INTERVAL_FREQUENCY]) /\
            (maximum_abs_freq * 2 - interval_row_data[maximum_abs_freq_idx - 1][ABS_INTERVAL_FREQUENCY] - interval_row_data[maximum_abs_freq_idx + 1][ABS_INTERVAL_FREQUENCY])\
                * interval_width
    return interval_mode

def median_interval(n, interval_row_data, interval_width): 
    relative_event_chances_for_each_variant = []
    cumsum_abs_interval_frequency = 0
    for interval in interval_row_data:
        cumsum_abs_interval_frequency += interval[ABS_INTERVAL_FREQUENCY]
        interval_start = interval[INTERVAL_BEGIN]
        relative_event_chance = cumsum_abs_interval_frequency / n
        relative_event_chances_for_each_variant.append(relative_event_chance)
        # print(f"F*({interval_start:3.1f}) = {cumsum_abs_interval_frequency / n:3.2}")

    maximum, submaximum = None, None
    submaximum_variant = None
    for i, chance in enumerate(relative_event_chances_for_each_variant):
        if maximum is None:
            maximum = chance
        elif abs(0.5 - chance) < abs(0.5 - maximum):
            submaximum = maximum
            submaximum_variant = interval_row_data[i][INTERVAL_BEGIN]
            maximum = chance
        elif submaximum is None or abs(0.5 - chance) < abs(0.5 - submaximum):
            submaximum = chance

    interval_median = submaximum_variant +  ((0.5 - submaximum) / (maximum - submaximum)) * interval_width
    return interval_median


def get_discrete_row_data(x):
    variants = list(set(x))
    n = len(x)
    abs_frequencies = [count_a_in_x(var, x) for var in variants]
    rel_frequencies = [freq/n for freq in abs_frequencies]
    return {
        VARIANTS: variants,
        ABS_FREQUENCY: abs_frequencies,
        REL_FREQUENCY: rel_frequencies
    }, n

def get_interval_row_data(x, discrete_row_data):
    n = len(x)
    xmax, xmin = max(x), min(x)
    
    interval_bins = (5 * log(n)) # k 
    range_ = (xmax - xmin)
    interval_width = range_ / interval_bins
    
    ranges = []

    current = xmin
    while current < xmax - interval_width:
        current += interval_width
        idx_of_variants_in_range = in_range(discrete_row_data[VARIANTS], current, current + interval_width, return_indices=True)
        abs_frequencies = get_by_idx(discrete_row_data[ABS_FREQUENCY], idx_of_variants_in_range)
        rel_frequencies = get_by_idx(discrete_row_data[ABS_FREQUENCY], idx_of_variants_in_range)
        
        
        if not abs_frequencies: 
            abs_interval_sum = 0
            rel_interval_sum = 0 
            density = 0
        else: 
            abs_interval_sum = sum(abs_frequencies)
            rel_interval_sum = sum(rel_frequencies)
            density = (abs_interval_sum / n) / interval_width 
        
        ranges.append({
            INTERVAL_BEGIN: current,
            INTERVAL_END: current + interval_width,
            DENSITY: density,
            ABS_INTERVAL_FREQUENCY: abs_interval_sum,
        })
        
    return ranges, interval_width, n


discrete_row_data, n = get_discrete_row_data(x)
interval_row_data, interval_width, n = get_interval_row_data(x, discrete_row_data) 

pretty_print_table(discrete_row_data, title="Дискретний та варіаційний ряди", )
pretty_print_table(interval_row_data, title="Інтервальний ряд", )

# Me
discrete_median = median(discrete_row_data[VARIANTS])
interval_median = median_interval(n, interval_row_data, interval_width)

# R
discrete_range = get_range(discrete_row_data[VARIANTS]) 
interval_range = get_range([interval_row_data[0][INTERVAL_BEGIN], interval_row_data[-1][INTERVAL_END]])

# Mo 
discrete_mode = mode(x)
interval_mode = mode_interval(interval_width, interval_row_data)

print(f"""
Медіана
    Дискретний  : {discrete_median:3.1f}
    Інтервальний: {interval_median:3.1f}
Розмах
    Дискретний  : {discrete_range:3.1f}
    Інтервальний: {interval_range:3.1f}
Мода 
    Дискретний  : {discrete_mode:3.1f}
    Інтервальний: {interval_mode:3.1f}
""")




densities = []
midpoints = []
widths = []

for interval in interval_row_data: 
    begin = interval[INTERVAL_BEGIN]
    end = interval[INTERVAL_END]
    density = interval[DENSITY]

    widths.append(end - begin)
    midpoints.append((begin + end) / 2)
    densities.append(density)


plt.figure(figsize=(12, 6))
plt.bar(midpoints, densities, width=widths, align='center', edgecolor='black', color='orange', label='Гістограма щільності')
plt.plot(midpoints, densities, color='red', marker='o', linewidth=2, label='Полігон частот')

plt.xlabel('Інтервали')
plt.ylabel('Щільність')
plt.title('Гістограма та полігон щільності')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('lab1_analysis.png')
print("saved to.. 'lab1_analysis.png'")



n = len(x)
selected_sum = 0
for freq, var in zip(discrete_row_data[ABS_FREQUENCY], discrete_row_data[VARIANTS]):
    selected_sum += freq * var
selected_average = selected_sum / n


sum_for_dispertion = 0
for freq, var in zip(discrete_row_data[ABS_FREQUENCY], discrete_row_data[VARIANTS]):
    sum_for_dispertion += var ** 2 * freq


dispertion = sum_for_dispertion / n - selected_average ** 2

mean_std = sqrt(dispertion)
variation_coefficient = mean_std / selected_average * 100

print(f"""
    Середнє вибіркове: {selected_average:3.1f}
    Дисперсія: {dispertion:3.1f}
    Середнє квадратичне відхилення: {mean_std:3.1f}
    Коофіцієнт варіації: {variation_coefficient:3.1f} % 
""")

