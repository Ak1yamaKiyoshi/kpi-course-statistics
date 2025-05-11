from math import *
from statistics import *

import matplotlib.pyplot as plt

# Kolosov
# A = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
# B = [55, 60, 42, 43, 65, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 69, 47, 50, 58, 58, 42, 55, 51, 53, 58, 41, 30, 48, 54, 46, 50, 49, 62, 34, 35, 62, 41, 40, 38, 34, 63, 24, 41, 41, 46, 61, 64, 47, 54, 63, 53, 62, 56, 52, 51, 64, 36, 43, 52, 49, 47, 40, 35, 61, 38, 40, 55, 49, 62, 64]
# C = [52, 44, 40, 57, 45, 37, 53, 30, 33, 59, 50, 40, 39, 38, 45, 55, 57, 77, 44, 40, 46, 40, 28, 46, 43, 52, 50, 68, 48, 47, 60, 43, 34, 51, 40, 45, 38, 37, 47, 47, 33, 51, 58, 56, 25, 67, 47, 64, 45, 54, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69]
# D = [36, 64, 50, 67, 37, 48, 51, 54, 55, 28, 54, 47, 45, 57, 51, 46, 57, 50, 45, 54, 30, 47, 47, 41, 52, 49, 44, 57, 61, 54, 50, 47, 57, 52, 40, 59, 72, 47, 39, 39, 54, 57, 39, 57, 49, 57, 59, 39, 45, 33, 70, 64, 49, 48, 62, 35, 54, 42, 34, 49, 42, 48, 34, 54, 51, 70, 39, 44, 41, 41, 50, 62, 43, 47, 49]
# E = [48, 29, 52, 55, 63, 67, 64, 44, 53, 69, 32, 53, 44, 36, 45, 49, 63, 53, 40, 45, 66, 41, 43, 60, 61, 45, 51, 58, 76, 48, 59, 42, 62, 39, 51, 60, 66, 71, 73, 61, 46, 48, 50, 37, 34, 55, 53, 42, 26, 69, 72, 49, 59, 55, 52, 51, 53, 50, 46, 63, 50, 27, 61, 48, 51, 59, 37, 54, 46, 59, 41, 47, 44, 75, 48]


# Yaschenko
A = [61, 64, 47, 54, 63,
             53, 62, 56, 52, 51,
             64, 36, 43, 52, 49,
             47, 40, 35, 61, 38,
             40, 55, 49, 62, 64,
             57, 43, 51, 42, 51,
             50, 58, 50, 52, 42,
             51, 49, 65, 56, 67,
             56, 37, 63, 49, 32,
             43, 59, 63, 50, 53,
             52, 50, 68, 48, 47,
             60, 43, 34, 51, 40,
             45, 38, 37, 47, 47,
             33, 51, 58, 56, 25,
             67, 47, 64, 45, 54]

B = [45, 48, 45, 64, 51,
             58, 60, 52, 38, 38,
             48, 35, 43, 33, 54,
             51, 31, 43, 42, 56,
             58, 38, 57, 59, 52,
             48, 57, 50, 53, 54,
             31, 48, 55, 53, 60,
             58, 63, 47, 42, 65,
             53, 51, 43, 46, 57,
             44, 53, 45, 54, 46,
             44, 42, 60, 44, 58,
             44, 48, 52, 48, 56,
             56, 63, 58, 52, 60,
             36, 37, 42, 39, 39,
             57, 55, 66, 61, 40]

C = [52, 44, 40, 57, 45,
             37, 53, 30, 33, 59,
             50, 40, 39, 38, 45,
             55, 57, 77, 44, 40,
             46, 40, 28, 46, 43,
             48, 46, 49, 55, 52,
             47, 51, 36, 47, 56,
             55, 50, 39, 43, 55,
             49, 56, 44, 55, 59,
             52, 36, 62, 49, 32,
             74, 53, 71, 53, 32,
             54, 56, 46, 43, 56,
             61, 61, 61, 67, 60,
             56, 46, 56, 62, 65,
             54, 53, 50, 56, 43]

D = [59, 72, 47, 39, 39,
             54, 57, 39, 57, 49,
             57, 59, 39, 45, 33,
             70, 64, 49, 48, 62,
             52, 55, 55, 60, 46,
             35, 54, 42, 34, 49,
             35, 36, 49, 37, 38,
             42, 48, 34, 54, 51,
             70, 39, 44, 41, 41,
             50, 62, 43, 47, 49,
             72, 49, 59, 55, 52,
             51, 53, 50, 46, 63,
             50, 27, 61, 48, 51,
             59, 37, 54, 46, 59,
             41, 47, 44, 75, 48]

E = [57, 43, 51, 42, 51,
             50, 58, 50, 52, 42,
             51, 49, 65, 56, 67,
             56, 37, 63, 49, 32,
             43, 59, 63, 50, 53,
             52, 50, 68, 48, 47,
             60, 43, 34, 51, 40,
             45, 38, 37, 47, 47,
             33, 51, 58, 56, 25,
             67, 47, 64, 45, 54,
             45, 48, 45, 64, 51,
             58, 60, 52, 38, 38,
             48, 35, 43, 33, 54,
             51, 31, 43, 42, 56,
             58, 38, 57, 59, 52] 


VARIANTS = "Варіанти"
ABS_FREQUENCY = "Абсолютна частота"
ABS_INTERVAL_FREQUENCY = "Абсолютна частота інтервалу"
REL_FREQUENCY = "Відносна частота"
DENSITY = "Щільність"
INTERVAL_END = "Кінець інтервалу"
INTERVAL_BEGIN = "Початок інтервалу"
MEAN = "Середнє"
DISPERSION = "Дисперсія"
STD = "Середнє відхилення"



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


def calculate_mean_std(dataset):
    discrete_row_data, n = get_discrete_row_data(dataset)
    
    selected_sum = 0
    for freq, var in zip(discrete_row_data[ABS_FREQUENCY], discrete_row_data[VARIANTS]):
        selected_sum += freq * var
    selected_average = selected_sum / n
    
    sum_for_dispertion = 0
    for freq, var in zip(discrete_row_data[ABS_FREQUENCY], discrete_row_data[VARIANTS]):
        sum_for_dispertion += var ** 2 * freq
    
    dispertion = sum_for_dispertion / n - selected_average ** 2
    mean_std = sqrt(dispertion)
    
    return {
        MEAN: selected_average,
        DISPERSION: dispertion,
        STD: mean_std
    }

############## HTML 



def generate_html_report(x, dataset_name):
    discrete_row_data, n = get_discrete_row_data(x)
    interval_row_data, interval_width, n = get_interval_row_data(x, discrete_row_data)
    
    discrete_median = median(discrete_row_data[VARIANTS])
    interval_median = median_interval(n, interval_row_data, interval_width)
    
    discrete_range = get_range(discrete_row_data[VARIANTS])
    interval_range = get_range([interval_row_data[0][INTERVAL_BEGIN], interval_row_data[-1][INTERVAL_END]])
    
    discrete_mode = mode(x)
    interval_mode = mode_interval(interval_width, interval_row_data)
    
    stats = calculate_mean_std(x)
    
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
    plt.title(f'Гістограма та полігон щільності - {dataset_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    image_filename = f'lab1_analysis_{dataset_name}.png'
    plt.savefig(image_filename)
    plt.close()
    
    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Статистичний аналіз - {dataset_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .section {{
            margin-bottom: 40px;
        }}
    </style>
</head>
<body>
    <h1>Статистичний аналіз даних - {dataset_name}</h1>
    
    <div class="section">
        <h2>Дискретний та варіаційний ряди</h2>
        <table>
            <tr>"""
    
    for col in discrete_row_data.keys():
        html += f"<th>{col}</th>"
    html += """
            </tr>"""
    
    for i in range(len(discrete_row_data[VARIANTS])):
        html += "<tr>"
        for col in discrete_row_data.keys():
            value = discrete_row_data[col][i]
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            html += f"<td>{formatted_value}</td>"
        html += "</tr>"
    
    html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Інтервальний ряд</h2>
        <table>
            <tr>"""
    
    if interval_row_data:
        for col in interval_row_data[0].keys():
            html += f"<th>{col}</th>"
    
    html += """
            </tr>"""
    
    for interval in interval_row_data:
        html += "<tr>"
        for col in interval.keys():
            value = interval[col]
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            html += f"<td>{formatted_value}</td>"
        html += "</tr>"
    
    html += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>Статистичні характеристики</h2>
        <table>
            <tr>
                <th>Характеристика</th>
                <th>Дискретний</th>
                <th>Інтервальний</th>
            </tr>
            <tr>
                <td>Медіана</td>
                <td>{discrete_median:.2f}</td>
                <td>{interval_median:.2f}</td>
            </tr>
            <tr>
                <td>Розмах</td>
                <td>{discrete_range:.2f}</td>
                <td>{interval_range:.2f}</td>
            </tr>
            <tr>
                <td>Мода</td>
                <td>{discrete_mode:.2f}</td>
                <td>{interval_mode:.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Додаткові статистичні показники</h2>
        <table>
            <tr>
                <th>Показник</th>
                <th>Значення</th>
            </tr>
            <tr>
                <td>Середнє вибіркове</td>
                <td>{stats[MEAN]:.2f}</td>
            </tr>
            <tr>
                <td>Дисперсія</td>
                <td>{stats[DISPERSION]:.2f}</td>
            </tr>
            <tr>
                <td>Середнє квадратичне відхилення</td>
                <td>{stats[STD]:.2f}</td>
            </tr>
            <tr>
                <td>Коофіцієнт варіації</td>
                <td>{stats[STD] / stats[MEAN] * 100:.2f} %</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Гістограма та полігон щільності</h2>
        <img src="{image_filename}" alt="Гістограма та полігон щільності" style="max-width:100%;">
    </div>
</body>
</html>"""
    
    html_filename = f"report_{dataset_name}.html"
    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html)
    
    return html_filename

def generate_all_reports():
    report_files = []
    report_files.append(generate_html_report(A, "A"))
    report_files.append(generate_html_report(B, "B"))
    report_files.append(generate_html_report(C, "C"))
    report_files.append(generate_html_report(D, "D"))
    report_files.append(generate_html_report(E, "E"))
    
    # Generate index page with links to all reports
    index_html = """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Статистичний аналіз - Звіти</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .report-list {
            margin: 30px 0;
        }
        .report-item {
            margin: 15px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
        }
        .report-link {
            font-size: 18px;
            color: #3498db;
            text-decoration: none;
        }
        .report-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Статистичний аналіз - Звіти</h1>
    
    <div class="report-list">"""
    
    for i, file in enumerate(report_files):
        dataset_name = file.split("_")[1].split(".")[0]
        index_html += f"""
        <div class="report-item">
            <a class="report-link" href="{file}">Звіт для набору даних {dataset_name}</a>
        </div>"""
    
    index_html += """
    </div>
</body>
</html>"""
    
    # Save index HTML to file
    with open("index.html", "w", encoding="utf-8") as file:
        file.write(index_html)
    
    print("Усі звіти успішно згенеровано:")
    for file in report_files:
        print(f"  - {file}")
    print("  - index.html (головна сторінка)")

generate_all_reports()

def plot_mean_std_comparison_scatter():
    # Calculate statistics for each dataset
    stats_A = calculate_mean_std(A)
    stats_B = calculate_mean_std(B)
    stats_C = calculate_mean_std(C)
    stats_D = calculate_mean_std(D)
    stats_E = calculate_mean_std(E)
    
    datasets = ['A', 'B', 'C', 'D', 'E']
    means = [stats_A[MEAN], stats_B[MEAN], stats_C[MEAN], stats_D[MEAN], stats_E[MEAN]]
    std_devs = [stats_A[STD], stats_B[STD], stats_C[STD], stats_D[STD], stats_E[STD]]
    
    lower_bounds = [m - s for m, s in zip(means, std_devs)]
    upper_bounds = [m + s for m, s in zip(means, std_devs)]
    
    plt.figure(figsize=(10, 6))
    
    x = range(len(datasets))
    plt.scatter(x, means, s=100, color='blue', label='Середнє значення', zorder=3)
    plt.axhline(y=50, color='green', linestyle='--', linewidth=2, label='Базова лінія (50)', zorder=1)
    
    for i, (m, s) in enumerate(zip(means, std_devs)):
        plt.annotate(f'{m:.2f}±{s:.2f}', (i, m), 
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom', fontsize=14)
    
    max_y = max(upper_bounds) + 1
    min_y = min(lower_bounds) - 1
    plt.ylim(min_y, max_y)
    
    plt.xlabel('Набір даних')
    plt.ylabel('Значення')
    plt.title('Порівняння середніх значень та стандартних відхилень')
    plt.xticks(x, datasets)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='blue', markersize=8, linestyle='None'),
        Line2D([0], [0], color='red', linewidth=2),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2)
    ]
    plt.legend(custom_lines, ['Середнє значення', 'Діапазон ст. відхилення', 'Базова лінія (50)'])
    
    plt.savefig('dataset_comparison_scatter.png')
    plt.close()
    
    print("Графік збережено як 'dataset_comparison_scatter.png'")

if __name__ == "__main__":
    plot_mean_std_comparison_scatter()