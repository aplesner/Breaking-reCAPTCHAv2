import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# Specify the directory where the log files are stored
log_dir = './Manual_test_with_cookies' #Manual_test_with_cookies

# Find all log files in the specified directory
log_files = [os.path.join(log_dir, filename) for filename in os.listdir(log_dir) if filename.startswith('log') and filename.endswith('.csv')]

# Sort the log files by their name
log_files.sort()



def visualize_number_of_attempts_needed():
    # Count the number of rows in each log file
    row_counts = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader) - 1  # Subtract 1 to exclude the last row
            row_counts.append(row_count)

    # Create a bar plot of the number of rows in each log file
    plt.bar(log_files, row_counts)
    plt.xlabel('Log file')
    plt.ylabel('Number of rows')
    plt.title('Number of rows in each log file')
    plt.xticks(rotation=90)
    plt.show()




from matplotlib.lines import Line2D

def visualize_number_of_attempts_by_type(log_dir=log_dir, y_limit=None):
    plt.rcParams.update({'font.size': 20})  # Adjust the font size as needed for textwidth 0.5 20, 25 for 0.3
    # Define a color for each type
    type_colors = {'Type1': sns.color_palette("deep")[0], 'Type2': sns.color_palette("deep")[2], 'Type3': sns.color_palette("deep")[3]}
    
    # Find all log files in the specified directory
    logs = [os.path.join(log_dir, filename) for filename in os.listdir(log_dir) if filename.startswith('log') and filename.endswith('.csv')]

    # Sort the log files by their name
    logs.sort()

    type_counts = {}
    total_counts = [0] * len(logs)
    for log_file in logs:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            rows = rows[:-1]  # Exclude the last row
            for row in rows:
                type = row[1]
                if type == 'dynamic':
                    type = 'Type3'  # Rename 'dynamic' to 'Type3'
                if type not in type_counts:
                    type_counts[type] = [0] * len(logs)
                type_counts[type][logs.index(log_file)] += 1
                total_counts[logs.index(log_file)] += 1

    # Create a stacked bar plot of the number of rows of each type in each log file
    bottom = [0] * len(logs)
    for type, counts in type_counts.items():
        plt.bar(logs, counts, bottom=bottom, color=type_colors.get(type, 'gray'))
        bottom = [b + c for b, c in zip(bottom, counts)]

    plt.xlabel('Run Index')
    plt.ylabel('Number of Attempts')
    #plt.title('Number of Attempts Needed to Solve Captchas per Run')

    # Set the y-axis limit if specified
    if y_limit is not None:
        plt.ylim(0, y_limit)

    # Set the xticks to label every nth tick
    n = 10  # Change this to control the frequency of the labels
    plt.xticks(np.arange(0, len(logs), n), np.arange(0, len(logs), n), rotation=90)

    # Manually add the legend entries
    legend_elements = [Line2D([0], [0], color=type_colors[type], lw=4, label=type) for type in ['Type1', 'Type2', 'Type3']]
    plt.legend(handles=legend_elements)

    plt.show()
    return total_counts  # Return total counts instead of type counts



def visualize_number_of_attempts_by_object():
    # Count the number of rows of each type in each log file
    type_counts = {}
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            rows = rows[:-1]  # Exclude the last row
            for row in rows:
                type = row[2]
                if type not in type_counts:
                    type_counts[type] = [0] * len(log_files)
                type_counts[type][log_files.index(log_file)] += 1

    # Create a stacked bar plot of the number of rows of each type in each log file
    bottom = [0] * len(log_files)
    for type, counts in type_counts.items():
        plt.bar(log_files, counts, bottom=bottom, label=type)
        bottom = [b + c for b, c in zip(bottom, counts)]

    plt.xlabel('Log file')
    plt.ylabel('Number of rows')
    plt.title('Number of rows of each type in each log file')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()



def calculate_average_rows():
    # Count the number of rows in each log file
    row_counts = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            row_count = len(rows) - 1  # Subtract 1 to exclude the last row
            row_counts.append(row_count)

    # Calculate and return the average number of rows
    average_rows = sum(row_counts) / len(log_files)
    return average_rows



def calculate_variance_rows():
    # Count the number of rows in each log file
    row_counts = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            row_count = len(rows) - 1  # Subtract 1 to exclude the last row
            row_counts.append(row_count)

    # Calculate the average number of rows
    average_rows = sum(row_counts) / len(log_files)

    # Calculate the variance
    variance = sum((x - average_rows) ** 2 for x in row_counts) / len(row_counts)
    return variance

def calculate_range_rows():
    # Count the number of rows in each log file
    row_counts = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            row_count = len(rows) - 1  # Subtract 1 to exclude the last row
            row_counts.append(row_count)

    # Calculate the range
    range_rows = max(row_counts) - min(row_counts)
    return range_rows



def calculate_iqr_rows():
    # Count the number of rows in each log file
    row_counts = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            row_count = len(rows) - 1  # Subtract 1 to exclude the last row
            row_counts.append(row_count)

    # Calculate the IQR
    # The Interquartile Range (IQR) is a measure of statistical dispersion, 
    # being equal to the difference between the upper and lower quartiles, Q3 - Q1.
    q3, q1 = np.percentile(row_counts, [75 ,25])
    iqr = q3 - q1
    return iqr

def visualize_item_percentage():
    # Count the number of occurrences of each item in the column with index 2
    item_counts = {}
    for log_file in log_files:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            rows = rows[:-1]  # Exclude the last row
            for row in rows:
                item = row[2]
                if item not in item_counts:
                    item_counts[item] = 0
                item_counts[item] += 1

    # Calculate the total number of items
    total_items = sum(item_counts.values())

    # Calculate the percentage of each item
    item_percentages = {item: count / total_items * 100 for item, count in item_counts.items()}

    # Create a pie chart of the item percentages
    plt.pie(item_percentages.values(), labels=item_percentages.keys(), autopct='%1.1f%%')
    plt.title('Percentage of each object')
    plt.show()


def calculate_transition_probabilities():
        # Count the number of transitions from each item to each other item
        transition_counts = {}
        for log_file in log_files:
            with open(log_file, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                rows = rows[:-1]  # Exclude the last row
                for i in range(len(rows) - 1):
                    item = rows[i][2]
                    next_item = rows[i + 1][2]
                    if item not in transition_counts:
                        transition_counts[item] = {}
                    if next_item not in transition_counts[item]:
                        transition_counts[item][next_item] = 0
                    transition_counts[item][next_item] += 1

        # Calculate the total number of transitions from each item
        total_transitions = {item: sum(next_item_counts.values()) for item, next_item_counts in transition_counts.items()}

        # Calculate the transition probabilities
        transition_probabilities = {item: {next_item: count / total_transitions[item] for next_item, count in next_item_counts.items()} for item, next_item_counts in transition_counts.items()}

        return transition_probabilities

def visualize_transition_probabilities():
    transition_probabilities = calculate_transition_probabilities()

    # Convert the transition probabilities to a DataFrame
    df = pd.DataFrame(transition_probabilities).fillna(0)

    # Sort the DataFrame to ensure the x and y axes have the same order
    df = df.sort_index(axis=0).sort_index(axis=1)

    # Create a heatmap of the transition probabilities
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True, cmap='viridis')
    plt.title('Transition probabilities')
    plt.xlabel('Next item')
    plt.ylabel('Current item')
    plt.show()



def independent_sample_tests(log_dir1, log_dir2):
    # Visualize the number of attempts by type for the two directories
    total_counts1 = visualize_number_of_attempts_by_type(log_dir=log_dir1)
    total_counts2 = visualize_number_of_attempts_by_type(log_dir=log_dir2)

    # Perform an independent two-sample t-test to compare the means of the total number of attempts for the two directories
    t_stat, p_val = stats.ttest_ind(total_counts1, total_counts2, equal_var=False)
    print(f'Test for total counts: t-statistic = {t_stat:.2f}, p-value = {p_val:.2f}')


visualize_number_of_attempts_by_type(y_limit=20)
#visualize_item_percentage()
#visualize_transition_probabilities()
#visualize_number_of_attempts_by_object()
#independent_sample_tests('./Session20', './Session9')
#visualize_item_percentage()

print(f'Average attempts: {calculate_average_rows():.2f}')
print(f'Variance of attempts: {calculate_variance_rows():.2f}')
print(f'Range of attempts: {calculate_range_rows():.2f}')
print(f'Interquartile Range of attempts: {calculate_iqr_rows():.2f}')
