import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# Daten einlesen
def read_data(path_with_music='data/group_with_music.csv', path_without_music='data/group_without_music.csv'):
    group_with_music = pd.read_csv(path_with_music)
    group_without_music = pd.read_csv(path_without_music)
    return group_with_music, group_without_music


# Standardization of the precision-values
def standardize_precision(precision_values):
    min = precision_values.min()
    max = precision_values.max()
    return (precision_values - min) / (max - min) * 100

def normalize_precision(precision_values):
    return precision_values * 100

# Descriptive statistics
def generate_statistics(group_with_music, group_without_music):

    # standardization
    #group_with_music['Precision'] = standardize_precision(group_with_music['Precision'])
    #group_without_music['Precision'] = standardize_precision(group_without_music['Precision'])
    # rescale to 0-100
    group_with_music['Precision'] = normalize_precision(group_with_music['Precision'])
    group_without_music['Precision'] = normalize_precision(group_without_music['Precision'])

    # statistics per respondent
    # statistics_with_music = group_with_music.groupby('Respondent').agg({
    #     'Precision': ['mean', 'median', 'std'],
    #     'Time': ['mean', 'median', 'std']
    # }).reset_index()
    #
    # statistics_without_music = group_without_music.groupby('Respondent').agg({
    #     'Precision': ['mean', 'median', 'std'],
    #     'Time': ['mean', 'median', 'std']
    # }).reset_index()
    # 
    # # Inferential statistics (t-test for Normalized Precision-value and effect size)
    # t_stat, p_value = stats.ttest_ind(group_with_music['Precision'], group_without_music['Precision'])
    # effect_size = (statistics_with_music['Precision']['mean'].mean() - statistics_without_music['Precision']['mean'].mean()) / statistics_without_music['Precision']['std'].mean()

    mean_precision_with_music = group_with_music['Precision'].mean()
    median_precision_with_music = group_with_music['Precision'].median()
    std_precision_with_music = group_with_music['Precision'].std()

    mean_time_with_music = group_with_music['Time'].mean()
    median_time_with_music = group_with_music['Time'].median()
    std_time_with_music = group_with_music['Time'].std()

    mean_precision_without_music = group_without_music['Precision'].mean()
    median_precision_without_music = group_without_music['Precision'].median()
    std_precision_without_music = group_without_music['Precision'].std()

    mean_time_without_music = group_without_music['Time'].mean()
    median_time_without_music = group_without_music['Time'].median()
    std_time_without_music = group_without_music['Time'].std()

    # Inferential statistics (t-test for Precision-value and effect size)
    t_stat, p_value = stats.ttest_ind(group_with_music['Precision'], group_without_music['Precision'])
    effect_size = (mean_precision_with_music - mean_precision_without_music) / std_precision_without_music

    # Plots
    plt.figure(figsize=(12, 6))
    # Histogramm für Precision in group with music
    plt.subplot(2, 2, 1)
    plt.hist(group_with_music['Precision'], bins=20, alpha=0.5, color='blue', label='with music')
    plt.hist(group_without_music['Precision'], bins=20, alpha=0.5, color='red', label='without music')
    plt.xlabel('Precision')
    plt.ylabel('Frequency')
    plt.legend()

    # Boxplot for Precision in both groups
    plt.subplot(2, 2, 2)
    plt.boxplot([group_with_music['Precision'], group_without_music['Precision']],
                labels=['with music', 'without music'])
    plt.ylabel('Precision')

    # Histogram for the time in group with music
    plt.subplot(2, 2, 3)
    plt.hist(group_with_music['Time'], bins=20, alpha=0.5, color='blue', label='with music')
    plt.hist(group_without_music['Time'], bins=20, alpha=0.5, color='red', label='without music')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()

    # Boxplot for Time in both groups
    plt.subplot(2, 2, 4)
    plt.boxplot([group_with_music['Time'], group_without_music['Time']], labels=['with music', 'without music'])
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

    # Print out Results
    print(f'Descriptive Statistics - group with music:\n'
          f'Mean Precision: {mean_precision_with_music}\n'
          f'Median Precision: {median_precision_with_music}\n'
          f'Standard Deviation Precision: {std_precision_with_music}\n'
          f'Mean Time: {mean_time_with_music}\n'
          f'Median Time: {median_time_with_music}\n'
          f'Standard Deviation Time: {std_time_with_music}\n')

    print(f'Descriptive Statistics - group without music:\n'
          f'Mean Precision: {mean_precision_without_music}\n'
          f'Median Precision: {median_precision_without_music}\n'
          f'Standard Deviation Precision: {std_precision_without_music}\n'
          f'Mean Time: {mean_time_without_music}\n'
          f'Median Time: {median_time_without_music}\n'
          f'Standard Deviation Time: {std_time_without_music}\n')

    print(f'Inferential Statistics:\n'
          f'T-statistic: {t_stat}\n'
          f'P-value: {p_value}\n'
          f'Effect Size (Cohen\'s d): {effect_size}\n')
