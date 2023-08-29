import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os


# Read in data
def read_data(path_with_music='data/group_with_music.csv', path_without_music='data/group_without_music.csv',
              path_to_questionnaire='data/questionnaire.csv'):
    group_with_music = pd.read_csv(path_with_music)
    group_without_music = pd.read_csv(path_without_music)
    questionnaire_data = pd.read_csv(path_to_questionnaire)
    return group_with_music, group_without_music, questionnaire_data


# Standardization of the precision-values
def standardize_precision(precision_values):
    min = precision_values.min()
    max = precision_values.max()
    return (precision_values - min) / (max - min) * 1000


def normalize_precision(precision_values):
    return precision_values * 1000


def calculate_average_per_respondent(data):
    # Gruppierung nach Probanden und Berechnung des Durchschnitts
    return data.groupby(['Respondent'])[['Precision', 'Time']].mean().reset_index()


def generate_statistics_per_respondent(group_with_music, group_without_music):
    # rescale to 0-100
    group_with_music['Precision'] = normalize_precision(group_with_music['Precision'])
    group_without_music['Precision'] = normalize_precision(group_without_music['Precision'])

    # statistics per respondent
    statistics_with_music = group_with_music.groupby('Respondent').agg({
        'Precision': ['mean', 'median', 'std'],
        'Time': ['mean', 'median', 'std']
    }).reset_index()

    statistics_without_music = group_without_music.groupby('Respondent').agg({
        'Precision': ['mean', 'median', 'std'],
        'Time': ['mean', 'median', 'std']
    }).reset_index()

    print("Statistics without Music:")
    print(statistics_without_music)
    print("Statistics with Music:")
    print(statistics_with_music)

    # Inferential statistics (t-test for Normalized Precision-value and effect size)
    t_stat, p_value = stats.ttest_ind(group_with_music['Precision'], group_without_music['Precision'])
    effect_size = (statistics_with_music['Precision']['mean'].mean() - statistics_without_music['Precision'][
        'mean'].mean()) / statistics_without_music['Precision']['std'].mean()

    print(f'Inferential Statistics:\n'
          f'T-statistic: {t_stat}\n'
          f'P-value: {p_value}\n'
          f'Effect Size (Cohen\'s d): {effect_size}\n')


# Descriptive statistics
def generate_dataframe(group_with_music, group_without_music, questionnaire_data):
    # standardization
    # group_with_music['Precision'] = standardize_precision(group_with_music['Precision'])
    # group_without_music['Precision'] = standardize_precision(group_without_music['Precision'])
    # rescale to 0-100
    group_with_music['Precision'] = normalize_precision(group_with_music['Precision'])
    group_without_music['Precision'] = normalize_precision(group_without_music['Precision'])

    # Average per Respondent
    group_with_music = calculate_average_per_respondent(group_with_music)
    group_without_music = calculate_average_per_respondent(group_without_music)

    # Assignment of the components of the questionnaire to the questions.
    components = {
        'Competence': ['Q_2', 'Q_10', 'Q_15', 'Q_17', 'Q_21'],
        'Sensory_and_Imaginative_Immersion': ['Q_3', 'Q_12', 'Q_18', 'Q_19', 'Q_27', 'Q_30'],
        'Flow': ['Q_5', 'Q_13', 'Q_25', 'Q_28', 'Q_31'],
        'Tension_Annoyance': ['Q_22', 'Q_24', 'Q_29'],
        'Challenge': ['Q_11', 'Q_23', 'Q_26', 'Q_32', 'Q_33'],
        'Negative_Affect': ['Q_7', 'Q_8', 'Q_9', 'Q_16'],
        'Positive_Affect': ['Q_1', 'Q_4', 'Q_6', 'Q_14', 'Q_20']
    }

    # Mean Aggregation of the components
    questionnaire_evaluation = pd.DataFrame()
    for component, items in components.items():
        questionnaire_evaluation[component] = questionnaire_data[items].mean(axis=1)

    # Grouping the two groups of respondents.
    questionnaire_evaluation["music"] = questionnaire_data["music"]
    group_with_music_questionnaire = questionnaire_evaluation[questionnaire_evaluation['music'] == 'yes'].copy()
    group_without_music_questionnaire = questionnaire_evaluation[questionnaire_evaluation['music'] == 'no'].copy()

    # Delete the column that is no longer needed.
    group_with_music_questionnaire.drop(columns="music", inplace=True)
    group_without_music_questionnaire.drop(columns="music", inplace=True)

    return group_with_music, group_without_music, group_with_music_questionnaire, group_without_music_questionnaire


# Method for generating the statistical analysis and plots.
def analyse(group_with_music, group_without_music, group_with_music_questionnaire, group_without_music_questionnaire):
    # Deskriptive Statistiken für Precision und Time
    print("\n\n---------- Descriptive stats for Precision with music ------------\n\n")
    print(group_with_music['Precision'].describe(include='all'))
    print("\n\n---------- Descriptive stats for Precision without music ------------\n\n")
    print(group_without_music['Precision'].describe(include='all'))

    print("\n\n---------- Descriptive stats for Time with music ------------\n\n")
    print(group_with_music['Time'].describe(include='all'))
    print("\n\n---------- Descriptive stats for Time without music ------------\n\n")
    print(group_without_music['Time'].describe(include='all'))

    # Output statistics for each column in the Questionnaire DataFrame individually
    print("\n\n---------- Descriptive Stats Questionnaire with music ------------\n\n")
    for column in group_with_music_questionnaire.columns:
        print(f"Column: {column}")
        print(group_with_music_questionnaire[column].describe())
        print("\n")

    print("\n\n---------- Descriptive Stats Questionnaire with music ------------\n\n")
    for column in group_without_music_questionnaire.columns:
        print(f"Column: {column}")
        print(group_without_music_questionnaire[column].describe())
        print("\n")

    # Plots
    plot_dir = 'statistics'  # directory to save the plots
    # check if the directory exists, if not create it
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # save the plots as png files in the directory
    time_precision_plot_filename = os.path.join(plot_dir, 'time_precision_boxplot.png')
    questionnaire_barplot_filename = os.path.join(plot_dir, 'questionnaire_barplots.png')
    questionnaire_boxplot_filename = os.path.join(plot_dir, 'questionnaire_boxplots.png')

    plt.figure(figsize=(12, 12))

    # Boxplot for Precision in group with music and without music
    plt.subplot(3, 2, 1)
    plt.boxplot([group_with_music['Precision'], group_without_music['Precision']],
                labels=['with music', 'without music'])
    plt.ylabel('Precision (%)')
    plt.title('Boxplot for Precision')


    # Boxplot for Time in group with music and without music
    plt.subplot(3, 2, 2)
    plt.boxplot([group_with_music['Time'], group_without_music['Time']],
                labels=['with music', 'without music'])
    plt.ylabel('Time (s)')
    plt.title('Boxplot for Time')
    plt.savefig(time_precision_plot_filename)

    questionnaire_columns = ['Competence', 'Sensory_and_Imaginative_Immersion', 'Flow',
                             'Tension_Annoyance', 'Challenge', 'Negative_Affect', 'Positive_Affect']

    num_subplots = len(questionnaire_columns)
    num_rows = 3  # Number of rows in the subplot grid
    num_cols = 3  # Number of colums in the subplot grid
    ylim = (0, 5)  # Scaling of the axis

    # Create the first figure with boxplots for Time and Precision.
    plt.figure(figsize=(12, 12))
    for i, column in enumerate(questionnaire_columns):
        if i >= num_subplots:
            break  # Exit the loop if there are more bar charts than subplots
        plt.subplot(num_rows, num_cols, i + 1)  # i + 1, to begin with 1
        plt.bar(['with music', 'without music'],
                [group_with_music_questionnaire[column].mean(), group_without_music_questionnaire[column].mean()],
                color=['red', 'blue'])
        plt.ylabel(column)
        plt.ylim(ylim)
        column_label = column.replace('_', ' ')
        plt.title(f'Mean {column_label}')
    plt.tight_layout()
    plt.savefig(questionnaire_barplot_filename)

    # Boxplots for questionnaires components
    plt.figure(figsize=(12, 12))
    plt.suptitle('Boxplots for Questionnaire', fontsize=16)

    for i, column in enumerate(questionnaire_columns):
        if i >= num_subplots:
            break  # Verlassen Sie die Schleife, wenn wir mehr Balkendiagramme haben als Subplots
        plt.subplot(num_rows, num_cols, i + 1)
        plt.boxplot([group_with_music_questionnaire[column], group_without_music_questionnaire[column]],
                    labels=['with music', 'without music'])
        plt.ylabel(column)
        plt.ylim(ylim)
        column_label = column.replace('_', ' ')
        plt.title(f'Boxplot {column_label}')
    plt.tight_layout()
    plt.savefig(questionnaire_boxplot_filename)

    plt.show()
    plt.close()


def independent_t_tests(group_with_music, group_without_music):
    print("\n\n---------- T-Test for Precision and Time ------------\n\n")
    # Perform separate t-tests for Precision and Time
    t_stat_precision, p_value_precision = stats.ttest_ind(group_with_music['Precision'],
                                                          group_without_music['Precision'])
    t_stat_time, p_value_time = stats.ttest_ind(group_with_music['Time'], group_without_music['Time'])

    # Output results
    print("Independent t-Tests:")
    print(f'Test for Precision - T-statistic: {t_stat_precision}, P-value: {p_value_precision}')
    print(f'Test for Time - T-statistic: {t_stat_time}, P-value: {p_value_time}\n\n')
    return t_stat_precision, p_value_precision, t_stat_time, p_value_time


def anova_questionnaire(group_with_music_questionnaire, group_without_music_questionnaire):
    print("\n\n---------- ANOVA Game Experience Questionnaire ------------\n\n")
    # Perform a one-factor ANOVA for each component of the Game Experience Questionnaire.

    # Components of the questionnaire
    geq_columns = ['Competence', 'Sensory_and_Imaginative_Immersion', 'Flow',
                   'Tension_Annoyance', 'Challenge', 'Negative_Affect', 'Positive_Affect']

    anova_results = {}
    # Iterate over each component
    for column in geq_columns:
        group_with_music_data = group_with_music_questionnaire[column]
        group_without_music_data = group_without_music_questionnaire[column]
        # Calculate F-value
        f_statistic, p_value = stats.f_oneway(group_with_music_data, group_without_music_data)
        anova_results[column] = {'F-statistic': f_statistic, 'P-value': p_value}

    # Output results
    for column, result in anova_results.items():
        print(f'Variable: {column}')
        print(f'F-statistic: {result["F-statistic"]}, P-value: {result["P-value"]}\n')
    print("\n\n")
    return anova_results


def correlation(group_with_music, group_without_music, group_with_music_questionnaire,
                group_without_music_questionnaire):
    print("\n\n---------- Correlation ------------\n\n")

    # Perform correlation analyses
    correlation_results = {}
    for column in group_with_music_questionnaire.columns:
        correlation_with_music_precision = group_with_music_questionnaire[column].corr(group_with_music['Precision'])
        correlation_without_music_precision = group_without_music_questionnaire[column].corr(
            group_without_music['Precision'])
        correlation_with_music_time = group_with_music_questionnaire[column].corr(group_with_music['Time'])
        correlation_without_music_time = group_without_music_questionnaire[column].corr(group_without_music['Time'])

        correlation_results[column] = {
            'Correlation with Music (Precision)': correlation_with_music_precision,
            'Correlation without Music (Precision)': correlation_without_music_precision,
            'Correlation with Music (Time)': correlation_with_music_time,
            'Correlation without Music (Time)': correlation_without_music_time
        }

    # Output results
    print("Correlation Analysis between GEQ Components, Precision, and Time:")
    for column, result in correlation_results.items():
        print(f'Variable: {column}')
        print(f'Correlation with Music (Precision): {result["Correlation with Music (Precision)"]}')
        print(f'Correlation without Music (Precision): {result["Correlation without Music (Precision)"]}')
        print(f'Correlation with Music (Time): {result["Correlation with Music (Time)"]}')
        print(f'Correlation without Music (Time): {result["Correlation without Music (Time)"]}\n')

    # Plot correlations as a heatmap
    columns = list(correlation_results.keys())
    correlations_precision = np.array([
        [result['Correlation with Music (Precision)'], result['Correlation without Music (Precision)']]
        for result in correlation_results.values()
    ])

    correlations_time = np.array([
        [result['Correlation with Music (Time)'], result['Correlation without Music (Time)']]
        for result in correlation_results.values()
    ])

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Breiteres Plot-Fenster mit zwei Subplots

    im_precision = ax[0].imshow(correlations_precision, cmap='coolwarm')
    im_time = ax[1].imshow(correlations_time, cmap='coolwarm')

    # Set labels and ticks
    for ax_idx, im, title in zip([0, 1], [im_precision, im_time], ['Precision', 'Time']):
        ax[ax_idx].set_xticks([0, 1])
        ax[ax_idx].set_yticks(range(len(columns)))
        ax[ax_idx].set_xticklabels(['With Music', 'Without Music'])
        ax[ax_idx].set_yticklabels(columns)
        ax[ax_idx].set_title(f'Correlations between GEQ Components and {title}')

        # Add colorbar
        cbar = ax[ax_idx].figure.colorbar(im, ax=ax[ax_idx])
        cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

    # Show plot
    plt.tight_layout()
    plt.savefig("statistics/correlation_plot.png")
    plt.show()

    return correlation_results
