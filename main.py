import sys
import os

import analysis


# Function to redirect the output to a file
def redirect_output_to_file(file_path):
    # Saving the current sys.stdout
    original_stdout = sys.stdout
    # check if the directory exists, if not create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # Open the file in write mode
    sys.stdout = open(file_path+"/statistics.txt", 'w')
    return original_stdout


# Function to restore the standard output
def restore_output(original_stdout):
    sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    original_stdout = redirect_output_to_file('statistics')
    group_with_music, group_without_music, questionnaire = analysis.read_data()
    group_with_music, group_without_music, group_with_music_questionnaire, group_without_music_questionnaire = \
        analysis.generate_dataframe(group_with_music, group_without_music, questionnaire)
    analysis.analyse(group_with_music, group_without_music, group_with_music_questionnaire,
                     group_without_music_questionnaire)

    analysis.independent_t_tests(group_with_music, group_without_music)

    analysis.anova_questionnaire(group_with_music_questionnaire, group_without_music_questionnaire)

    analysis.correlation(group_with_music, group_without_music, group_with_music_questionnaire,
                         group_without_music_questionnaire)

    # Write output to the file
    restore_output(original_stdout)
    print("All statistics have been saved to 'statistics.txt'")
    # analysis.generate_statistics_per_respondent(group_with_music,group_without_music)
