import analysis

if __name__ == '__main__':
    group_with_music, group_without_music = analysis.read_data()
    analysis.generate_statistics(group_with_music, group_without_music)
