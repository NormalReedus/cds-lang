import os, pandas as pd


def load_got_data():
    got_filepath = os.path.join('data', 'Game_of_Thrones_Script.csv')
    got_df = pd.read_csv(got_filepath, usecols=['Sentence', 'Season'])

    sentences = got_df['Sentence'].values
    seasons = got_df['Season'].values

    return sentences, seasons
