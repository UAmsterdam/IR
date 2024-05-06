import pandas as pd
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def read_split(file_path):
    return [el for el in open(file_path).read().split('\n') if el != '']

def map_doc(row):
    return row['sent_id'].split('_')[0]

def stratify(df):
    from sklearn.model_selection import train_test_split
    g = df.groupby('label')
    return df, g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
