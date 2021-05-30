import src.utils.import_util as imports
import numpy as np
import pandas as pd


def get_process_data():
    seq_length = 17
    matrix_embedding = np.load(imports.MATRIX_EMBEDDING)
    df_p = pd.read_csv(imports.POSITIVE_TWEETS_PATH_PROCESS)
    df_n = pd.read_csv(imports.NEGATIVE_TWEETS_PATH_PROCESS)

    df_p.Vetores = df_p.Vetores.apply(lambda x: __convert_strig_to_vector(x))
    df_n.Vetores = df_n.Vetores.apply(lambda x: __convert_strig_to_vector(x))

    return matrix_embedding, df_p, df_n, seq_length


def __convert_strig_to_vector(txt):
    txt = txt.replace('[', '')
    txt = txt.replace(']', '')
    txt = txt.split(',')
    lst = [int(i) for i in txt]
    return lst


def cut_dataset(df_pos, df_neg, train, valid):
    train_len = round(len(df_pos) * train)
    valid_len = round(len(df_pos) * valid)

    train_len_n = round(len(df_neg) * train)
    valid_len_n = round(len(df_neg) * valid)

    df1 = pd.concat([df_pos.iloc[:train_len], df_neg.iloc[:train_len_n]])
    df2 = pd.concat([df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len_n + valid_len_n)]])
    df3 = pd.concat([df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len_n + valid_len_n):]])

    return df1, df2, df3

