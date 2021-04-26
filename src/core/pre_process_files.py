import src.utils.imports_util as imports
import src.utils.graphic_util as graphic_util

import pandas as pd


def get_dataframes_min():
    df_pos = pd.read_csv(imports.POSITIVE_TWEETS_PATH)
    df_neg = pd.read_csv(imports.NEGATIVE_TWEETS_PATH)

    # df_pos = df_pos.iloc[:2000]
    # df_neg = df_neg.iloc[:2000]
    df_pos = df_pos.iloc[:28000]
    df_neg = df_neg.iloc[:28000]

    df_pos.loc[:, 'Polaridade'] = 1
    df_neg.loc[:, 'Polaridade'] = 0

    graphic_util.plot_hist_length_dataframe(df_pos, imports.POSITIVE_HISTOGRAMA_PATH)
    graphic_util.plot_hist_length_dataframe(df_neg, imports.NEGATIVE_HISTOGRAMA_PATH)
    graphic_util.plot_hist_length_dataframe(pd.concat([df_pos, df_neg]), imports.HISTOGRAMA_PATH)

    # train_len = int(len(df_pos) * 0.8)
    # valid_len = int(len(df_pos) * 0.1)

    train_len = round(len(df_pos) * 0.7)
    valid_len = round(len(df_pos) * 0.2)

    df1 = pd.merge(df_pos.iloc[:train_len], df_neg.iloc[:train_len], how = 'outer')
    df2 = pd.merge(df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len + valid_len)], how = 'outer')
    df3 = pd.merge(df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len + valid_len):], how = 'outer')
    return df1, df2, df3
    # return df_pos, df_neg


def get_dataframe():
    df_pos = pd.read_csv(imports.POSITIVE_TWEETS_PATH)
    df_neg = pd.read_csv(imports.NEGATIVE_TWEETS_PATH)

    df_pos = df_pos.dropna()
    df_neg = df_neg.dropna()

    if len(df_pos) > len(df_neg):
        df_pos = df_pos.iloc[:len(df_neg)]
    else:
        df_neg = df_neg.iloc[:len(df_pos)]

    df_pos.loc[:, 'Polaridade'] = 1
    df_neg.loc[:, 'Polaridade'] = 0

    graphic_util.plot_hist_length_dataframe(df_pos, imports.POSITIVE_HISTOGRAMA_PATH)
    graphic_util.plot_hist_length_dataframe(df_neg, imports.NEGATIVE_HISTOGRAMA_PATH)

    train_len = int(len(df_pos) * 0.8)
    valid_len = int(len(df_pos) * 0.1)

    df1 = pd.merge(df_pos.iloc[:train_len], df_neg.iloc[:train_len], how='outer')
    df2 = pd.merge(df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len + valid_len)],
                   how='outer')
    df3 = pd.merge(df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len + valid_len):], how='outer')
    return df1, df2, df3

    # return df_pos, df_neg


def get_histogram_min():
    df_pos = pd.read_csv(imports.POSITIVE_TWEETS_PATH)
    df_neg = pd.read_csv(imports.NEGATIVE_TWEETS_PATH)

    # df_pos = df_pos.iloc[:2000]
    # df_neg = df_neg.iloc[:2000]
    df_pos = df_pos.iloc[:28000]
    df_neg = df_neg.iloc[:28000]

    df_pos.loc[:, 'Polaridade'] = 1
    df_neg.loc[:, 'Polaridade'] = 0

    # graphic_util.plot_hist_length_dataframe(df_pos, imports.POSITIVE_HISTOGRAMA_PATH)
    # graphic_util.plot_hist_length_dataframe(df_neg, imports.NEGATIVE_HISTOGRAMA_PATH)
    graphic_util.plot_hist_length_dataframe(pd.concat([df_pos, df_neg]), imports.HISTOGRAMA_PATH)

    return
