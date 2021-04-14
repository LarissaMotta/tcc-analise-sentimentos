import src.utils.imports_util as imports

import pandas as pd

# def plot_hist_length_dataframe(dataframe, filename):
#     data = __phrases_dataframe_size(dataframe)
#
#     plt.figure(figsize=[10, 8])
#     plt.hist(x=data, bins=10, color='#D24324', alpha=0.9, rwidth=1)
#     plt.xlabel("Sentence Length", fontsize=15)
#     # plt.xticks(np.arange(0, max(data) + 1, 20), fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.ylabel("Frequency", fontsize=15)
#     plt.title("Sentences Distribution Histogram", fontsize=15)
#     plt.legend(handles=__statistics_mpatches(data), handlelength=0, handletextpad=0, fancybox=True)
#     plt.savefig(filename)
#
#
# def __phrases_dataframe_size(dataframe):
#     tweet_size = []
#
#     for column in dataframe['Texto']:
#         series_phrase = dataframe[column].str.split().str.len().fillna(0)
#         tweet_size.extend(series_phrase.tolist())
#
#     return tweet_size
#
#
# def __statistics_mpatches(data):
#     qnt_patch = mpatches.Patch(label="Qnt. - {qnt}".format(qnt=len(data)))
#     min_patch = mpatches.Patch(label="Min. - {min}".format(min=int(min(data))))
#     max_patch = mpatches.Patch(label="Max. - {max}".format(max=int(max(data))))
#     mean_patch = mpatches.Patch(label="Mean - {:.2f}".format(statistics.mean(data)))
#     mode_patch = mpatches.Patch(label="Mode - {mode}".format(mode=statistics.mode(data)))
#     median_patch = mpatches.Patch(label="Median - {:.2f}".format(statistics.median(data)))
#     stddev_patch = mpatches.Patch(label="Std. Dev. - {:.2f}".format(statistics.stdev(data)))
#     variance_patch = mpatches.Patch(label="Variance - {:.2f}".format(statistics.variance(data)))
#
#     patches = [qnt_patch, min_patch, max_patch, mean_patch, mode_patch, median_patch, stddev_patch, variance_patch]
#
#     return patches


def get_dataframes_min():
    df_pos = pd.read_csv(imports.POSITIVE_TWEETS_PATH)
    df_neg = pd.read_csv(imports.NEGATIVE_TWEETS_PATH)

    df_pos = df_pos.iloc[:2000]
    df_neg = df_neg.iloc[:2000]

    train_len = int(len(df_pos) * 0.8)
    valid_len = int(len(df_pos) * 0.1)

    df1 = pd.merge(df_pos.iloc[:train_len], df_neg.iloc[:train_len], how = 'outer')
    df2 = pd.merge(df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len + valid_len)], how = 'outer')
    df3 = pd.merge(df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len + valid_len):], how = 'outer')

    return df1, df2, df3


def get_dataframe():
    df_pos = pd.read_csv(imports.POSITIVE_TWEETS_PATH)
    df_neg = pd.read_csv(imports.NEGATIVE_TWEETS_PATH)

    if len(df_pos) > len(df_neg):
        df_pos = df_pos.iloc[:len(df_neg)]
    else:
        df_neg = df_neg.iloc[:len(df_pos)]

    train_len = int(len(df_pos) * 0.8)
    valid_len = int(len(df_pos) * 0.1)

    df1 = pd.merge(df_pos.iloc[:train_len], df_neg.iloc[:train_len], how='outer')
    df2 = pd.merge(df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len + valid_len)],
                   how='outer')
    df3 = pd.merge(df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len + valid_len):], how='outer')

    return df1, df2, df3