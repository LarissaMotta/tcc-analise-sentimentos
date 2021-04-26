import pandas as pd
import seaborn as sn
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# HISTOGRAMA DO DATAFRAME
def plot_hist_length_dataframe(dataframe, filename):
    data = __phrases_dataframe_size(dataframe)

    plt.figure(figsize=[10, 8])
    plt.hist(x=data, bins=10, color='#D24324', alpha=0.9, rwidth=1)
    plt.xlabel("Sentence Length", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title("Sentences Distribution Histogram", fontsize=15)
    plt.legend(handles=__statistics_mpatches(data), handlelength=0, handletextpad=0, fancybox=True)
    plt.savefig(filename)


def __phrases_dataframe_size(dataframe):
    tweet_size = []

    series_phrase = dataframe['Texto'].str.split().str.len().fillna(0)
    tweet_size.extend(series_phrase.tolist())

    return tweet_size


def __statistics_mpatches(data):
    qnt_patch = mpatches.Patch(label="Qnt. - {qnt}".format(qnt=len(data)))
    min_patch = mpatches.Patch(label="Min. - {min}".format(min=int(min(data))))
    max_patch = mpatches.Patch(label="Max. - {max}".format(max=int(max(data))))
    mean_patch = mpatches.Patch(label="Mean - {:.2f}".format(statistics.mean(data)))
    mode_patch = mpatches.Patch(label="Mode - {mode}".format(mode=statistics.mode(data)))
    median_patch = mpatches.Patch(label="Median - {:.2f}".format(statistics.median(data)))
    stddev_patch = mpatches.Patch(label="Std. Dev. - {:.2f}".format(statistics.stdev(data)))
    variance_patch = mpatches.Patch(label="Variance - {:.2f}".format(statistics.variance(data)))

    patches = [qnt_patch, min_patch, max_patch, mean_patch, mode_patch, median_patch, stddev_patch, variance_patch]

    return patches


def plot_confusion_matrix(confusion_matrix, filename):
    # df_cm = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10, 8))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.savefig(filename)


def create_plot_multiple(x, y, y2, title_graph, x_label, y_label, y_label2, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, label=y_label)
    plt.plot(x, y2, label=y_label2)
    plt.xlabel(x_label)
    plt.title(title_graph)
    plt.legend()
    plt.savefig(filename)
