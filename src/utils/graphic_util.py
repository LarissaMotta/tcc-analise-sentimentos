import seaborn as sn
import matplotlib.pyplot as plt


def plot_confusion_matrix(confusion_matrix):
    # df_cm = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10, 8))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.show()
