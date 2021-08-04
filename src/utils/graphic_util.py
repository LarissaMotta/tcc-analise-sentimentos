import seaborn as sn
import matplotlib.pyplot as plt
import stylecloud
import src.utils.df_util as df_util


def plot_confusion_matrix(confusion_matrix):
    # df_cm = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10, 8))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.show()


def create_adjetives_cloud():
    pos_words, neg_words = df_util.get_adjetives()
    file_pos = 'D:\\Documentos\\FACULDADE\\TCC\\Codigo\\tcc-analise-sentimentos\\src\\datas\\adjetives\\wordcloud_positive.png'
    file_neg = 'D:\\Documentos\\FACULDADE\\TCC\\Codigo\\tcc-analise-sentimentos\\src\\datas\\adjetives\\wordcloud_negative.png'

    stylecloud.gen_stylecloud(text=pos_words,
                              icon_name="fas fa-thumbs-up",
                              output_name=file_pos)

    stylecloud.gen_stylecloud(text=neg_words,
                              icon_name="fas fa-thumbs-down",
                              output_name=file_neg)
