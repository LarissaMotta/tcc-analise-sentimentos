import seaborn as sn
import matplotlib.pyplot as plt
import src.utils.import_util as imports
import src.utils.df_util as df_util


def plot_confusion_matrix():
    cf_matrix = df_util.get_data_confusion_matrix()
    plt.figure()
    mt_cf = sn.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', yticklabels=['NEGATIVO', 'POSITIVO'],
                     xticklabels=['NEGATIVO', 'POSITIVO'], cbar=False, square=True)

    mt_cf.set_ylabel('VALOR REAL', fontdict=dict(weight='bold'))
    mt_cf.set_xlabel('VALOR PREVISTO', fontdict=dict(weight='bold'))

    # salvando a figura
    plt.savefig(imports.MATRIX_CONFUSION)
    return


def plot_graphic_train_valid():
    # PEGAR OS VALORES
    acc_train, acc_valid, loss_train, loss_valid = df_util.get_datas_graphic_train_valid()
    # PLOTAR GRAFICO
    # Plot acuracia
    plt.figure()
    plt.subplot(211)
    plt.plot(acc_train)
    plt.plot(acc_valid)
    plt.title('Accuracy' + ' LSTM')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(loss_train)
    plt.plot(loss_valid)
    plt.title('Loss' )
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig(imports.GRAPHIC_TRAIN)
    return
