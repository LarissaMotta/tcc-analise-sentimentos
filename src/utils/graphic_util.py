import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import utils.import_util as imports
import utils.df_util as df_util


def plot_confusion_matrix():
    cf_matrix = df_util.get_data_confusion_matrix()
    plt.figure()
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['({0:.1%})'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    mt_cf = sn.heatmap(cf_matrix, annot=labels, fmt='', alpha=0.8, cmap=['#ff3385', '#cceeff'], yticklabels=['NEGATIVO', 'POSITIVO'],
                     xticklabels=['NEGATIVO', 'POSITIVO'], cbar=False, square=True, annot_kws={"fontsize":20})

    mt_cf.set_ylabel('VALOR REAL', fontdict=dict(weight='bold'))
    mt_cf.set_xlabel('VALOR PREDITO', fontdict=dict(weight='bold'))

    # salvando a figura
    plt.savefig(imports.MATRIX_CONFUSION, bbox_inches='tight', pad_inches=0.1)
    return

## para printar os valores máximos de treinamento e validação
# def plot_graphic_train_valid(legend_graphic):
#     acc_train, acc_valid, loss_train, loss_valid = df_util.get_datas_graphic_train_valid()
#     print(legend_graphic, ' -> ', imports.GRAPHIC_TRAIN)
#     print("acc_train_max: ", acc_train[-1])
#     print("acc_valid_max: ", acc_valid[-1])
#     print("loss_train_min: ", loss_train[-1])
#     print("loss_valid_min: ", loss_valid[-1])
#     print('\n')


def plot_graphic_train_valid(legend_graphic):
    # PEGAR OS VALORES
    acc_train, acc_valid, loss_train, loss_valid = df_util.get_datas_graphic_train_valid()
    # PLOTAR GRAFICO
    # Plot acuracia
    plt.figure()
    plt.subplot(211)
    plt.plot(acc_train)
    plt.plot(acc_valid)
    plt.title('Acurácia média' + ' LSTM - ' + legend_graphic)
    plt.ylabel('Acurácia média')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(loss_train)
    plt.plot(loss_valid)
    plt.title('Loss médio' + ' LSTM - ' + legend_graphic)
    plt.ylabel('Loss médio')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig(imports.GRAPHIC_TRAIN)
    return
