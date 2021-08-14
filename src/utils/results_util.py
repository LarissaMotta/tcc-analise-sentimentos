import os
import csv
import pandas as pd
import utils.import_util as imports
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


# hiperparametros, net, pred
def save_results(model, train, pred, df_test):
    # Adicionando os dados do modelo de hiperparametro
    lst = [model.date_now, model.n_hidden, model.lr, model.drop_1, model.initializer, model.drop_recurrent,
           model.activation, model.loss, model.optimizer, model.n_epochs, model.batch_size, model.len_train,
           model.len_valid]

    # Adicionando dados de previsao
    lst = lst + __get_values_pred(pred, df_test)

    # Adicionando dado da embedding
    lst.append(imports.MATRIX_EMBEDDING)


    # Salvando os dados no arquivo
    insert_informations_file(lst, imports.RESULT_PATH)
    return


def __get_values_pred(pred, df3):
    # Tratando o retorno da predicao
    # colocando a predicao na dataframe de teste
    df3['Predict'] = pred

    # salvando no csv antes de mudar o valor caso queiramos consultar futuramente
    df3.to_csv(imports.PREDICT_DATASET, index=False)

    # setando 0 e 1 para fazer a comparacao com a classificacao esperada
    df_clas_p = df3.loc[df3['Predict'] >= 0.5]
    df_clas_n = df3.loc[df3['Predict'] < 0.5]
    df_clas_n.loc[:, 'Predict'] = 0
    df_clas_p.loc[:, 'Predict'] = 1
    df_new = pd.concat([df_clas_n, df_clas_p])

    # montando a matriz de confusão
    confusion_mt = confusion_matrix(df_new.Polaridade, df_new.Predict)
    tn, fp, fn, tp = confusion_matrix(df_new.Polaridade, df_new.Predict).ravel()  # apenas pra facilitar o IoU
    print('confusion_mt:\n', confusion_mt)
    cf = 'tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn) + ' tp=' + str(tp)

    # calculando a acurácia
    accuracy = accuracy_score(df_new.Polaridade, df_new.Predict)
    print('accuracy: ', accuracy)

    # calculando a precisão
    precision = precision_score(df_new.Polaridade, df_new.Predict)
    print('precision: ', precision)

    # calculando o recall
    recall = recall_score(df_new.Polaridade, df_new.Predict)
    print('recall: ', recall)

    # calculando o f1-score
    f1 = f1_score(df_new.Polaridade, df_new.Predict)
    print('f1: ', f1)

    # calculando o IoU
    iou = tp / (tp + fp + fn)
    print('IoU: ', iou)

    return [accuracy, tn, fp, fn, tp, precision, recall, f1, iou]


def save_infos_train_valid(net):
    insert_informations_file(net.history['accuracy'], imports.ACC_TRAIN_PATH)
    insert_informations_file(net.history['val_accuracy'], imports.ACC_VALID_PATH)
    insert_informations_file(net.history['loss'], imports.LOSS_TRAIN_PATH)
    insert_informations_file(net.history['val_loss'], imports.LOSS_VALID_PATH)


def insert_informations_file(lst, file):
    append_write = 'a' if os.path.exists(file) else 'w'

    with open(file, append_write) as outfile:
        file = csv.writer(outfile)
        file.writerow(lst)
    return

