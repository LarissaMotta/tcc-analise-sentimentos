# Imports
import src.core.training as training
import src.core.testing as testing
import src.utils.df_util as df_util
import src.models.hiperparametros as params
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd



import src.utils.import_util as imports

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # convertendo os arquivos para as variáveis
    matrix_embedding, df_p, df_n, seq_length = df_util.get_process_data()
    # obtendo os hiperparametros setados
    model = params.hiperparams()
    # separando o dataset em treinamento, validação e teste
    df, df2, df3 = df_util.cut_dataset(df_p, df_n, model.len_train, model.len_valid)
    # treinando a rede
    net, batch_size = training.training(df, df2, matrix_embedding, seq_length, model)
    # testando a rede
    pred, net = testing.testing(df3, 20)

    # Tratando o retorno da predição
    # colocando a predição na dataframe de teste
    df3['Predict'] = pred
    # salvando no csv antes de mudar o valor caso queiramos consultar futuramente
    df3.to_csv(imports.PREDICT_DATASET, index=False)
    # setando 0 e 1 para fazer a comparação com a classificação esperada
    df_clas_p = df3.loc[df3['Predict'] >= 0.5]
    df_clas_n = df3.loc[df3['Predict'] < 0.5]
    df_clas_n.loc[:, 'Predict'] = 0
    df_clas_p.loc[:, 'Predict'] = 1
    df_new = pd.concat([df_clas_n, df_clas_p])
    # montando a matriz de confusão
    confusion_mt = confusion_matrix(df_new.Polaridade, df_new.Predict)
    tn, fp, fn, tp = confusion_matrix(df_new.Polaridade, df_new.Predict).ravel() #apenas pra facilitar o IoU
    print('confusion_mt:\n', confusion_mt)
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
    iou = tp/(tp+fp+fn)
    print('IoU: ', iou)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
