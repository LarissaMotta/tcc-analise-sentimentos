# Imports
import src.core.training as training
import src.core.testing as testing
import src.utils.df_util as df_util
import src.models.hiperparametros as params
import src.utils.results_util as results_util
import src.utils.import_util as imports


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # obtendo os hiperparametros setados
    model = params.hiperparams()
    dic_part = {'first': {'train': 0.7, 'valid': 0.15},
                'second': {'train': 0.8, 'valid': 0.1}}
    dic_database = {'zero': {'seq_length': 14, 'cut': 0},
                    'one': {'seq_length': 13, 'cut': 1},
                    'two': {'seq_length': 12, 'cut': 2}}

    # execute(model)

    for particao in dic_part.values():
        model.len_train = particao['train']
        model.len_valid = particao['valid']

        for dado in dic_database.values():
            __set_path_files(dado, particao)

            for i in range(20):
                execute(model)

    return 0


def execute(model):
    # convertendo os arquivos para as variáveis
    matrix_embedding, df_p, df_n, seq_length = df_util.get_process_data()
    # separando o dataset em treinamento, validação e teste
    df, df2, df3 = df_util.cut_dataset(df_p, df_n, model.len_train)
    # df, df2, df3 = df_util.cut_dataset(df_p, df_n, model.len_train, model.len_valid)
    # treinando a rede
    net, batch_size = training.training(df, df2, matrix_embedding, seq_length, model)
    # __set_datas_on_model(model, net)
    results_util.save_infos_train_valid(net)
    # testando a rede
    pred, net = testing.testing(df3, seq_length)
    # salvando os resultados e printando os resultados de predicao
    results_util.save_results(model, net, pred, df3)
    return 0


def __set_path_files(dado, particao):
    imports.SEQ_LENGTH = dado['seq_length']
    imports.MATRIX_EMBEDDING = imports.MATRIX_EMBEDDING_BASE.format(dado['cut'], dado['seq_length'])
    imports.POSITIVE_TWEETS_PATH_PROCESS = imports.POSITIVE_TWEETS_PATH_PROCESS_BASE.format(dado['cut'],
                                                                                            dado['seq_length'])
    imports.NEGATIVE_TWEETS_PATH_PROCESS = imports.NEGATIVE_TWEETS_PATH_PROCESS_BASE.format(dado['cut'],
                                                                                            dado['seq_length'])
    imports.ACC_TRAIN_PATH = imports.ACC_TRAIN_PATH_BASE.format(dado['cut'], particao['train'])
    imports.ACC_VALID_PATH = imports.ACC_VALID_PATH_BASE.format(dado['cut'], particao['train'])
    imports.LOSS_TRAIN_PATH = imports.LOSS_TRAIN_PATH_BASE.format(dado['cut'], particao['train'])
    imports.LOSS_VALID_PATH = imports.LOSS_VALID_PATH_BASE.format(dado['cut'], particao['train'])
    imports.RESULT_PATH = imports.METRIC_PATH_BASE.format(dado['cut'], particao['train'])
    return


def __set_datas_on_model(model, net):
    model.acc_train = "{:.6f}".format(sum(net.history['accuracy']) / len(net.history['accuracy']))
    model.acc_valid = "{:.6f}".format(sum(net.history['val_accuracy']) / len(net.history['val_accuracy']))
    model.loss_train = "{:.6f}".format(sum(net.history['loss']) / len(net.history['loss']))
    model.loss_valid = "{:.6f}".format(sum(net.history['val_loss']) / len(net.history['val_loss']))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
