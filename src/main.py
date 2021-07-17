# Imports
import src.core.training as training
import src.core.testing as testing
import src.utils.df_util as df_util
import src.models.hiperparametros as params
import src.utils.results_util as results_util
import src.utils.model_util as model_util


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # rho = [0.9, 0.95, 0.97, 0.99]
    # obtendo os hiperparametros setados
    model_util.get_activation_function()
    # hp.try_hyperas()
    model = params.hiperparams()
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
    __set_datas_on_model(model, net)
    # testando a rede
    pred, net = testing.testing(df3, seq_length)
    # salvando os resultados e printando os resultados de predicao
    results_util.save_results(model, net, pred, df3)
    return 0


def __set_datas_on_model(model, net):
    model.acc_train = "{:.6f}".format(sum(net.history['accuracy']) / len(net.history['accuracy']))
    model.acc_valid = "{:.6f}".format(sum(net.history['val_accuracy']) / len(net.history['val_accuracy']))
    model.loss_train = "{:.6f}".format(sum(net.history['loss']) / len(net.history['loss']))
    model.loss_valid = "{:.6f}".format(sum(net.history['val_loss']) / len(net.history['val_loss']))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
