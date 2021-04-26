# Imports
import src.core.pre_process_files as process_file
import src.core.training_w2v as training_w2v
import src.models.hiperparametros as params
import src.core.training as training
import src.core.testing as testing
import src.utils.file_util as file_util

import torch
from torch import nn


import src.utils.imports_util as imports

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # file_util.remove_duplicates_datas()
    # file_util.clean_crude_dataset()
    # file_util.remove_stopwords_of_datasets()
    # file_util.lemmatization_of_datasets()
    # process_file.get_histogram_min()

    # pega os dataframes, sendo um de treinamento, um de validacao e um de teste
    df, df2, df3 = process_file.get_dataframes_min()
    # df, df2, df3 = process_file.get_dataframe()

    # treinamento da rede
    net, test_loader, batch_size, criterion, csv_list = training_w2v.training(df, df2, df3)
    # net, test_loader, batch_size, criterion = training.training(df, df2)

    # testando a rede
    testing.testing(net, test_loader, batch_size, criterion, csv_list)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
