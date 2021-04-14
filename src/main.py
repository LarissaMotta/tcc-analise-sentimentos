# Imports
import src.core.pre_process_files as process_file
import src.core.training_w2v as training_w2v
import src.core.training as training
import src.core.testing as testing

import torch
from torch import nn


import src.utils.imports_util as imports

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():

    # pega os dataframes, sendo um de treinamento, um de validacao e um de teste
    df, df2, df3 = process_file.get_dataframes_min()
    # df, df2, df3 = process_file.get_dataframe()

    # treinamento da rede
    #net, test_loader, batch_size, criterion = training_w2v.training(df, df2, df3)
    net, test_loader, batch_size, criterion = training.training(df, df2, df3)

    # testando a rede
    testing.testing(net, test_loader, batch_size, criterion)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
