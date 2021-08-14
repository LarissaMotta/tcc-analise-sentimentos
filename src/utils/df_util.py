from sklearn.model_selection import train_test_split
import utils.import_util as imports
import numpy as np
import pandas as pd


def get_process_data():
    seq_length = imports.SEQ_LENGTH
    matrix_embedding = np.load(imports.MATRIX_EMBEDDING)
    df_p = pd.read_csv(imports.POSITIVE_TWEETS_PATH_PROCESS)
    df_n = pd.read_csv(imports.NEGATIVE_TWEETS_PATH_PROCESS)

    df_p.Vetores = df_p.Vetores.apply(lambda x: __convert_strig_to_vector(x))
    df_n.Vetores = df_n.Vetores.apply(lambda x: __convert_strig_to_vector(x))

    return matrix_embedding, df_p, df_n, seq_length


def __convert_strig_to_vector(txt):
    txt = txt.replace('[', '')
    txt = txt.replace(']', '')
    txt = txt.split(',')
    lst = [int(i) for i in txt]
    return lst


# def cut_dataset(df_pos, df_neg, train, valid):
#     train_len = round(len(df_pos) * train)
#     valid_len = round(len(df_pos) * valid)
#
#     train_len_n = round(len(df_neg) * train)
#     valid_len_n = round(len(df_neg) * valid)
#
#     df1 = pd.concat([df_pos.iloc[:train_len], df_neg.iloc[:train_len_n]])
#     df2 = pd.concat([df_pos.iloc[train_len: (train_len + valid_len)], df_neg.iloc[train_len: (train_len_n + valid_len_n)]])
#     df3 = pd.concat([df_pos.iloc[(train_len + valid_len):], df_neg.iloc[(train_len_n + valid_len_n):]])
#
#     return df1, df2, df3

def cut_dataset(df_pos, df_neg, train):

    df = pd.concat([df_pos, df_neg])

    df_train, df_valid_test = train_test_split(df, train_size=train, random_state=0, stratify=df['Polaridade'])
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=0, stratify=df_valid_test['Polaridade'])

    return df_train, df_valid, df_test

def get_datas_graphic_train_valid():
    df_acc_train = pd.read_csv(imports.ACC_TRAIN_PATH)
    df_acc_valid = pd.read_csv(imports.ACC_VALID_PATH)
    df_loss_train = pd.read_csv(imports.LOSS_TRAIN_PATH)
    df_loss_valid = pd.read_csv(imports.LOSS_VALID_PATH)

    lst_acc_train = __get_lst_mean_colums(df_acc_train)
    lst_acc_valid = __get_lst_mean_colums(df_acc_valid)
    lst_loss_train = __get_lst_mean_colums(df_loss_train)
    lst_loss_valid = __get_lst_mean_colums(df_loss_valid)

    return lst_acc_train, lst_acc_valid, lst_loss_train, lst_loss_valid


def __get_lst_mean_colums(df):
    # limpar as colunas vazias
    df = df.dropna()
    # pegando as medias das colunas
    lst = []
    for i in df.columns:
        lst.append(df[i].mean())
    return lst

def get_data_confusion_matrix():
    header = ['DateTime', 'n_hidden', 'learning_rate', 'drop_1', 'inicializador', 'drop_recorrente', 'ativação', 'loss',
              'otimizador', 'n_epocas', 'batch_size', 'len_train', 'len_valid', 'acc_teste', 'true_neg', 'false_pos',
              'false_neg', 'true_pos', 'precision', 'recall', 'f1', 'IoU', 'embedding']
    df_m = pd.read_csv(imports.RESULT_PATH, names=header)
    df_m = df_m.dropna()
    return np.array([[int(df_m.true_neg.mean()), int(df_m.false_pos.mean())], [int(df_m.false_neg.mean()), int(df_m.true_pos.mean())]])