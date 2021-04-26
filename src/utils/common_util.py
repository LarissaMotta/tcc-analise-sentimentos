import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import re
import nltk


# download dependencias de processamento de texto
def loading_dependences():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    return 0


def get_input_loader(df, batch_size=1):
    # separando os dados de treinamento e validacao
    tweet, polarity = np.array(df.Vetores.tolist()), df.Polaridade.tolist()
    # transformando os dados em tensor
    tensor_data = TensorDataset(torch.from_numpy(tweet.astype(np.int)),
                               torch.from_numpy(np.array(polarity, dtype=np.int)))
    # dados de entrada para a rede
    loader_data = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    return loader_data


# Pre processamento e conversão de texto para lista de palavras convert texts to a list of words
def texto_para_lista_palavras(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"br", " ", text)

    return text.split()


# deixando os vetores com mesmo tamanho
def pad_vec_tweet(tweet, seq_length=20):
    retorno = []
    if tweet != np.nan and len(tweet) >= seq_length:
        retorno = tweet[:seq_length]
    else:
        retorno = [0] * (seq_length - len(tweet)) + tweet
    return retorno


def clean_datas(df_p, df_n):
    # remove os na
    df_p = df_p.dropna()
    df_n = df_n.dropna()
    # balanceia os dados
    df_p, df_n = get_data_balanced(df_p, df_n)
    # separa os dados de treinamento, validacao e teste
    df1_p, df2_p, df3_p = get_cut_data(df_p)
    df1_n, df2_n, df3_n = get_cut_data(df_n)

    df = pd.merge(df1_p, df1_n, how='outer')
    df2 = pd.merge(df2_p,df2_n, how='outer')
    df3 = pd.merge(df3_p, df3_n, how='outer')

    return df, df2, df3


def equal_data(df, df2):
    if len(df) > len(df2):
        df = df.iloc[:len(df2)]
    else:
        df2 = df2.iloc[:len(df)]
    return df, df2


def get_data_balanced(df, df2, div=100):
    # pegando o menor tamanho entre os dataframes
    cut_len = min(len(df), len(df2))
    # deixando o corte divisivel pelo parametro passado
    cut_len = cut_len - (cut_len % div)
    # cortando os dataframes
    df = df.iloc[:cut_len]
    df2 = df2.iloc[:cut_len]

    return df, df2


def get_cut_data(df_):
    train_len = int(len(df_) * 0.8)
    valid_len = int(len(df_) * 0.1)

    df1 = df_.iloc[:train_len]
    df2 = df_.iloc[train_len: (train_len + valid_len)]
    df3 = df_.iloc[(train_len + valid_len):]

    return df1, df2, df3


def pre_process_tweet(text):
    # remove URLs
    text = re.sub("((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", "", text)
    text = re.sub(r'http\S+', '', text)
    # remove usernames
    text = re.sub('@[^\s]+', '', text)
    # remove all #
    text = re.sub(r'#([^\s]+)', '', text)
    # remove
    text = re.sub(r"[^\w\s]", " ", text)

    return text.split()

