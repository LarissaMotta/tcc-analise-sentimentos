import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re
# from string import punctuation


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
    if len(tweet) >= seq_length:
        retorno = tweet[:seq_length]
    else:
        retorno = [0] * (seq_length - len(tweet)) + tweet
    return retorno


# def preprocess(text):
#     text = text.lower()
#     text = "".join([ch for ch in text if ch not in punctuation])
#     all_reviews = text.split("\n")
#     text = " ".join(all_reviews)
#     all_words = text.split()
#
#     return all_reviews, all_words