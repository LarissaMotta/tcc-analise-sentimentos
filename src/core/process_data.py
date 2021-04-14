import src.utils.common_util as util

import numpy as np
import nltk
import nltk.corpus as corpus


# download dependencias de processamento de texto
def loading_dependences():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    return 0


# gerar dicionario com frequencia de todas as palavras no dataset
def generate_dictionary(vocabs, tweets, stopwords):
    for tweet in tweets:
        for word in tweet:
            # pular as palabras indesejadas
            if word in stopwords:
                continue

            # Add a palavra e frequencia no vocabulario
            if word not in vocabs:
                vocabs[word] = 1
            else:
                vocabs[word] = vocabs[word] + 1
    return vocabs


# funcao que vetoriza o tweet, caso n possua nenhuma palavra vetorizada retorna NaN
def tweet_to_vector(tweet, dic):
    tweet_embedding = []
    for word in tweet:
        if word in dic:
            tweet_embedding.append(dic[word])
    if len(tweet_embedding) > 0:
        return tweet_embedding
    return np.nan


def process_data_freq(df, df2, df3):
    # fazendo download das dependencias
    loading_dependences()
    stopwords = set(corpus.stopwords.words('english'))

    # Aplicando a tokenização do texto
    df.Texto = df.Texto.apply(lambda x: util.texto_para_lista_palavras(x))
    df2.Texto = df2.Texto.apply(lambda x: util.texto_para_lista_palavras(x))
    df3.Texto = df3.Texto.apply(lambda x: util.texto_para_lista_palavras(x))

    # gerar dicionario com as frequências
    dic_values_word = {}
    dic_values_word = generate_dictionary(dic_values_word, df.Texto, stopwords)
    dic_values_word = generate_dictionary(dic_values_word, df2.Texto, stopwords)
    dic_values_word = generate_dictionary(dic_values_word, df3.Texto, stopwords)

    # Cria a coluna com os tweets vetorizados
    df['Vetores'] = df.Texto.apply(lambda x: tweet_to_vector(x, dic_values_word))
    df2['Vetores'] = df2.Texto.apply(lambda x: tweet_to_vector(x, dic_values_word))
    df3['Vetores'] = df3.Texto.apply(lambda x: tweet_to_vector(x, dic_values_word))

    # Aplicando a tokenização do texto
    seq_length = 20
    df.Vetores = df.Vetores.apply(lambda x: util.pad_vec_tweet(x))
    df2.Vetores = df2.Vetores.apply(lambda x: util.pad_vec_tweet(x))
    df3.Vetores = df3.Vetores.apply(lambda x: util.pad_vec_tweet(x))

    return dic_values_word, df, df2, df3
