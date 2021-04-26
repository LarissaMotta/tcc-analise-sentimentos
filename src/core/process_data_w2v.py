import src.utils.common_util as util
import src.utils.imports_util as imports

import numpy as np
import nltk
import nltk.corpus as corpus
from gensim.models import KeyedVectors


# download dependencias de processamento de texto
def loading_dependences():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    return 0


# carregando o word2vec
def loading_w2v():

    embedding_dim = 300
    empty_w2v = False

    # carregando word2vec
    print("Carregando o modelo word2vec (isso pode demorar cerca de 2-3 minutos)...")
    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format(imports.W2V_PATH, binary=True)
    print("word2vec carregado")
    return word2vec


# criando o dicionario da primeira entrada (treinamento)
def create_w2v_dict(dataframe, stopwords, word2vec, vocabs={}, vocabs_cnt=0, vocabs_not_w2v={}, vocabs_not_w2v_cnt=0):
    dataframe['Vetores'] = np.nan
    dataframe = dataframe.astype({'Vetores': 'object'})
    for index, row in dataframe.iterrows():
        # representacao numerica dos tweet
        tweet_vec = []
        for word in row['Texto']:
            # pular as palabras indesejadas
            if word in stopwords:
                continue

            # caso a palavra não esteja no modelo word2vec
            if word not in word2vec.vocab:
                if word not in vocabs_not_w2v:
                    vocabs_not_w2v_cnt += 1
                    vocabs_not_w2v[word] = 1

            # caso a palavra não estiver no vocabulario
            if word not in vocabs:
                vocabs_cnt += 1
                vocabs[word] = vocabs_cnt
                tweet_vec.append(vocabs_cnt)
            else:
                tweet_vec.append(vocabs[word])

        # add a representacao numerica no dataframe
        dataframe.at[index, 'Vetores'] = tweet_vec

    # retornando o dataframe  e a embedding matrix
    return dataframe, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt


# criando a matriz embedding
def create_embedding_matrix(vocabs, word2vec, embedding_dim=300):
    # iniciando a criacao da embedding matrix
    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)

    # primeira posicao ignorada (as palavra comeca com index 1)
    embeddings[0] = 0

    # montando a embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)

    return embeddings


def delete_w2v(word2vec):
    del word2vec
    return


def process_data_w2v(df, df2, df3):

    # fazendo download das dependencias
    loading_dependences()
    stopwords = set(corpus.stopwords.words('english'))

    # Aplicando a tokenização do texto
    df.Texto = df.Texto.apply(lambda x: util.texto_para_lista_palavras(x))
    df2.Texto = df2.Texto.apply(lambda x: util.texto_para_lista_palavras(x))
    df3.Texto = df3.Texto.apply(lambda x: util.texto_para_lista_palavras(x))

    # carregando word2vec
    word2vec = loading_w2v()
    # criando o dicionario e vetorizando os tweets
    df, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt = create_w2v_dict(df, stopwords, word2vec)
    df2, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt = create_w2v_dict(df2, stopwords, word2vec, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt)
    df3, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt = create_w2v_dict(df3, stopwords, word2vec, vocabs, vocabs_cnt, vocabs_not_w2v, vocabs_not_w2v_cnt)
    # criando a matrix embedding
    matrix_embedding = create_embedding_matrix(vocabs, word2vec)
    # deletando o word2vec
    delete_w2v(word2vec)

    # Aplicando a tokenização do texto
    seq_length = 17
    df.Vetores = df.Vetores.apply(lambda x: util.pad_vec_tweet(x, seq_length))
    df2.Vetores = df2.Vetores.apply(lambda x: util.pad_vec_tweet(x, seq_length))
    df3.Vetores = df3.Vetores.apply(lambda x: util.pad_vec_tweet(x, seq_length))

    # with open(imports.DATAS_PATH + "/mt_emb.txt", "w") as output:
    #     output.write(str(matrix_embedding))

    return matrix_embedding, df, df2, df3, seq_length


class EmptyWord2Vec:
    vocab = {}
    word_vec = {}

