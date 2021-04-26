import src.utils.imports_util as imports
import src.utils.common_util as util
import pandas as pd
import numpy as np
import csv
import nltk
import nltk.stem as stem
import nltk.corpus as corpus


# PROCESSAMENTO DO DATASET

def remove_duplicates_datas():
    # importando o csv
    df1 = pd.read_csv(imports.FIRST_DATASET_TWEETS_PATH)
    df2 = pd.read_csv(imports.SECOND_DATASET_TWEETS_PATH)

    print("Tamanho do dataset sem remoção dos duplicados = ", len(df1))
    print("Tamanho do dataset sem remoção dos pré processados = ", len(df2))

    df = pd.concat([df1,df2])
    print("Tamanho dos datasets unidos = ", len(df))

    # ordenando o dataframe
    df.sort_values("Texto", inplace = True)
    # removendo os duplicados baseado na coluna Texto
    df.drop_duplicates(subset="Texto", keep=False, inplace=True)
    print("Tamanho do dataset depois da remoção dos duplicados = ", len(df))

    print("---------------")
    print("iniciando a criação dos novos dataset")
    df_pos_1 = df.loc[df['Polaridade'] > 0]
    df_neg_1 = df.loc[df['Polaridade'] < 0]
    print("1 - Com corte '0': pos = ", len(df_pos_1), " neg = ", len(df_neg_1))

    df_pos_2 = df.loc[df['Polaridade'] > 0.1]
    df_neg_2 = df.loc[df['Polaridade'] < -0.1]
    print("2 - Com corte '0.1': pos = ", len(df_pos_2), " neg = ", len(df_neg_2))

    df_pos_3 = df.loc[df['Polaridade'] > 0.2]
    df_neg_3 = df.loc[df['Polaridade'] < -0.2]
    print("3 - Com corte '0.2': pos = ", len(df_pos_3), " neg = ", len(df_neg_3))

    df_pos_4 = df.loc[df['Polaridade'] > 0.3]
    df_neg_4 = df.loc[df['Polaridade'] < -0.3]
    print("4 - Com corte '0.3': pos = ", len(df_pos_4), " neg = ", len(df_neg_4))

    df_pos_5 = df.loc[df['Polaridade'] > 0.5]
    df_neg_5 = df.loc[df['Polaridade'] < -0.5]
    print("5 - Com corte '0.5': pos = ", len(df_pos_5), " neg = ", len(df_neg_5))

    print("Finalizando a criação dos novos dataset")
    print("---------------")

    print("Salvando os arquivos")
    df.to_csv(imports.DATAS_PATH + "sem_duplicacoes.csv", index=False)

    df_pos_1.to_csv(imports.DATAS_PATH + '/dataset_pos_0.csv', index=False)
    df_neg_1.to_csv(imports.DATAS_PATH + '/dataset_neg_0.csv', index=False)

    df_pos_2.to_csv(imports.DATAS_PATH + '/dataset_pos_1.csv', index=False)
    df_neg_2.to_csv(imports.DATAS_PATH + '/dataset_neg_1.csv', index=False)

    df_pos_3.to_csv(imports.DATAS_PATH + '/dataset_pos_2.csv', index=False)
    df_neg_3.to_csv(imports.DATAS_PATH + '/dataset_neg_2.csv', index=False)

    df_pos_4.to_csv(imports.DATAS_PATH + '/dataset_pos_3.csv', index=False)
    df_neg_4.to_csv(imports.DATAS_PATH + '/dataset_neg_3.csv', index=False)

    df_pos_5.to_csv(imports.DATAS_PATH + '/dataset_pos_5.csv', index=False)
    df_neg_5.to_csv(imports.DATAS_PATH + '/dataset_neg_5.csv', index=False)
    print("Arquivos salvos")
    print("---------------")

    return 0


def clean_crude_dataset():
    # carregando o csv no dataframe
    df_ = pd.read_csv(imports.CRUDE_TWEETS_PATH)
    # df_ = pd.read_csv(imports.ALL_TWEETS_PATH)
    print("Tamanho do dataset carregado: ", len(df_))
    #pegando apenas as colunas de texto e polaridade
    df_ = df_[['Texto', 'Polaridade']]
    # pegando apenas os tweets q nao sao retweets
    df_unic = df_[~df_['Texto'].str.contains('RT ')]
    print("Tamanho do dataset sem RT: ", len(df_unic))
    # limpando o dataset de #, @, www
    df_unic.Texto = df_unic.Texto.apply(lambda x: util.pre_process_tweet(x))
    print("Tamanho do dataset limpo: ", len(df_unic))
    df_unic.to_csv(imports.DATAS_PATH + "/novo_dataset/saida_limpa.csv", index=False)
    print("Arquivos salvos")
    print("---------------")

    return 0


def remove_stopwords_of_datasets():
    # carregar dataset
    df_pos_0 = pd.read_csv(imports.DATAS_PATH + '/dataset_pos_0.csv')
    df_neg_0 = pd.read_csv(imports.DATAS_PATH + '/dataset_neg_0.csv')

    df_pos_1 = pd.read_csv(imports.DATAS_PATH + '/dataset_pos_1.csv')
    df_neg_1 = pd.read_csv(imports.DATAS_PATH + '/dataset_neg_1.csv')

    df_pos_2 = pd.read_csv(imports.DATAS_PATH + '/dataset_pos_2.csv')
    df_neg_2 = pd.read_csv(imports.DATAS_PATH + '/dataset_neg_2.csv')

    #loading depencences
    util.loading_dependences()
    stopwords = set(corpus.stopwords.words('english'))
    # retirar os stopwords
    df_pos_0.Texto = df_pos_0.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    df_neg_0.Texto = df_neg_0.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    df_pos_1.Texto = df_pos_1.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    df_neg_1.Texto = df_neg_1.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    df_pos_2.Texto = df_pos_2.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    df_neg_2.Texto = df_neg_2.Texto.apply(lambda x: __tweet_without_stopword(x, stopwords))
    # remover os nan
    df_pos_0 = df_pos_0.dropna()
    df_neg_0 = df_neg_0.dropna()
    df_pos_1 = df_pos_1.dropna()
    df_neg_1 = df_neg_1.dropna()
    df_pos_2 = df_pos_2.dropna()
    df_neg_2 = df_neg_2.dropna()
    # printar os tamanhos
    print("Tamanho do dataset sw pos com corte 0 = ", len(df_pos_0))
    print("Tamanho do dataset sw neg com corte 0 = ", len(df_neg_0))
    print("Tamanho do dataset sw pos com corte 0.1 = ", len(df_pos_1))
    print("Tamanho do dataset sw neg com corte 0.1 = ", len(df_neg_1))
    print("Tamanho do dataset sw pos com corte 0.2 = ", len(df_pos_2))
    print("Tamanho do dataset sw neg com corte 0.2 = ", len(df_neg_2))
    # salvar os arquivos
    df_pos_0.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_0.csv', index=False)
    df_neg_0.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_0.csv', index=False)
    df_pos_1.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_1.csv', index=False)
    df_neg_1.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_1.csv', index=False)
    df_pos_2.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_2.csv', index=False)
    df_neg_2.to_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_2.csv', index=False)

    return 0


def lemmatization_of_datasets():
    # carregar dataset
    print("carregando dataset")
    df_pos_0 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_0.csv')
    df_neg_0 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_0.csv')
    df_pos_1 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_1.csv')
    df_neg_1 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_1.csv')
    df_pos_2 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_pos_2.csv')
    df_neg_2 = pd.read_csv(imports.DATAS_PATH + '/sw' + '/dataset_neg_2.csv')
    #loading depencences
    util.loading_dependences()
    lemmatizer = stem.WordNetLemmatizer()  # Lemmatization
    #lematizar o dataset
    print("iniciando a lematização")
    df_pos_0.Texto = df_pos_0.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    df_neg_0.Texto = df_neg_0.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    df_pos_1.Texto = df_pos_1.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    df_neg_1.Texto = df_neg_1.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    df_pos_2.Texto = df_pos_2.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    df_neg_2.Texto = df_neg_2.Texto.apply(lambda x: __tweet_lemmatization(x, lemmatizer))
    # printar os tamanhos
    print("Tamanho do dataset sw_lm pos com corte 0 = ", len(df_pos_0))
    print("Tamanho do dataset sw_lm neg com corte 0 = ", len(df_neg_0))
    print("Tamanho do dataset sw_lm pos com corte 0.1 = ", len(df_pos_1))
    print("Tamanho do dataset sw_lm neg com corte 0.1 = ", len(df_neg_1))
    print("Tamanho do dataset sw_lm pos com corte 0.2 = ", len(df_pos_2))
    print("Tamanho do dataset sw_lm neg com corte 0.2 = ", len(df_neg_2))
    # salvar os arquivos
    print("salvando arquivos")
    df_pos_0.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_pos_0.csv', index=False)
    df_neg_0.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_neg_0.csv', index=False)
    df_pos_1.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_pos_1.csv', index=False)
    df_neg_1.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_neg_1.csv', index=False)
    df_pos_2.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_pos_2.csv', index=False)
    df_neg_2.to_csv(imports.DATAS_PATH + '/sw_lm' + '/dataset_neg_2.csv', index=False)
    print("acabou")
    return 0


def __tweet_without_stopword(tweet, stopwords):
    tt = util.texto_para_lista_palavras(tweet)
    tt_wtt_sw = [word for word in tt if word not in stopwords]
    if len(tt_wtt_sw) > 0:
        return tt_wtt_sw
    else:
        return np.nan


def __tweet_lemmatization(tweet, lemmatizer):
    tt = util.texto_para_lista_palavras(tweet)
    tt_lm = [lemmatizer.lemmatize(w, __get_wordnet_pos(w)) for w in tt]
    if len(tt_lm) == 0:
        print("vazio")
    return tt_lm


def __get_wordnet_pos(word):
    wordnet = corpus.wordnet

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


def save_dict(dic, filename):
    with open(filename, 'w') as outfile:
        file = csv.writer(outfile)
        for key, val in dic.items():
            file.writerow([key, val])


def save_infos_in_csv(list):
    with open(imports.RESULT_PATH, 'a') as outfile:
        file = csv.writer(outfile)
        file.writerow(list)
