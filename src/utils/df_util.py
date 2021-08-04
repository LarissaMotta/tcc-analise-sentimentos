from sklearn.model_selection import train_test_split
import src.utils.import_util as imports
import numpy as np
import pandas as pd
import nltk
import nltk.corpus as corpus


def get_process_data():
    seq_length = imports.SEQ_LENGTH
    matrix_embedding = np.load(imports.MATRIX_EMBEDDING)
    df_p = pd.read_csv(imports.POSITIVE_TWEETS_PATH_PROCESS)
    df_n = pd.read_csv(imports.NEGATIVE_TWEETS_PATH_PROCESS)

    df_p.Vetores = df_p.Vetores.apply(lambda x: __convert_string_to_vector(x))
    df_n.Vetores = df_n.Vetores.apply(lambda x: __convert_string_to_vector(x))

    return matrix_embedding, df_p, df_n, seq_length


def __convert_string_to_vector(txt):
    txt = txt.replace('[', '')
    txt = txt.replace(']', '')
    txt = txt.split(',')
    lst = [int(i) for i in txt]
    return lst


def cut_dataset(df_pos, df_neg, train):

    df = pd.concat([df_pos, df_neg])

    df_train, df_valid_test = train_test_split(df, train_size=train, random_state=0, stratify=df['Polaridade'])
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=0, stratify=df_valid_test['Polaridade'])

    return df_train, df_valid, df_test


def get_adjetives():
    #download nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

    df_p = pd.read_csv(imports.POSITIVE_TWEETS_PATH_PROCESS)
    df_n = pd.read_csv(imports.NEGATIVE_TWEETS_PATH_PROCESS)

    df_p.Texto = df_p.Texto.apply(lambda x: [str(i) for i in x.replace('[', '').replace(']', '').replace('\'', '').split(',')])
    df_n.Texto = df_n.Texto.apply(lambda x: [str(i) for i in x.replace('[', '').replace(']', '').replace('\'', '').split(',')])

    pos_words = create_adj_list(df_p)
    neg_words = create_adj_list(df_n)

    return pos_words, neg_words


# criando uma string unica contendo todos os adjetivos
def create_adj_list(df):
    adjs = ""
    for tweet in df.Texto:
        for word in tweet:
            if __get_wordnet_pos(word) == "a":
                adjs += (" " + word)
    return adjs


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
