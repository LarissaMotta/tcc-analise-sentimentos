import os

SRC_PATH = os.path.dirname(os.path.abspath("src"))
DATAS_PATH = SRC_PATH + "/datas"

# ARQUIVO DE RESULTADO
RESULT_PATH = DATAS_PATH + "/resultados/results.csv"

## COM STOPWORD

MATRIX_EMBEDDING = DATAS_PATH + '/process_data_25/matrix_embedding_25.npy'
POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_25/process_data_pos_25.csv"
NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_25/process_data_neg_25.csv"
GRAPHIC_TRAIN = DATAS_PATH + '/figures/figure_'


# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_1/matrix_embedding_1.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/process_data_pos_1.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/process_data_neg_1.csv"


# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_0/matrix_embedding_0.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/process_data_pos_0.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/process_data_neg_0.csv"

## COM STOPWORD E LEMATIZAÇÃO
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_sw_lm_0/matrix_embedding_0.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_0/process_data_pos_0.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_0/process_data_neg_0.csv"

## DADOS COM O W2V ANTIGO

# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_1/w2v_old/matrix_embedding_1_300.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/w2v_old/process_data_pos_1_300.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/w2v_old/process_data_neg_1_300.csv"


# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_0/w2v_old/matrix_embedding_0_300.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/w2v_old/process_data_pos_0_300.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/w2v_old/process_data_neg_0_300.csv"



PREDICT_DATASET = DATAS_PATH + "/predict_dataset.csv"