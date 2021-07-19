import os

SRC_PATH = os.path.dirname(os.path.abspath("src"))
DATAS_PATH = SRC_PATH + "/datas"

# ARQUIVO DE RESULTADO
RESULT_PATH = DATAS_PATH + "/resultados/results.csv"
GRAPHIC_TRAIN = DATAS_PATH + '/figures/figure_'

## COM STOPWORD
# SEQ_LENGTH = 12
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_2/matrix_embedding_2_12.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_2/process_data_pos_2_12.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_2/process_data_neg_2_12.csv"
# #
# SEQ_LENGTH = 13
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_1/matrix_embedding_1_13.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/process_data_pos_1_13.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_1/process_data_neg_1_13.csv"

# SEQ_LENGTH = 14
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_0/matrix_embedding_0_14.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/process_data_pos_0_14.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_0/process_data_neg_0_14.csv"


## COM STOPWORD E LEMATIZAÇÃO
# SEQ_LENGTH = 14
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_sw_lm_0/matrix_embedding_0_14.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_0/process_data_pos_0_14.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_0/process_data_neg_0_14.csv"

# SEQ_LENGTH = 13
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_sw_lm_1/matrix_embedding_1_13.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_1/process_data_pos_1_13.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_1/process_data_neg_1_13.csv"

SEQ_LENGTH = 12
MATRIX_EMBEDDING = DATAS_PATH + '/process_data_sw_lm_2/matrix_embedding_2_12.npy'
POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_2/process_data_pos_2_12.csv"
NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_2/process_data_neg_2_12.csv"



PREDICT_DATASET = DATAS_PATH + "/predict_dataset.csv"