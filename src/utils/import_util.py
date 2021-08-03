import os

SRC_PATH = os.path.dirname(os.path.abspath("src"))
DATAS_PATH = SRC_PATH + "/datas"

# ARQUIVO DE RESULTADO
RESULT_PATH = DATAS_PATH + "/resultados/results.csv"
GRAPHIC_TRAIN = DATAS_PATH + '/figures/figure_'

# Para substituição no for
SEQ_LENGTH_BASE = 14
MATRIX_EMBEDDING_BASE = DATAS_PATH + '/process_data_{0}/matrix_embedding_{0}_{1}.npy'
POSITIVE_TWEETS_PATH_PROCESS_BASE = DATAS_PATH + "/process_data_{0}/process_data_pos_{0}_{1}.csv"
NEGATIVE_TWEETS_PATH_PROCESS_BASE = DATAS_PATH + "/process_data_{0}/process_data_neg_{0}_{1}.csv"

ACC_TRAIN_PATH_BASE = DATAS_PATH + "/results_experiment/acc_train_{0}_{1}.csv"
ACC_VALID_PATH_BASE = DATAS_PATH + "/results_experiment/acc_valid_{0}_{1}.csv"
LOSS_TRAIN_PATH_BASE = DATAS_PATH + "/results_experiment/loss_train_{0}_{1}.csv"
LOSS_VALID_PATH_BASE = DATAS_PATH + "/results_experiment/loss_valid_{0}_{1}.csv"
METRIC_PATH_BASE = DATAS_PATH + "/results_experiment/metrics_{0}_{1}.csv"

SEQ_LENGTH = 14
MATRIX_EMBEDDING = ''
POSITIVE_TWEETS_PATH_PROCESS = ''
NEGATIVE_TWEETS_PATH_PROCESS = ''
ACC_TRAIN_PATH = ''
ACC_VALID_PATH = ''
LOSS_TRAIN_PATH = ''
LOSS_VALID_PATH = ''
RESULT_PATH = ''

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

# SEQ_LENGTH = 12
# MATRIX_EMBEDDING = DATAS_PATH + '/process_data_sw_lm_2/matrix_embedding_2_12.npy'
# POSITIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_2/process_data_pos_2_12.csv"
# NEGATIVE_TWEETS_PATH_PROCESS = DATAS_PATH + "/process_data_sw_lm_2/process_data_neg_2_12.csv"



PREDICT_DATASET = DATAS_PATH + "/predict_dataset.csv"