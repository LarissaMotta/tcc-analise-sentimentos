import os

SRC_PATH = os.path.dirname(os.path.abspath("src"))
DATAS_PATH = SRC_PATH + "/datas"
W2V_PATH = DATAS_PATH + "/w2v/GoogleNews-vectors-negative300.bin.gz"

RESULT_PATH = DATAS_PATH + "/results.csv"
W2V_DIC_PATH = DATAS_PATH + "/dic_w2v.csv"

# Para limpeza do dataset
ALL_TWEETS_PATH = DATAS_PATH + "/Final.csv"
CRUDE_TWEETS_PATH = DATAS_PATH + "/novo_dataset/Saida.csv"
# para segunda limpeza dos dataset e geracao dos novos
FIRST_DATASET_TWEETS_PATH = DATAS_PATH + "/novo_dataset/final_saida_limpa.csv"
SECOND_DATASET_TWEETS_PATH = DATAS_PATH + "/novo_dataset/saida_limpa.csv"

# # primeiro teste (so serviu para o minimo)
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/TweetsSw_p.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/TweetsSw_n.csv"
# testes do datasets com remocao de stopwords
# corte 0
POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_pos_0.csv"
NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_neg_0.csv"
POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_pos_0.jpg"
NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_neg_0.jpg"
HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_0.jpg"
CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw/confusion_matrix_0.jpg"
GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw/loss_0.jpg"
GRAPHIC_ACC_PATH = DATAS_PATH + "/sw/acc_0.jpg"
METRICS_PATH = DATAS_PATH + "/sw/metricas_0.csv"
# #corte 0.1
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_pos_1.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_neg_1.csv"
# POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_pos_1.jpg"
# NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_neg_1.jpg"
# HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_1.jpg"
# CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw/confusion_matrix_1.jpg"
# GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw/loss_1.jpg"
# GRAPHIC_ACC_PATH = DATAS_PATH + "/sw/acc_1.jpg"
# METRICS_PATH = DATAS_PATH + "/sw/metricas_1.csv"
# # corte 0.2
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_pos_2.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw/dataset_neg_2.csv"
# POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_pos_2.jpg"
# NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_neg_2.jpg"
# HISTOGRAMA_PATH = DATAS_PATH + "/sw/histograma_2.jpg"
# CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw/confusion_matrix_2.jpg"
# GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw/loss_2.jpg"
# GRAPHIC_ACC_PATH = DATAS_PATH + "/sw/acc_2.jpg"
# METRICS_PATH = DATAS_PATH + "/sw/metricas_2.csv"


# # testes do datasets com remocao de stopwords e lematizacao
# # corte 0
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_pos_0.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_neg_0.csv"
# POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_pos_0.jpg"
# NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_neg_0.jpg"
# HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_0.jpg"
# CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw_lm/confusion_matrix_0.jpg"
# GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw_lm/loss_0.jpg"
# GRAPHIC_ACC_PATH = DATAS_PATH + "/sw_lm/acc_0.jpg"
# METRICS_PATH = DATAS_PATH + "/sw_lm/metricas_0.csv"
# # corte 0.1
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_pos_1.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_neg_1.csv"
# POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_pos_1.jpg"
# NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_neg_1.jpg"
# HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_1.jpg"
# CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw_lm/confusion_matrix_1.jpg"
# GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw_lm/loss_1.jpg"
# GRAPHIC_ACC_PATH = DATAS_PATH + "/sw_lm/acc_1.jpg"
# METRICS_PATH = DATAS_PATH + "/sw_lm/metricas_1.csv"
# # corte 0.2
# POSITIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_pos_2.csv"
# NEGATIVE_TWEETS_PATH = DATAS_PATH + "/sw_lm/dataset_neg_2.csv"
# POSITIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_pos_2.jpg"
# NEGATIVE_HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_neg_2.jpg"
# HISTOGRAMA_PATH = DATAS_PATH + "/sw_lm/histograma_2.jpg"
# CONFUSION_MATRIX_PATH = DATAS_PATH + "/sw_lm/confusion_matrix_2.jpg"
# GRAPHIC_LOSS_PATH = DATAS_PATH + "/sw_lm/loss_2.jpg"
# GRAPHIC_ACC_PATH = DATAS_PATH + "/sw_lm/acc_2.jpg"
# METRICS_PATH = DATAS_PATH + "/sw_lm/metricas_2.csv"
