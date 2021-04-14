import os

SRC_PATH = os.path.dirname(os.path.abspath("src"))
DATAS_PATH = SRC_PATH + "/datas"
W2V_PATH = DATAS_PATH + "/w2v/GoogleNews-vectors-negative300.bin.gz"
POSITIVE_TWEETS_PATH = DATAS_PATH + "/TweetsSw_p.csv"
NEGATIVE_TWEETS_PATH = DATAS_PATH + "/TweetsSw_n.csv"
