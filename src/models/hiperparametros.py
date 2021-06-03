from datetime import datetime

class hiperparams:
    def __init__(self):
        self.n_hidden = 64
        self.lr = 0.01
        self.drop_p = 0.4
        self.n_epochs = 5
        self.batch_size = 256
        self.len_train = 0.75
        self.len_valid = 0.15
        self.n_embedding = 400
        self.date_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.acc_train = ''
        self.acc_valid = ''
        self.loss_train = ''
        self.loss_valid = ''
