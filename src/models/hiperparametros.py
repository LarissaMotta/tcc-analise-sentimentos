from datetime import datetime

class hiperparams:
    def __init__(self):
        self.n_hidden = 516
        self.lr = 0.1
        self.rho = 0.95
        self.drop_1 = 0.14
        self.drop_recurrent = 0.47
        self.n_epochs = 40
        self.batch_size = 16
        self.len_train = 0.70
        self.len_valid = 0.10
        self.n_embedding = 400

        self.initializer = 'he'
        # self.initializer = 'xavier'

        self.loss ='binary_crossentropy'
        # self.loss ='mse'

        # self.optimizer = 'adam'
        self.optimizer = 'adagrad'
        # self.optimizer = 'adadelta'
        # self.optimizer = 'sgd'
        # self.optimizer = 'rsm'

        self.activation = 'elu'

        self.date_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.acc_train = ''
        self.acc_valid = ''
        self.loss_train = ''
        self.loss_valid = ''
