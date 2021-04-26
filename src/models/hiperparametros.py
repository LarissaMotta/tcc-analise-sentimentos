class hiperparams:
    def __init__(self):
        self.n_hidden = 24
        self.n_layer = 3
        self.optimizer = 'RMSProp'
        self.loss = 'MSE'
        self.lr = 0.005
        self.drop_p = 0.2
        self.n_epochs = 20
        self.batch_size = 20
        self.clip = 5

    def set_n_hidden(self, n_hidden):
        self.n_hidden = n_hidden
        return 0

    def set_n_layer(self, n_layer):
        self.n_layer = n_layer
        return 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return 0

    def set_loss(self, loss):
        self.loss = loss
        return 0

    def set_lr(self, lr):
        self.lr = lr
        return 0

    def set_drop_p(self, drop_p):
        self.drop_p = drop_p
        return 0

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs
        return 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return 0

    def set_clip(self, clip):
        self.clip = clip
        return 0

    def print_class(self):
        print("n_hidden: ", self.n_hidden,
              "n_layers: ", self.n_layer,
              "optimizer: ", self.optimizer,
              "loss: ", self.loss,
              "lr: ", self.lr,
              "drop_p: ", self.drop_p,
              "n_epochs: ", self.n_epochs,
              "batch_size: ", self.batch_size,
              "clip: ", self.clip)

    def set_hiperparams(self, entrada):
        while True:
            if entrada == 1:
                valor = input("n_hidden: ")
                self.set_n_hidden(valor)
            elif entrada == 2:
                valor = input("n_layer: ")
                self.set_n_layer(valor)
            elif entrada == 3:
                valor = input("optimizer ('',''): ")
                self.set_optimizer(valor)
            elif entrada == 4:
                valor = input("loss ('',''): ")
                self.set_loss(valor)
            elif entrada == 5:
                valor = input("lr: ")
                self.set_lr(valor)
            elif entrada == 6:
                valor = input("drop_p: ")
                self.set_drop_p(valor)
            elif entrada == 7:
                valor = input("n_epochs: ")
                self.set_n_epochs(valor)
            elif entrada == 8:
                valor = input("batch_size: ")
                self.set_batch_size(valor)
            elif entrada == 9:
                valor = input("clip: ")
                self.set_clip(valor)
            elif entrada == 10:
                self.print_class()
            elif entrada == 0:
                return self
            entrada = input("Alterar algum parâmetro? (1 a 9 são parametros, 10 printar os parametros, 0 finalizar")
