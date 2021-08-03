import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

import src.utils.import_util as imports
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Activation, Dense, GRU
np.random.seed(1)


def training(df, df2, matrix_embedding, seq_length, hyperparams):
    gpus = 1
    batch_size = hyperparams.batch_size * gpus
    model = Sequential()
    model.add(Embedding(len(matrix_embedding), hyperparams.n_embedding, weights=[matrix_embedding], input_length=seq_length))
    model.add(Dropout(hyperparams.drop_1))

    if hyperparams.initializer == 'xavier':
        initializer = tf.keras.initializers.glorot_normal(seed=1)
    else:
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal',
                                                        seed=1)
    model.add(LSTM(hyperparams.n_hidden,
                   kernel_initializer=initializer,
                   activation='softsign', recurrent_activation='sigmoid',
                   dropout=0.0, recurrent_dropout=hyperparams.drop_recurrent,
                   implementation=1
                   ))

    model.add(Activation(hyperparams.activation))
    model.add(Dense(1, activation='sigmoid'))

    if gpus >= 2:
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    # outros otimizadores para teste estão comentados
    model.compile(loss=hyperparams.loss,
                  optimizer=__get_optimizer(hyperparams),
                  metrics=['accuracy'])

    plot_model(model, to_file=imports.DATAS_PATH + '/resultados/model_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    training_start_time = time()
    net = model.fit(df.Vetores.tolist(), df.Polaridade.tolist(), batch_size=batch_size, epochs=hyperparams.n_epochs,
                    validation_data=(df2.Vetores.tolist(), df2.Polaridade.tolist()))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (hyperparams.n_epochs,
                                                            training_end_time - training_start_time))
    model.save('LSTM.h5')

    # Plot accuracy
    # plt.subplot(211)
    # plt.plot(net.history['accuracy'])
    # plt.plot(net.history['val_accuracy'])
    # plt.title('Accuracy' + ' LSTM: opt=' + str(hyperparams.optimizer) + ' lr=' + str(hyperparams.lr))
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    #
    # # Plot loss
    # plt.subplot(212)
    # plt.plot(net.history['loss'])
    # plt.plot(net.history['val_loss'])
    # plt.title('Loss' + ' LSTM: loss=' + str(hyperparams.loss))
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper right')
    #
    # plt.tight_layout(h_pad=1.0)
    # date = hyperparams.date_now.replace('/', '').replace(':', '').replace(' ', '_')
    # plt.savefig(imports.GRAPHIC_TRAIN + date + '.png')
    # plt.show()

    return net, batch_size


def __get_optimizer(hyperparams):
    if hyperparams.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=hyperparams.lr, beta_1=0.9, beta_2=0.999,
                                                    epsilon=1e-08, decay=0.0, amsgrad=False, clipnorm=1.5)
    elif hyperparams.optimizer == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=hyperparams.lr,  initial_accumulator_value=0.1,
                                            epsilon=1e-07, name='Adagrad', clipnorm=0.1)
    elif hyperparams.optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=hyperparams.lr, momentum=0.0, nesterov=False, name='SGD')

    elif hyperparams.optimizer == 'rsm':
        return tf.keras.optimizers.RMSprop(lr=hyperparams.lr, clipnorm=1)

    elif hyperparams.optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=hyperparams.lr, rho=hyperparams.rho,
                                            epsilon=1e-07, name='Adadelta', clipnorm=1.5)

