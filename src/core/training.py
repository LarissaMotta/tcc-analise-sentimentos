import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Activation, Dense, GRU
from tensorflow.python.keras.regularizers import l2
from time import time


def training(df, df2, matrix_embedding, seq_length, hyperparams):
    gpus = 1
    batch_size = hyperparams.batch_size * gpus
    model = Sequential()
    model.add(Embedding(len(matrix_embedding), hyperparams.n_embedding, weights=[matrix_embedding], input_length=seq_length))
    model.add(Dropout(0.2))
    #LSTM com
    # model.add(LSTM(units=hyperparams.n_hidden,
    #                activation='tanh', recurrent_activation='sigmoid',
    #                use_bias=True, kernel_initializer='glorot_uniform',
    #                recurrent_initializer='orthogonal',
    #                bias_initializer='zeros', unit_forget_bias=True,
    #                kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    #                activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    #                bias_constraint=None, dropout=0.0, recurrent_dropout=hyperparams.drop_p,
    #                return_sequences=False, return_state=False, go_backwards=False, stateful=False,
    #                time_major=False, unroll=False
    #                ))
    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal',
                                                        seed=None)
    model.add(LSTM(hyperparams.n_hidden,
                   kernel_initializer=initializer,
                   activation='tanh', recurrent_activation='sigmoid',
                   dropout=0.0, recurrent_dropout=0.0,
                   implementation=1
                   ))
    model.add(Dropout(hyperparams.drop_p))
    model.add(Activation('elu'))
    model.add(Dense(1, activation='sigmoid'))

    if gpus >= 2:
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    # outros otimizadores para teste estão comentados
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.lr, beta_1=0.9, beta_2=0.999,
                                                     epsilon=1e-08, decay=0.0, amsgrad=False),
                  # optimizer=tf.keras.optimizers.Adagrad(learning_rate=hyperparams.lr,  initial_accumulator_value=0.1,
                  # epsilon=1e-07, name='Adagrad'),
                  # optimizer=tf.keras.optimizers.SGD(
                  #     learning_rate=hyperparams.lr, momentum=0.0, nesterov=False, name='SGD'),
                  metrics=['accuracy'])
    # otimizador e loss antigos para um futuro teste
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(lr=0.01, clipnorm=1), metrics=['accuracy'])
    print(model.summary())
    training_start_time = time()
    net = model.fit(df.Vetores.tolist(), df.Polaridade.tolist(), batch_size=batch_size, epochs=hyperparams.n_epochs,
                    validation_data=(df2.Vetores.tolist(), df2.Polaridade.tolist()))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (hyperparams.n_epochs,
                                                            training_end_time - training_start_time))
    model.save('LSTM.h5')

    # Plot accuracy
    plt.subplot(211)
    plt.plot(net.history['accuracy'])
    plt.plot(net.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(net.history['loss'])
    plt.plot(net.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('history-graph.png')
    plt.show()

    return net, batch_size


