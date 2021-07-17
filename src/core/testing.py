import tensorflow as tf


# predição -> teste
def testing(df3, batch_size):
    model = tf.keras.models.load_model('./LSTM.h5')
    print(model.summary())
    predictions = model.predict(df3.Vetores.tolist(), batch_size=batch_size, verbose=1)

    # prints para entender como funciona o retorno da predicao
    # print(predictions)
    # print(len(predictions))
    # print(len(predictions[0]))

    return predictions, model
