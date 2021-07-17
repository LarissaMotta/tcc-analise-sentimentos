import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import backend as k


def comb_h_sine(x):
    return k.sum(tf.math.sinh(x), tf.math.asinh(x))


def get_activation_function():
    get_custom_objects().update({'comb-H-sine': Activation(comb_h_sine)})
