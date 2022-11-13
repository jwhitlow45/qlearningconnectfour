import tensorflow as tf
import keras.layers as tfl

from board import BOARD_WIDTH, BOARD_HEIGHT

def create_conv2d_model():
    
    model = tf.keras.Sequential()
    model.add(tfl.Input((BOARD_HEIGHT, BOARD_WIDTH, 1)))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same', activation=None))
    model.add(tfl.ReLU())
    model.add(tfl.LayerNormalization())
    model.add(tfl.Conv2D(filters=64, kernel_size=2, padding='same', activation=None))
    model.add(tfl.ReLU())
    model.add(tfl.LayerNormalization())
    model.add(tfl.Conv2D(filters=64, kernel_size=2, padding='same', activation=None))
    model.add(tfl.ReLU())
    model.add(tfl.LayerNormalization())
    model.add(tfl.Flatten())
    model.add(tfl.Dense(64, activation=None))
    model.add(tfl.LeakyReLU())
    model.add(tfl.LayerNormalization())
    model.add(tfl.Dense(BOARD_WIDTH, activation='linear'))
    
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    return model