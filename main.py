from ExperienceReplay import Experience, Memory
import tensorflow as tf

import Networks
from board import BOARD_HEIGHT, BOARD_WIDTH

TARGET_NET: tf.keras.Sequential = Networks.create_conv2d_model()
PRIMARY_NET: tf.keras.Sequential = Networks.create_conv2d_model()

def main():
    
    
    
    pass


if __name__ == '__main__':
    main()