import numpy as np

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
EMPTY = 0

def isValidMove(state: np.array, col: int):
    return state[0][col] == EMPTY