import numpy as np

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
EMPTY = 0

def is_valid_move(state: np.array, col: int):
    return state[col] == EMPTY

def count_empties(state):
    empties = 0
    for value in state:
        if value == EMPTY:
            empties += 1
    return empties