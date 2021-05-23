# Using this to try and implement Recursion but running into problems

import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, avail_cols, connected_four, check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from typing import Optional, Callable, Tuple


def minimax2(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[int, Optional[SavedState]]:

    action = recursive(board_i, player, depth=2)
    return action, saved_state


def recursive(board: np.ndarray, player: BoardPiece, depth: int):
    columns = [0, 1, 2, 3, 4, 5, 6]
    max = 0
    min = 0

    if GameState.IS_DRAW or GameState.IS_WIN:
        return
    else:
        if player == PLAYER1:
            other_player = PLAYER2
        else:
            other_player = PLAYER1

        for i in avail_cols(board):
            danger_col = []

            board_i = apply_player_action(board, i, player, True)
            if connected_four(board_i, player) == True:
                danger_col.append(i)
                return i
            else:
                recursive(board_i, other_player, depth-1)


        if len(danger_col) != 0:
            columns.remove(i)  # don't use columns i in random.choice if they will lead to an other_player win
        cols = np.array(columns)
        action = np.random.choice(cols)
