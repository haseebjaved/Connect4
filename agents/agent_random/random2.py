import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, NO_PLAYER, GenMove
from typing import Optional, Callable, Tuple

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[PlayerAction, Optional[SavedState]]:
    '''
    Choose a valid, non-full column randomly and return it as the random move to be made be the player on the
    given board
    '''
    #
    action = np.random.randint(0, high=7)
    # find available columns first, then while loop on that
    while board[5, action] != NO_PLAYER:  # until an available column is found, generate a new random column
            action = np.random.randint(0, high=7)

    return action, saved_state