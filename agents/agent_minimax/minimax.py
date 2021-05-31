import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, available_columns, opponent, connected_four, check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from typing import Optional, Callable, Tuple
"""
part A: basics
check if any of the legal moves (open columns) will lead to a win for me or for my opponent
if not, make a random move and check in the next iteration if any of the moves results in a win for one of us
run a for loop for 4 iterations to check 4 moves into the future
extra: if working, have a look at recursion

part B: heuristics
assign a value to each move based on status of board: perhaps how many of my pieces are together, or pieces in the
middle of the board

part C :
use alpha beta pruning to optimize. Stop iterating if loss inevitable on certain moves? How?'''
"""
def minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[int, Optional[SavedState]]:
    """
    This function returns the best position for the agent to play by returning the appropriate column index. The
    agent checks to see if there is a win in any of the available columns, and if so, makes that move. If not, it iterates
    through every column, makes a move there, and checks whether the opposing player can win subsequently. If the opposing
    player can win given the current player's first move, the current player chooses not to make that move in the first
    place.

    Keyword arguments:
        board: the board that the player is playing and trying to win
        player: current player
        saved_state: Optional Saved State

    Returns:
        Tuple: consisting of the location of the column of the best move, and the Optional Saved State
    """

    danger_col = []
    columns = [0, 1, 2, 3, 4, 5, 6]
    score = 0

    other_player = opponent(player)

    for i in available_columns(board):
        board_i = apply_player_action(board, i, player, True)
        if connected_four(board_i, player) == True:
            return i, saved_state
        else:
            danger_col = []
            for j in available_columns(board_i):
                board_i_j = apply_player_action(board_i, j, other_player, True)
                if connected_four(board_i_j, other_player) == True:
                    danger_col.append(j)  # these columns will lead to the opposing player's win

        if len(danger_col) != 0:
            columns.remove(i)   # don't use columns i in random.choice if they will lead to an other_player win
    cols = np.array(columns)
    action = np.random.choice(cols)  # randomly choose a column that will avoid a loss in the
                                     # opposing player's next move

    return action, saved_state