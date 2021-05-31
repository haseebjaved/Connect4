import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, available_columns, opponent, connected_four, check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from typing import Optional, Callable, Tuple

def mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[int, Optional[SavedState]]:
    """
    Build a Monte Carlo Tree Search algorithm that takes a board at any state (beginning or middle of the game
    and determines the most valuable move for a player.
    The MCTS algorithm at any board state builds out the search tree iteratively and determines the most valuable move
    which will give the most wins.
    Start at the root node which is the given state of the board, and build out different branches corresponding to
    available moves (columns).
    Within each branch, make random moves until you reach the end, i.e. a leaf node i.e. no
    more moves or indices available to make a legitimate move.
    Then determine what the board state is at the end, i.e. a Win, Loss, or a Draw. Score each end state and associate
    this value to the branch, store it, and also provide it to the root.
    The root can do the same for each available branch (column).
    After one iteration of each branch, the root has an idea of the value of each branch (for Exploitation), but based
    on random moves within each branch. We would get different outcomes if the random values were different.
    Therefore run more iterations on each branch (Exploration) to find new outcomes and update each branch score
    for instance after one round and one win (1/1), and after three rounds and two wins (2/3), etc. and store in
    root node as well.
    Determine how many iterations are needed before the statistics for each branch are more reliable than just one
    iteration.
    Then choose the highest value branch and provide the column number (action) to the player.

    Keyword arguments:
        board: the board that the player is playing and trying to win
        player: current player
        saved_state: Optional Saved State

    Returns:
        Tuple: consisting of the location of the column of the best move, and the Optional Saved State
    """