from enum import Enum

import numpy as np

from typing import Optional

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

from typing import Callable, Tuple


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    board[:] = NO_PLAYER
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    print_board = '|=======|'
    for i in range(board.shape[0] - 1, -1, -1):
        print_board += '\n|'
        for j in range(board.shape[1]):
            if board[i, j] == BoardPiece(1):
                print_board += PLAYER1_PRINT
            elif board[i, j] == BoardPiece(2):
                print_board += PLAYER2_PRINT
            else:
                print_board += NO_PLAYER_PRINT
        print_board += '|'
    print_board += '\n|=======|\n|0123456|'
    return print_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    count = 0

    while pp_board[count] == '\n' or pp_board[count] == '|' or pp_board[count] == '=':
        count += 1
    for row in reversed(range(6)):
        for col in range(7):
            if pp_board[count] == NO_PLAYER_PRINT:
                board[row, col] = NO_PLAYER
                count += 1
            elif pp_board[count] == PLAYER1_PRINT:
                board[row, col] = PLAYER1
                count += 1
            elif pp_board[count] == PLAYER2_PRINT:
                board[row, col] = PLAYER2
                count += 1
            else:
                count += 1  # for remaining characters
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    i = 0
    while i < board.shape[0] and board[i, action] != NO_PLAYER:
        i += 1

    if copy == True:
        board_copy = board.copy()
        board_copy[i, action] = player
        return board_copy
    else:
        board[i, action] = player
        return board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    # check horizontal win
    for i in range(6):
        for j in range(4):
            if board[i, j] == board[i, j + 1] == board[i, j + 2] == board[i, j + 3] == player:
                return True

    # check vertical win
    for i in range(3):
        for j in range(7):
            if board[i, j] == board[i + 1, j] == board[i + 2, j] == board[i + 3, j] == player:
                return True

    # check / diagonal
    for i in reversed(range(3)):
        for j in range(4):
            if board[i, j] == board[i + 1, j + 1] == board[i + 2, j + 2] == board[i + 3, j + 3] == player:
                return True

    # check \ diagonal
    for i in range(3, 6):
        for j in range(4):
            if board[i, j] == board[i - 1, j + 1] == board[i - 2, j + 2] == board[i - 3, j + 3] == player:
                return True

    return False


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
        # how to end the game
    elif (board == NO_PLAYER).any():
        return GameState.STILL_PLAYING
    else:
        return GameState.IS_DRAW


def available_columns(board: np.ndarray) -> list:
    """
    Takes a board as input and returns a list of all the available columns where a legitimate move can be made

    Keyword arguments:
        board: the board that the player is playing and trying to win

    Returns:
        list: a list of the available column indices
    """
    columns = []
    for col in range(board.shape[1]):
        if board[-1, col].any() == NO_PLAYER:
            columns.append(col)
    return columns


def opponent(player: BoardPiece) -> BoardPiece:
    """
    Given a player, return their opponent
    :param player: BoardPiece
    :return: BoardPiece
    """
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2
    return other_player
