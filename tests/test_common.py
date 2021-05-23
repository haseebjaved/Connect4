import numpy as np
from agents.common import *

def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board():
    from agents.common import pretty_print_board
    """
    Test the given boards and see if they output the correct string representation
    """

    board = np.array([[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
                    [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
                    [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    test_pretty_board = '''|=======|
|XOXOXOX|
|O O X X|
|XOXOXOX|
|O O X X|
|XOXOXOX|
|O O X X|
|=======|
|0123456|'''

    assert pretty_print_board(board) == test_pretty_board

    empty_board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    empty_board[:] = NO_PLAYER

    test_empty_pretty_board = '''|=======|
|       |
|       |
|       |
|       |
|       |
|       |
|=======|
|0123456|'''

    assert pretty_print_board(empty_board) == test_empty_pretty_board

def test_string_to_board():
    '''
    Test if the given string representation of the board maps to the correct Board configuration
    '''
    from agents.common import string_to_board

    test_board = np.array([[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
                    [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
                    [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
                    [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])
    test_string = '''|=======|
|XOXOXOX|
|O O X X|
|XOXOXOX|
|O O X X|
|XOXOXOX|
|O O X X|
|=======|
|0123456|'''

    assert (string_to_board(test_string) == test_board).all()

def test_apply_player_action():
    from agents.common import apply_player_action
    test_board = np.array(
        [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    assert (apply_player_action(test_board, 3, PLAYER1, False) == np.array(
        [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])).all()

    # test if puts a piece in empty column, in a non empty column, and if copy is true

def test_connected_four():
    from agents.common import connected_four

    # 5th column has connected 4 for PLAYER1 - testing vertical win
    test_board = np.array(
        [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    assert connected_four(test_board, PLAYER1) == True
    assert connected_four(test_board, PLAYER2) == False

    # row 0, first 4 units are PLAYER2 - testing horizontal win
    test_board_2 = np.array(
        [[BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    assert connected_four(test_board_2, PLAYER2) == True
    assert connected_four(test_board_2, PLAYER1) == False

    # test \ diagonal - Player 2 wins
    test_board_3 = np.array(
        [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1)]])

    assert connected_four(test_board_3, PLAYER2) == True

    # test / diagonal - Player 2 wins
    test_board_4 = np.array(
        [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(1)]])

    assert connected_four(test_board_4, PLAYER2) == True

def test_check_end_state():
    from agents.common import check_end_state
    # row 0, first 4 units are PLAYER2 - horizontal win
    test_board = np.array(
        [[BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])
    assert check_end_state(test_board, PLAYER2) == GameState.IS_WIN

    # checking for a draw
    test_board_2 = np.array(
        [[BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(2)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1)]])
    assert check_end_state(test_board_2, PLAYER2) == GameState.IS_DRAW

    # checking for an ongoing game
    test_board_3 = np.array(
        [[BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])
    assert check_end_state(test_board_3, PLAYER2) == GameState.STILL_PLAYING
