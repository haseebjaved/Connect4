from agents.common import *
from agents.agent_minimax import minimax

def test_minimax():

    # column 2 is nearing a win for Player 2. Player 1 should
    test_board = np.array(
        [[BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(2)],
         [BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(2)],
         [BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
         [BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0), BoardPiece(0)]])

    assert minimax(test_board, PLAYER1) == (1, None)
