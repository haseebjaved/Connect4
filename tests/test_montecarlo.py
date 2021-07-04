from agents.common import *
from agents.agent_MCTS import *

test_board = np.array(
    [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

test_board_modified = test_board.copy()
#test_board_modified[1, 3] = PLAYER1

testNode = Node(test_board_modified, PLAYER1)
testNode.simulations = 10
testNode.wins = 5
testNode.parent.simulations = 50

def test_value():
    from agents.agent_MCTS import value

    assert value(testNode) == 1.38453637635

