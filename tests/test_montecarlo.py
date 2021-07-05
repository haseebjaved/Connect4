from agents.agent_MCTS.montecarlo import Node
from agents.common import *
from agents.agent_MCTS import montecarlo

test_board = np.array(
    [[BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

test_board_modified = test_board.copy()

testNode = Node(test_board_modified, PLAYER1)
testNode.simulations = 10
testNode.wins = 5
test_board_modified[3, 0] = NO_PLAYER
testNode.parent = Node(test_board_modified, PLAYER2)
testNode.parent.simulations = 50

def test_value():
    from agents.agent_MCTS import montecarlo


    assert montecarlo.Node.value(testNode) == 1.3845363763495708
'''
def test_create_child():

def test_rollout():

def test_rolloutHelper():

def test_isTerminal():

def test_determineWin():

def test_update_Tree():

def test_expand():

def test_select():
'''