from agents.agent_MCTS.montecarlo import Node
from agents.common import *
from agents.agent_MCTS import montecarlo

test_board = np.array(
    [[BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
     [BoardPiece(2), BoardPiece(0), BoardPiece(2), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1)],
     [BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(0), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

test_board_modified = test_board.copy()


def test_value():
    # from agents.agent_MCTS import montecarlo
    testNode = Node(test_board_modified, PLAYER1)
    testNode.simulations = 10
    testNode.wins = 5
    test_board_modified[3, 0] = NO_PLAYER
    testNode.parent = Node(test_board_modified, PLAYER2)
    testNode.parent.simulations = 50
    assert montecarlo.Node.value(testNode) == 1.3845363763495708


def test_create_child():
    avail_cols = [1, 3]  # only 2 available columns
    test_Node = Node(test_board_modified, PLAYER2)

    test_board_modified[1, 2] = PLAYER1
    test_node_child1 = Node(test_board_modified, PLAYER1)  # create a child node with opponent player and column 1 of
    # avail_cols

    test_board_modified2 = test_board.copy()
    test_board_modified2[3, 1] = PLAYER1
    test_node_child2 = Node(test_board_modified2, PLAYER1)  # create a child node with opponent player and column 3 of
    # avail_cols

    # a child will be created in one of the two available columns with random choice
    assert test_Node.create_child(avail_cols) == test_node_child1 or test_node_child2


def test_isTerminal():
    terminal_board = np.array(
        [[BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    terminal_node = Node(terminal_board, PLAYER1)  # win in column 1 for Player 1, i.e. terminal Node
    assert terminal_node.isTerminal(terminal_board, PLAYER1) == connected_four(terminal_board, PLAYER1)

    terminal_board[1, 5] = PLAYER2  # draw game, no more moves left, i.e. terminal node
    terminal_node2 = Node(terminal_board, PLAYER2)
    terminal = None
    if not available_columns(terminal_board):
        terminal = True
    assert terminal_node2.isTerminal(terminal_board, PLAYER2) == terminal

def test_determineWin():
    win_1_board = np.array(
        [[BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])

    win_node = Node(win_1_board, PLAYER1)  # test win for Player 1 in column 1
    assert win_node.determineWin(win_1_board, PLAYER1) is True

    assert win_node.determineWin(win_1_board, PLAYER2) is False  # test loss for Player 2 in Column 1

    # replace Player 1 with Player 2 in Column 1, 5 and force a draw
    draw_board = np.array(
        [[BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(2)],
         [BoardPiece(2), BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(1)],
         [BoardPiece(1), BoardPiece(2), BoardPiece(2), BoardPiece(1), BoardPiece(1), BoardPiece(2), BoardPiece(1)]])
    draw_node = Node(draw_board, PLAYER2)
    assert draw_node.determineWin(draw_board, PLAYER1) is None

'''
def test_rollout():





def test_rolloutHelper():





def test_update_Tree():

def test_expand():

def test_select():
'''
