from agents.agent_MCTS.montecarlo import Node, Tree
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


def test_rollout():
    roll_node = Node(test_board, PLAYER1)
    initial_simulations = roll_node.simulations
    initial_wins = roll_node.wins

    roll_node.rollout(opponent(PLAYER1))

    assert initial_simulations != roll_node.simulations  # after one simulation, the number is incremented and changed
    assert initial_wins == roll_node.wins  # node did not win
    assert roll_node.state == 2  # most likely assumption that it was a draw, is true



def test_update_Tree():
    tree = Tree(test_board_modified, PLAYER1)
    initial_root_sims = tree.root.simulations
    child_node = tree.root.create_child(available_columns(test_board_modified))
    child_node.rollout(opponent(child_node.nodePlayer))

    tree.update_Tree(child_node)

    assert tree.root.simulations == initial_root_sims + 1  # backpropagate simulation statistics up the tree

    if child_node.state == 1:  # if child wins
        assert tree.root.state == 0  # parent lost
    elif child_node.state == 0:  # if child lost
        assert tree.root.state == 1  # parent won
    else:  # if draw for child
        assert tree.root.state == 2  # draw for parent as well


def test_expand():
    tree = Tree(test_board_modified, PLAYER1)
    initial_children_length = len(tree.root.children)
    child_node, child_player = tree.expand(tree.root, tree.root.nodePlayer)  # add one child to the root node

    assert len(tree.root.children) == initial_children_length + 1  # list of children is increased by one


def test_select():
    zero_board = np.zeros_like(test_board)
    tree = Tree(zero_board, PLAYER1)

    for i in range(len(available_columns(tree.root.board))):  # create all children to select from
        child_node, child_player = tree.expand(tree.root, tree.root.nodePlayer)
        child_node.rollout(opponent(child_player))  # rollout on opponent
        tree.update_Tree(child_node)

        if len(available_columns(tree.root.board)) == 7 and i == 1:
            assert tree.select(tree.root) == tree.root  # if all children not created but available, return self
    print(tree.root)

    assert tree.select(tree.root) == tree.root.children[1]  # why is it choosing this child and failing on others?



