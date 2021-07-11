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


def test_wins():
    tree = Tree(test_board_modified, PLAYER1)
    new_node = Node(tree.root.board, PLAYER1)
    initial_wins = new_node.wins()
    for i in range(100):
        winner = new_node.rollout(new_node.board, opponent(new_node.nodePlayer))
        tree.update_Tree(new_node, winner)

    assert new_node.wins() != initial_wins  # expecting at least 1 win in 100 rollouts


def test_value():
    testNode = Node(test_board_modified, PLAYER1)
    testNode.simulations = 10
    testNode.rolloutResults[testNode.nodePlayer] = 5  # 5 wins for Player1
    test_board_modified[3, 0] = NO_PLAYER
    testNode.parent = Node(test_board_modified, PLAYER2)
    testNode.parent.simulations = 50
    assert montecarlo.Node.value(testNode) == 1.3845363763495708  # applying the UCB1 formula


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
    tree = Tree(test_board_modified, PLAYER1)
    roll_node = Node(tree.root.board, PLAYER1)
    initial_simulations = roll_node.simulations
    initial_wins = roll_node.wins()  # wins of the node's player

    winner = roll_node.rollout(roll_node.board, opponent(roll_node.nodePlayer))
    tree.update_Tree(roll_node, winner)

    assert initial_simulations != roll_node.simulations  # rollout caused an increase in simulations
    assert initial_wins == roll_node.wins() or roll_node.wins()+1  # either node won or did not win - Random moves


def test_update_Tree():
    tree = Tree(test_board_modified, PLAYER1)
    initial_root_sims = tree.root.simulations
    child_node = tree.root.create_child(available_columns(test_board_modified))
    winner = child_node.rollout(child_node.board, opponent(child_node.nodePlayer))

    tree.update_Tree(child_node, winner)

    assert tree.root.simulations == initial_root_sims + 1  # after one rollout, simulations increase by 1

    if child_node.wins() == 1:  # if child wins
        assert tree.root.wins() == 0  # parent lost
    else:  # if child did not win
        assert tree.root.wins() == 1 or tree.root.wins() == 0  # parent won or game drawn


def test_expand():
    tree = Tree(test_board_modified, PLAYER1)
    initial_children_length = len(tree.root.children)
    child_node, child_player = tree.expand(tree.root, tree.root.nodePlayer)  # add one child to the root node

    assert len(tree.root.children) == initial_children_length + 1  # list of children is increased by one


def test_select():
    zero_board = np.zeros_like(test_board)
    tree = Tree(zero_board, PLAYER1)  # create an empty board, i.e. new game

    for i in range(len(available_columns(tree.root.board))):  # create all children to select from
        child_node, child_player = tree.expand(tree.root, tree.root.nodePlayer)
        winner = child_node.rollout(child_node.board, child_player)  # rollout on opponent which is returned by expand()
        tree.update_Tree(child_node, winner)

        if len(available_columns(tree.root.board)) == 7 and i == 1:
            assert tree.select(tree.root) == tree.root  # if all possible children not created, return self

    scores = []
    for c in tree.root.children:
        scores.append(c.value())
    ind = np.argmax(scores)
    child = tree.root.children[ind]
    assert tree.select(tree.root) == child  # select child with max UCB1 score



