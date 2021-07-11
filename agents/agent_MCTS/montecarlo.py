import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, available_columns, opponent, connected_four, \
    check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from agents.agent_random.random2 import generate_move_random
from typing import Optional, Callable, Tuple


class Node:
    def __init__(self, board: np.ndarray, player: BoardPiece):  # constructor - each node is a state of the board
        self.simulations = 0  # keep track of total simulations
        self.rolloutResults = {PLAYER1: 0, PLAYER2: 0, NO_PLAYER: 0}
        self.parent = None
        self.children = []
        self.board = board
        self.nodePlayer = player  # PLAYER1 or PLAYER2

    def wins(self) -> int:
        """
        :return: number of wins of the Node's player
        """
        return self.rolloutResults[self.nodePlayer]

    def value(self, c=np.sqrt(2)) -> float:  # UCB1 algorithm
        """
        Determine the UCB1 value of a given Node (Board state)
        :param c: constant used in the UCB1 algorith
        :return: float value of the UCB1 algorithm outcome
        """
        if self.simulations == 0:
            return np.inf
        else:
            return self.wins() / self.simulations + (c * np.sqrt(np.log(self.parent.simulations) / self.simulations))

    def create_child(self, avail_cols: list):
        """
        Create a new child node during Expansion at a random column within the available moves (columns)
        :param avail_cols:
        :return: the new child node
        """
        col = np.random.choice(avail_cols)
        board_copy = apply_player_action(self.board, col, self.nodePlayer, True)
        new_node = Node(board_copy, opponent(self.nodePlayer))
        new_node.parent = self
        self.children.append(new_node)
        return new_node

    def rollout(self, board: np.ndarray, player: BoardPiece) -> BoardPiece:
        """
        Recursive call with opponent, determineWin() check for and return win
        :param board:
        :param player:
        :return: player if won, otherwise continue the rollout with recursive call
        """
        if self.isTerminal(board, player):
            if self.determineWin(board, player):  # if win
                return player
            elif self.determineWin(board, opponent(player)):  # if loss
                return opponent(player)
            else:  # if draw
                return BoardPiece(0)  # None
        else:  # generate a random move, create a new board state, apply random move
            random_move, saved_state = generate_move_random(board, player)
            random_board = apply_player_action(board, random_move, player, True)
            return self.rollout(random_board, opponent(player))  # recursive call, passing in new board & opponent

    def isTerminal(self, board: np.ndarray, player: BoardPiece) -> bool:
        """
        Take in a board, and a player, and make a determination if the board is terminal or not.
        Check if no available columns (moves) or if either player has won.
        :param self:
        :param board:
        :param player:
        :return: True or False
        """
        if available_columns(board) == [] or self.determineWin(board, player) or self.determineWin(board,
                                                                                                   opponent(player)):
            return True
        else:
            return False

    def determineWin(self, board: np.ndarray, player: BoardPiece) -> bool:
        """
        # Case 1: player == 1, and wins -> return true
        # Case 2: player == 2, and wins -> return false
        """
        if check_end_state(board, player) == GameState.IS_WIN:
            return True
        elif check_end_state(board, opponent(player)) == GameState.IS_WIN:
            return False
        else:  # draw
            return None


class Tree:
    def __init__(self, board: np.ndarray, player: BoardPiece):  # constructor
        self.root = Node(board, player)

    def update_Tree(self, node: Node, winner: BoardPiece):
        """
        Backpropagate the results of a rollout up the tree to the root node
        :param winner: BoardPiece
        :param node: Node
        """
        while node is not None:
            node.rolloutResults[winner] += 1
            node.simulations += 1
            node = node.parent

    def expand(self, node: Node, player: BoardPiece) -> Tuple[Node, BoardPiece]:  # insert node
        """
        Create one child node of the given node. If no more children left to create, return self.
        :param: parent node
        :return: True
        """
        avail_cols = available_columns(node.board)
        if avail_cols is None:  # no available moves
            print('board is full')
            return node, player  # return self
        else:
            new_node = Node.create_child(node, avail_cols)
            return new_node, opponent(player)  # child node has board with opponent's turn to play

    def select(self, node: Node, saved_state: Optional[SavedState] = None) \
            -> Node:  # selection needs to start from root all the way down
        """
        Only select if all children have been created and have had at least one rollout. Select based on
        exploration vs. exploitation.
        :param saved_state:
        :param node:
        :return child node of highest UCB1 value if all children present. Otherwise select itself and expand in mcts()
        """
        if len(node.children) == len(available_columns(node.board)):
            values = []
            for n in node.children:
                values.append(n.value())
            ind = np.argmax(values)
            child = node.children[ind]
            return child  # value of the highest child node
        else:  # return self and expand in mcts()
            return node


def mcts(tree: Tree):
    """
    Run many iterations of the MCTS tree and build out its statistics
    :param tree:
    :return None
    """
    for _ in range(3000):
        node = tree.select(tree.root)
        test_node, test_player = tree.expand(node, node.nodePlayer)
        winner = test_node.rollout(test_node.board, opponent(test_player))  # rollout on opponent
        tree.update_Tree(test_node, winner)
    return


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None) -> \
        Tuple[int, Optional[SavedState]]:
    """
    Choose the child with most simulations instead of win percentage because could have a child with just 1
    simulation and 1 win for a win percentage of 100% but a different child with 1000 iterations and 600 wins for 60%
    :param board:
    :param player:
    :param saved_state:
    :return: Tuple with next move (column)
    """
    tree = Tree(board, player)
    mcts(tree)
    move = tree.root.children.index(max(tree.root.children, key=lambda c: c.simulations))
    for c in tree.root.children:
        print('Num of simulations: ', c.simulations, 'and Win %: ', c.wins() * 100 / c.simulations)
    return move, None
