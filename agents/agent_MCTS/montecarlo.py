import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, available_columns, opponent, connected_four, check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from agents.agent_random.random2 import generate_move_random
from typing import Optional, Callable, Tuple

class Node:
    def __init__(self, board, nodePlayer):  # constructor - each node is a state of the board
        #self.score = 0  # keep track of score for node - do I need this if already have value function below?
        self.simulations = 0  # keep track of total simulations
        self.wins = 0  # keep track of total wins, integer
        self.win = None # keep track of current win or loss - True or False
        self.parent = None
        self.children = []
        self.board = board
        self.nodePlayer = nodePlayer  # PLAYER1 or PLAYER2

    def value(self):  # UCB1 algorithm
        return self.wins / self.simulations + (c * np.sqrt(np.log(self.parent.simulations) / self.simulations))

    def create_child(self, avail_cols):
        col = np.random.choice(avail_cols)
        board_copy = apply_player_action(self.board, col, self.nodePlayer, True)
        new_node = Node(board_copy, opponent(self.nodePlayer))  # create_child function in Node instead
        new_node.parent = self
        self.children.append(new_node)
        return new_node

    def rollout(self, player):
        # call our recursive function helper
        if self.rolloutHelper(self.board, self.nodePlayer) == 1:
            self.wins += 1
            self.win = True
        self.win = False
        self.simulations += 1

    def rolloutHelper(self, board, player):
        """
        IS THIS CORRECT? In recursive call with opponent, will determineWin() check for and return win for original
        player or opponent? How to keep track of and return a 1 only for the original player of Node's rollout?
        :param board:
        :param player:
        :return: 1 for a win and 0 for a loss or draw
        """
        if self.isTerminal(board, player):
            if self.determineWin(board, player):  # if win
                return 1
            else:  # loss or draw
                return 0
        else:  # generate a random move, create a new board state, apply random move.
            random_move, saved_state = generate_move_random(board, player)
            random_board = apply_player_action(board, random_move, player, True)
            self.rolloutHelper(random_board, opponent(player))  # recursive call, passing in new board & opponent

    def isTerminal(self, board, player):
        """
        Take in a board, and a player, and make a determination if the board is terminal or not.
        Check if no available columns (moves) or if either player has won.
        :param board:
        :param player:
        :return: True or False
        """
        if available_columns(board) is None or determineWin(board, player) or determineWin(board, opponent(player)):
            return True
        else:
            return False

    def determineWin(self, board, player):
        """
        # Case 1: player == 1, and wins -> return true
        # Case 2: player == 2, and wins -> return false
        """
        if check_end_state(board, player) == GameState.IS_WIN:
            return True
        elif check_end_state(board, opponent(player)) == GameState.IS_WIN:
            return False
        else:  # draw
            return False


class MCTS:
    def __init__(self):  # constructor
        self.root = None  # empty tree
        self.height = -1

    def expand(self, node, player):  # insert node
        """
        Create one child node at a time, run a simulation/rollout, and back propagate the results.
        Since simulations of each child node are at random, we don't save those grandchildren nodes as they were
        created randomly and used randomly. We only save nodes in the MCTS that have some logic such as UCB1.
        Expand from a node (make it a parent) only when it has had at least one simulation
        :param: parent node
        :return:
        """
        if self.root is None:
            self.root = Node(node.board, player)
            self.height += 1
        else:
            if node.value() != np.inf:  # node has had at least one simulation
                avail_cols = available_columns(node.board)
                if avail_cols is None: # board full, check for win, loss, or draw?
                    print('board is full')
                    return # do i need to return None or 0 or something
                else:
                    new_node = Node.create_child(node, avail_cols)
                    new_node.rollout(opponent(player))  # should rollout happen on current player or opponent?
                    self.update_Tree(new_node)

    def update_Tree(self, node):
        """

        :type node: object
        """
        while node.parent is not None:  # go back up the tree to root
            if not node.win: # how to check for this?
                node.parent.wins += 1
            node.parent.simulations += 1
            node = node.parent

    def select(self, node):  # selection needs to start from root all the way down
        #
        """
        Only select if all children have been created and have had at least one rollout. If not, then do rollout instead
        Then select based on exploration vs. exploitation.
        :parameter given parent node
        :return child node of highest UCB1 value if all children present. Otherwise create a new child
        """
        if len(node.children) == 7: #and value_arr.any() != np.inf: assuming each child has at least 1 rollout
            values = []
            for n in node.children:
                values.append = n.value()
             value_arr = np.array(values)
            return np.argmax(value_arr)  # value of the highest non-infinity child node
        else: # need to expand by one and rollout on that new node
            self.expand(node, node.nodePlayer)


if __name__ == "__main__":
    tree = MCTS()
    tree.expand()
    return tree.select()

# how is the tree built?
# in expand(), should rollout happen on existing player or opponent?
# is the recursive call in the rolloutHelper checking for the original player or opponent, and which one is it updating?
# where to iterate and continue updating the tree statistics?