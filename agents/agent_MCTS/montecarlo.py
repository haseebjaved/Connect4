import numpy as np
from agents.common import PlayerAction, apply_player_action, BoardPiece, available_columns, opponent, connected_four, check_end_state, PLAYER2, PLAYER1, NO_PLAYER, SavedState, GenMove, GameState
from typing import Optional, Callable, Tuple

class Node:
    def __init__(self, board): # constructor
        self.wins = None # or 0
        self.simulations = None # or 0
        self.parent = None
        self.children = [] # or a single child as in self.child = None ?
        self.board = board

    def value(self):
        return self.wins / self.parent.wins + (c * np.sqrt(np.log(self.parent.simulations) / self.simulations)

class MCTS:   # how do I pass the board into MCTS?
    def __init__(self):  # constructor
        self.root = None  # empty tree
        self.height = -1

    def expand(self, parent):  # insert node
        """
        Create one child node at a time, run a simulation, and back propagate the results.
        Since simulations of each child node are at random, we don't save those grandchildren nodes as they were
        created randomly and used randomly.
        We only save nodes in the MCTS that have some logic such as UCB1.
        :param: parent node
        :return:
        """
        if self.root is None:
            self.root = Node()
            self.height += 1
        else:
            avail_cols = available_columns(board)  # how do I pass the board into MCTS?

        find available_columns, create a child in a column as long as columns are available

    def simulate(self):
        """
        for each newly created child node, run a simulation of random moves all the way down to a win/draw/loss.
        save results and back propagate the results all the way to root.
        :return:
        """


    def select(self, Node, c):
        """
        Don't select until all children have been create for the parent with one simulation each

        """
        chosen_node = Node.children.index(max(Node.children.value)) # will .value work on a list or should I iterate?

        return chosen_node


#def mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
#) -> Tuple[int, Optional[SavedState]]:
    """
    Build a Monte Carlo Tree Search algorithm that takes a board at any state (beginning or middle of the game
    and determines the most valuable move for a player.

    Keyword arguments:
        board: the board that the player is playing and trying to win
        player: current player
        saved_state: Optional Saved State

    Returns:
        Tuple: consisting of the location of the column of the best move, and the Optional Saved State
    """


