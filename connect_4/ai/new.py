import numpy as np
import random
from math import sqrt, log
from connect_4.game.engine import Connect4GameEngine


def find_lefter_object(children):
    sort_by_s = sorted(children, key=lambda c: c.total_reward, reverse=True)
    sort_by_c = sorted(children, key=lambda c: c.visits, reverse=True)
    # Create a dictionary to store the indices
    position_sum = {}

    # Add positions from sort_by_s
    for index, obj in enumerate(sort_by_s):
        if obj not in position_sum:
            position_sum[obj] = 0
        position_sum[obj] += index

    # Add positions from sort_by_c
    for index, obj in enumerate(sort_by_c):
        if obj not in position_sum:
            position_sum[obj] = 0
        position_sum[obj] += index

    # Find the object with the minimum combined position
    return min(position_sum, key=position_sum.get)


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Connect4GameEngine instance
        self.parent = parent  # Parent node
        self.move = move  # The move that led to this state
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits to this node
        self.total_reward = 0  # Total reward accumulated
        self.untried_moves = self.get_legal_moves()  # Moves not tried yet

    def get_legal_moves(self):
        return [col for col in range(7) if self.state.is_valid_location(col)]

    def ucb1(self, exploration_param=1.41):
        if self.visits == 0:
            return - float('inf')  # Prioritize unvisited nodes
        return (self.total_reward / self.visits) + exploration_param * sqrt(log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, game_engine, player_piece):
        self.game_engine = game_engine
        self.player_piece = player_piece
        self.opponent_piece = 1 if player_piece == 2 else 2

    def run(self, root_state, simulations=1000):
        root_node = MCTSNode(root_state)

        for _ in range(simulations):
            # Selection
            node = self.select(root_node)

            # Expansion
            if node.untried_moves:
                node = self.expand(node)

            # Simulation
            reward = self.simulate(node.state)

            # Backpropagation
            self.backpropagate(node, reward)

        # Choose the best move based on visit count
        # best_child = max(root_node.children, key=lambda n: n.visits)
        best_child = find_lefter_object(root_node.children)
        return best_child.move

    def select(self, node):
        # Traverse the tree using UCB1 until a leaf node is reached
        while node.children and not node.untried_moves:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def expand(self, node):
        # Expand a child node for one of the untried moves
        move = node.untried_moves.pop()
        next_state = self.simulate_move(node.state, move, self.player_piece)
        child_node = MCTSNode(next_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def simulate(self, state):
        # Perform a random simulation to the end of the game
        current_state = state
        current_piece = self.player_piece

        while not current_state.is_game_over():
            legal_moves = current_state.get_legal_moves()
            move = random.choice(legal_moves)
            current_state = self.simulate_move(current_state, move, current_piece)
            current_piece = 1 if current_piece == 2 else 2

        # Assign rewards based on the outcome
        if current_state.get_winner() == self.player_piece:
            return 1  # Win
        elif current_state.get_winner() == self.opponent_piece:
            return -1  # Loss
        else:
            return 0  # Draw

    def backpropagate(self, node, reward):
        # Update statistics for all nodes in the path
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def simulate_move(self, state, move, piece):
        # Create a copy of the state and apply the move
        new_state = Connect4GameEngine()
        new_state.board = state.board.copy()
        new_state.make_move(move, piece)
        return new_state
