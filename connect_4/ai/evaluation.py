import numpy as np
import random
from collections import deque
from connect_4.game.engine import Connect4GameEngine
import copy
import math


def copy_engine(engine):
    new_engine = Connect4GameEngine()
    new_engine.board = engine.get_board().copy()
    new_engine.turn = engine.turn
    return new_engine


class Node:

    def __init__(self, board: Connect4GameEngine, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent

    def get_legal_actions(self):
        return self.board.get_legal_moves()

    def act(self, action):
        self.board.make_move(action, self.player)
        return self.board.get_winner()

    def expand(self):
        childes = []
        if not self.board.is_game_over():
            for action in self.get_legal_actions():
                board = copy_engine(self.board)
                node = Node(board, self.player, self)
                node.act(action)
                childes.append(node)
        return childes


class MCTS:

    def __init__(self, board: Connect4GameEngine, player):
        self.board = board
        self.player = player
        self.nodes = []

    def expand(self):
        if len(self.nodes) == 0:
            nodes = []
            for move in self.board.get_legal_moves():
                board = copy_engine(self.board)
                node = Node(board, board.turn + 1)
                node.act(move)
                nodes.append(node)
            self.nodes = nodes
        else:
            nodes = []
            for node in self.nodes:
                nodes.extend(node.expand())
            self.nodes = nodes

    def best_move(self, debt=3):
        self.nodes = []
        for i in debt:
            self.expand()


