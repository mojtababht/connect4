import numpy as np
import random
import tensorflow as tf
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

    def __init__(self, state, action, player, sim, parent=None):
        self.state = copy_engine(state)
        self.action = action
        self.player = player
        self.parent = parent
        self.sim = sim

    @staticmethod
    def create_node(state, action, sim, parent=None):
        player = state.turn
        state.make_move(action, player)
        return Node(state, action, player, sim, parent)

    def get_winner(self):
        return self.state.winner

    def dict(self):
        return {'parent': self.parent.dict() if self.parent is not None else {},}


class MCTS:

    def __init__(self, player):
        self.player = player
        self.layers = {}
        self.x = {}

    def map_winner(self, winner):
        match winner - 1:
            case self.player:
                return 'self'
            case None:
                return
            case _:
                return 'opponent'

    def search(self, engine, debt=4, sim=0):
        legal_moves = engine.get_legal_moves()
        moves = {}
        if debt > sim and not engine.game_over:
            sim += 1
            for move in legal_moves:
                new_engin = copy_engine(engine)
                new_engin.make_move(move, engine.turn + 1)
                if new_engin.game_over:
                    moves[move] = self.map_winner(new_engin.get_winner())
                else:
                    moves[move] = self.search(new_engin, debt, sim)
        return moves

    def get_wins_loss(self, moves, wins=0, loss=0):
        for winner in moves.values():
            match winner:
                case 'self':
                    wins += 1
                case 'opponent':
                    loss += 1
                case _:
                    if type(winner) == dict:
                        wins_loss = self.get_wins_loss(winner, wins, loss)
                        wins += wins_loss[0]
                        loss += wins_loss[1]
        return wins, loss

    def get_best_move(self, engin, debt=4):
        moves = self.search(engin, debt)
        moves_win_loss = {}
        for move, winner in moves.copy().items():
            if winner == 'self':
                return move
            else:
                if 'opponent' in winner.values():
                    moves.pop(move)
                else:
                    moves_win_loss[move] = self.get_wins_loss(winner)
        print(moves_win_loss)
        print(moves)

