from game.engine import Connect4GameEngine
from game.gui import Connect4GUI
from ai.evaluation import Connect4AI
import pygame
import numpy as np
import time
import random


ACTION_SIZE = 7  # Number of columns
STATE_SIZE = 6 * 7  # Flattened board


if __name__ == "__main__":
    game_engine = Connect4GameEngine()
    gui = Connect4GUI(game_engine)
    mcts = Connect4AI(game_engine)
    while not game_engine.is_game_over():
        for event in pygame.event.get():
            gui.handel_event(event, game_engine.turn)
            if game_engine.turn == 1 or True:
                best_move = mcts.get_best_move(2)
                # best_move = random.randint(0, 7)
                game_engine.make_move(best_move, piece=game_engine.turn+1)
                gui.draw_board()

    time.sleep(2)


