import numpy as np
from connect_4.game.engine import Connect4GameEngine


class Connect4AI:

    def __init__(self, game_engine, depth=6):
        self.engine = game_engine
        self.depth = depth

    @property
    def game_engine(self):
        engin = Connect4GameEngine()
        engin.turn = self.engine.turn
        engin.board = self.engine.board
        return engin

    def minimax(self, board, depth, maximizing_player, alpha, beta, piece):
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            board (ndarray): Current board state.
            depth (int): Current depth in the decision tree.
            maximizing_player (bool): True if it's the maximizing player's turn, False otherwise.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            piece (int): The player's piece (1 or 2).

        Returns:
            tuple: (best_score, best_move)
        """
        opponent_piece = 2 if piece == 1 else 1

        # Check for terminal state (win/loss/draw) or depth limit
        if self.game_engine.check_winner() == piece:
            return 100, None
        elif self.game_engine.check_winner() == opponent_piece:
            return -100, None
        elif self.game_engine.is_draw() or depth == 0:
            return 0, None

        valid_moves = self.game_engine.get_legal_moves()

        if maximizing_player:
            best_score = -float("inf")
            best_move = None
            for col in valid_moves:
                # Simulate the move
                row = self.game_engine.get_next_open_row(col)
                board[row][col] = piece

                # Recursive call
                score, _ = self.minimax(board, depth - 1, False, alpha, beta, piece)

                # Undo the move
                board[row][col] = 0

                if score > best_score:
                    best_score = score
                    best_move = col

                # Alpha-beta pruning
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

            return best_score, best_move
        else:
            best_score = float("inf")
            best_move = None
            for col in valid_moves:
                # Simulate the move
                row = self.game_engine.get_next_open_row(col)
                board[row][col] = opponent_piece

                # Recursive call
                score, _ = self.minimax(board, depth - 1, True, alpha, beta, piece)

                # Undo the move
                board[row][col] = 0

                if score < best_score:
                    best_score = score
                    best_move = col

                # Alpha-beta pruning
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            return best_score, best_move

    def get_best_move(self, piece):
        """
        Get the best move for the current player.

        Args:
            piece (int): The player's piece (1 or 2).

        Returns:
            int: The column of the best move.
        """
        _, best_move = self.minimax(
            self.game_engine.get_board(),
            self.depth,
            True,
            -float("inf"),
            float("inf"),
            piece,
        )
        return best_move