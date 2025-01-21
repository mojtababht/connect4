import numpy as np


class Connect4GameEngine:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = self.create_board()
        self.game_over = False
        self.winner = None
        self.turn = 0

    def create_board(self):
        return np.zeros((self.ROWS, self.COLS), dtype=int)

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.ROWS - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.ROWS):
            if self.board[r][col] == 0:
                return r

    def check_winner(self):
        # Check horizontal locations
        for c in range(self.COLS - 3):
            for r in range(self.ROWS):
                if self.board[r][c] == self.board[r][c+1] == self.board[r][c+2] == self.board[r][c+3] != 0:
                    self.game_over = True
                    self.winner = self.board[r][c]
                    return self.winner

        # Check vertical locations
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                if self.board[r][c] == self.board[r+1][c] == self.board[r+2][c] == self.board[r+3][c] != 0:
                    self.game_over = True
                    self.winner = self.board[r][c]
                    return self.winner

        # Check positively sloped diagonals
        for c in range(self.COLS - 3):
            for r in range(self.ROWS - 3):
                if self.board[r][c] == self.board[r+1][c+1] == self.board[r+2][c+2] == self.board[r+3][c+3] != 0:
                    self.game_over = True
                    self.winner = self.board[r][c]
                    return self.winner

        # Check negatively sloped diagonals
        for c in range(self.COLS - 3):
            for r in range(3, self.ROWS):
                if self.board[r][c] == self.board[r-1][c+1] == self.board[r-2][c+2] == self.board[r-3][c+3] != 0:
                    self.game_over = True
                    self.winner = self.board[r][c]
                    return self.winner

        return None

    def is_draw(self):
        for x in self.board:
            for y in x:
                if y == 0:
                    return False
        return True

    def is_game_over(self):
        return self.game_over or all(self.board[self.ROWS - 1, :] != 0)

    def make_move(self, col, piece):
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.drop_piece(row, col, piece)
            self.check_winner()
            if winner := self.check_winner():
                self.winner = winner
            self.turn = 1 - self.turn
        else:
            raise ValueError("Invalid move: Column is full")

    def get_board(self):
        return self.board

    def get_winner(self):
        return self.winner

    def reset(self):
        self.board = self.create_board()
        self.game_over = False
        self.winner = None
        self.turn = 0

    def step(self, action, piece):
        """
        Perform one step in the Connect 4 environment.

        Args:
            action (int): Column where the player wants to drop their piece.
            piece (int): The player's piece (1 for Red, 2 for Yellow).

        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (ndarray): The updated board state.
                - reward (float): The reward for the action.
                - done (bool): Whether the game has ended.
                - info (dict): Additional information (e.g., the winner).
        """
        # Check if the action is valid
        if not self.is_valid_location(action):
            # Invalid move penalty and opponent wins
            reward = -1
            winner = None
            return self.get_board(), reward, True, {"winner": winner}

        # Drop the piece and update the board
        row = self.get_next_open_row(action)
        self.drop_piece(row, action, piece)

        # Check for a winner
        winner = self.check_winner()
        if winner:
            # Reward the winning player
            reward = 1 if winner == piece else -1
            return self.get_board(), reward, True, {"winner": winner}

        # Check for a draw
        if self.is_draw():
            return self.get_board(), 0, True, {"winner": None}  # Draw has neutral reward

        # If no winner and no draw, game continues
        return self.get_board(), 0, False, {"winner": None}

    def get_legal_moves(self):
        return [col for col in range(7) if self.is_valid_location(col)]

