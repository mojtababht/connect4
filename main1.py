import random
import tensorflow as tf

import chess
from chessboard import display
from sklearn.preprocessing import StandardScaler, LabelEncoder

from engin.utils import convert_board_to_tensor
from engin.models import eval_model


board = chess.Board()


board_tensor = convert_board_to_tensor(board)


b = display.start(board.fen())
while not display.check_for_quit() and not board.outcome():
    move = random.choice(list(board.legal_moves))
    board.push_san(str(move))
    display.update(board.fen(), b)

board_tensor = convert_board_to_tensor(board)
p = eval_model.predict(board_tensor)
print(p)

print(board.outcome())

display.terminate()
