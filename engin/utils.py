import tensorflow as tf
# import numpy as np
#
#
# pieces_alt = np.array([-1, -5, -2, -3, -9, -10, 1, 5, 2, 3, 9, 10])
# piece_map = {char: idx for idx, char in enumerate('prnbqkPRNBQK')}
#
# def fen_to_tensor(fen_board):
#     board_array = np.zeros((8, 8), dtype=np.int32)
#     row_idx = 0
#     col_idx = 0
#     for row in fen_board.split('/'):
#         for char in row:
#             if char.isdigit():
#                 col_idx += int(char)
#             else:
#                 board_array[row_idx, col_idx] = pieces_alt[piece_map[char]]
#                 col_idx += 1
#             if col_idx == 8:
#                 row_idx += 1
#                 col_idx = 0
#     board_tensor = tf.convert_to_tensor(board_array)
#     board_tensor = tf.reshape(board_tensor, (1, 8, 8, 1))
#     return board_tensor
#
#
# def convert_board_to_tensor(board):
#     return fen_to_tensor(board.board_fen())
#
import tensorflow as tf
import numpy as np

def fen_to_tensor(fen):
    piece_to_int = {
        'r': -5, 'n': -2, 'b': -3, 'q': -9, 'k': -10, 'p': -1,
        'R': 5, 'N': 2, 'B': 3, 'Q': 9, 'K': 10, 'P': 1
    }
    tensor = tf.zeros([8, 8], dtype=tf.int32)
    numpy_array = tensor.numpy()
    rows = fen.split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                numpy_array[i, col] = piece_to_int[char]
                col += 1
    tensor = tf.convert_to_tensor(numpy_array)
    return tensor
