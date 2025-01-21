import numpy as np


def convert_board_to_model_input(board):
    # Convert the board into a 3D array with shape (6, 7, 1)
    # We will treat player 1 (red) as 1 and player 2 (yellow) as -1.
    # Empty spaces will be 0.

    # Create a 3D array with the board and add the channel dimension
    model_input = np.expand_dims(board, axis=-1)

    # Normalize the board to have values between -1 (player 2) and 1 (player 1)
    # Empty spaces remain as 0.
    model_input = np.where(model_input == 1, 1, model_input)
    model_input = np.where(model_input == 2, -1, model_input)
    model_input = model_input.astype(np.float32)

    return model_input
