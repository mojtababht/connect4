import tensorflow as tf
from chessboard import display

Evaluation = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)  # Scalar evaluation
])


Prediction = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1968, activation='softmax')  # 1968 possible moves
])

# b_display = display.start('8/8/8/8/8/8/8/8')


def get_combined_model():
    input_layer = tf.keras.Input(shape=(8, 8, 12))
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Policy Head
    policy_head = tf.keras.layers.Dense(1968, activation='softmax', name='policy')(x)

    # Value Head
    value_head = tf.keras.layers.Dense(256, activation='relu')(x)
    value_head = tf.keras.layers.Dense(1, activation='tanh', name='value')(value_head)

    model = tf.keras.Model(inputs=input_layer, outputs=[policy_head, value_head])
    return model


Combined = get_combined_model()





import chess
import chess.engine
import numpy as np


def move_to_index(move):
    """
    Convert a chess.Move into a unique index based on the
    (from_square, to_square, promotion_piece) encoding.

    Index calculation:
    - from_square: 0–63 (6 bits)
    - to_square: 0–63 (6 bits)
    - promotion: 0–3 (2 bits) (0=None, 1=Knight, 2=Bishop, 3=Rook, 4=Queen)

    Args:
        move (chess.Move): A move object.

    Returns:
        int: A unique index for the move in the range 0–1967.
    """
    from_square = move.from_square
    to_square = move.to_square

    # Map promotion to an index (0 for no promotion)
    promotion_map = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    promotion_piece = promotion_map.get(move.promotion, 0)

    # Calculate the index
    return from_square * 5 + to_square * 5 + promotion_piece


class Node:
    def __init__(self, board, parent=None):
        self.board = board  # Current board state
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of visits to this node
        self.value_sum = 0  # Sum of evaluation scores for this node
        self.prior = 0  # Prior probability of this move

    def get_value(self, exploration_weight=1.0):
        """Calculate the UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have infinite value for exploration
        exploitation = self.value_sum / self.visits
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visits + 1) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self):
        """Check if the node has been fully expanded."""
        return len(self.children) == len(list(self.board.legal_moves))


def encode_board(board):
    """Convert a python-chess board into the (8, 8, 12) tensor format."""
    encoded_board = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        encoded_board[row, col, channel] = 1
    return encoded_board


def select_child(node):
    """Select the child node with the highest UCB1 value."""
    return max(node.children, key=lambda child: child.get_value())


def expand_node(node, model):
    """Expand a node by creating a child node for every legal move."""
    if node.is_fully_expanded():
        return
    for move in node.board.legal_moves:
        try:
            child_board = node.board.copy()
            child_board.push(move)
            child_node = Node(child_board, parent=node)
            node.children.append(child_node)
        except ValueError as e:
            print(f"Skipping move {move}: {e}")
            continue
    # Assign prior probabilities to children
    if node.children:
        encoded_board = encode_board(node.board).reshape(1, 8, 8, 12)
        policy_preds, _ = model(encoded_board)
        policy_preds = policy_preds.numpy().flatten()
        for child in node.children:
            try:
                move_idx = move_to_index(child.board.peek())
                child.prior = policy_preds[move_idx]
            except ValueError as e:
                print(e)
                continue



def simulate_game(node, model):
    """Simulate a game from the current node using the model's Value Head."""


    current_board = node.board.copy()
    while not current_board.is_game_over():
        encoded_board = encode_board(current_board).reshape(1, 8, 8, 12)
        policy_preds, _ = model(encoded_board)
        policy_preds = policy_preds.numpy().flatten()
        legal_moves = list(current_board.legal_moves)
        move_probs = [policy_preds[move_to_index(move)] for move in legal_moves]
        move_probs = np.array(move_probs) / sum(move_probs)  # Normalize probabilities
        selected_move = np.random.choice(legal_moves, p=move_probs)
        current_board.push(selected_move)

        # display.update(current_board.fen(), b_display)

    # Return the game outcome as a value in [-1, 0, 1]
    result = current_board.result()
    if result == "1-0":
        return 1  # White wins
    elif result == "0-1":
        return -1  # Black wins
    else:
        return 0  # Draw


def backpropagate(node, value):
    """Update node statistics along the path from the leaf to the root."""
    current = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        value = -value  # Alternate perspective for the opponent
        current = current.parent


def run_mcts(board, model, simulations=100, exploration_weight=1.0):
    """Perform Monte Carlo Tree Search to find the best move."""
    root = Node(board)
    for _ in range(simulations):
        # Selection: Traverse the tree
        node = root
        while node.is_fully_expanded() and node.children:
            node = select_child(node)

        # Expansion: Expand the node
        if not node.board.is_game_over():
            expand_node(node, model)

        # Simulation: Simulate a game to get the outcome
        leaf = select_child(node) if node.children else node
        value = simulate_game(leaf, model)

        # Backpropagation: Update statistics
        backpropagate(leaf, value)

    # Return move probabilities based on visit counts
    move_probs = np.zeros(1968)  # Total possible moves
    for child in root.children:
        move = child.board.peek()
        move_idx = move_to_index(move)
        move_probs[move_idx] = child.visits
    move_probs /= move_probs.sum()  # Normalize

    return move_probs


def select_move(board, move_probs, temperature=1.0):
    """
    Select a move based on move probabilities.

    Args:
        board (chess.Board): The current board position.
        move_probs (np.ndarray): Probability distribution over all 1968 possible moves.
        temperature (float): Controls exploration.
                             - High temperature (~1.0) encourages exploration.
                             - Low temperature (~0.1) encourages exploitation.
                             - Zero temperature always selects the move with the highest probability.

    Returns:
        chess.Move: The selected move.
    """
    legal_moves = list(board.legal_moves)  # Get legal moves from the board
    legal_move_indices = [move_to_index(move) for move in legal_moves]  # Map to indices

    # Extract probabilities for legal moves
    legal_probs = move_probs[legal_move_indices]
    if temperature == 0:
        # Select move with highest probability
        selected_index = np.argmax(legal_probs)
    else:
        # Adjust probabilities with temperature for exploration
        scaled_probs = np.power(legal_probs, 1 / temperature)
        scaled_probs /= np.sum(scaled_probs)  # Normalize
        selected_index = np.random.choice(len(legal_moves), p=scaled_probs)

    return legal_moves[selected_index]


import numpy as np
import tensorflow as tf
import random

# Example hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_SELF_PLAY_GAMES = 100
MCTS_SIMULATIONS = 100

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Loss functions
policy_loss_fn = tf.keras.losses.CategoricalCrossentropy()
value_loss_fn = tf.keras.losses.MeanSquaredError()

def play_self_play_game(model, mcts_simulations):
    import chess  # Use python-chess for game logic

    board = chess.Board()
    game_data = []

    b2 = display.start(board.fen())


    while not board.is_game_over():
        # Generate MCTS probabilities
        mcts_policy = run_mcts(board, model, mcts_simulations)
        board_state = encode_board(board)  # Encode the board into (8, 8, 12)

        # Select move based on MCTS probabilities
        move = select_move(board, mcts_policy)
        board.push(move)
        display.update(board.fen(), b2)

        # Store data
        game_data.append([board_state, mcts_policy, None])  # Game result filled later

    # Assign game results (-1, 0, 1 for loss, draw, win)
    result = board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
    outcome = 1 if result == "1-0" else -1 if result == "0-1" else 0
    for data in game_data:
        data[2] = outcome

    return game_data


# Training loop
for iteration in range(1000):  # Training iterations
    game_data = []  # Store (board_state, MCTS_policy, game_result) tuples

    # Self-play phase
    for _ in range(NUM_SELF_PLAY_GAMES):
        # Simulate a game using the Combined model and MCTS
        game = play_self_play_game(Combined, MCTS_SIMULATIONS)
        game_data.extend(game)  # Collect training data

    # Prepare data for training
    board_states, mcts_policies, game_results = zip(*game_data)
    board_states = np.array(board_states)  # Shape: (num_samples, 8, 8, 12)
    mcts_policies = np.array(mcts_policies)  # Shape: (num_samples, 1968)
    game_results = np.array(game_results)  # Shape: (num_samples,)

    # Training phase
    dataset = tf.data.Dataset.from_tensor_slices((board_states, mcts_policies, game_results))
    dataset = dataset.shuffle(len(game_data)).batch(BATCH_SIZE)

    for batch in dataset:
        states, target_policies, target_values = batch

        with tf.GradientTape() as tape:
            policy_preds, value_preds = Combined(states, training=True)
            policy_loss = policy_loss_fn(target_policies, policy_preds)
            value_loss = value_loss_fn(target_values, value_preds)
            loss = policy_loss + value_loss

        gradients = tape.gradient(loss, Combined.trainable_weights)
        optimizer.apply_gradients(zip(gradients, Combined.trainable_weights))

    print(f"Iteration {iteration + 1}: Loss = {loss.numpy():.4f}")
