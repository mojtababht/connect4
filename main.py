import chess
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from chessboard import display

b_display = display.start('8/8/8/8/8/8/8/8')


# Model Definitions
def get_combined_model():
    input_layer = tf.keras.Input(shape=(8, 8, 12))
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Policy Head
    policy_head = tf.keras.layers.Dense(1968, activation='softmax', name='policy')(x)

    # Value Head
    value_head = tf.keras.layers.Dense(128, activation='relu')(x)
    value_head = tf.keras.layers.Dense(1, activation='tanh', name='value')(value_head)

    model = tf.keras.Model(inputs=input_layer, outputs=[policy_head, value_head])
    return model

Combined = get_combined_model()

PROMOTION_MAP = np.array([0, 1, 2, 3, 4])  # None, Knight, Bishop, Rook, Queen

def move_to_index(move):
    """Optimized move_to_index function."""
    from_square = move.from_square
    to_square = move.to_square
    promotion_piece = 0 if move.promotion is None else move.promotion - 1
    return from_square * 5 + to_square * 5 + promotion_piece

def encode_board(board):
    """Convert a python-chess board into the (8, 8, 12) tensor format."""
    encoded_board = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        encoded_board[row, col, channel] = 1
    return encoded_board

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.prior = 0

    def get_value(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value_sum / self.visits
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visits + 1) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

def expand_node_with_batch(node, model):
    """Expand node and assign priors using batch processing."""
    children = []
    boards = []
    for move in node.board.legal_moves:
        child_board = node.board.copy()
        child_board.push(move)
        boards.append(encode_board(child_board))
        children.append(Node(child_board, parent=node))

    # Batch process boards
    if boards:
        board_batch = np.array(boards)
        policy_preds, _ = model(board_batch)
        policy_preds = policy_preds.numpy()

        # Assign priors
        for i, child in enumerate(children):
            move_idx = move_to_index(child.board.peek())
            child.prior = policy_preds[i][move_idx]
            node.children.append(child)

def simulate_game(node, model):
    current_board = node.board.copy()
    while not current_board.is_game_over():
        encoded_board = encode_board(current_board).reshape(1, 8, 8, 12)
        policy_preds, _ = model(encoded_board)
        policy_preds = policy_preds.numpy().flatten()

        legal_moves = list(current_board.legal_moves)
        move_probs = [policy_preds[move_to_index(move)] for move in legal_moves]
        move_probs = np.array(move_probs) / sum(move_probs)
        selected_move = np.random.choice(legal_moves, p=move_probs)
        current_board.push(selected_move)

    result = current_board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0

def backpropagate(node, value):
    current = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        value = -value
        current = current.parent

def run_mcts(board, model, simulations=100, exploration_weight=1.0):
    root = Node(board)
    for _ in range(dynamic_mcts_simulations(board)):
        node = root
        while node.is_fully_expanded() and node.children:
            node = max(node.children, key=lambda child: child.get_value(exploration_weight))

        if not node.board.is_game_over():
            expand_node_with_batch(node, model)

        leaf = node if not node.children else max(node.children, key=lambda child: child.get_value())
        value = simulate_game(leaf, model)
        backpropagate(leaf, value)

    move_probs = np.zeros(1968)
    for child in root.children:
        move = child.board.peek()
        move_idx = move_to_index(move)
        move_probs[move_idx] = child.visits
    return move_probs / move_probs.sum()

def dynamic_mcts_simulations(board):
    if board.fullmove_number < 10:
        return 20
    elif board.fullmove_number < 30:
        return 50
    else:
        return 100

def play_self_play_game(model):
    board = chess.Board()
    game_data = []

    while not board.is_game_over():
        simulations = dynamic_mcts_simulations(board)
        mcts_policy = run_mcts(board, model, simulations)
        board_state = encode_board(board)
        legal_moves = list(board.legal_moves)
        move_probs = [mcts_policy[move_to_index(move)] for move in legal_moves]
        move_probs = np.array(move_probs) / sum(move_probs)
        selected_move = np.random.choice(legal_moves, p=move_probs)
        board.push(selected_move)
        display.update(board.fen(), b_display)

        game_data.append([board_state, mcts_policy, None])

    result = board.result()
    outcome = 1 if result == "1-0" else -1 if result == "0-1" else 0
    for data in game_data:
        data[2] = outcome

    return game_data

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_SELF_PLAY_GAMES = 10  # Reduce for testing
MCTS_SIMULATIONS = 100

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
policy_loss_fn = tf.keras.losses.CategoricalCrossentropy()
value_loss_fn = tf.keras.losses.MeanSquaredError()

for iteration in range(10):  # Adjust for production
    game_data = []
    for _ in range(NUM_SELF_PLAY_GAMES):
        game = play_self_play_game(Combined)
        game_data.extend(game)

    board_states, mcts_policies, game_results = zip(*game_data)
    board_states = np.array(board_states)
    mcts_policies = np.array(mcts_policies)
    game_results = np.array(game_results)

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
