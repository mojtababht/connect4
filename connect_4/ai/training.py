import numpy as np
from connect_4.game.engine import Connect4GameEngine
from connect_4.game.gui import Connect4GUI
from model import DQNAgent

# Hyperparameters
EPISODES = 100000000+9
BATCH_SIZE = 1000
# BATCH_SIZE = 64
SAVE_INTERVAL = 1  # Save models every 10 episodes
STATE_SIZE = 6 * 7  # Flattened board
ACTION_SIZE = 7  # Number of columns

# Initialize the game and agents
game = Connect4GameEngine()
agent_red = DQNAgent(STATE_SIZE, ACTION_SIZE, color='red')  # Red agent
agent_yellow = DQNAgent(STATE_SIZE, ACTION_SIZE, color='yellow')  # Yellow agent
# gui = Connect4GUI(game)

# Training Loop
for episode in range(EPISODES):
    # print(f"Episode {episode + 1}/{EPISODES}")

    # Reset the game
    game.reset()
    state = np.reshape(game.get_board(), [1, STATE_SIZE])
    done = False
    total_rewards = {"red": [], "yellow": []}

    while not done:
        agent_red.epsilon = 1.0
        # Red's Turn
        action_red = agent_red.act(state)
        next_state, reward_red, done, info = game.step(action_red, piece=1)  # Red is piece=1
        next_state = np.reshape(next_state, [1, STATE_SIZE])

        # Record Red's experience
        agent_red.remember(state, action_red, reward_red, next_state, done)
        total_rewards["red"].append(reward_red)
        if reward_red:
            total_rewards["yellow"].append(reward_red * -1)

        if done:
            break  # Game ended after Red's move

        # Update state for Yellow's turn
        state = next_state

        # Yellow's Turn
        agent_yellow.epsilon = 1.0

        action_yellow = agent_yellow.act(state)
        next_state, reward_yellow, done, info = game.step(action_yellow, piece=2)  # Yellow is piece=2
        next_state = np.reshape(next_state, [1, STATE_SIZE])

        # Record Yellow's experience
        agent_yellow.remember(state, action_yellow, reward_yellow, next_state, done)
        total_rewards["yellow"].append(reward_yellow)
        if reward_yellow:
            total_rewards["red"].append(reward_yellow * -1)

        # Update state for next turn
        state = next_state
        # gui.draw_board()

    # Train both agents
    # if len(agent_red.memory) > BATCH_SIZE:
    if (episode + 1) % 1000 == 0:
        print('replaying red')
        agent_red.replay(BATCH_SIZE)
        print('done')
    # if len(agent_yellow.memory) > BATCH_SIZE:
    if (episode + 1) % 1000 == 0:
        print('replaying yellow')
        agent_yellow.replay(BATCH_SIZE)
        print('done')

    # Save models periodically
    if (episode + 1) % SAVE_INTERVAL == 0:
        agent_red.model.save("trained_data/red.keras")
        agent_yellow.model.save("trained_data/yellow.keras")
        if (episode + 1) % 1000 == 0:
            print(f"Models saved at Episode {episode + 1}")

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: "
              f"Red Reward: {total_rewards['red']}, "
              f"Yellow Reward: {total_rewards['yellow']}, "
              f"Red Epsilon: {agent_red.epsilon:.2f}, "
              f"Yellow Epsilon: {agent_yellow.epsilon:.2f}")
