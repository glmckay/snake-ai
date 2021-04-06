from Agent_Snake import (
    aggregate_memories,
    choose_action,
    Memory,
    train_step,
    discount_rewards,
    create_snake_model,
)
import numpy
from snake import SnakeGame, play_game

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

width = 10
height = 10

learning_rate = 1e-3
gamma = 0.99
batch_size = 69
activation_func = None  # tf.keras.activations.softmax
optimizer = tf.keras.optimizers.Adam(learning_rate)
NUM_FRUITS = 1

# Initialize things
memory = Memory()
snake_model = create_snake_model(width, height, activation_func)


####### clear memory at some point

game_actions = [
    SnakeGame.Move.UP,
    SnakeGame.Move.DOWN,
    SnakeGame.Move.LEFT,
    SnakeGame.Move.RIGHT,
]

GAME_OVER_REWARD = -1.0


def reward(game):
    if game.game_over:
        return GAME_OVER_REWARD
    elif game.moves_since_last_fruit == 0:
        return 1.0
    elif game.moves_since_last_fruit % 20 == 0:
        return -0.1520
    else:
        return 0


def train(num_episodes, episode_length):

    for episode_no in range(num_episodes):

        print(f"\rEpisode {episode_no} out of {num_episodes}", end="\r")

        games = [SnakeGame(width, height, num_fruit=NUM_FRUITS) for i in range(batch_size)]
        memories = [Memory() for i in range(batch_size)]
        memories_for_training = []

        for _ in range(episode_length):
            observations = numpy.array([numpy.copy(game.board) for game in games])
            actions = choose_action(snake_model, observations, single=False)
            for game, action in zip(games, actions):
                game.tick(game_actions[action])

            for memory, observation, action, game in zip(
                memories, observations, actions, games
            ):
                memory.add_to_memory(observation, action, reward(game))

            for i in range(batch_size):
                if games[i].game_over:
                    # memories_for_training.append(memories[i])
                    # memories[i] = Memory()

                    games[i] = SnakeGame(width, height, num_fruit=NUM_FRUITS)

        batch_memory = aggregate_memories(memories)

        train_step(
            snake_model,
            optimizer,
            observations=numpy.stack(batch_memory.observations, 0),
            actions=numpy.array(batch_memory.actions),
            discounted_rewards=discount_rewards(batch_memory.rewards, GAME_OVER_REWARD),
        )


train(700, 116)

game = SnakeGame(width, height, num_fruit=NUM_FRUITS)
play_game(game, snake_model)