from Agent_Snake import (
    aggregate_memories,
    choose_action,
    Memory,
    train_step,
    discount_rewards,
    create_snake_model,
)
import numpy
from snake import SnakeGame
from snake_terminal import play_game
from game_options import game_options
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

opts = game_options

learning_rate = 1e-3
gamma = 0.99
batch_size = 69
activation_func = None  # tf.keras.activations.softmax
optimizer = tf.keras.optimizers.Adam(learning_rate)

width = game_options["width"]
height = game_options["height"]

# Initialize things
memory = Memory()
snake_model = create_snake_model(
    (
        (
            tf.keras.layers.Dense,
            {"units": width * height * 10, "activation": tf.keras.activations.relu},
        ),
        (
            tf.keras.layers.Dense,
            {"units": width * height * 10, "activation": tf.keras.activations.relu},
        ),
        (
            tf.keras.layers.Dense,
            {"units": width * height * 5, "activation": tf.keras.activations.relu},
        ),
        (tf.keras.layers.Dense, {"units": 4, "activation": None}),
    )
)


game_actions = [
    SnakeGame.Move.UP,
    SnakeGame.Move.DOWN,
    SnakeGame.Move.LEFT,
    SnakeGame.Move.RIGHT,
]

GAME_OVER_REWARD = -4.0


def reward(game):
    if game.game_over:
        return GAME_OVER_REWARD
    elif game.moves_since_last_fruit == 0:
        return max(1.0, min(3.5 - game.moves_to_get_last_fruit * 0.1, 3.0))
    # elif game.moves_since_last_fruit % 20 == 0:
    #     return -0.1520
    else:
        return 0


def train(num_episodes, episode_length, gamma=0.7):

    for episode_no in range(num_episodes):

        print(f"\rEpisode {episode_no} out of {num_episodes}", end="\r")

        games = [SnakeGame(**opts) for i in range(batch_size)]
        memories = [Memory() for i in range(batch_size)]

        for _ in range(episode_length):
            observations = numpy.array([game.get_board() for game in games])
            actions = choose_action(snake_model, observations, single=False)
            for game, action in zip(games, actions):
                game.tick(game_actions[action])

            for memory, observation, action, game in zip(
                memories, observations, actions, games
            ):
                memory.add_to_memory(observation, action, reward(game))

            for i in range(batch_size):
                if games[i].game_over:
                    games[i] = SnakeGame(**opts)

        batch_memory = aggregate_memories(memories)

        train_step(
            snake_model,
            optimizer,
            observations=numpy.stack(batch_memory.observations, 0),
            actions=numpy.array(batch_memory.actions),
            discounted_rewards=discount_rewards(
                batch_memory.rewards, GAME_OVER_REWARD, gamma
            ),
        )


train(100, 100, 0.8)

game = SnakeGame(**opts)
play_game(game, snake_model)
