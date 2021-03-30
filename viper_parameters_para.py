from Agent_Snake import aggregate_memories, choose_action, Memory, train_step, discount_rewards, create_snake_model
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
episodes = 10000

# Initialize things
memory = Memory()
snake_model = create_snake_model(width, height, activation_func)


####### clear memory at some point

game_actions = [SnakeGame.Move.UP,SnakeGame.Move.DOWN,SnakeGame.Move.LEFT,SnakeGame.Move.RIGHT]


def reward(game):
    if game.game_over:
        return -10
    elif game.just_ate_fruit:
        return 1
    else:
        return 0


def train(num_episodes, episode_length):

    for episode_no in range(num_episodes):

        if episode_no % 50 == 49:
            print(episode_no)

        games = [SnakeGame(width, height, num_fruit=3) for i in range(batch_size)]
        memories = [Memory() for i in range(batch_size)]

        for _ in range(episode_length):
            observations = numpy.array([numpy.copy(game.board) for game in games])
            actions = choose_action(snake_model, observations, single=False)
            for game, action in zip(games, actions):
                game.tick(game_actions[action])

            for memory, observation, action in zip(memories, observations, actions):
                memory.add_to_memory(observation, action, reward(game))

            for i in range(batch_size):
                if games[i].game_over:
                    games[i] = SnakeGame(width, height, num_fruit=3)

        batch_memory = aggregate_memories(memories)

        train_step(
            snake_model,
            optimizer,
            observations=numpy.stack(batch_memory.observations, 0),
            actions=numpy.array(batch_memory.actions),
            discounted_rewards=discount_rewards(batch_memory.rewards, -10)
        )


train(1000, 100)

game = SnakeGame(width, height, num_fruit=3)
play_game(game, snake_model)





