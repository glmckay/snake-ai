from Agent_Snake import choose_action, Memory, train_step, discount_rewards, create_snake_model
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

#### Do we parallelize later??

def train(num_episodes):
    max_moves_without_fruit = 15

    for i in range(num_episodes):
        game = SnakeGame(width, height, num_fruit=3)
        num_moves_without_fruit = 0
        while not game.game_over:
            observation = numpy.copy(game.board)
            action = choose_action(snake_model, observation)
            game.tick(game_actions[action])
            ## next_observation = numpy.copy(game.board)

            num_moves_without_fruit += 1

            if game.game_over:
                reward = -10
            elif game.just_ate_fruit:
                reward = 1
            elif num_moves_without_fruit > max_moves_without_fruit:
                reward = -1
                num_moves_without_fruit = 0
            else:
                reward = 0

            memory.add_to_memory(observation, action, reward)

            if game.game_over:
                #### total_reward  = sum(memory.rewards)
                total_observation = numpy.stack(memory.observations, 0)
                total_action = numpy.array(memory.actions)
                total_rewards = discount_rewards(memory.rewards, gamma)
                train_step(snake_model, optimizer, total_observation, total_action, total_rewards)

                memory.clear()
                break



for i in range(16):
    train(1250)
    print(i)


game = SnakeGame(width, height, num_fruit=3)
play_game(game, snake_model)





