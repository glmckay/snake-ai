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
activation_func = tf.keras.activations.softmax
optimizer = tf.keras.optimizers.Adam(learning_rate)
episodes = 10000

# Initialize things
memory = Memory()
snake_model = create_snake_model(width, height, activation_func)


####### clear memory at some point

game_actions = [SnakeGame.Move.UP,SnakeGame.Move.DOWN,SnakeGame.Move.LEFT,SnakeGame.Move.RIGHT]

#### Do we parallelize later??

def train(num_episodes):
    max_moves = 101

    for i in range(num_episodes):
        game = SnakeGame(width, height, num_fruit=3)
        num_moves = 0
        while not game.game_over:
            observation = numpy.copy(game.board)
            action = choose_action(snake_model, observation)
            game.tick(game_actions[action])
            ## next_observation = numpy.copy(game.board)
            if game.game_over:
                reward = -10
            elif game.just_ate_fruit:
                reward = 1
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

            num_moves += 1
            if num_moves > max_moves:
                break


for i in range(4):
    train(2500)
    print(i)


game = SnakeGame(width, height, num_fruit=3)
play_game(game, snake_model)





