from Agent_Snake import choose_action, Memory, train_step, discount_rewards, create_snake_model
import numpy
from snake import SnakeGame, play_game
import tensorflow as tf

width = 10
height = 10

learning_rate = 1
gamma = 0.95
batch_size = 69
activation_func = tf.keras.activations.softmax
optimizer = tf.keras.optimizers.Adam(learning_rate)
episodes = 100

# Initialize things
memory = Memory()
snake_model = create_snake_model(width, height, activation_func)


####### clear memory at some point

game_actions = [SnakeGame.Move.UP,SnakeGame.Move.DOWN,SnakeGame.Move.LEFT,SnakeGame.Move.RIGHT]

#### Do we parallelize later??

for i in range(episodes):
    game = SnakeGame(width,height)
    while not game.game_over:       
        observation = numpy.copy(game.board)
        action = choose_action(snake_model,observation)
        game.tick(game_actions[action])
        ## next_observation = numpy.copy(game.board)
        if game.game_over:
            reward = -1
        elif game.just_ate_fruit:
            reward = 1
        else:
            reward = 0

        memory.add_to_memory(observation, action, reward)

        if game.game_over:
            #### total_reward  = sum(memory.rewards)
            total_observation = numpy.stack(memory.observations)
            total_action = numpy.array(memory.actions)
            total_rewards = discount_rewards(memory.rewards, gamma)
            train_step(snake_model, optimizer, total_observation, total_action, total_rewards)

            memory.clear()
            break

game = SnakeGame(width, height)
play_game(game,snake_model)
     



    
