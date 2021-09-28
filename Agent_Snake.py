import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
from game_options import game_options
from snake import SnakeGame

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

opts = {
    "width": game_options["width"],
    "height": game_options["height"],
    "num_fruits": game_options["num_fruits"],
    "walls": game_options["walls"],
    "grows": game_options["grows"],
}

game_actions = [
    SnakeGame.Move.UP,
    SnakeGame.Move.DOWN,
    SnakeGame.Move.LEFT,
    SnakeGame.Move.RIGHT,
]

width = opts["width"]
height = opts["height"]

class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


# Helper function to combine a list of Memory objects into a single Memory.
# This will be very useful for batching.
def aggregate_memories(memories):
    batch_memory = Memory()
    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)
    return batch_memory



### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x = x - np.mean(x)
    std = np.std(x)
    x = x / std if std != 0 else x
    return x.astype(np.float32)


# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, game_over_reward, gamma=0.5):
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    R = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] == game_over_reward:
            # reset before contributing to sum because we are iterating in reverse
            R = 0
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)


# A reward function

def reward(game, GameOverReward, UsualReward, TimeOutReward):
    if game.game_over:
        return GameOverReward
    elif game.moves_since_last_fruit == 0:
        return UsualReward
    elif game.moves_since_last_fruit >= 20:
        return TimeOutReward
    else:
        return 0

### Loss function ###

# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels= actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


### Training step (forward and backpropagation) ###

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# a function that creates the snake model
# Sample format for a layers input:
# [
#     [tf.keras.layers.Dense, {"units": width*height*10, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": width*height*10, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": width*height*5, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": 4, "activation": tf.keras.activations.sigmoid}]
# ]
def create_snake_model(layers):
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(), *(layer(**attrs) for layer, attrs in layers)]
    )
    return model


# a function that takes in a model, and an observation, then output an action that the model wants to do
def choose_action(model, observation, single = True):
    observation = np.expand_dims(observation, axis =0) if single else observation
    logits = model(observation)
    action = tf.random.categorical(logits, num_samples=1)
    # action = tf.random.categorical(np.log(logits), num_samples=1)
    action = action.numpy().flatten()
    return action[0] if single else action



# A functions that trains a model given various parameters
def train_model(model, learning_rate, optimizer, batch_size, episode_length, num_episodes, rewards, gamma):
    GameOverReward, UsualReward, TimeOutReward = rewards
    for episode_no in range(num_episodes):
        print(f"\rEpisode {episode_no} out of {num_episodes}", end="\r")

        games = [SnakeGame(**opts) for i in range(batch_size)]
        memories = [Memory() for i in range(batch_size)]
        for _ in range(episode_length):
            observations = np.array([game.get_board() for game in games])
            actions = choose_action(model, observations, single=False)
            for game, action in zip(games, actions):
                game.tick(game_actions[action])
            for memory, observation, action, game in zip(
                memories, observations, actions, games
            ):
                memory.add_to_memory(observation, action, reward(game, GameOverReward, UsualReward, TimeOutReward))

            for i in range(batch_size):
                if games[i].game_over:
                    games[i] = SnakeGame(**opts)

        batch_memory = aggregate_memories(memories)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(learning_rate),
            decay_steps = 10000,
            decay_rate = 0.96,
            staircase = True)
        train_step(
            model,
            optimizer = optimizer(lr_schedule),
            observations= np.stack(batch_memory.observations, 0),
            actions = np.array(batch_memory.actions),
            discounted_rewards = discount_rewards(batch_memory.rewards, GameOverReward, gamma),
        )


# A function that calculates how good a particular model is
# It will output two metrics:
#       the average number of turns it takes for the model to game over (maximum number of turn is episode_length)
#       the average number of fruits the model picks up before game over

def score_model(model, episode_length = 500, num_episodes = 100, batch_size = 100):
    print(f"Begin scoring with episode length {episode_length}, number of episodes {num_episodes} and batch size {batch_size}.")
    turns_game_over = 0
    number_of_fruits = 0
    for episode_no in range(num_episodes):

        games = [SnakeGame(**opts) for i in range(batch_size)]
        game_finished = [False for i in range(batch_size)]

        for i in range(episode_length):
            observations = np.array([game.get_board() for game in games])
            actions = choose_action(model, observations, single=False)
            for game, action, game_done in zip(games, actions, game_finished):
                game.tick(game_actions[action])
                if not game_done and game.game_over:
                    if not game_finished:
                        game_finished[i] = True
                        turns_game_over += i

        turns_game_over += sum([0 if fin else episode_length for fin in game_finished])
        number_of_fruits += sum([game.score for game in games])
    AvgTurnsGG = turns_game_over / (episode_length * batch_size)
    AvgGameScore = number_of_fruits/ (episode_length * batch_size)
    print(f"The average number of turns it takes to game over is {AvgTurnsGG}. \n The average score is {AvgGameScore}.")
    return (AvgTurnsGG, AvgGameScore)