import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

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
#  This will be very useful for batching.
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

### Cartpole training! ###


def create_snake_model(width, height, activation_func):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(width*height*10, activation = 'relu'),
        # tf.keras.layers.Dense(width*height*10, activation = 'sigmoid'),
        # tf.keras.layers.Dense(width*height*5, activation = 'sigmoid'),
        tf.keras.layers.Dense(4, activation = activation_func)
    ])
    return model


def choose_action(model, observation, single = True):
    observation = np.expand_dims(observation, axis =0) if single else observation
    logits = model(observation)
    action = tf.random.categorical(logits, num_samples=1)
    # action = tf.random.categorical(np.log(logits), num_samples=1)
    action = action.numpy().flatten()
    return action[0] if single else action




