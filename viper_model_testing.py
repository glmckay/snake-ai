from Agent_Snake import (
    aggregate_memories,
    choose_action,
    Memory,
    train_step,
    discount_rewards,
    create_snake_model,
    train_model,
    score_model
)
import numpy
from snake import SnakeGame
from snake_terminal import play_game
from main import game_options
import os
import pandas as pd
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

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

# Initialize things
memory = Memory()

####### clear memory at some point
game_actions = [
    SnakeGame.Move.UP,
    SnakeGame.Move.DOWN,
    SnakeGame.Move.LEFT,
    SnakeGame.Move.RIGHT,
]

# A function that creates a DataFrame containing the results of trying out various different parameters into the model
def test_models(layers_struct, learning_rates, optimizers, losses, batches, episode_lengths, num_episodes_list, rewards, gammas):
    results = defaultdict(list)
    for layers in layers_struct:
        for learning_rate in learning_rates:
            for optimizer in optimizers:
                for loss in losses:
                    for batch_size in batches:
                        for episode_length in episode_lengths:
                            for num_episodes in num_episodes_list:
                                for reward in rewards:
                                    for gamma in gammas:
                                        print(f"Creating model with layers {layers}")
                                        snake_model = create_snake_model(layers)
                                        train_model(snake_model, learning_rate, optimizer, batch_size, episode_length, num_episodes, reward, gamma)

                                        for i in range(len(layers)):
                                            layer_type, layer_attributes = layers[i]
                                            results[f"Layer{i+1}_Name"].append(layer_type.__name__)
                                            for elem in layer_attributes:
                                                if layer_attributes[elem] == None:
                                                    results[f"Layer{i+1}_{elem}"].append("None")
                                                elif callable(layer_attributes[elem]):
                                                    results[f"Layer{i+1}_{elem}"].append(layer_attributes[elem].__name__)
                                                else:
                                                    results[f"Layer{i+1}_{elem}"].append(layer_attributes[elem])
                                        results["LearningRate"].append(learning_rate)
                                        results["Optimizer"].append(optimizer.__name__)
                                        results["Loss"].append(loss.__name__)
                                        results["Batch"].append(batch_size)
                                        results["EpisodeLength"].append(episode_length)

                                        game_over, points = score_model(snake_model)
                                        results["AvgTurnsGG"] = game_over
                                        results["AvgPoints"] = points                                        
    return pd.DataFrame(results)


# Sample format for the different parameters to test for the test_models function above. Note, for layer structure, all of them must have the same number of layers and layer attributes (or else we can't construct the data frame because of different column size)
test_parameters = {
    "layers_struct": [
        [
            [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*20, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*10, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*5, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 4, "activation": tf.keras.activations.sigmoid}]
        ]
    ],
    "learning_rates": [1],
    "optimizers": [tf.keras.optimizers.SGD], 
    "losses": [tf.keras.losses.CategoricalCrossentropy],
    "batches": [100],
    "episode_lengths": [100], 
    "num_episodes_list": [100],
    "rewards": [[-1,1,-0.152]],
    "gammas": [0.7]
} 

# A function that calls on generates the results from varies test parameters and stores the results in a csv file
def test_parameters_to_csv(test_parameters):
    test_results = test_models(**test_parameters)

    # add the results to previously computed ones
    #previous_result = pd.read_csv("Results/TrainingResults.csv")
    #test_results = pd.concat([test_results, previous_result])

    # subset = list(test_results.columns)
    # subset.remove("SuccessRate")
    # subset.remove("WinRate")
    # test_results = test_results.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

    # save the results
    test_results.to_csv("Results/TrainingResults.csv", index=False)

