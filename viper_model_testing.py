from Agent_Snake import (
    Memory,
    create_snake_model,
    train_model,
    score_model
)
from snake import SnakeGame
from game_options import game_options
import os
import pandas as pd
from collections import defaultdict
from itertools import product
import math

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


def dict_product(**keyed_iterables):
    for value_combination in product(*keyed_iterables.values()):
        yield dict(zip(keyed_iterables.keys(), value_combination))


# A function that creates a DataFrame containing the results of trying out various different parameters into the model
# training_params must be a dictionary of iterables where the keys are arguments to train_model
def test_models(layers_struct, **training_params):
    results = defaultdict(list)

    for layers in layers_struct:
        for params in dict_product(**training_params):
            print(f"Creating model with layers {layers}")
            snake_model = create_snake_model(layers)
            train_model(snake_model, **params)

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
            results["LearningRate"].append(params["learning_rate"])
            results["Optimizer"].append(params["optimizer"].__name__)
            results["Batch"].append(params["batch_size"])
            results["EpisodeLength"].append(params["episode_length"])
            results["NumEpisodes"].append(params["num_episodes"])
            results["GameOverReward"].append(params["rewards"][0])
            results["UsualReward"].append(params["rewards"][1])
            results["TimeOutReward"].append(params["rewards"][2])
            results["Gamma"].append(params["gamma"])
            game_over, points = score_model(snake_model)
            results["AvgTurnsGG"].append(game_over)
            results["AvgPoints"].append(points)
    return pd.DataFrame(results)


# Sample format for the different parameters to test for the test_models function above. Note, for layer structure, all of them must have the same number of layers and layer attributes (or else we can't construct the data frame because of different column size)
test_parameters = {
    "layers_struct": [
        [
            [tf.keras.layers.Reshape, {"target_shape": (opts["width"],opts["height"],1)}],
            [tf.keras.layers.Conv2D, {"filters": 48, "kernel_size": 3, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Flatten, {}],
            [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*10, "activation": tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 4, "activation": None}]
        ]
    ],
    "learning_rate": [1,0.1,5],
    "optimizer": [tf.keras.optimizers.SGD],
    "batch_size": [100],
    "episode_length": [500],
    "num_episodes": [100],
    "rewards": [[-1,1,-0.1520]],
    "gamma": [0.75]
}

# Other possible layer structure:
# [
#     [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*20, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*10, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": opts["width"]*opts["height"]*5, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Dense, {"units": 4, "activation": tf.keras.activations.sigmoid}]
# ]
# [
#     [tf.keras.layers.Reshape, {"target_shape": (opts["width"],opts["height"],1)}],
#     [tf.keras.layers.Conv2D, {"filters": 48, "kernel_size": 4, "activation":  tf.keras.activations.relu}],
#     [tf.keras.layers.Flatten, {}],
#     [tf.keras.layers.Dense, {"units": 4, "activation": None}]
# ]

# A function that calls on generates the results from varies test parameters and stores the results in a csv file
def test_parameters_to_csv(test_parameters):
    test_results = test_models(**test_parameters)

    # add the results to previously computed ones
    previous_result = pd.read_csv("Results/TrainingResults.csv")
    test_results = pd.concat([test_results, previous_result])

    # subset = list(test_results.columns)
    # subset.remove("SuccessRate")
    # subset.remove("WinRate")
    # test_results = test_results.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

    # save the results
    test_results.to_csv("Results/TrainingResults.csv", index=False)


# A function that reads in a row from the TrainingResults.csv file, and outputs the corresponding model
# There must be a better way to input the layer type than how it's done here. However, tf.keras.layers is not callable...
def csv_to_model(row):
    layer_type = {
        "Dense": tf.keras.layers.Dense,
        "Conv2D": tf.keras.layers.Conv2D,
        "Flatten": tf.keras.layers.Flatten,
        "Reshape": tf.keras.layers.Reshape
    }
    Optimize = {
        "SGD": tf.keras.optimizers.SGD,
        "Adam": tf.keras.optimizers.Adam
    }
    layers = []
    k = 1
    # reconstruct all the parameters for the layers
    while f"Layer{k}_Name" in row.index and str(row[f"Layer{k}_Name"]) != "None" and str(row[f"Layer{k}_Name"]) != "nan":
        columns = [col for col in row.index if col.startswith(f"Layer{k}")]
        length = len(f"Layer{k}_")
        parameters = {}
        for attribute in columns:
            param = attribute[length:]
            if param == "Name" or row[attribute] is None or pd.isnull(row[attribute]):
                continue
            elif param == "kernel_size":
                parameters[param] = int(row[attribute])
            elif param == "target_shape":
                a,b,c = row[attribute].split(',')
                parameters[param] = (int(a[1:]),int(b),int(c[:-1]))
            else:
                if row[attribute] == "None":
                    parameters[param] = None
                else:
                    parameters[param] = row[attribute]
        layers.append([layer_type[row[f"Layer{k}_Name"]] , parameters])
        k += 1
    print(f"Creating model with layers {layers}")
    model = create_snake_model(layers)
    train_model(model, row["LearningRate"], Optimize[row["Optimizer"]], row["Batch"], row["EpisodeLength"], row["NumEpisodes"],
    [row["GameOverReward"], row["UsualReward"], row["TimeOutReward"]], row["Gamma"])
    return model


# A function that outputs the the model with the highest value in a particular column in the TrainingResults.csv file
# By default, this is the AvgPoints column
def best_model(type = "AvgPoints"):
    # import the previously computed results
    results = pd.read_csv("Results/TrainingResults.csv")

    row = results.iloc[results[type].idxmax()]
    print(row)
    model = csv_to_model(row)
    game_over, points = score_model(model)
    return model
