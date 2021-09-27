import os
import numpy as np
from typing import Optional, Tuple
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  
from viper_model_testing import test_parameters_to_csv, best_model, test_parameters_to_csv
from snake import SnakeGame
from snake_terminal import play_game
from game_options import game_options, change_options

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def play(model: Optional["tf.keras.Model"] = None):
    game = SnakeGame(game_options["width"], game_options["height"], game_options["num_fruits"])
    play_game(game, model)