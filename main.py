import os
import numpy as np
from typing import Optional, Tuple
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from viper_model_testing import test_parameters_to_csv, best_model
from snake import SnakeGame
from snake_terminal import play_game
from game_options import game_options, change_options
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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


def play(model: Optional["tf.keras.Model"] = None, **override_game_opts):
    # with python 3.9, this could be SnakeGame(game_options | override_game_opts)
    opts = game_options.copy()
    opts.update(override_game_opts)
    game = SnakeGame(**opts)
    play_game(game, model)