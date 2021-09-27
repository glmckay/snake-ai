# snake-ai
An AI that plays the Snake Game built with neural networks with reinforcement learning. 

* Download the repository
* On the command line, go to the directory of the download
* Type `python -i Main.py`


# About the network
In this section, we will explain how we train our networks. For instructions of how to play with the game, and see our models in action, skip to the next section.

First, we have the model play a full game (up to some fixed number of moves, because the AI becomes really good at not losing pretty fast). While playing the game, we store every more the AI made. If the AI loses at a particular step, then the reward for making that move is some GameOverReward (eg. -1). Conversely, if the AI eats a fruit by making a particular move, that move has an associated reward of UsualReward (eg. 1). If the AI hasn't gotten a fruit in more than 20 turns, every move it makes before it eats another fruit has associated reward TimeOutReward (eg. -0.15). 

Since the moves that were made leading up to a fruit being eaten should be rewarded (at a higher rate the closer the move is made to the fruit-eating move), we use a discounted reward instead, which is calculated based on the following formula:

><img src="https://render.githubusercontent.com/render/math?math=\text{discounted reward of move k} = \text{normalization}(\sum_{i=0}^{k} \gamma^{k-i} \left(\text{reward of move i})\right)">

for some gamma close to but smaller than 1 (eg. 0.7). The idea is that a move made just before picking up a fruit would then have reward 0.7. The move before that 0.7^2, and before that 0.7^3.... This way, we are properly encouraging all moves leading up to picking up a fruit as well. 


# To play a game of Snake:
Type in the function 
>play()


or 
>play(width= , height= , num_fruits= )

The default is currently set to width = 10, height = 10, num_fruits =  1. 

To play the game, simplying press the keys w,a,s,d for up, left, down and right respectively. 


# Model Creation
To create a trained model with parameters that give the highest successrate (out of all the parameters that we tested and stored in Results/TrainingResults.csv), run
>model = best_model()

To create a **trained** model, run 
>model = csv_to_model(**parameter)

where a sample parameter is the following:
```Python
{
    "layers_struct": [
        [
            [tf.keras.layers.Reshape, {"target_shape": (opts["width"],opts["height"],1)}],
            [tf.keras.layers.Conv2D, {"filters": 48, "kernel_size": 4, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Flatten, {}],
            [tf.keras.layers.Dense, {"units": 4, "activation": tf.keras.activations.sigmoid}]
        ]
    ],
    "learning_rates": [1],
    "optimizers": [tf.keras.optimizers.SGD], 
    "batches": [100],
    "episode_lengths": [100], 
    "num_episodes_list": [100],
    "rewards": [[-1,1,-0.152]],
    "gammas": [0.7]
} 
```


# Model Prediction
To have the model play the game, run
>play(model)
