from othello_env import OthelloEnv
from tqdm import tqdm
import os
import tensorflow as tf
import keras
import numpy as np
from random import random

def run_ct_n_games_and_return_mean_score(gamescount: int) -> float:
    env = OthelloEnv()
    
    total = 0.0
    wins = 0
    losses = 0
    sum_steps = 0
    mean_scores = []
    mean_steps = []

    
    for i in tqdm(range(1,gamescount+1)):
        env.reset()
        steps = 0
        while not env.is_game_over():
            steps += 1
            aa = env.available_actions_ids()
            chosen_a = np.random.choice(aa)
            env.act_with_action_id(chosen_a)

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1

        total += env.blkplayer.score
        sum_steps += steps
        if i == 1000:
            mean_scores.append(total/i)
            mean_steps.append(sum_steps/i)
        if i == 10000:
            mean_scores.append(total/i)
            mean_steps.append(sum_steps/i)
        if i%100000 == 0:
            mean_scores.append(total/100000)
            mean_steps.append(sum_steps/100000)


    print(f"Random agent - wins : {wins}, losses : {losses}")
    print(f"Random agent - mean_scores : {mean_scores}")
    print(f"Random agent - mean_steps : {mean_steps}")
    return mean_scores, mean_steps

if __name__ == '__main__':
    mean_score = run_ct_n_games_and_return_mean_score(10000)
    