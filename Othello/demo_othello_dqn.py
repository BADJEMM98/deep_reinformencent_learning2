from othello_env import OthelloEnv
from tqdm import tqdm
import os
import tensorflow as tf
import keras
import numpy as np

def run_ct_n_games_and_return_mean_score(gamescount: int) -> float:
    env = OthelloEnv()
    q = keras.models.load_model(os.path.join("models", "Othello","dqn.h5"))
    total = 0.0
    wins = 0
    losses = 0
    sum_steps = 0
    mean_scores = []
    mean_steps = []
    # draws = 0

    
    for i in tqdm(range(1,gamescount+1)):
        env.reset()
        steps = 0
        while not env.is_game_over():
            steps += 1
            aa = env.available_actions_ids()
            mask = np.zeros((env.max_action_count(),))
            mask[aa] = 1.0
            s = env.state_description()

            q_pred = q(np.array([s]))[0]
            chosen_a = aa[np.argmax(q_pred.numpy()[aa])]

            env.act_with_action_id(chosen_a)

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1

        total += env.blkplayer.score
        sum_steps += steps
        if i == 1000:
            mean_scores.append(total/1000)
            mean_steps.append(sum_steps/1000)
        if i == 10000:
            mean_scores.append(total/10000)
            mean_steps.append(sum_steps/10000)
        # if i%100000 == 0:
        #     mean_scores.append(total/100000)
        #     mean_steps.append(sum_steps/100000)


    print(f"DQN - wins : {wins}, losses : {losses}")
    print(f"DQN - mean_scores : {mean_scores}")
    print(f"DQN - mean_steps : {mean_steps}")
    return mean_scores, mean_steps

if __name__ == '__main__':
    mean_score = run_ct_n_games_and_return_mean_score(10000)
    