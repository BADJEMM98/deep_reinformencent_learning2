from othello_env import OthelloEnv
from tqdm import tqdm
import os
import tensorflow as tf
import keras
import numpy as np

def run_ct_n_games_and_return_mean_score(gamescount: int) -> float:
    env = OthelloEnv()
    pi = keras.models.load_model(os.path.join("Othello",'models',"ppo_one_step_sctor_critic.h5"))
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
            pi_s = pi([np.array([s]), np.array([mask])])[0].numpy()

            allowed_pi_s = pi_s[aa]
            sum_allowed_pi_s = np.sum(allowed_pi_s)
            if sum_allowed_pi_s == 0.0:
                probs = np.ones((len(aa),)) * 1.0 / (len(aa))
            else:
                probs = allowed_pi_s / sum_allowed_pi_s

            action_id = np.random.choice(aa, p=probs)

            assert(not env.is_game_over())
            i = action_id // env.board_size
            j = action_id % env.board_size
            assert (env.is_legal_move((i, j)))
            env.apply_move((i,j))

            

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
        # if i%100000 == 0:
        #     mean_scores.append(total/100000)
        #     mean_steps.append(sum_steps/100000)


    print(f"Reinforce With baseline - wins : {wins}, losses : {losses}")
    print(f"Reinforce With baseline - mean_scores : {mean_scores}")
    print(f"Reinforce With baseline - mean_steps : {mean_steps}")
    return mean_scores, mean_steps

if __name__ == '__main__':
    mean_score = run_ct_n_games_and_return_mean_score(10000)
    