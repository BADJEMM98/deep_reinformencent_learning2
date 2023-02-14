import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

import sys
#sys.path.append('/')

from env.deep_single_agent_env import DeepSingleAgentEnv
from env.env_grid_world import GridWorld
from env.env_line_world import LineWorld

def REINFORCE(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                                 gamma: float = 0.99,
                                 alpha: float = 0.1):
    pi = tf.keras.models.Sequential()
    pi.add(tf.keras.layers.Dense(env.max_action_count(),
                                activation=tf.keras.activations.softmax,
                                use_bias=True
                                ))

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    episode_states_buffer = []
    episode_actions_buffer = []
    episode_rewards_buffer = []

    for _ in tqdm(range(max_iter_count)):
        if env.is_game_over():
            ### TRAINING TIME !!!
            G = 0.0

            for t in reversed(range(0, len(episode_states_buffer))):
                G = episode_rewards_buffer[t] + gamma * G

                with tf.GradientTape() as tape:
                    pi_s_a_t = pi(np.array([episode_states_buffer[t]]))[0][episode_actions_buffer[t]]
                    log_pi_s_a_t = tf.math.log(pi_s_a_t)

                grads = tape.gradient(log_pi_s_a_t, pi.trainable_variables)

                for (v, g) in zip(pi.trainable_variables, grads):
                    if g is not None:
                        v.assign_add(alpha * (gamma ** t) * G * g)

            if first_episode:
                ema_score = env.score()
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score
                ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps
                ema_score_progress.append(ema_score)
                ema_nb_steps_progress.append(ema_nb_steps)

            env.reset()
            episode_states_buffer.clear()
            episode_actions_buffer.clear()
            episode_rewards_buffer.clear()
            step = 0

        s = env.state_description()

        episode_states_buffer.append(s)

        aa = env.available_actions_ids()

        pi_s = pi(np.array([s]))[0].numpy()
        allowed_pi_s = pi_s[aa]
        sum_allowed_pi_s = np.sum(allowed_pi_s)
        if sum_allowed_pi_s == 0.0:
            probs = np.ones((len(aa),)) * 1.0 / (len(aa))
        else:
            probs = allowed_pi_s / sum_allowed_pi_s

        a = np.random.choice(aa, p=probs)

        episode_actions_buffer.append(a)

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        episode_rewards_buffer.append(r)

        step += 1
    pi.save('models/reinforcebaseline.h5')
    return pi, ema_score_progress, ema_nb_steps_progress

def REINFORCE_with_mean_baseline(env, iter_count):

    pi = tf.keras.models.load_model(os.path.join("models", "","reinforcebaseline.h5"))

    pi, scores, steps = REINFORCE(env, max_iter_count=iter_count)
    print(pi.weights)
    #print("Score moyen")
    #print(np.mean(scores))
    #print("longeur moyenne step")
    #print(np.mean(steps))
    print(f"Reinforce With baseline - Grid World - iter count : {iter_count}")
    print(f"Reinforce With baseline  - Grid World - mean_scores : {np.mean(scores)}")
    print(f"Reinforce With baseline - Grid World  - mean_steps : {np.mean(steps)}")
    plt.plot(scores)
    plt.savefig("graphics/reinforceBaseLineLWScores.png")
    plt.show()
    plt.plot(steps)
    plt.savefig("graphics/reinforceBaseLineLWSteps.png")

    #plt.show()

REINFORCE_with_mean_baseline(GridWorld(5,5),10000)
#REINFORCE_with_mean_baseline(LineWorld(5),1000)