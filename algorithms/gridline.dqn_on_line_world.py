import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm


from env.deep_single_agent_env import DeepSingleAgentEnv
from env.env_grid_world import GridWorld
from env.env_line_world import LineWorld


def dqn(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                                 gamma: float = 0.99,
                                 alpha: float = 0.1,
                                 epsilon: float = 0.2):
    q = tf.keras.models.Sequential()
    q.add(tf.keras.layers.Dense(env.max_action_count(),
                                activation=tf.keras.activations.linear,
                                use_bias=True
                                ))

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    opt = tf.keras.optimizers.SGD(alpha)

    for _ in tqdm(range(max_iter_count)):
        if env.is_game_over():
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
            step = 0

        s = env.state_description()
        aa = env.available_actions_ids()

        q_pred = q(np.array([s]))[0]
        if np.random.random() < epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax(q_pred.numpy()[aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_description()
        aa_p = env.available_actions_ids()

        if env.is_game_over():
            y = r
        else:
            q_pred_p = q(np.array([s_p]))[0]
            max_q_s_p = np.max(q_pred_p.numpy()[aa_p])
            y = r + gamma * max_q_s_p

        with tf.GradientTape() as tape:
            q_s_a = q(np.array([s]))[0][a]
            loss = tf.reduce_mean((y - q_s_a) ** 2)

        grads = tape.gradient(loss, q.trainable_variables)
        opt.apply_gradients(zip(grads, q.trainable_variables))

        step += 1
    q.save('models/dqn.h5')
    return q, ema_score_progress, ema_nb_steps_progress

def dqn_algo(env, iter_count):
    q, scores, steps = dqn(env, epsilon=0.01, max_iter_count=iter_count)
    q = tf.keras.models.load_model(os.path.join("models", "","dqn.h5"))
    print(q.weights)
    #print("Score moyen")
    #print(np.mean(scores))
    #print("longeur moyenne step")
    #print(np.mean(steps))
    print(f"DQN- Grid World - iter count : {iter_count}")
    print(f"DQN  - Grid World - mean_scores : {np.mean(scores)}")
    print(f"DQN - Grid World  - mean_steps : {np.mean(steps)}")
    plt.plot(scores)
    plt.savefig("graphics/dqnscore.png")
    plt.show()
    plt.plot(steps)
    plt.savefig("graphics/dqsteps.png")


dqn_algo(GridWorld(5,5), 10000)
#dqn_algo(LineWorld(10), 10000)