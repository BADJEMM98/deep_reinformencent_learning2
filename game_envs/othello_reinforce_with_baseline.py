from deep_single_agent_env import DeepSingleAgentEnv
from othello_env import OthelloEnv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os



def REINFORCE_with_learned_baseline(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                                    gamma: float = 0.99,
                                    alpha_pi: float = 0.001,
                                    alpha_v: float = 0.005):
    pi_input_state_desc = tf.keras.layers.Input((env.state_dim(),))
    pi_input_mask = tf.keras.layers.Input((env.max_action_count(),))

    hidden_tensor = pi_input_state_desc
    for _ in range(3):
        hidden_tensor = tf.keras.layers.Dense(128,
                                              activation=tf.keras.activations.tanh,
                                              use_bias=True
                                              )(hidden_tensor)

    output_tensor = tf.keras.layers.Dense(env.max_action_count(),
                                          activation=tf.keras.activations.linear,
                                          use_bias=True
                                          )(hidden_tensor)

    output_probs = tf.keras.layers.Softmax()(output_tensor, pi_input_mask)

    pi = tf.keras.models.Model([pi_input_state_desc, pi_input_mask], output_probs)

    v = tf.keras.models.Sequential()
    for _ in range(3):
        v.add(tf.keras.layers.Dense(128,
                                    activation=tf.keras.activations.tanh,
                                    use_bias=True
                                    ))
    v.add(tf.keras.layers.Dense(1,
                                activation=tf.keras.activations.linear,
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
    episode_mask_buffer = []

    for _ in tqdm(range(max_iter_count)):
        if env.is_game_over():
            ### TRAINING TIME !!!
            G = 0.0

            for t in reversed(range(0, len(episode_states_buffer))):
                G = episode_rewards_buffer[t] + gamma * G

                with tf.GradientTape() as tape_v:
                    v_s_pred = v(np.array([episode_states_buffer[t]]))[0][0]

                delta = G - tf.constant(v_s_pred)

                grad_v_s_pred = tape_v.gradient(v_s_pred, v.trainable_variables)
                for (var, grad) in zip(v.trainable_variables, grad_v_s_pred):
                    if grad is not None:
                        var.assign_add(alpha_v * delta * grad)

                with tf.GradientTape() as tape_pi:
                    pi_s_a_t = pi([
                        np.array([episode_states_buffer[t]]),
                        np.array([episode_mask_buffer[t]])
                    ])[0][episode_actions_buffer[t]]
                    log_pi_s_a_t = tf.math.log(pi_s_a_t)

                grads = tape_pi.gradient(log_pi_s_a_t, pi.trainable_variables)

                for (var, grad) in zip(pi.trainable_variables, grads):
                    if grad is not None:
                        var.assign_add(alpha_pi * (gamma ** t) * delta * grad)

            if first_episode:
                ema_score = env.score()
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.99) * env.score() + 0.99 * ema_score
                ema_nb_steps = (1 - 0.99) * step + 0.99 * ema_nb_steps
                ema_score_progress.append(ema_score)
                ema_nb_steps_progress.append(ema_nb_steps)

            env.reset()
            episode_states_buffer.clear()
            episode_actions_buffer.clear()
            episode_rewards_buffer.clear()
            episode_mask_buffer.clear()
            step = 0

        s = env.state_description()

        episode_states_buffer.append(s)

        aa = env.available_actions_ids()

        mask = np.zeros((env.max_action_count(),))
        mask[aa] = 1.0

        episode_mask_buffer.append(mask)

        pi_s = pi([np.array([s]), np.array([mask])])[0].numpy()
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
    return pi, v, ema_score_progress, ema_nb_steps_progress


pi, v, scores, steps = REINFORCE_with_learned_baseline(OthelloEnv(), max_iter_count=10000)
print(pi.weights)
pi.save(os.path.join("Othello",'models',"reinforce_with_baseline.h5"))
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()
