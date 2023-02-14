from deep_single_agent_env import DeepSingleAgentEnv
from othello_env import OthelloEnv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random
import os

def REINFORCE(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                                 gamma: float = 0.99,
                                 alpha: float = 0.001):
    pi = tf.keras.models.Sequential()
    pi.add(tf.keras.layers.Input((env.state_dim(),)))
    pi.add(tf.keras.layers.Dense(128,
                                activation=tf.keras.activations.relu,
                                use_bias=True
                                ))
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
    return pi, ema_score_progress, ema_nb_steps_progress


pi, scores, steps = REINFORCE(OthelloEnv(), max_iter_count=10000)
print(pi.weights)
pi.save(os.path.join("Othello",'models',"reinforce.h5"))
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()


# episodes: int = 10_000

# class REINFORCEAgent():
#     def __init__(
#         self,
#         env:DeepSingleAgentEnv,
#         gamma: float = 0.99,
#         alpha: float = 0.01,
        
#     ):
#         self.states_buffer = []
#         self.actions_buffer = []
#         self.rewards_buffer = []
#         self.gamma = gamma
#         self.alpha = alpha
#         self.env = env
#         self.pi = self.create_model()
    
#     def create_model(self):
#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Input((self.env.state_dim(),)))
#         model.add(tf.keras.layers.Dense(self.env.max_action_count(),
#                                     activation=tf.keras.activations.softmax,
#                                     use_bias=True
#                                     ))
#         return model

#     def clear_buffer(self):
#         self.states_buffer.clear()
#         self.actions_buffer.clear()
#         self.rewards_buffer.clear()
    
#     def store_transition(self,transition):
#         self.states_buffer.append(transition[0])
#         self.actions_buffer.append(transition[1])
#         self.rewards_buffer.append(transition[2])

#     def train(self):
#         G = 0.0
#         for t in reversed(range(0, len(self.states_buffer))):
#             G = self.rewards_buffer[t] + self.gamma * G

#             with tf.GradientTape() as tape:
#                 pi_s_a_t = self.pi(np.array([self.states_buffer[t]]).reshape(-1,self.env.state_dim()))[0][self.actions_buffer[t]]
#                 log_pi_s_a_t = tf.math.log(pi_s_a_t)

#             grads = tape.gradient(log_pi_s_a_t, self.pi.trainable_variables)

#             for (v, g) in zip(self.pi.trainable_variables, grads):
#                 if g is not None:
#                     v.assign_add(self.alpha * (self.gamma ** t) * G * g)


# def reinforce(env: DeepSingleAgentEnv,episodes = episodes):

#     agent = REINFORCEAgent(env=env)
#     ema_score = 0.0
#     ema_nb_steps = 0.0
#     first_episode = True

#     step = 0
#     ema_score_progress = []
#     ema_nb_steps_progress = []

#     for _ in tqdm(range(episodes)):

#         if env.is_game_over():
            
#             agent.train()

#             if first_episode:
#                 ema_score = env.score()
#                 ema_nb_steps = step
#                 first_episode = False
#             else:
#                 ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score
#                 ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps
#                 ema_score_progress.append(ema_score)
#                 ema_nb_steps_progress.append(ema_nb_steps)
#             env.reset()
#             agent.clear_buffer()
#             step = 0

#         curr_state = env.state_description()
#         actions = env.available_actions_ids()

#         # Get action from Q table
#         pi_s = agent.pi(np.array(curr_state).reshape(-1,env.state_dim()))[0].numpy()
#         # print("pi_s ",pi_s)
#         allowed_pi_s = pi_s[actions]
#         # print("allowed_pis ", allowed_pi_s)
#         sum_allowed_pi_s = np.sum(allowed_pi_s)
#         if sum_allowed_pi_s == 0.0:
#             probs = np.ones((len(actions),)) * 1.0 / (len(actions))
#         else:
#             probs = allowed_pi_s / sum_allowed_pi_s
        

#         action = np.random.choice(actions, p=probs)

#         old_score = env.score()
#         env.act_with_action_id(action)
#         new_score = env.score()
#         reward = new_score - old_score

#         agent.store_transition((curr_state,action,reward))

#         step += 1

#     return agent.pi, ema_score_progress, ema_nb_steps_progress

# if __name__ == "__main__":

#     curr_dir = os.path.dirname(os.path.realpath('__file__'))
#     workspace = os.path.dirname(curr_dir)

#     model_name = "reinforce"
#     pi_model, scores, steps = reinforce(env=OthelloEnv())
#     print(pi_model.get_weights())
#     pi_model.save(os.path.join('models',"Othello",f"{model_name}.h5"))
#     plt.plot(scores)
#     plt.show()
#     # plt.savefig(os.path.join('models',"Othello",f"{model_name}_scores.png"))
#     plt.plot(steps)
#     plt.show()  
#     # plt.savefig(os.path.join('models',"Othello",f"{model_name}_steps.png"))

            