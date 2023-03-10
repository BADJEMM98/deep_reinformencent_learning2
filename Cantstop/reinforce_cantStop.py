from agents.Deep_Single_Agent_Env import DeepSingleAgentEnv
from Cantstop.cant_stop_multiplayers import CantStopEnv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random
import os

episodes: int = 10_000
class REINFORCEAgent():
    def __init__(
            self,
            env: DeepSingleAgentEnv,
            gamma: float = 0.99,
            alpha: float = 0.01,

    ):
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.pi = self.create_model()

    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input((self.env.state_dim(),)))
        model.add(tf.keras.layers.Dense(self.env.max_action_count(),
                                        activation=tf.keras.activations.softmax,
                                        use_bias=True
                                        ))
        return model

    def clear_buffer(self):
        self.states_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()

    def store_transition(self, transition):
        self.states_buffer.append(transition[0])
        self.actions_buffer.append(transition[1])
        self.rewards_buffer.append(transition[2])

    def train(self):
        G = 0.0
        for t in reversed(range(0, len(self.states_buffer))):
            G = self.rewards_buffer[t] + self.gamma * G

            with tf.GradientTape() as tape:
                pi_s_a_t = self.pi(np.array([self.states_buffer[t]]).reshape(-1, self.env.state_dim()))[0][
                    self.actions_buffer[t]]
                log_pi_s_a_t = tf.math.log(pi_s_a_t)

            grads = tape.gradient(log_pi_s_a_t, self.pi.trainable_variables)

            for (v, g) in zip(self.pi.trainable_variables, grads):
                if g is not None:
                    v.assign_add(self.alpha * (self.gamma ** t) * G * g)


def reinforce(env: DeepSingleAgentEnv, episodes=episodes):
    agent = REINFORCEAgent(env=env)
    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    for _ in tqdm(range(episodes)):

        if env.is_game_over():

            agent.train()

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
            agent.clear_buffer()
            step = 0

        curr_state = env.state_description()
        actions = env.available_actions_ids()

        # Get action from Q table
        pi_s = agent.pi(np.array(curr_state).reshape(-1, env.state_dim()))[0].numpy()
        #print("pi_s ",pi_s)
        allowed_pi_s = pi_s[actions]
        # print("allowed_pis ", allowed_pi_s)
        sum_allowed_pi_s = np.sum(allowed_pi_s)
        if sum_allowed_pi_s == 0.0:
            probs = np.ones((len(actions),)) * 1.0 / (len(actions))
        else:
            probs = allowed_pi_s / sum_allowed_pi_s
        #print(actions)
        action = np.random.choice(actions, p=probs)

        old_score = env.score()
        print(old_score)
        env.act_with_action_id(action)
        new_score = env.score()
        reward = new_score - old_score

        agent.store_transition((curr_state, action, reward))

        step += 1

    return agent.pi, ema_score_progress, ema_nb_steps_progress


if __name__ == "__main__":
    n=4
    curr_dir = os.path.dirname(os.path.realpath('__file__'))
    workspace = os.path.dirname(curr_dir)

    model_name = "reinforce"
    pi_model, scores, steps = reinforce(env=CantStopEnv(n))
    print(pi_model.get_weights())
    pi_model.save(os.path.join('models', "canStop", f"{model_name}.h5"))
    plt.plot(scores)
    plt.show()

    plt.plot(steps)
    plt.show()
    print('Score moyen :', sum(scores))
    print('Longueur moyenne :', sum(steps))
