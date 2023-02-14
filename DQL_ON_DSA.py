import os
import random
from collections import deque

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

from env.cant_stop_multiplayers import CantStopMultiplayer

episodes: int = 10_000
epsilon: float = 0.2
n_players = 4

class DQNAgent():
    def __init__(
            self,
            env: CantStopMultiplayer,
            gamma: float = 0.99,
            alpha: float = 1e-3,

    ):
        self.min_replay_memory_size = 128
        self.max_replay_memory_size = 1000
        self.replay_memory = deque(maxlen=self.max_replay_memory_size)
        self.target_update_counter = 0
        self.update_target_every = 20
        self.gamma = gamma
        self.minibacth_size = 128
        self.alpha = alpha
        self.env = env
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        nb_actions = self.env.action_space.n
        nb_obs = self.env.observation_space.n

        input_layer = Input(shape=(1, nb_obs,))
        x = Flatten()(input_layer)
        x = Dense(50, activation='relu')(x)
        output_layer = Dense(nb_actions, activation='linear')(x)
        print(output_layer, "output_layer", input_layer)
        model = Model(input_layer, output_layer)

        model.compile(loss='mse',optimizer=Adam(learning_rate=self.alpha) )
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, steps):

        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        minibatch = random.sample(self.replay_memory, self.minibacth_size)
        curr_states = np.array([transition[0] for transition in minibatch])
        curr_qs_list = self.model.predict(curr_states.reshape(-1, self.env.state_dim()), verbose=0)

        new_curr_states = np.array([transition[3] for transition in minibatch])
        future_qs = self.target_model.predict(new_curr_states.reshape(-1, self.env.state), verbose=0)

        X = []
        y = []

        for i, (curr_state, action, reward, new_curr_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[i])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            curr_qs = curr_qs_list[i]
            curr_qs[action] = new_q

            X.append(curr_state)
            y.append(curr_qs)

        self.model.fit(np.array(X).reshape(-1, self.env.observation_space),
                       np.array(y).reshape(-1, self.env.action_space),
                       batch_size=self.minibacth_size, verbose=0)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



    def get_qs(self, state):
        return self.model.predict(state, verbose=0)


def dqn(env: CantStopMultiplayer, episodes=episodes, epsilon=epsilon):
    agent = DQNAgent(env=env)

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    for _ in tqdm(range(episodes)):

        if env.is_game_over():
            if first_episode:
                ema_score = env.board
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score
                ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps
                ema_score_progress.append(ema_score)
                ema_nb_steps_progress.append(ema_nb_steps)
            env.reset()
            step = 0

        env.reset()
        curr_state = env.state
        actions = env.available_actions_ids()

        if np.random.random() > epsilon and not first_episode:
            # Get action from Q table
            weigths_action_sorted = np.argsort(agent.get_qs(curr_state.reshape(-1, env.state_dim())))[0]
            for i in range(len(weigths_action_sorted) - 1, -1, -1):
                if weigths_action_sorted[i] in actions:
                    action = weigths_action_sorted[i]
                    break
        else:
            # Get random action 
            action = random.randint(2, 12)

        old_score = env.score()
        env.act_with_action_id(action)
        new_score = env.score()
        reward = new_score - old_score
        done = env.is_game_over()


        next_curr_state = env.state
        next_actions = env.available_actions_ids()

        agent.update_replay_memory((curr_state, action, reward, env.state, env.is_game_over()))
        agent.train(env.is_game_over(), step)

        curr_state = env.state_description()

        step += 1

    return agent.model, agent.target_model, ema_score_progress, ema_nb_steps_progress


if __name__ == "__main__":
    n=4
    curr_dir = os.path.dirname(os.path.realpath('__file__'))
    workspace = os.path.dirname(curr_dir)

    checkpoint_name = f"weights/dnq_2.h5"
    q_model, q_target_model, scores, steps = dqn(env=CantStopMultiplayer(n))
    print(q_model.get_weights())
    q_model.save(os.path.join('models', checkpoint_name))
    q_target_model.save(os.path.join('models', "CantStop", "target_model.h5"))
    plt.plot(scores)
    plt.savefig(os.path.join('models', "CantStop", "dqn_scores.png"))
    plt.plot(steps)
    plt.savefig(os.path.join('models', "CantStop", "dqn_steps.png"))
