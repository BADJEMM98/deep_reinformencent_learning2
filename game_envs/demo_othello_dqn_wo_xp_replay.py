from deep_single_agent_env import DeepSingleAgentEnv
from othello_env import OthelloEnv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random
import os

episodes: int = 10_000
epsilon: float = 0.2

class DQNAgent():
    def __init__(
        self,
        env:DeepSingleAgentEnv,
        gamma: float = 0.99,
        alpha: float = 0.1,
        
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
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input((self.env.state_dim(),)))
        model.add(tf.keras.layers.Dense(self.env.max_action_count(),
                                    activation=tf.keras.activations.linear,
                                    use_bias=True
                                    ))
        model.compile(
            loss= 'mse',
            optimizer= tf.keras.optimizers.SGD(learning_rate=self.alpha)
        )
        return model

    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)

    def train(self,terminal_state,steps):
        if len(self.replay_memory)< self.min_replay_memory_size:
            return
        minibatch = random.sample(self.replay_memory,self.minibacth_size)
        curr_states = np.array([transition[0] for transition in minibatch])
        curr_qs_list = self.model.predict(curr_states.reshape(-1,self.env.state_dim()),verbose=0)


        new_curr_states = np.array([transition[3] for transition in minibatch])
        future_qs = self.target_model.predict(new_curr_states.reshape(-1,self.env.state_dim()),verbose=0)

        X = []
        y = []

        for i, (curr_state,action,reward,new_curr_state,done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[i])
                new_q = reward +self.gamma *max_future_q
            else:
                new_q = reward

            curr_qs = curr_qs_list[i]
            curr_qs[action] = new_q

            X.append(curr_state)
            y.append(curr_qs)

        self.model.fit(np.array(X).reshape(-1,self.env.state_dim()),
                       np.array(y).reshape(-1,self.env.max_action_count()), 
                       batch_size=self.minibacth_size,verbose=0)

        if terminal_state:
            self.target_update_counter+=1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
    def get_qs(self,state):
        return self.model.predict(state,verbose=0)


def dqn(env: DeepSingleAgentEnv,episodes = episodes,epsilon=epsilon):

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

        env.reset()
        curr_state = env.state_description()
        actions = env.available_actions_ids()

        if np.random.random() > epsilon and not first_episode:
                # Get action from Q table
                weigths_action_sorted = np.argsort(agent.get_qs(curr_state.reshape(-1,env.state_dim())))[0]
                for i in range(len(weigths_action_sorted)-1,-1,-1):
                    if weigths_action_sorted[i] in actions:
                        action = weigths_action_sorted[i]
                        break
        else:
            # Get random action 
            action = random.choice(actions)

        old_score = env.score()
        env.act_with_action_id(action)
        new_score = env.score()
        reward = new_score - old_score
        done = env.is_game_over()

        next_curr_state = env.state_description()
        next_actions = env.available_actions_ids()

        agent.update_replay_memory((curr_state,action,reward,env.state_description(),env.is_game_over()))
        agent.train(env.is_game_over(),step)
            
        curr_state = env.state_description()

        step += 1

    return agent.model, agent.target_model, ema_score_progress, ema_nb_steps_progress

if __name__ == "__main__":

    curr_dir = os.path.dirname(os.path.realpath('__file__'))
    workspace = os.path.dirname(curr_dir)

    checkpoint_name = f"Othello/dnq_2.h5"
    q_model, q_target_model, scores, steps = dqn(env=OthelloEnv())
    print(q_model.get_weights())
    q_model.save(os.path.join('models',checkpoint_name))
    q_target_model.save(os.path.join('models',"Othello","target_model.h5"))
    plt.plot(scores)
    plt.savefig(os.path.join('models',"Othello","dqn_scores.png"))
    plt.plot(steps)
    plt.savefig(os.path.join('models',"Othello","dqn_steps.png"))

            