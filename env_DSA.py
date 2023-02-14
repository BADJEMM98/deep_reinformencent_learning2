import numpy as np

class DeepSingleAgentEnv:
    def max_action_count(self) -> int:
        pass

    def state_description(self) -> np.ndarray:
        pass

    def state_dim(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass






import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from env.cant_stop_multiplayers import CantStopMultiplayer



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return action

# Train the agent
if __name__ == '__main__':
    nplayer = 4
    env = CantStopMultiplayer(nplayer)
    nb_actions = env.action_space.n
    nb_obs = env.observation_space.n

    state_size = nb_actions         # define the state size based on your implementation of Can't Stop
    action_size = nb_obs            # define the action size based on your implementation of Can't Stop
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for episode in range(episodes + 1):
        state = env.reset() # reset the environment
        state = np.array(state, [1, state_size])
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action) # take a
            next_state = np.array(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 100 == 0:
            print(f'Episode: {episode}, e: {agent.epsilon}')
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)