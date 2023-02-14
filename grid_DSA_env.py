import math
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

class GridWorld(DeepSingleAgentEnv):
    def __init__(self, grid_size: int = 4, termination_states= [(3,3)]):
        self.grid_size = grid_size
        self.termination_states = termination_states
        self.step_count = 0
        self.current_state = (0, 0)
        self.reset()

    def max_action_count(self) -> int:
        return 4

    def state_description(self) -> np.ndarray:
        state_one_hot = np.zeros(self.grid_size * self.grid_size)
        state_one_hot[self.current_state[0] * self.grid_size + self.current_state[1]] = 1
        return state_one_hot

    def state_dim(self) -> int:
        return self.grid_size * self.grid_size

    def is_game_over(self) -> bool:
        if self.step_count > self.grid_size * self.grid_size:
          return True
        return self.current_state in self.termination_states or  self.current_state == ( 0, 0)

    def act_with_action_id(self, action_id: int):
        if action_id == 0:
            self.current_state = (self.current_state[0] - 1, self.current_state[1])
        elif action_id == 1:
            self.current_state = (self.current_state[0] + 1, self.current_state[1])
        elif action_id == 2:
            self.current_state = (self.current_state[0], self.current_state[1] - 1)
        else:
            self.current_state = (self.current_state[0], self.current_state[1] + 1)

    def score(self) -> float:
        if self.current_state in self.termination_states:
            return 1.0
        elif self.current_state in [(0,0), (3,0), (0,3)]:
           return -1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        if self.current_state[0] > 0:
            actions.append(0)
        if self.current_state[0] < self.grid_size - 1:
            actions.append(1)
        if self.current_state[1] > 0:
            actions.append(2)
        if self.current_state[1] < self.grid_size - 1:
            actions.append(3)
        return np.array(actions)

    def reset(self):
        self.current_state = (0, 0)

    def view(self):
        print('Game Over:', self.is_game_over())
        print('Score:', self.score())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.current_state:
                    print("X", end=' ')
                else:
                    print("_", end=' ')


















































































































os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class GridWorld(DeepSingleAgentEnv):
    def __init__(self, grid_size: int = 5, termination_states= [(3,3)]):
        self.grid_size = grid_size
        self.termination_states = termination_states
        self.step_count = 0
        self.current_state = (0, 0)
        self.reset()

    def max_action_count(self) -> int:
        return 4

    def state_description(self) -> np.ndarray:
        state_one_hot = np.zeros(self.grid_size * self.grid_size)
        state_one_hot[self.current_state[0] * self.grid_size + self.current_state[1]] = 1
        return state_one_hot

    def state_dim(self) -> int:
        return self.grid_size * self.grid_size

    def is_game_over(self) -> bool:
        if self.step_count > self.grid_size * self.grid_size:
          return True
        return self.current_state in self.termination_states or  self.current_state == ( 0, 0)

    def act_with_action_id(self, action_id: int):
        if action_id == 0:
            self.current_state = (self.current_state[0] - 1, self.current_state[1])
        elif action_id == 1:
            self.current_state = (self.current_state[0] + 1, self.current_state[1])
        elif action_id == 2:
            self.current_state = (self.current_state[0], self.current_state[1] - 1)
        else:
            self.current_state = (self.current_state[0], self.current_state[1] + 1)

    def score(self) -> float:
        if self.current_state in self.termination_states:
            return 1.0
        elif self.current_state in [(0,0), (3,0), (0,3)]:
           return -1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        if self.current_state[0] > 0:
            actions.append(0)
        if self.current_state[0] < self.grid_size - 1:
            actions.append(1)
        if self.current_state[1] > 0:
            actions.append(2)
        if self.current_state[1] < self.grid_size - 1:
            actions.append(3)
        return np.array(actions)

    def reset(self):
        self.current_state = (0, 0)

    def view(self):
        print('Game Over:', self.is_game_over())
        print('Score:', self.score())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.current_state:
                    print("X", end=' ')
                else:
                    print("_", end=' ')