from .deep_single_agent_env import DeepSingleAgentEnv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

class GridWorld(DeepSingleAgentEnv):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.nb_cells = height * width
        self.current_x = 0
        self.current_y = 0
        self.step_count = 0
        self.current_score = 0.0
        self.grid = np.zeros((height, width))

    def state_id(self) ->  np.array([int, int]):
        return (self.current_x, self.current_y)

    def max_action_count(self) -> int:
        return 4

    def state_description(self) -> np.ndarray:
        return np.array([self.current_x  / (self.height - 1), self.current_y / (self.width - 1)])
    
    def state_dim(self) -> int:
        return 2

    def is_game_over(self) -> bool: # ajouter des conditions
        if self.current_x < 0 or self.current_x >= self.height or self.current_y < 0 or self.current_y >= self.width:
            return True
        return False

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_x -= 1 # left
        elif action_id == 1:
            self.current_x += 1 # right
        elif action_id == 2:
            self.current_y -= 1 # up
        else:
            self.current_y += 1 # down

    def score(self) -> float:
        if self.current_x == 0 and self.current_y == self.width - 1:
            self.current_score -= 1.0
        
        elif self.current_x == self.height - 1 and self.current_y == self.width - 1:
            self.current_score += 1.0

        return self.current_score 
            
    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1, 2, 3])

    def reset(self):
        self.current_x = 0
        self.current_y = np.random.randint(0, self.width - 1)
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.height):
            for j in range(self.width):
                if i == self.current_x and j == self.current_y:
                    print("X", end='')
                else:
                    print("_", end='')
            print()

    def clone(self) -> 'DeepSingleAgentEnv':
        cloned_env = GridWorld(self.height, self.width)
        cloned_env.height = self.height
        cloned_env.width = self.width
        cloned_env.nb_cells = np.copy(self.nb_cells)
        cloned_env.current_x = self.current_x
        cloned_env.current_y = self.current_y
        cloned_env.step_count = self.step_count
        cloned_env.current_score = self.current_score
        cloned_env.grid = self.grid
        return cloned_env