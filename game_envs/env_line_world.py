from .deep_single_agent_env import DeepSingleAgentEnv
import math

import numpy as np



class LineWorld(DeepSingleAgentEnv):
    def __init__(self, nb_cells: int = 5):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.current_score = 0
        self.step_count = 0
    
    def state_id(self) -> int:
        return self.current_cell

    def max_action_count(self) -> int:
        return 8

    def state_description(self) -> np.ndarray:
        return np.array([self.current_cell / (self.nb_cells - 1) * 2.0 - 1.0])

    def state_dim(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells ** 2:
            return True
        return self.current_cell == 0 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_cell -= 1 # left
        else:
            self.current_cell += 1# right

    def score(self) -> float:
        if self.current_cell == 0:
            self.current_score -= 1.0
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            self.current_score += 1.0
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1])

    def reset(self):
        self.current_cell = math.floor(self.nb_cells / 2)
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            else:
                print("_", end='')
        print()

    def clone(self) -> 'DeepSingleAgentEnv':
        cloned_env = LineWorld()
        cloned_env.nb_cells = np.copy(self.nb_cells)
        cloned_env.current_cell = self.current_cell
        cloned_env.current_score = self.current_score
        cloned_env.step_count = self.step_count
        return cloned_env