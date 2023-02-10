import random


import gym
from gym import spaces
import numpy as np
import random


class CantStopEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4])
        self.observation_space = spaces.Box(low=0, high=4, shape=(8,), dtype=int)
        self.board = np.zeros((8,), dtype=int)
        self.done = False
        self.turn = 0

    def step(self, action):
        assert self.action_space.contains(action)
        if self.done:
            raise Exception("CantStopEnv - Episode is done")

        # update the board based on the action
        for i in range(len(action)):
            if action[i] > 0:
                self.board[i] += action[i]

        # check if the game is won or lost
        if self.check_game_over():
            self.done = True

        # update the turn and get the reward
        self.turn += 1
        reward = self.get_reward()

        return self.board, reward, self.done, {}

    def reset(self):
        self.board = np.zeros((8,), dtype=int)
        self.done = False
        self.turn = 0
        return self.board

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def check_game_over(self):
        # check if the game is won or lost
        return True if np.max(self.board) >= 4 else False

    def get_reward(self):
        # return the reward based on the game result
        return 1 if not self.done else -1


if __name__ == "__main__":

    env = CantStopEnv()  # create an instance of the environment
    state = env.reset()  # reset the environment to start a new episode
    done = False

    while not done:
        action = random.randint(2, 12)# choose an action (e.g., using an agent or random actions)
        state, reward, done, _ = env.step(action)  # take a step in the environment based on the action
        # render the environment or update some variables (e.g., total reward)
        env.render()
        env.clock.tick(30)
    env.close()  # close the environment when the episode is done
"""
----------------------------------
import tkinter as tk


class CantStopGame(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Can't Stop Game")
        self.geometry("400x400")

        self.dice = [0, 0, 0, 0]
        self.scores = [0, 0, 0, 0]

        self.message = tk.StringVar()
        self.message.set("Bienvenue dans le jeu Can't Stop!")

        label = tk.Label(self, textvariable=self.message)
        label.pack(pady=20)

        roll_dice_button = tk.Button(self, text="Lancer les dés", command=self.roll_dice)
        roll_dice_button.pack(pady=10)

    def roll_dice(self):
        self.dice = [random.randint(1, 6) for i in range(4)]
        self.message.set("Les dés sont lancés: " + str(self.dice))


if __name__ == "__main__":
    game = CantStopGame()
    game.mainloop()

---------------------------------------
import pygame
import random

class CantStopEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        self.clock = pygame.time.Clock()
        self.done = False

    def step(self, action):
        # Your code for taking the action and updating the state

        # Example:
        dice_roll = random.randint(1, 6) + random.randint(1, 6)
        if dice_roll == action:
            reward = 1
        else:
            reward = -1
        # Return the new state, reward, and done flag
        return new_state, reward, self.done

    def reset(self):
        # Your code for resetting the environment to its initial state

        # Example:
        self.done = False
        return initial_state

    def render(self):
        # Your code for rendering the environment using Pygame
        self.screen.fill((255, 255, 255))
        pygame.display.update()

    def close(self):
        # Clean up Pygame
        pygame.quit()

if __name__ == "__main__":
    env = CantStopEnv()
    state = env.reset()
    while not env.done:
        action = random.randint(2, 12)
        state, reward, done = env.step(action)
        env.render()
        env.clock.tick(30)
    env.close()
"""

class CanStopEnv:
    def __init__(self):
        self.board_size = 11
        self.num_dice = 2
        self.dice_sides = 6
        self.max_moves = 4
        self.win_squares = [55, 56, 57, 58, 59, 60]

    def reset(self):
        self.state = [0 for i in range(self.board_size)]
        self.dice = [0 for i in range(self.num_dice)]

    def step(self, action):
        if action == "roll":
            self.dice = [random.randint(1, self.dice_sides) for i in range(self.num_dice)]
        elif action == "move":
            for i in range(self.max_moves):
                if self.dice[i] == 0:
                    break
                else:
                    self.state[self.dice[i]] += 1
        elif action == "stop":
            pass
        else:
            raise ValueError("Invalid action")

        done = False
        if any(pos >= 4 for pos in self.state):
            done = True

        reward = 0
        if done:
            if all(self.state[square] >= 4 for square in self.win_squares):
                reward = 1
            else:
                reward = -1

        return self.state, reward, done, {}