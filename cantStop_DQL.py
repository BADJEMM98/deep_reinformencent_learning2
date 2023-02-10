import random
import logging

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from drl_lib.to_do.cant_stop_env import CantStopMultiplayer

from drl_lib.to_do.cantstop_agent import InterleavedAgent

logging.getLogger().setLevel(logging.INFO)
logging.info('test')
nplayers = 4

def main():
