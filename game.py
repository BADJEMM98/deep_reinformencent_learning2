import random

from env.cant_stop_multiplayers import CantStopMultiplayer

if __name__ == "__main__":
    numplay = 4
    env = CantStopMultiplayer(numplay)  # create an instance of the environment
    state = env.reset()  # reset the environment to start a new episode
    done = False

    while not done:
        action = random.randint(2, 12)# choose an action (e.g., using an agent or random actions)
        state, reward, done, _ = env.step(action)  # take a step in the environment based on the action
        # render the environment or update some variables (e.g., total reward)
        #env.render(mode='human')

    env.close()  # close th