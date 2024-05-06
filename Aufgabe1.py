import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from matplotlib import pyplot as plt

from Mars_Rover_env import MarsRoverEnv

number_episodes = 10
env = MarsRoverEnv()
V = np.zeros(7)
alpha = 0.1
gamma = 1
value_storage = {state: [] for state in range(env.nS)}

for episode in range(number_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.random.choice([0, 1])
        next_state, reward, done, info = env.step(action)
        V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
        value_storage[state].append(V[state])
        state = next_state

env.reset()


#Q-learning
