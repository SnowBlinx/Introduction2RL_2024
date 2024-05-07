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

#print(env.step(0))

x = np.linspace(0, len(value_storage[4]), len(value_storage[4]))
y= value_storage[4]

#print(len(x),len(y))

plt.plot(x, y)
plt.title("C:\\Users\\Rapha\\OneDrive\\Dokumente\\VS-Code\\Uni\\Semester4\\Reinforced\\Übung6\\Introduction2RL_2024\\Value of state 4")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.savefig("Value of state 4.png")
plt.show()
plt.close()

#q-learning

for episode in range(number_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(V[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
        env.render()
