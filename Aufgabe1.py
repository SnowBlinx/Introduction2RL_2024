import numpy as np
from matplotlib import pyplot as plt

from Mars_Rover_env import MarsRoverEnv

env = MarsRoverEnv()

alpha = 0.1
gamma = 0.9
numberofepisodes = 1000

# Q-learning

Q = np.zeros([env.nS, env.nA])
valueStorageQ = {state: [] for state in range(env.nS)}

def select_action(state, Q, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action

for episode in range(numberofepisodes):
    state = env.reset()
    done = False
    while not done:
        action = select_action(state, Q, 0.1)
        next_state, reward, done, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        valueStorageQ[state].append(Q[state, action])
        state = next_state
print(f"Q-Table Q-Learning: {Q}")

# SARSA

Q = np.zeros([env.nS, env.nA])
valueStorageSARSA = {state: [] for state in range(env.nS)}

for episode in range(numberofepisodes):
    state = env.reset()
    done = False
    while not done:
        action = select_action(state, Q, 0.1)
        next_state, reward, done, info = env.step(action)
        next_action = select_action(next_state, Q, 0.1)
        error = reward + gamma * Q[next_state, next_action] - Q[state, action]
        Q[state, action] = Q[state, action] + alpha * error
        valueStorageSARSA[state].append(Q[state, action])
        state = next_state
print(f"Q-Table SARSA: {Q}")

# First Visit MC

Q = np.zeros([env.nS, env.nA])
returns_sum = np.zeros([env.nS, env.nA])
returns_count = np.zeros([env.nS, env.nA])
valueStorageFirstVisitMC = {state: [] for state in range(env.nS)}

for episode in range(numberofepisodes):
    state = env.reset()
    episode_states_actions = []
    episode_rewards = []
    done = False
    while not done:
        action = select_action(state, Q, epsilon=0.1)
        next_state, reward, done, _ = env.step(action)
        episode_states_actions.append((state, action))
        episode_rewards.append(reward)
        state = next_state

    G = 0
    visited_states_actions = set()
    for t in range(len(episode_states_actions) - 1, -1, -1):
        state, action = episode_states_actions[t]
        if (state, action) not in visited_states_actions:
            G = gamma * G + episode_rewards[t]
            returns_sum[state, action] += G
            returns_count[state, action] += 1
            Q[state, action] = returns_sum[state, action] / returns_count[state, action]
            valueStorageFirstVisitMC[state].append(Q[state, action])
            visited_states_actions.add((state, action))

print(f"Q-Table First Visit MC: {Q}")


# Berechne die durchschnittlichen Q-Werte für jeden nicht-terminalen Zustand
def calculate_average_q_values(value_storage):
    average_q_values = []
    for state, q_values in value_storage.items():
        if len(q_values) > 0:
            average_q_values.append(np.mean(q_values))
    return average_q_values


# Berechne die bewegten Durchschnittswerte
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Berechne und plotte die durchschnittlichen Q-Werte für jeden nicht-terminalen Zustand
def plot_average_q_values(value_storage, method_name):
    average_q_values = calculate_average_q_values(value_storage)
    episodes = np.arange(len(average_q_values))
    plt.plot(episodes, average_q_values, label=method_name)


# Plotte die durchschnittlichen Q-Werte für jeden nicht-terminalen Zustand für alle drei Methoden
plt.figure(figsize=(10, 6))
plt.title("Average Q-Values for Non-terminal States")
plt.xlabel("Episodes")
plt.ylabel("Average Q-Value")

# Q-Learning
plot_average_q_values(valueStorageQ, "Q-Learning")

# SARSA
plot_average_q_values(valueStorageSARSA, "SARSA")

# First Visit MC

plot_average_q_values(valueStorageFirstVisitMC, "First Visit MC")

plt.show()

# Aufgabe 2.


# Evaluierung der gelernten Richtlinien während des Trainings
# Funktion zur Berechnung der Erfolgsrate
def calculate_success_rate(env, Q):
    success_count = 0
    total_episodes = 100  # Anzahl der Episoden, um die Erfolgsrate zu berechnen
    for _ in range(total_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, _, done, _ = env.step(action)
        if state == 0 or state == 6:
            success_count += 1
    success_rate = success_count / total_episodes
    return success_rate


# Funktion zur Berechnung des bewegten Durchschnitts
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Berechne und plotte die Erfolgsrate für jede Methode
def plot_success_rate(Q_values, method_name):
    success_rates = []
    for Q in Q_values:
        success_rate = calculate_success_rate(env, Q)
        success_rates.append(success_rate)
    episodes = np.arange(len(success_rates))
    plt.plot(episodes, moving_average(success_rates, window_size=10), label=method_name)


# Q-Learning
#plot_success_rate([Q], "Q-Learning")

# SARSA
#plot_success_rate([Q], "SARSA")

# First Visit Monte Carlo
#plot_success_rate([Q], "First Visit MC")

#plt.xlabel("Episodes")
#plt.ylabel("Success Rate")
#plt.title("Success Rate During Training")
#plt.legend()
#plt.grid(True)
#plt.show()


# Aufgabe 3


# Initialize variables for storing rewards
rewards_Q_learning = []
rewards_SARSA = []
rewards_FirstVisitMC = []


# Function to calculate moving average
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


# Training loop for Q-learning
for episode in range(numberofepisodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = select_action(state, Q, 0.1)
        next_state, reward, done, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        total_reward += reward
        state = next_state
    rewards_Q_learning.append(total_reward)

# Training loop for SARSA
for episode in range(numberofepisodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = select_action(state, Q, 0.1)
        next_state, reward, done, info = env.step(action)
        next_action = select_action(next_state, Q, 0.1)
        error = reward + gamma * Q[next_state, next_action] - Q[state, action]
        Q[state, action] = Q[state, action] + alpha * error
        total_reward += reward
        state = next_state
    rewards_SARSA.append(total_reward)

# Training loop for First Visit MC
for episode in range(numberofepisodes):
    state = env.reset()
    episode_states_actions = []
    episode_rewards = []
    done = False
    total_reward = 0
    while not done:
        action = select_action(state, Q, epsilon=0.1)
        next_state, reward, done, _ = env.step(action)
        episode_states_actions.append((state, action))
        episode_rewards.append(reward)
        total_reward += reward
        state = next_state

    rewards_FirstVisitMC.append(total_reward)

# Calculate moving averages
window_size = 10
moving_avg_Q_learning = moving_average(rewards_Q_learning, window_size)
moving_avg_SARSA = moving_average(rewards_SARSA, window_size)
moving_avg_FirstVisitMC = moving_average(rewards_FirstVisitMC, window_size)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(moving_avg_Q_learning, label="Q-learning")
plt.plot(moving_avg_SARSA, label="SARSA")
plt.plot(moving_avg_FirstVisitMC, label="First Visit MC")
plt.title("Mean Reward per Episode (Moving Average)")
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.legend()
plt.show()
