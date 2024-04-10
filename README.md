# Introduction2RL_2024
Additional material for the lecture and exercise.
This is our little Mars rover from the lecture:
<img width="2024" alt="image" src="https://github.com/MathPhysSim/Introduction2RL_2024/assets/22523245/419f30be-12f0-4445-a077-56b0c8f03eda">



# Mars Rover Environment

The Mars Rover Environment is a simple, custom environment designed for reinforcement learning experiments. It simulates a rover navigating a linear track on Mars, where the objective is to reach a specific target location. The environment is structured to provide a straightforward yet challenging scenario for reinforcement learning algorithms to navigate, with rewards assigned based on the rover's ability to reach the designated goal.

## Environment Structure

The environment is represented as a linear space with a defined number of states, including two terminal states at both ends. The rover can occupy any state in this linear space, moving left or right based on the actions taken. The actions have probabilistic outcomes, introducing uncertainty into the rover's movement and making the navigation task more complex.

### States

- The environment consists of `n_states + 2` discrete states, where `n_states` is the configurable size of the environment, excluding the terminal states.
- The two additional states represent the terminal states at the left and right ends of the environment.

### Actions

The rover can perform two actions:
- **Left (0)**: Attempt to move one state to the left.
- **Right (1)**: Attempt to move one state to the right.

The actual movement of the rover is subject to a probabilistic model, where there is a chance the rover may stay in place or move in the opposite direction of the intended action.

### Transition Probabilities

The outcome of each action is determined by the following probabilities:
- **p_stay**: The probability that the rover stays in the same state after an action.
- **p_backward**: The probability that the rover moves in the opposite direction of the intended action.
- The probability of moving in the intended direction is calculated as `1 - p_stay - p_backward`.

### Rewards

Rewards are assigned based on the rover's movement and its ability to reach the terminal states:
- **left_side_reward**: The reward received for reaching the left terminal state.
- **right_side_reward**: A higher reward received for reaching the right terminal state. The goal is to encourage the rover to navigate towards the right side of the environment.

### Terminal States

The environment includes two terminal states at both ends. Reaching either terminal state concludes an episode, with the rover receiving a corresponding reward based on the terminal state reached.

## Usage

This environment is designed to be compatible with OpenAI Gym, allowing it to be easily integrated into existing reinforcement learning workflows. Users can instantiate the environment, reset it to start a new episode, and interact with it by sending actions and receiving observations, rewards, and done signals in return.

## Customization

The environment allows for customization of its size (`n_states`), transition probabilities (`p_stay`, `p_backward`), and rewards (`left_side_reward`, `right_side_reward`). This flexibility makes it suitable for a wide range of experiments, from teaching basic concepts of reinforcement learning to testing different algorithms and strategies.

## Implementation Note

To use this environment, ensure you have OpenAI Gym installed and properly configured in your Python environment. The Mars Rover Environment extends the `DiscreteEnv` class from Gym's `toy_text` module, leveraging its discrete state and action space functionality.
