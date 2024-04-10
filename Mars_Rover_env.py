import numpy as np
from gymnasium.envs.toy_text import discrete

class MarsRoverEnv(discrete.DiscreteEnv):
    """
    A simple Mars Rover environment designed for reinforcement learning experiments within the Gymnasium framework.
    
    This environment simulates a rover navigating a linear track on Mars. The rover can move left or right across the landscape,
    aiming to reach a high-reward state at the right end of the environment. Movement actions have probabilistic outcomes, adding
    complexity to the task.
    
    Attributes:
    - n_states (int): The number of non-terminal states in the environment.
    - p_stay (float): The probability that the rover's action does not change its state.
    - p_backward (float): The probability that the rover moves opposite to the requested direction.
    - left_side_reward (float): The reward for reaching the left terminal state.
    - right_side_reward (float): The reward for reaching the right terminal state.
    """
    
    LEFT, RIGHT = 0, 1  # Actions
    
    def __init__(self, n_states=5, p_stay=1/3, p_backward=1/6, left_side_reward=1, right_side_reward=10):
        self.shape = (1, n_states + 2)  # Including terminal states
        self.nS = np.prod(self.shape)
        self.nA = 2  # Left, Right actions
        
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(1, self.nS - 1):  # Skipping terminal states for transitions
            for a in [self.LEFT, self.RIGHT]:
                outcomes = self._calculate_outcomes(s, a, p_stay, p_backward, left_side_reward, right_side_reward)
                P[s][a] = outcomes
        
        isd = np.zeros(self.nS)  # Initial state distribution
        isd[self.nS // 2] = 1.0  # Start in the middle
        
        super().__init__(self.nS, self.nA, P, isd)
    
    def _calculate_outcomes(self, s, action, p_stay, p_backward, left_side_reward, right_side_reward):
        p_forward = 1.0 - p_stay - p_backward
        move = -1 if action == self.LEFT else 1
        s_forward = np.clip(s + move, 0, self.nS - 1)
        s_backward = np.clip(s - move, 0, self.nS - 1)
        
        r_forward = right_side_reward if s_forward == self.nS - 1 else 0
        r_backward = left_side_reward if s_backward == 0 else 0
        
        d_forward = s_forward in [0, self.nS - 1]
        d_backward = s_backward in [0, self.nS - 1]
        
        return [
            (p_forward, s_forward, r_forward, d_forward),
            (p_stay, s, 0.0, s in [0, self.nS - 1]),
            (p_backward, s_backward, r_backward, d_backward)
        ]

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

if __name__ == '__main__':
    env = MarsRoverEnv()
    env.reset()
