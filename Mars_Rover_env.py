import numpy as np
from gym.envs.toy_text import discrete

class MarsRoverEnv(discrete.DiscreteEnv):
    """
    A simple Mars rover environment for reinforcement learning.
    
    The rover can move left or right across a linear landscape with terminal states at both ends.
    The goal is to reach the right side, which offers a higher reward compared to the left side.
    
    Parameters
    ----------
    n_states : int
        The number of states in the environment, excluding terminal states.
    p_stay : float
        The probability that the rover stays in the same state after an action.
    p_backward : float
        The probability that the rover moves in the opposite direction of the action.
    left_side_reward : float
        The reward for reaching the left terminal state.
    right_side_reward : float
        The reward for reaching the right terminal state.
    """
    
    LEFT, RIGHT = 0, 1  # Action indices

    def __init__(self, n_states=7, p_stay=1/3, p_backward=1/6, left_side_reward=1, right_side_reward=10):
        self.shape = (1, n_states + 2)  # Account for two terminal states
        self.nS = np.prod(self.shape)  # Total number of states
        self.nA = 2  # Number of actions (left, right)

        # Transition probabilities
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(1, self.nS - 1):  # Exclude terminal states for transitions
            for a in [self.LEFT, self.RIGHT]:
                outcomes = self._calculate_outcomes(s, a, p_stay, p_backward, left_side_reward, right_side_reward)
                P[s][a] = outcomes

        # Initial state distribution: centered
        isd = np.zeros(self.nS)
        isd[self.nS // 2] = 1.0

        super().__init__(self.nS, self.nA, P, isd)

    def _calculate_outcomes(self, s, action, p_stay, p_backward, left_side_reward, right_side_reward):
        """
        Calculate the outcomes (probabilities, next states, rewards, done) for all possible transitions from state `s` using action `action`.
        """
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
        # You can implement this to visualize the environment if needed.
        pass

if __name__ == '__main__':
    env = MarsRoverEnv()
    env.reset()
