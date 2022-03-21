from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import gym
import numpy as np


class ConnectFourEnv(gym.Env):
    def __init__(self):
        """
        Constructor
        """
        super(ConnectFourEnv, self).__init__()

        # Describe the observation space
        self.observation_space = Box(low=-1, high=1, shape=(6, 7), dtype=int)

        # Describe possible actions (Player can insert in column 1 to 7)
        self.action_space = Discrete(7)

        # Custom members to keep track of the states
        self.__board = np.zeros((6, 7))
        self.__current_player = 1

    def step(self, action):
        """
        Move on the game

        :param action: Action to take for next step
        :returns: Tuple (board_state, reward, done, {})
        :returns: numpy.array(6,7) board_state - the current board state
        :returns: int reward - reward value corresponding to the step taken
        :returns: bool done - is the game completed?
        """
        pass

    def reset(self):
        """
        Reset the board
        :returns: numpy.array(6,7) board state after its being reset
        """
        pass

    def render(self):
        """
        To render the board.
        """
        print(self.__board)
        pass

    def close(self):
        """
        To unrender a board.
        """
        pass
