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
        reward = 0
        done = False
        if action < 0 or action > 6:
            raise Exception("Invalid action")

        # Check and perform action
        for index in list(reversed(range(6))):
            if self.__board[index][action] == 0:
                self.__board[index][action] = self.__current_player
                break

        # Check for draw
        if np.count_nonzero(self.__board[0]) == 7:
            reward = 0
        # Else Check for win
        elif self.check_win():
            reward = self.__current_player
            done = True
        
        self.__current_player = -self.__current_player
        return self.__board, reward, done, {}

    def check_win(self) -> bool:
        """
        Checks if the board have any winning players
        """

        # Test rows
        for i in range(6):
            for j in range(7 - 3):
                value = sum(self.__board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.__board)]
        for i in range(7):
            for j in range(6 - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(6 - 3):
            for j in range(7 - 3):
                value = 0
                for k in range(4):
                    value += self.__board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(self.__board)
        # Test reverse diagonal
        for i in range(6 - 3):
            for j in range(7 - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def reset(self):
        """
        Reset the board
        :returns: numpy.array(6,7) board state after its being reset
        """
        self.__board = np.zeros((6, 7))
        return self.__board

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
