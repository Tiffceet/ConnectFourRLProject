from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import gym
import numpy as np
from typing import Optional
import pygame
from time import sleep
class ConnectFourEnv(gym.Env):
    def __init__(self):
        """
        Constructor
        """
        super(ConnectFourEnv, self).__init__()

        # Describe the observation space
        # self.observation_space = Box(low=-1, high=1, shape=(6, 7), dtype=int)
        self.observation_space = Box(low=-1, high=1, shape=(42, ), dtype=int)

        self.num_envs = 1

        # Describe possible actions (Player can insert in column 1 to 7)
        self.action_space = Discrete(7)

        # Custom members to keep track of the states
        self.__board = np.zeros((6, 7))
        self.__current_player = 1

    def get_flatten_board(self):
        """
        Flatten the board into 1D space for stable-baselines3 algorithms to work        
        """
        flat = np.zeros((42, ), dtype=int)
        for row in range(len(self.__board)):
            for col in range(len(self.__board[row])):
                np.append(flat, self.__board[row][col])
        return flat

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
            reward = 0.5
        # Else Check for win
        elif self.check_win():
            reward = self.__current_player
            done = True

        self.__current_player = -self.__current_player
        return self.get_flatten_board(), reward, done, {}

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
        return self.get_flatten_board()

    def draw_board(self, board):
        BLUE = (0, 0, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        YELLOW = (255, 255, 0)

        ROW_COUNT = 6
        COLUMN_COUNT = 7

        PLAYER = 0
        AI = 1

        EMPTY = 0
        PLAYER_PIECE = -1
        AI_PIECE = 1
        SQUARESIZE = 100

        width = COLUMN_COUNT * SQUARESIZE
        height = (ROW_COUNT+1) * SQUARESIZE

        size = (width, height)

        RADIUS = int(SQUARESIZE/2 - 5)

        screen = pygame.display.set_mode(size)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r *
                                                SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (int(
                    c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == PLAYER_PIECE:
                    pygame.draw.circle(screen, RED, (int(
                        c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif board[r][c] == AI_PIECE:
                    pygame.draw.circle(screen, YELLOW, (int(
                        c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()

    def render(self, mode = 'human'):
        """
        To render the board.
        """
        self.draw_board(np.flip(self.__board, 0))
        sleep(1)
        pass

    def close(self):
        """
        To unrender a board.
        """
        pass
