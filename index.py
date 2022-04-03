from ConnectFourEnv import ConnectFourEnv
from gym.wrappers import FlattenObservation
import gym
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# env = VecFrameStack(env, n_stack=4)
env = ConnectFourEnv()
env = FlattenObservation(env)
log_path = os.path.join('Training', 'Logs')
a2c_path = os.path.join('Training', 'Saved Models', 'C4')

def verifyEnv():
    check_env(env, warn=True)

# verifyEnv()

def train():
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=50000)
    model.save(a2c_path)

def load_model():
    model = A2C.load(a2c_path, env=env)
    print(model.predict([0]*42))

load_model()