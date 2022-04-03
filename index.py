from ConnectFourEnv import ConnectFourEnv
import gym
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

def verifyEnv():
    env = ConnectFourEnv()
    check_env(env, warn=True)

verifyEnv()
# env = VecFrameStack(env, n_stack=4)
# log_path = os.path.join('Training', 'Logs')
# a2c_path = os.path.join('Training', 'Saved Models', 'C4')
# model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
# # model = A2C.load(a2c_path, env=env)
# # print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
# model.learn(total_timesteps=50000)
# model.save(a2c_path)