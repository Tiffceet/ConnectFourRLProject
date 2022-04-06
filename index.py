from re import match
from ConnectFourEnv import ConnectFourEnv
from gym.wrappers import FlattenObservation
import gym
import os
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from random import randint
# env = VecFrameStack(env, n_stack=4)
env = ConnectFourEnv()
# env = FlattenObservation(env)
log_path = os.path.join('Training', 'Logs')
a2c_path = os.path.join('Training', 'Saved Models', 'C4')
dqn_path = os.path.join('Training', 'Saved Models', 'C4_DQN')
ppo_path = os.path.join('Training', 'Saved Models', 'C4_PPO')

def verifyEnv():
    check_env(env, warn=True)


def train():
    # model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # model = A2C.load(a2c_path, env=env)
    # model = DQN.load(dqn_path, env=env)
    model.learn(total_timesteps=100000)
    model.save(dqn_path)


def load_model():
    model = DQN.load(dqn_path, env=env)
    print(model.predict([0]*42))

def sample_game():
    model = DQN.load(dqn_path, env=env)
    match_count = 0
    win = 0
    lose = 0
    while match_count <= 100:
        while True:
            action = model.predict(env.get_flatten_board())[0]
            env.step(action)
            env.render()

            if env.check_win():
                break

            p_move = randint(0, 6)
            env.step(p_move)
            env.render()

            if env.check_win():
                break
        if env.get_last_player() == 1:
            win += 1
        elif env.get_last_player() == -1:
            lose += 1
        match_count += 1
        env.reset()
        print(f"Win: {win} | Lose: {lose}")
    input()

# verifyEnv()
sample_game()
# train() 
# model = PPO.load(ppo_path, env=env)
# model = DQN.load(dqn_path, env=env)
# print(evaluate_policy(model, env, n_eval_episodes=10, render=True))