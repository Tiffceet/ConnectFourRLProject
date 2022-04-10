from ConnectFourEnv import ConnectFourEnv
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import ARS
import matplotlib.pyplot as plt


# env = VecFrameStack(env, n_stack=4)
env = ConnectFourEnv()
# env = FlattenObservation(env)
log_path = os.path.join('Training', 'Logs')
a2c_path = os.path.join('Training', 'Saved Models', 'C4')
dqn_path = os.path.join('Training', 'Saved Models', 'C4_DQN')
ppo_path = os.path.join('Training', 'Saved Models', 'C4_PPO')
ars_path = os.path.join('Training', 'Saved Models', 'C4_ARS')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    # model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model = ARS('LinearPolicy', env, verbose=1, tensorboard_log=log_path)

    model.learn(total_timesteps=50000000)
    model.save(ars_path)


def sample_game(model_a, model_b, plot_x, plot_y):
    match_count = 0
    win = 0
    lose = 0
    while match_count < 100:
        while True:
            action = model_a.predict(env.get_flatten_board())[0]
            env.step(action)
            env.render()

            if env.check_win():
                break

            if env.check_draw():
                break

            p_move = model_b.predict(env.get_flatten_board(True))[0]
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

        plot_x.append(match_count)
        plot_y.append(win)


# train()
ars_model = ARS.load(ars_path, env=env)
dqn_model = DQN.load(dqn_path, env=env)

ars_plot_x = []
ars_plot_y = []
dqn_plot_x = []
dqn_plot_y = []

sample_game(ars_model, dqn_model, ars_plot_x, ars_plot_y)
plt.plot(ars_plot_x, ars_plot_y, color='r', label='ARS as first player')
sample_game(dqn_model, ars_model, dqn_plot_x, dqn_plot_y)
plt.plot(dqn_plot_x, dqn_plot_y, color='b', label='DQN as first player')
plt.xlabel('Round')
plt.ylabel('Win Rate')
plt.legend()
plt.show()
