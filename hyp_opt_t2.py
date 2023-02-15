import optuna
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


# define your multi-input policy class
class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)


# define your environment, with a custom observation space that has multiple inputs
class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-10, high=10, shape=(1,)), gym.spaces.Box(low=-10, high=10, shape=(1,))))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state = None

    def reset(self):
        self.state = (np.random.uniform(low=-10, high=10, size=(1,)), np.random.uniform(low=-10, high=10, size=(1,)))
        return self.state

    def step(self, action):
        reward = np.sum(action * self.state)
        self.state = (np.random.uniform(low=-10, high=10, size=(1,)), np.random.uniform(low=-10, high=10, size=(1,)))
        done = False
        return self.state, reward, done, {}


# define your objective function for Optuna
def objective(trial):
    # set the hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    n_steps = trial.suggest_int('n_steps', 16, 256, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1, step=0.01)

    # create the environment
    env = CustomEnv()
    env = DummyVecEnv([lambda: env])

    # create the PPO model with the custom policy and tuned hyperparameters
    model = PPO(CustomPolicy, env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                ent_coef=ent_coef)

    # train the model
    model.learn(total_timesteps=1000)

    # evaluate the model and return the mean episode reward
    episode_rewards = []
    obs = env.reset()
    for i in range(10):
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


# run Optuna to find the best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# print the best hyperparameters found
print(study.best_trial.params)
