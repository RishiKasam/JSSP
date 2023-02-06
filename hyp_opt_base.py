import gym
import stable_baselines3
import optuna
from JsspEnvironment.envs.Jssp import Jssp


def _make_env(instance_id, parameters, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param instance_id: (str) the instance ID
    :param parameters: (int) the hyperparameters of the training
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        s_env = gym.make('Jssp-v0', instance_id=instance_id, hyper_parameters=parameters,
                         roll_out_timestep=config['roll out'])
        s_env.seed(seed + rank)
        return s_env

    set_random_seed(seed)
    return _init()


env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.5)
    model = stable_baselines3.PPO(policy="MlpPolicy", env= env, learning_rate=learning_rate, clip_range=clip_range)
    # Train the model for a certain number of timesteps
    model.learn(total_timesteps=10000)
    # Evaluate the model
    reward = model.get_avg_reward()
    #reward2 = model
    return reward


study = optuna.create_study()
study.optimize(objective, n_trials=100)

best_hyperparams = study.best_params
best_reward = study.best_value

# Train the final model with the best hyperparameters
model = stable_baselines3.PPO(policy="MlpPolicy", learning_rate=best_hyperparams["learning_rate"], clip_range=best_hyperparams["clip_range"])
model.learn(total_timesteps=20000)
