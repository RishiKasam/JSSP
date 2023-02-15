
from typing import Any
from typing import Dict

import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
import torch as th
import torch.nn as nn
from JsspEnvironment.envs.Jssp import Jssp
from main import hyper_parameters, config

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


N_TRIALS = 100
N_JOBS = 2
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3


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


'''env = SubprocVecEnv(
            [lambda: _make_env(config['instance id'], config['hyper parameters'], i) for i in range(config['n_cpu'])])
env = VecMonitor(env)'''

env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])

DEFAULT_HYPERPARAMS = dict(policy="MultiInputPolicy", env=env)


def sample_ppo_params(trail: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams

    :param trail:
    :return:
    """
    #initial_learning_rate = trail.suggest_loguniform("initial_learning_rate", low=1e-15, high=1)
    #final_learning_rate = trail.suggest_loguniform("final_learning_rate", low=1e-15, high=1e-4)
    learning_rate = trail.suggest_loguniform("learning_rate", 1e-15, 1)
    lr_schedule = trail.suggest_categorical('lr_schedule', ['linear', 'constant'])
    n_steps = trail.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    batch_size = trail.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    gamma = trail.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.9995])
    gae_lambda = trail.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    clip_range = trail.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    ent_coef = trail.suggest_loguniform("ent_coef", low=1e-8, high=0.1)
    vf_coef = trail.suggest_uniform("vf_coef", low=0, high=1)
    max_grad_norm = trail.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    n_epochs = trail.suggest_categorical("n_epochs", [1, 5, 10, 20])
    net_arch = trail.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trail.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu"])
    #ortho_init = False
    ortho_init = trail.suggest_categorical('ortho_init', [False, True])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu" : nn.LeakyReLU,}[activation_fn]

    net_arch = {"tiny": [dict(pi=[64], vf=[64])], "small": [dict(pi=[64, 64], vf=[64, 64])],}[net_arch]

    '''# TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":

        learning_rate = linear_schedule(initial_value=learning_rate)

    if lr_schedule == "linear2":
        learning_rate = _linear_schedule(initial_value=initial_learning_rate, final_value=final_learning_rate)

        assert initial_learning_rate != final_learning_rate
        assert initial_learning_rate > final_learning_rate'''

    '''net_arch = {
        "tiny": dict(pi=[8,8],  vf=[8,8]),
        "v_small": dict(pi=[16,16], vf=[16,16]),
        "small": dict(pi=[64,64], vf=[64,64]),
        "medium": dict(pi=[256,256], vf=[256,256]),
    }[net_arch]'''

    #activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        #"initial_learning_rate": initial_learning_rate,
        #"final_learning_rate": final_learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs" : n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm" : max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),

    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super(TrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model
    model = PPO(**kwargs)
    # Create env used for evaluation
    eval_env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])
    #eval_env = gym.make()
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        # Prune hyperparams that generate NaNs
        print(e)
        raise optuna.exceptions.TrialPruned()
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))