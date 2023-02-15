import gym
import stable_baselines3.common.monitor
from stable_baselines3 import PPO, DQN, A2C, HerReplayBuffer,DDPG, TD3, SAC
import imageio.v2
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from PIL import ImageFile
from stable_baselines3.common.utils import configure_logger
from typing import Callable
import numpy as np
from JsspEnvironment.envs.Jssp import Jssp
from validation import valid
import warnings
from typing import Any, Optional
from typing import Dict
import optuna
import torch
from torch import nn as nn
from stable_baselines3.common.callbacks import EvalCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


ImageFile.LOAD_TRUNCATED_IMAGES = True
configure_logger()

hyper_parameters = {  # don't change any parameters for initial step
    'learning_rate': 3,
    'machine status': 1,
    'machining progression': 1,
    'machine time left': 0,
    'job progression': 1,
    'job status': 1,
    'job left': 0,
    'machine index': 0,
    'total job status': 1,
    'job availability': 1,
    'available job times': 0,
    'machine current job': 0,
    'job current machine': 0,
    'job time left': 0,
    "occupancy average": 1,
    'ent_coeff info': 4
}

config = {
    'roll out': 2000,  # for ta01 use 2000 and for ft 06 use 110
    'instance id': "ta01",  # use ta01 or ft06
    'model file': r"models1\ta01_def_params6M.zip",  # final model file ta01_makespan_1355 or ft06_makespan
    'train timestep': 6_000_000,  # training time step
    'n_steps': 512,  # change based on ur training performance, was 448 for ta01
    'hyper parameters': hyper_parameters,  # don't change this for now
    'visu': False,  # don't use it while training
    'verbose': False,
    'n_cpu': 4,  # multiple run
    'final overview': False,  # check final performance of the model
    'hpo iter': 50  # not necessary for now

}


def _linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    #Conversion to float
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        #print(initial_value)
        #print(final_value)
        #print(progress_remaining)
        progress = 1 - progress_remaining
        #print(progress)
        p_1 = progress_remaining * initial_value
        #print(p_1)
        p_2 = progress * final_value
        #print(final_value)
        #print(p_2)
        rate = p_1 + p_2
        #print(rate)
        return rate
    return func


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


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


if __name__ == "__main__":
    train = False
    evaluate = False
    validation_process = False
    hyperparam_opt = True


    if validation_process:
        validator = valid.Validate()

        validator._start_validation()


    if train:
        env = SubprocVecEnv(
            [lambda: _make_env(config['instance id'], config['hyper parameters'], i) for i in range(config['n_cpu'])])
        env = VecMonitor(env)

        '''env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])'''

        '''model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log='./Logs1/', n_steps=config['n_steps'],
                    learning_rate=_linear_schedule(initial_value=1e-3, final_value=1e-10)
                    ,batch_size=128,gamma=0.99)'''

        model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log='./Logs1/')  #, n_steps=config['n_steps'])
        #model = A2C("MultiInputPolicy", env, verbose=0, learning_rate=1e-4, tensorboard_log='./Logs/', rms_prop_eps=1e-7, use_rms_prop=False)
                    #,gae_lambda=0.9, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True)

        '''model = DQN("MultiInputPolicy", env, verbose=0, tensorboard_log='./Logs/', learning_rate=_linear_schedule(initial_value=1e-4, final_value=1e-10),
                    exploration_fraction=0.6, exploration_initial_eps=1.0, exploration_final_eps=0.0005, tau=0.5)'''

        model.learn(total_timesteps=config['train timestep'])
        model.save(config['model file'])

        env.close()

    if evaluate:

        env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])
        model = PPO.load(config['model file'], print_system_info=True)
        state = env.reset()
        done = False
        images = []
        while not done:
            actions, _ = model.predict(state, deterministic=True)
            # actions = env.action_space.sample()
            state, reward, done, _ = env.step(actions)

            if config['visu'] and env.time_step > 0:
                plt = env.render()
                plt.savefig('gant1.png', facecolor='#36454F')
                images.append(imageio.v2.imread('gant1.png'))
            if done:
                if config['visu']:      # To create gif
                    imageio.mimsave(f'./visualization/{config["instance id"]}_{datetime.now().strftime("%M_%S")}.gif',
                                    images)
                if config['final overview']:    # To create final assignment image
                    plt = env.render()
                    plt.savefig(
                        f'./visualization/finalOverview/{config["instance id"]}_{datetime.now().strftime("%M_%S")}.png',
                        facecolor='#36454F')
                images = []
        env.close()

    if hyperparam_opt:

        def objective(trail):
            # set hyper-parameters to tune

            learning_rate = trail.suggest_float("learning_rate", 1e-15, 1) # suggest_loguniform
            lr_schedule = trail.suggest_categorical('lr_schedule', ['linear', 'constant'])
            n_steps = trail.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
            batch_size = trail.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
            gamma = trail.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.9995])
            gae_lambda = trail.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
            clip_range = trail.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
            ent_coef = trail.suggest_float("ent_coef", low=1e-8, high=0.1) # suggest_loguniform
            vf_coef = trail.suggest_float("vf_coef", low=0, high=1) # suggest_uniform
            max_grad_norm = trail.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
            n_epochs = trail.suggest_categorical("n_epochs", [1, 5, 10, 20])

            return {
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "clip_range": clip_range,
                "n_epochs": n_epochs,
                "gae_lambda": gae_lambda,
                "max_grad_norm": max_grad_norm,
                "vf_coef": vf_coef,


            }

            # create environment

            '''env = SubprocVecEnv(
                [lambda: _make_env(config['instance id'], config['hyper parameters'], i) for i in range(config['n_cpu'])])
            env = VecMonitor(env)'''

            env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])

            # create PPO model with tuned hyper parameters

            model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, n_epochs=n_epochs)

            # train the model

            model.learn(total_timesteps=100000)

            # evaluation of model

            state = env.reset()
            done = False
            while not done:
                actions, _ = model.predict(state, deterministic=True)
                # actions = env.action_space.sample()
                state, reward, done, _ = env.step(actions)

        # run Optuna to find the best parameters

        study = optuna.create_study(direction= 'maximize')
        study.optimize(objective, n_trials=2)

        # print the best hyper-parameters
        print(study.best_trial.params)
