import gym
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
import warnings
from validation import valid

hyper_parameters = {  # don't change any parameters for initial step
    'learning_rate': 3,
    'machine status': 1,
    'machining progression': 1,
    'machine time left': 0,
    'job progression': 0,
    'job status': 0,
    'job left': 1,
    'machine index': 0,
    'total job status': 1,
    'job availability': 1,
    'available job times': 0,
    'machine current job': 1,
    'job current machine': 0,
    'job time left': 0,
    "occupancy average": 0,
    'ent_coeff info': 4
}

config = {
    'roll out': 110,  # for ta01 use 2000 and for ft 06 use 110
    'instance id': "ft06",  # use ta01 or ft06
    'model file': r"models\ft06_makespan",  # final model file ta01_makespan_1355
    'train timestep': 1_000_000,  # training time step
    'n_steps': 128 ,  # change based on ur training performance, was 448 for ta01
    'hyper parameters': hyper_parameters,  # don't change this for now
    'visu': False,  # don't use it while training
    'verbose': False,
    'n_cpu': 4,  # multiple run
    'final overview': False,  # check final performance of the model
    'hpo iter': 50  # not necessary for now
}

def _make_env(instance_id, parameters, rank, seed=0):

    def _init():
        s_env = gym.make('Jssp-v0', instance_id=instance_id, hyper_parameters=parameters,
                         roll_out_timestep=config['roll out'])
        s_env.seed(seed + rank)
        return s_env

    set_random_seed(seed)
    return _init()

if __name__ == "__main__":
    validation_process = True

    validator = valid.Validate()



