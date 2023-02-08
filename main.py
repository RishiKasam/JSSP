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
    'model file': r"models1\ta01_def_param_6M_448steps.zip",  # final model file ta01_makespan_1355 or ft06_makespan
    'train timestep': 6_000_000,  # training time step
    'n_steps': 448,  # change based on ur training performance, was 448 for ta01
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
    net_arch = trail.suggest_categorical("net_arch", ["tiny", "v_small", "small", "medium"])
    activation_fn = trail.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu"])
    ortho_init = False
    #ortho_init = trail.suggest_categorical('ortho_init', [False, True])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":

        learning_rate = linear_schedule(initial_value=learning_rate)

    '''if lr_schedule == "linear2":
        learning_rate = _linear_schedule(initial_value=initial_learning_rate, final_value=final_learning_rate)

        assert initial_learning_rate != final_learning_rate
        assert initial_learning_rate > final_learning_rate'''

    '''net_arch = {
        "tiny": dict(pi=[8,8],  vf=[8,8]),
        "v_small": dict(pi=[16,16], vf=[16,16]),
        "small": dict(pi=[64,64], vf=[64,64]),
        "medium": dict(pi=[256,256], vf=[256,256]),
    }[net_arch]'''

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation_fn]

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
        #"policy_kwargs": dict(
            #net_arch=net_arch,
            #activation_fn=activation_fn,
            #ortho_init=ortho_init,
        #),

    }


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

        model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log='./Logs1/', n_steps=config['n_steps'])
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
        '''env = SubprocVecEnv(
            [lambda: _make_env(config['instance id'], config['hyper parameters'], i) for i in range(config['n_cpu'])])
        env = VecMonitor(env)'''

        env = gym.make('Jssp-v0', instance_id=config['instance id'], hyper_parameters=config['hyper parameters'],
                       roll_out_timestep=config['roll out'])

        N_TRIALS = 10 #100
        N_STARTUP_TRAILS = 2 #5
        N_EVALUATIONS = 10000 #2
        N_TIMESTEPS = int(1e6)
        EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
        N_EVAL_EPISODES = 0 #3

        ENV_ID = "Jssp-v0"

        DEFAULT_HYPERPARAMS = {
            "policy": "MultiInputPolicy",
            "env": env,
        }


        class TrialEvalCallback(EvalCallback):
            """Callback used for evaluating and reporting a trial."""

            def __init__(
                    self,
                    eval_env: gym.Env,
                    trial: optuna.Trial,
                    n_eval_episodes: int = 5,
                    eval_freq: int = 10000,
                    log_path: Optional[str] = None,
                    best_model_save_path: Optional[str] = None,
                    deterministic: bool = True,
                    verbose: int = 0,
            ):

                super().__init__(
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
                    super()._on_step()
                    self.eval_idx += 1
                    self.trial.report(self.last_mean_reward, self.eval_idx)
                    # Prune trial if need
                    if self.trial.should_prune():
                        self.is_pruned = True
                        return False
                return True

        def objective(trail: optuna.Trial) -> float:
            kwargs = DEFAULT_HYPERPARAMS.copy()
            kwargs.update(sample_ppo_params(trail))
            model = PPO(**kwargs)
            # Create env used for evaluation
            #eval_env = gym.make(ENV_ID)
            eval_env = env

            eval_callback = TrialEvalCallback(
                eval_env, trail, n_eval_episodes=N_EVAL_EPISODES,
                 eval_freq=EVAL_FREQ, deterministic=True, best_model_save_path="./best_model_logs/"
            )

            nan_encountered = False
            try:
                model.learn(N_TIMESTEPS, callback=eval_callback)
            except AssertionError as e:
                # sometimes random hyperparams can generate NaN
                print(e)
                nan_encountered = True
            finally:
                # Free memory
                model.env.close()
                eval_env.close()

            # Tell optimizer that the trail failed
            if nan_encountered:
                return float("nan")
            if eval_callback.is_pruned:
                raise optuna.exceptions.TrialPruned()

            return eval_callback.last_mean_reward

         # Set pytorch num threads to 1 for faster training
        torch.set_num_threads(1)

        sampler = TPESampler(n_startup_trials=N_STARTUP_TRAILS)
        # Do not prune before 1/3 of the max budget is used
        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRAILS, n_warmup_steps=N_EVALUATIONS // 3)

        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
        try:
            study.optimize(objective, n_trials=N_TRIALS, timeout=5500)
        except KeyboardInterrupt:
            pass

        '''study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=100)'''

        #study.best_params

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))