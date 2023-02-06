from typing import Any
from typing import Dict

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import configure_logger

config = {
    'roll out': 5500,  # for ta01 use 2000 and for ft 06 use 110
    'instance id': "ta01",  # use ta01 or ft06
    'model file': r"models1\ta01_PPO_2400_128_E_3_13_5500_param.zip",  # final model file ta01_makespan_1355 or ft06_makespan
    'train timestep': 6_000_000,  # training time step
    'n_steps': 2400,  # change based on ur training performance, was 448 for ta01
    #'hyper parameters': hyper_parameters,  # don't change this for now
    'visu': False,  # don't use it while training
    'verbose': False,
    'n_cpu': 4,  # multiple run
    'final overview': False,  # check final performance of the model
    'hpo iter': 50,# not necessary for now
}
def _make_env(instance_id, parameters, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param instance_id: (str) the instance ID
    :param parameters: (int) the hyperparameters of the training
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        s_env = gym.make('Jssp-v0', instance_id=instance_id,
                         roll_out_timestep=config['roll out'])
        s_env.seed(seed + rank)
        return s_env

    set_random_seed(seed)
    return _init()

env = SubprocVecEnv(
            [lambda: _make_env(config['instance id'], config['hyper parameters'], i) for i in range(config['n_cpu'])])
env = VecMonitor(env)

DEFAULT_HYPERPARAMS = {
    "policy" : "MultiInputPolicy",
    "env" : env,
}

def sample_ppo_params(trail: optuna.Trial) -> Dict[str, Any]:

    learning_rate = trail.suggest_float("lr", 1e-10, 1, log=True)
    n_steps = trail.suggest_int("n_steps", 16, 5000, 4)
    batch_size = trail.suggest_int("batch_size", low=4, high=2048)
    gamma = 1.0 - trail.suggest_float("gamma", low=1e-6, high=0.1, log=True)
    gae_lambda = 1.0 - trail.suggest_float("gae_lambda", low=1e-4, high=0.2, log=True)
    clip_range = trail.suggest_float("clip_range", low=1e-3, high= 0.85, log=True)
    ent_coef = trail.suggest_float("ent_coef", low=1e-8, high=0.1, log=True)
    vf_coef = trail.suggest_float("vf_coef", low=1e-8, high=0.1, log=True)
    max_grad_norm = trail.suggest_float("max_grad_norm", low=0.3, high=5.0, log=True)

    # Display True values

    trail.set_user_attr("gamma_", gamma)
    trail.set_user_attr("n_steps", n_steps)
    trail.set_user_attr("batch_size", batch_size)

    return {
        "n_steps" : n_steps,
        "gamma" : gamma,
        "gae_lambda" : gae_lambda,
        "learning_rate" : learning_rate,
        "ent_coef" : ent_coef,
        "max_grad_norm" : max_grad_norm,
        "clip_range" : clip_range,
        "batch_size" : batch_size,
        "vf_coef" : vf_coef,
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
    # Sample hyperparams
    kwargs.update(sample_ppo_params(trail))
    # RL model
    model = PPO(**kwargs)

    nan_encountered = False
    try:
        model.learn(100000)
    except AssertionError as e:
        # sometimes random hyperparams can generate NaN
        print(e)
        nan_encountered =True
    finally:
        #Free memory
        model.env.close()

    #Tell optimizer that the trail failed
    if nan_encountered:
        return float("nan")


study = optuna.create_study()
try:
    study.optimize(objective, n_trials=100, timeout=5000)
except KeyboardInterrupt:
    pass

print("Number of finished trails:", len(study.trials))

print("Best trail:")
trail = study.best_trial

print("Value:", trail.value)

print("Params: ")

for key, value in trail.params.items():
    print(" {}:{}".format(key, value))

print(" User attrs:")
for key, value in trail.user_attrs.items():
    print("  {}: {}".format(key, value))