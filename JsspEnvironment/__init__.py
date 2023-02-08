from gym.envs.registration import register

register(
    id='Jssp-v0',
    entry_point='JsspEnvironment.envs:Jssp',
)
