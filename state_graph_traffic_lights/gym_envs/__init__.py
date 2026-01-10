from gymnasium.envs.registration import register

register(
    id='StateTrafficEnv',
    entry_point='gym_envs.envs:StateTrafficEnv'
)
register(
    id='TrafficEnv',
    entry_point='gym_envs.envs:TrafficEnv'
)