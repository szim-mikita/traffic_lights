from gymnasium.envs.registration import register

register(
    id='TrafficEnv-V0',
    entry_point='gym_envs.envs:TrafficEnv'
)