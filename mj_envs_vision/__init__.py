from gym.envs.registration import register

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs_vision.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200
)

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs_vision.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs_vision.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs_vision.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)