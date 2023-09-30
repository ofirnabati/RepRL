from gym.envs.registration import registry, register, make, spec

# Mujoco
# ----------------------------------------



register(
    id='SparseHalfCheetah-v0',
    entry_point='sparseMuJoCo.envs.mujoco.half_cheetah_v0:SparseHalfCheetahV0',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)




register(
    id='SparseHopper-v0',
    entry_point='sparseMuJoCo.envs.mujoco.hopper_v0:SparseHopperV0',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)



register(
    id='SparseWalker2d-v0',
    max_episode_steps=1000,
    entry_point='sparseMuJoCo.envs.mujoco.walker2d_v0:SparseWalker2dV0',
)




register(
    id='SparseAnt-v0',
    entry_point='sparseMuJoCo.envs.mujoco.ant_v0:SparseAntV0',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)



register(
    id='SparseHumanoid-v0',
    entry_point='sparseMuJoCo.envs.mujoco.humanoid_v0:SparseHumanoidV0',
    max_episode_steps=1000,
)

