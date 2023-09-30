from gym.envs.mujoco import Walker2dEnv
import numpy as np

class SparseWalker2dV0(Walker2dEnv):
    """Sparse Half-cheetah environment with target direction
    """
    def __init__(self, sparse_dist=2.1):
        self._goal_dir = 1.0
        self._sparse_dist = sparse_dist
        self.goals = [self._sparse_dist * x for x in range(1,10000)]
        super().__init__()

    def reset(self, *args, **kwargs):
        self.goals = [self._sparse_dist * x for x in range(1,10000)]
        return super().reset(*args, **kwargs)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel * (np.abs(xposafter) >= self._sparse_dist)
        if np.abs(xposafter) >= self.goals[0]:
            goal_reward = 10.0
            self.goals.pop(0)
        else:
            goal_reward = 0.0
        ctrl_cost = 0.1 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = goal_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,reward_ctrl=-ctrl_cost , sparse_dist = self._sparse_dist, xposafter=xposafter)
        return (observation, reward, done, infos)