import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

""" n x n gridworld
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GaussGridWorld(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, n=8, noise=0.0, terminal_reward=13.0,
          border_reward=0.0, step_reward=-2.0, start_state=0,
          bump_reward = 0.0, terminal_state_offset=0, gaussian_std=1.0): #'random'):
    self.n = n
    self.noise = noise
    self.terminal_reward = terminal_reward
    self.border_reward = border_reward
    self.bump_reward = bump_reward
    self.step_reward = step_reward
    self.n_states = self.n ** 2
    self.terminal_state = self.n_states - 1 - terminal_state_offset
    self.absorbing_state = self.n_states - 1
    self.done = False
    self.start_state = start_state #if not isinstance(start_state, str) else np.random.rand(n**2)


    #### Costum to 8x8
    # self.gaussian_center_1 = np.array([self.n // 2 + 1, self.n // 2 - 1])
    self.gaussian_center_1 = np.array([6 , 1])
    # self.gaussian_center_2 = np.array([self.n // 2 - 1 ,self.n // 2 + 1])
    self.gaussian_center_2 = np.array([2, 5])

    self.gaussian_means = np.zeros([self.n,self.n])
    reward_mean_cov1 = np.eye(2) * (8.0 / gaussian_std)
    reward_mean_cov2 = np.eye(2) * (1 / (gaussian_std * 8.0))
    for i in range(self.n):
      for j in range(self.n):
        p = np.array([i,j])
        vec1 = p - self.gaussian_center_1
        vec2 = p - self.gaussian_center_2
        log_val_1 = -0.5 * np.sum((vec1 * (reward_mean_cov1.dot(vec1))))
        log_val_2 = -0.5 * np.sum((vec2 * (reward_mean_cov2.dot(vec2))))
        self.gaussian_means[i,j] = 2.5 * np.exp(log_val_1) + 0.3 * np.exp(log_val_2)

    self.gaussian_means = self.gaussian_means.T.reshape(-1)

    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Discrete(self.n_states) # with absorbing state
    #self._seed()

  def one_hot(self, a):
    return np.squeeze(np.eye(self.n_states)[a])

  def step(self, action):
    assert self.action_space.contains(action)

    if self.state == self.terminal_state:
      # self.state = self.absorbing_state
      # self.done = True
      return self.one_hot(self.state), self._get_reward(), self.done, None

    [row, col] = self.ind2coord(self.state)

    if np.random.rand() < self.noise:
      action = self.action_space.sample()

    if action == UP:
      row = max(row - 1, 0)
    elif action == DOWN:
      row = min(row + 1, self.n - 1)
    elif action == RIGHT:
      col = min(col + 1, self.n - 1)
    elif action == LEFT:
      col = max(col - 1, 0)

    new_state = self.coord2ind([row, col])

    reward = self._get_reward(new_state=new_state)

    self.state = new_state

    return self.one_hot(self.state), reward, self.done, None

  def _get_reward(self, new_state=None):
    # if self.done:
    if new_state == self.terminal_state:
        return self.terminal_reward

    elif self.state == self.terminal_state:
      return 0.0
    # reward = self.step_reward

    #####Gaussian reward########
    reward = self.step_reward + np.random.rand() * 3.0 + self.gaussian_means[new_state]
    #############################


    if self.border_reward != 0 and self.at_border():
      reward = self.border_reward

    if self.bump_reward != 0 and self.state == new_state:
      reward = self.bump_reward

    return reward

  def at_border(self):
    [row, col] = self.ind2coord(self.state)
    return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

  def ind2coord(self, index):
    assert(index >= 0)
    #assert(index < self.n_states - 1)

    col = index // self.n
    row = index % self.n

    return [row, col]


  def coord2ind(self, coord):
    [row, col] = coord
    assert(row < self.n)
    assert(col < self.n)

    return col * self.n + row


  def reset(self, seed=None):
    super().reset(seed=seed)
    self.state = self.start_state if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 1)
    self.done = False
    return self.one_hot(self.state)

  def _render(self, mode='human', close=False):
    pass

  def test_me(self,H=20,T=10000):
    actions = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    actions_isbad = [2, 2, 2, 2, 2, 2, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3,1, 3, 1, 1, 3, 1, 3, 1, 3]
    actions_bad = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    mean_R = []
    for _ in range(T):
      i = 0
      R = 0
      self.reset()
      while i < H:
          s, r, done, _ = self.step(actions[i])
          # print(self.ind2coord(s.argmax()), r)
          R+= r
          i+= 1
      mean_R.append(R)

    mean_R_is_bad = []
    for _ in range(T):
      i = 0
      R = 0
      self.reset()
      while i < H:
          s, r, done, _ = self.step(actions_isbad[i])
          # print(self.ind2coord(s.argmax()), r)
          R+= r
          i+= 1
      mean_R_is_bad.append(R)


    mean_R_bad = []
    for _ in range(T):
      i = 0
      R = 0
      self.reset()
      while i < H:
          s, r, done, _ = self.step(actions_bad[i])
          # print(self.ind2coord(s.argmax()), r)
          R+= r
          i+= 1
      mean_R_bad.append(R)

    print(np.mean(mean_R))
    print(np.mean(mean_R_is_bad))
    print(np.mean(mean_R_bad))

