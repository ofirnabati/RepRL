'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''
from filter import get_filter
import torch
import torch.nn as nn
from torch.distributions import  Distribution, Normal, Categorical
import numpy as np
from functools import partial

class Policy(object):

    def __init__(self, policy_params, device='cpu:0'):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

        self.device = device
        
    def update_weights(self, new_weights):
        W, _ = new_weights
        self.weights[:] = W[0][:]
        return

    def get_weights(self):
        return [tuple([self.weights]), []]

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = torch.zeros(self.ac_dim, self.ob_dim, dtype = torch.float32, device=self.device)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        if len(ob.shape) > 1:
            ob = ob.flatten().astype(np.float32)
        ob = self.observation_filter(ob, update=self.update_filter)
        ob = torch.tensor(ob).float().to(self.device)
        out = torch.matmul(self.weights, ob)
        return out.cpu().numpy()

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = [self.weights, [], mu, std]
        return aux

    def get_stats(self):
        mu, std = self.observation_filter.get_stats()
        mu = torch.tensor(mu).float()
        std = torch.tensor(std).float() + 1e-8
        mu = -mu / std
        std = torch.diag(1 / std)
        return tuple([mu]), tuple([std])

class BasicModel(nn.Module):
    def __init__(self,):
        super(BasicModel, self).__init__()


    def forward(self, obs):
        raise NotImplementedError

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def get_weights(self):
        weights = tuple(
            [v.permute(1, 0).clone() for w, v in self.state_dict().items() if "weight" in w]
        )
        biases = tuple([v.clone() for w, v in self.state_dict().items() if "bias" in w])
        return [weights, biases]

    def update_weights(self, z):
        W, b = z
        D = self.state_dict()
        w_id, b_id = 0,0
        for w, v in D.items():
            if "weight" in w:
                D[w] = W[w_id].permute(1,0)
                w_id += 1
            if "bias" in w:
                D[w] = b[b_id]
                b_id += 1
        self.load_state_dict(D)

    def get_stats(self):
        mu, std = self.observation_filter.get_stats()
        mu = torch.tensor(mu).float()
        std = torch.tensor(std).float() + 1e-8
        mu = -mu / std
        std = torch.diag(1 / std)
        return tuple([mu]), tuple([std])


class DiscretePolicy(BasicModel):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params, device='cpu:0'):
        # Policy.__init__(self, policy_params)
        super(DiscretePolicy, self).__init__()
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        hidden_dim = 32
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        # self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)
        self.net = nn.Sequential(
            nn.Linear(self.ob_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.ac_dim),
        )
        self.device = device

        module_gains = {
            self.net: np.sqrt(2),
        }
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    def act(self, ob):
        if len(ob.shape) > 1:
            ob = ob.flatten().astype(np.float32)
        ob = self.observation_filter(ob, update=self.update_filter)
        ob = torch.tensor(ob).float().to(self.device)
        logits =  self.net(ob)
        probs = torch.softmax(logits, dim=-1)
        act_dist = Categorical(probs)
        act = act_dist.sample()
        return act.cpu().numpy()


    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        mean_filter = torch.eye(self.ob_dim) / (std)
        weights, biases = self.get_weights()
        aux = [weights, biases, mu, std]
        return aux

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


class StableBaselinePolicy(BasicModel):
  """
  Usage:

  ```
  policy = TanhGaussianPolicy(...)
  action, mean, log_std, _ = policy(obs)
  action, mean, log_std, _ = policy(obs, deterministic=True)
  action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
  ```
  Here, mean and log_std are the mean and log_std of the Gaussian that is
  sampled from.

  If deterministic is True, action = tanh(mean).
  If return_log_prob is False (default), log_prob = None
      This is done because computing the log_prob can be a bit expensive.
  """

  def __init__(
          self,
          obs_dim,
          action_dim,
          ob_filter,
          device,
          hidden_size=64,
          std=None,
          log_std_init=0.0,
          init_w=1e-3,
          **kwargs
  ):
    super(StableBaselinePolicy, self).__init__()
    self.log_std = None
    self.std = std
    self.deterministic = False
    self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, action_dim),
    )

    self.observation_filter = get_filter(ob_filter, shape=(obs_dim,))
    self.update_filter = True
    self.device = device
    # self.action_dist = DiagGaussianDistribution(action_dim)
    # self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=hidden_size,log_std_init=log_std_init)

    # module_gains = {
    #     self.net: np.sqrt(2),
    #     self.action_net: 0.0#0.01,
    # }
    # for module, gain in module_gains.items():
    #     module.apply(partial(self.init_weights, gain=gain))

  def get_weights_plus_stats(self):
      mu, std = self.observation_filter.get_stats()
      weights = self.get_weights()
      aux = np.asarray([weights, mu, std])
      return aux

  def act(self,obs):
    with torch.no_grad():
        obs = self.observation_filter(obs, update=self.update_filter)
        obs = torch.tensor(obs).float().to(self.device)
        return self.forward(obs)

  def forward(
          self,
          obs,
  ):
    """
    :param obs: Observation
    :param deterministic: If True, do not sample
    :param return_log_prob: If True, return a sample and its log probability
    """

    h = self.net(obs)
    # mean_actions = self.action_net(h)
    # dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
    # action = dist.get_actions(deterministic=self.deterministic)
    # return action.cpu().numpy()
    return h.cpu().numpy()