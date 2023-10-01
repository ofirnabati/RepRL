'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''
import ipdb
import numpy as np
from filter import get_filter

from torch.distributions import Distribution, Normal
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal, Categorical
import numpy as np
from functools import partial
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod


class Policy(object):

    def __init__(self, policy_params):
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

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
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = [self.weights, mu, std]
        return aux


class BasicModel(nn.Module):
    def __init__(self, ):
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
        z = []
        for param in self.parameters():
            if param.requires_grad:
                z.append(param.data.reshape(-1))
        return torch.cat(z).cpu().numpy()

    def update_weights(self, z):
        z = torch.tensor(z).float().to(self.device)
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                dim = np.prod(param.shape)
                param.data = z[idx:idx + dim].reshape(param.shape)
                idx += dim
        return

    def vec2set(self, z):
        z_dim = z.shape
        W = []
        b = []
        idx = 0
        for module in self.children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                dim = np.prod(module.weight.shape)
                if len(z_dim) > 1:
                    W.append(z[:, idx:idx + dim].reshape(-1, *(module.weight.shape)))
                else:
                    W.append(z[idx:idx + dim].reshape(module.weight.shape))
                idx += dim
                if module.bias is not None:
                    dim = np.prod(module.bias.shape)
                    if len(z_dim) > 1:
                        b.append(z[:, idx:idx + dim].reshape(-1, *(module.bias.shape)))
                    else:
                        b.append(z[idx:idx + dim].reshape(module.bias.shape))
                    idx += dim
        return [W, b]

    def get_weights_set(self):
        W = [];
        b = []
        for module in self.children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                W.append(module.weight.data.cpu().numpy())
                if module.bias is not None:
                    b.append(module.bias.data.cpu().numpy())

        return [W, b]

    def update_weights_set(self, z):
        W, b = z
        for idx, module in enumerate(self.children()):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data = W[idx]
                if module.bias is not None:
                    module.bias.data = b[idx]


class DiscretePolicy(BasicModel):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params, device='cpu:0'):
        # Policy.__init__(self, policy_params)
        super(DiscretePolicy, self).__init__()
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        # self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)
        self.net = nn.Sequential(
            nn.Linear(self.ob_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, self.ac_dim),
        )
        self.device = device

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        ob = torch.tensor(ob).float().to(self.device)
        # logits =  np.dot(self.weights, ob)
        logits = self.net(ob)
        # probs = np.exp(logits) / np.sum(np.exp(logits))
        probs = torch.softmax(logits, dim=-1)
        act_dist = Categorical(probs)
        act = act_dist.sample()
        # act = np.random.choice(np.arange(self.ac_dim), p=probs)
        return act.cpu().numpy()

    # def get_weights_plus_stats(self):
    #     mu, std = self.observation_filter.get_stats()
    #     aux = np.asarray([self.weights, mu, std])
    #     return aux

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        weights = self.get_weights()
        aux = np.asarray([weights, mu, std])
        return aux


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
        aux = [weights, mu, std]
        return aux

    def act(self, obs):
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


class BaselineDistribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


class DiagGaussianDistribution(BaselineDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=False)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor,
                            deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor