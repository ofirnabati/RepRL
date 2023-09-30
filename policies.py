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

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
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
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux




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
