# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a family of neural network architectures for bandits.

The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from muzero_models import DownSample
# import math
import ipdb

def google_nonlinear(x):
    return torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) + 0.001 * x

class NormalizationLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n = x.norm(dim=-1,keepdim=True)
        n = n.detach()
        x = x / n
        return x

def init_params_gauss(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class NeuralBanditModel(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModel, self).__init__()
        self.state_based = hparams.state_based_value
        if self.state_based:
            latent_dim = hparams.layers_size[0] // 2
        else:
            latent_dim = hparams.layers_size[0]
        self.hparams = hparams
        policy_layers = []
        policy_layers.append(nn.Linear(hparams.context_dim, latent_dim  ,bias=False))
        policy_layers.append(nn.BatchNorm1d( latent_dim))
        policy_layers.append(nn.ReLU())
        policy_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
        policy_layers.append(nn.BatchNorm1d( latent_dim))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0] * 2, hparams.layers_size[0] ))
        # policy_layers.append(nn.BatchNorm1d(hparams.layers_size[0]))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0] ))
        self.policy_embedder = nn.Sequential(*policy_layers)
        # if hparams.env == 'breakout':
        #     self.block_output_size_policy = (
        #                 16
        #                 * math.ceil(hparams.obs_shape[1] / 16)
        #                 * math.ceil(hparams.obs_shape[2] / 16)
        #         )
        #
        #     self.downsampler = DownSample(hparams.obs_shape[0], 64)
        #     self.conv1x1 = nn.Conv2d(64, 16, 1)
        #     self.bn_state =  nn.BatchNorm2d(16)
        #     self.state_fc = nn.Linear(self.block_output_size_policy, hparams.layers_size[0] // 2)
        #     self.state_embedder = nn.Sequential(self.downsampler, self.conv1x1, self.bn_state, nn.ReLU())
        # else:
        state_layers = []
        state_layers.append(nn.Linear(hparams.obs_dim, latent_dim, bias=False))
        state_layers.append(nn.BatchNorm1d( latent_dim))
        state_layers.append(nn.ReLU())
        state_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
        state_layers.append(nn.BatchNorm1d(latent_dim))
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Sequential(*state_layers)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0] // 2)
        layers = []

        for i in range(len(hparams.layers_size)-1):
            if i < len(hparams.layers_size) - 2:
                layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                layers.append(nn.ReLU())
            else:

                layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1]))
                # if i<len(hparams.layers_size)-2:
            #     layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
            #     layers.append(nn.ReLU())
            # else:
            #     layers.append(nn.Tanh())
        # layers.append(NormalizationLayer())
        self.feature_extractor = nn.Sequential(*layers)
        self.value_pred = nn.Linear(hparams.layers_size[-1], 1, bias=False)
        self.feat_dim = hparams.layers_size[-1]
        # Initialize parameters correctly
        # self.apply(init_params)

        # decoder_layers = []
        # for j in range(len(hparams.layers_size),1,-1):
        #         decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
        #         decoder_layers.append(nn.ReLU())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        # self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)


    def forward(self, state, policy):
        phi   = self.encode(state, policy)
        # phi   = self.encode(policy)
        value = self.value_pred(phi)
        value = google_nonlinear(value)
        # value = torch.zeros(phi.shape[0],1).to(phi.device)
        # phi_hat = self.decoder(phi)
        return value, phi

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.state_based:
            s = self.state_embedder(state)
        # # if self.hparams.env == 'breakout':
        # #     s = s.view(-1,self.block_output_size_policy)
        # #     s = self.state_fc(s)
        z = self.policy_embedder(policy)
        if self.state_based:
            x = torch.cat([s,z],dim=-1)
        else:
            x = z
        # # x = self.bn0(z)
        x = F.relu(x)
        # # return z
        # # x = z
        z =  self.feature_extractor(x)
        return z#torch.softmax(z,dim=-1)
        # return torch.cat([state,policy],dim=-1)

    # def decode(self, phi):
    #     z_hat = self.decoder(phi)
    #     return z_hat

    # def estimate_advantages(self, rewards, masks, values):
    #     """
    #
    #     :param rewards:   Time x 1
    #     :param masks:  Time x 1
    #     :param values:  Time x 1
    #     :return:  Time x 1
    #     """
    #     deltas = torch.zeros_like(rewards)
    #     advantages = torch.zeros_like(rewards)
    #
    #     prev_value = 0 #torch.zeros(B, device= self.device)
    #     prev_advantage = 0 #torch.zeros(B, device= self.device)
    #     for i in reversed(range(rewards.size(0))):
    #         deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
    #         advantages[i] = deltas[i] + self.gamma * self.gae_lambda * prev_advantage * masks[i]
    #
    #         prev_value = values[i]
    #         prev_advantage = advantages[i]
    #
    #     returns = values + advantages # Time x 1
    #     # advantages = (advantages - advantages.mean()) / advantages.std()
    #
    #     return returns

    def get_last_layer_weights(self):
        return self.value_pred.weight.data[0]



class NeuralBanditModelVAEGaussian(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModelVAEGaussian, self).__init__()
        self.state_based = hparams.state_based_value
        self.no_embedding = hparams.no_embedding
        # self.non_linear_func = nn.ReLU
        self.non_linear_func = nn.Tanh
        if self.state_based:
            latent_dim = hparams.layers_size[0] // 2
        else:
            latent_dim = hparams.layers_size[0]
        self.hparams = hparams
        policy_layers = []
        policy_layers.append(nn.Linear(hparams.context_dim, latent_dim  ,bias=True))
        if hparams.use_bn:
            policy_layers.append(nn.BatchNorm1d( latent_dim))
        policy_layers.append(self.non_linear_func())
        policy_layers.append(nn.Linear(latent_dim, latent_dim, bias=True))
        # policy_layers.append(nn.BatchNorm1d( latent_dim))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0] * 2, hparams.layers_size[0] ))
        # policy_layers.append(nn.BatchNorm1d(hparams.layers_size[0]))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0] ))
        self.policy_embedder = nn.Sequential(*policy_layers)
        # if hparams.env == 'breakout':
        #     self.block_output_size_policy = (
        #                 16
        #                 * math.ceil(hparams.obs_shape[1] / 16)
        #                 * math.ceil(hparams.obs_shape[2] / 16)
        #         )
        #
        #     self.downsampler = DownSample(hparams.obs_shape[0], 64)
        #     self.conv1x1 = nn.Conv2d(64, 16, 1)
        #     self.bn_state =  nn.BatchNorm2d(16)
        #     self.state_fc = nn.Linear(self.block_output_size_policy, hparams.layers_size[0] // 2)
        #     self.state_embedder = nn.Sequential(self.downsampler, self.conv1x1, self.bn_state, nn.ReLU())
        # else:
        state_layers = []
        state_layers.append(nn.Linear(hparams.obs_dim, latent_dim, bias=True))
        # state_layers.append(nn.BatchNorm1d( latent_dim))
        state_layers.append(self.non_linear_func())
        state_layers.append(nn.Linear(latent_dim, latent_dim, bias=True))
        state_layers.append(self.non_linear_func())
        # state_layers.append(nn.BatchNorm1d(latent_dim))
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Sequential(*state_layers)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0] // 2)
        layers = []

        for i in range(len(hparams.layers_size)-1):
                layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1], bias=True))
                if hparams.use_bn:
                    layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                layers.append(self.non_linear_func())

        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])
        self.fc_var = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])

        self.value_pred = nn.Linear(hparams.layers_size[-1], 1, bias=False)
        self.feat_dim = hparams.layers_size[-1]
        # Initialize parameters correctly
        # self.apply(init_params)

        decoder_layers = []
        for j in range(len(hparams.layers_size),1,-1):
                decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
                if hparams.use_bn:
                    decoder_layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)


    def forward(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy
        else:
            mu, _   = self.encode(state, policy)
            mean_value = self.value_pred(mu)
            mean_value = google_nonlinear(mean_value)
            return mean_value, mu

    def forward_sample(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy, None, None
        else:
            mu, log_var   = self.encode(state, policy)
            z = self.reparameterize(mu, log_var)
            value = self.value_pred(z)
            value = google_nonlinear(value)
            return value, z ,mu, log_var

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.no_embedding:
            return policy, None
        else:
            if self.state_based:
                s = self.state_embedder(state)
            z = self.policy_embedder(policy)
            if self.state_based:
                x = torch.cat([s,z],dim=-1)
            else:
                x = z
            # x = F.relu(x)
            y = self.feature_extractor(x)
            mu = self.fc_mu(y)
            log_var = self.fc_var(y)
            return mu, log_var

    # def decode(self, z):
    #     return self.decoder(z)

    def sample(self, state, policy):
        if self.no_embedding:
            return policy
        else:
            mu, log_var = self.encode(state, policy)
            z = self.reparameterize(mu, log_var)
            return z


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def get_last_layer_weights(self):
        return self.value_pred.weight.data[0]

class NeuralBanditModelVAEDiscrete(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModelVAEDiscrete, self).__init__()
        self.state_based = hparams.state_based_value
        self.no_embedding = hparams.no_embedding
        self.category_size = hparams.category_size
        self.class_size    = hparams.class_size
        # self.non_linear_func = nn.ReLU
        self.non_linear_func = nn.Tanh
        if self.state_based:
            latent_dim = hparams.layers_size[0] // 2
        else:
            latent_dim = hparams.layers_size[0]
        self.hparams = hparams
        policy_layers = []
        policy_layers.append(nn.Linear(hparams.context_dim, latent_dim  ,bias=True))
        # policy_layers.append(nn.BatchNorm1d( latent_dim))
        policy_layers.append(self.non_linear_func())
        policy_layers.append(nn.Linear(latent_dim, latent_dim, bias=True))
        # policy_layers.append(nn.BatchNorm1d( latent_dim))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0] * 2, hparams.layers_size[0] ))
        # policy_layers.append(nn.BatchNorm1d(hparams.layers_size[0]))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0] ))
        self.policy_embedder = nn.Sequential(*policy_layers)
        # if hparams.env == 'breakout':
        #     self.block_output_size_policy = (
        #                 16
        #                 * math.ceil(hparams.obs_shape[1] / 16)
        #                 * math.ceil(hparams.obs_shape[2] / 16)
        #         )
        #
        #     self.downsampler = DownSample(hparams.obs_shape[0], 64)
        #     self.conv1x1 = nn.Conv2d(64, 16, 1)
        #     self.bn_state =  nn.BatchNorm2d(16)
        #     self.state_fc = nn.Linear(self.block_output_size_policy, hparams.layers_size[0] // 2)
        #     self.state_embedder = nn.Sequential(self.downsampler, self.conv1x1, self.bn_state, nn.ReLU())
        # else:
        state_layers = []
        state_layers.append(nn.Linear(hparams.obs_dim, latent_dim, bias=True))
        # state_layers.append(nn.BatchNorm1d( latent_dim))
        state_layers.append(self.non_linear_func())
        state_layers.append(nn.Linear(latent_dim, latent_dim, bias=True))
        state_layers.append(self.non_linear_func())
        # state_layers.append(nn.BatchNorm1d(latent_dim))
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Sequential(*state_layers)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0] // 2)
        layers = []

        for i in range(len(hparams.layers_size)-1):
                layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1], bias=True))
                # layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                layers.append(self.non_linear_func())

        layers.append(nn.Linear(hparams.layers_size[-1], hparams.category_size * hparams.class_size))
        self.feature_extractor = nn.Sequential(*layers)

        self.value_pred = nn.Linear(hparams.category_size * hparams.class_size, 1, bias=False)
        self.feat_dim = hparams.layers_size[-1]
        # Initialize parameters correctly
        # self.apply(init_params)

        decoder_layers = []
        decoder_layers.append(nn.Linear(hparams.category_size * hparams.class_size, hparams.layers_size[-1]))
        # decoder_layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
        decoder_layers.append(self.non_linear_func())
        for j in range(len(hparams.layers_size),1,-1):
                decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
                # decoder_layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0]))
        decoder_layers.append(self.non_linear_func())
        decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)


    def forward(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy
        else:
            logits  = self.encode(state, policy)
            mean_value = self.value_pred(logits)
            mean_value = google_nonlinear(mean_value)
            return mean_value, logits

    def forward_sample(self, state, policy):
        if self.no_embedding:
            return torch.zeros_like(policy[:,0]), policy, None, None
        else:
            logits  = self.encode(state, policy)
            z = self.reparameterize(logits)
            value = self.value_pred(z)
            value = google_nonlinear(value)
            return value, z, logits

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.no_embedding:
            return policy
        else:
            if self.state_based:
                s = self.state_embedder(state)
            z = self.policy_embedder(policy)
            if self.state_based:
                x = torch.cat([s,z],dim=-1)
            else:
                x = z
            # x = F.relu(x)
            logits = self.feature_extractor(x)
            return logits

    # def decode(self, z):
    #     return self.decoder(z)

    def sample(self, state, policy):
        if self.no_embedding:
            return policy
        else:
            logits = self.encode(state, policy)
            z = self.reparameterize(logits)
            return z


    def reparameterize(self, logit):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        shape = logit.shape
        logit = torch.reshape(logit, shape=(*shape[:-1], self.category_size, self.class_size))
        dist = torch.distributions.OneHotCategorical(logits=logit)
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)


    def get_last_layer_weights(self):
        return self.value_pred.weight.data[0]


class NeuralBanditModelDirichlet(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModelDirichlet, self).__init__()
        self.state_based = hparams.state_based_value
        if self.state_based:
            latent_dim = hparams.layers_size[0] // 2
        else:
            latent_dim = hparams.layers_size[0]
        self.hparams = hparams
        policy_layers = []
        policy_layers.append(nn.Linear(hparams.context_dim, latent_dim  ,bias=False))
        policy_layers.append(nn.BatchNorm1d( latent_dim))
        policy_layers.append(nn.ReLU())
        policy_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
        policy_layers.append(nn.BatchNorm1d( latent_dim))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0] * 2, hparams.layers_size[0] ))
        # policy_layers.append(nn.BatchNorm1d(hparams.layers_size[0]))
        # policy_layers.append(nn.ReLU())
        # policy_layers.append(nn.Linear(hparams.layers_size[0], hparams.layers_size[0] ))
        self.policy_embedder = nn.Sequential(*policy_layers)
        # if hparams.env == 'breakout':
        #     self.block_output_size_policy = (
        #                 16
        #                 * math.ceil(hparams.obs_shape[1] / 16)
        #                 * math.ceil(hparams.obs_shape[2] / 16)
        #         )
        #
        #     self.downsampler = DownSample(hparams.obs_shape[0], 64)
        #     self.conv1x1 = nn.Conv2d(64, 16, 1)
        #     self.bn_state =  nn.BatchNorm2d(16)
        #     self.state_fc = nn.Linear(self.block_output_size_policy, hparams.layers_size[0] // 2)
        #     self.state_embedder = nn.Sequential(self.downsampler, self.conv1x1, self.bn_state, nn.ReLU())
        # else:
        state_layers = []
        state_layers.append(nn.Linear(hparams.obs_dim, latent_dim, bias=False))
        state_layers.append(nn.BatchNorm1d( latent_dim))
        state_layers.append(nn.ReLU())
        state_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
        state_layers.append(nn.BatchNorm1d(latent_dim))
        # self.state_embedder = nn.Linear(hparams.obs_dim, hparams.layers_size[0] // 2)
        self.state_embedder = nn.Sequential(*state_layers)
        # self.bn0 = nn.BatchNorm1d(hparams.layers_size[0] // 2)
        layers = []

        for i in range(len(hparams.layers_size)-1):
                layers.append(nn.Linear(hparams.layers_size[i], hparams.layers_size[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
                layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])
        self.fc_var = nn.Linear(hparams.layers_size[-1], hparams.layers_size[-1])

        self.value_pred = nn.Linear(hparams.layers_size[-1], 1, bias=False)
        self.feat_dim = hparams.layers_size[-1]
        # Initialize parameters correctly
        # self.apply(init_params)

        # decoder_layers = []
        # for j in range(len(hparams.layers_size),1,-1):
        #         decoder_layers.append(nn.Linear(hparams.layers_size[j-1], hparams.layers_size[j-2]))
        #         decoder_layers.append(nn.BatchNorm1d(hparams.layers_size[i + 1]))
        #         decoder_layers.append(nn.ReLU())
        # decoder_layers.append(nn.Linear(hparams.layers_size[0], hparams.context_dim))
        # self.decoder = torch.nn.Sequential(*decoder_layers)

        # self.apply(init_params_gauss)

        # Dir prior
        self.prior_mean, self.prior_var = map(nn.Parameter, self.prior(hparams.layers_size[-1], 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def prior(self, K, alpha):
        """
        Prior for the model.
        :K: number of categories
        :alpha: Hyper param of Dir
        :return: mean and variance tensors
        """
        # ラプラス近似で正規分布に近似
        # Approximate to normal distribution using Laplace approximation
        a = torch.Tensor(1, K).float().fill_(alpha)
        mean = a.log().t() - a.log().mean(1)
        var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
        return mean.t(), var.t()  # Parameters of prior distribution after approximation

    def forward(self, state, policy):
        mu, _   = self.encode(state, policy)
        mu = torch.softmax(mu, dim=-1)
        mean_value = self.value_pred(mu)
        mean_value = google_nonlinear(mean_value)
        return mean_value, mu

    def forward_sample(self, state, policy):
        mu, log_var   = self.encode(state, policy)
        z = self.reparameterize(mu, log_var)
        z = F.softmax(z,dim=-1)
        value = self.value_pred(z)
        value = google_nonlinear(value)
        return value, z ,mu, log_var

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.state_based:
            s = self.state_embedder(state)
        z = self.policy_embedder(policy)
        if self.state_based:
            x = torch.cat([s,z],dim=-1)
        else:
            x = z
        x = F.relu(x)
        y = self.feature_extractor(x)
        mu = self.fc_mu(y)
        log_var = self.fc_var(y)
        return mu, log_var

    # def decode(self, z):
    #     return self.decoder(z)

    def sample(self, state, policy):
        mu, log_var = self.encode(state, policy)
        z = self.reparameterize(mu, log_var)
        z = F.softmax(z,dim=-1)
        return z


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def get_last_layer_weights(self):
        return self.value_pred.weight.data[0]