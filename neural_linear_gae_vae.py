"""Thompson Sampling with linear posterior over a learnt deep representation."""

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from neural_bandit_model import NeuralBanditModelVAEGaussian, NeuralBanditModelVAEDiscrete
import torch.distributions as td
from exp_dws.data import ReprlDatasetVae
from scipy.stats import invgamma

class NeuralLinearPosteriorSampling:
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, storage,  device, hparams):

    self.hparams = hparams
    self.storage = storage
    # self.latent_dim = self.hparams.context_dim + self.hparams.obs_dim
    self.no_embedding = self.hparams.no_embedding
    self.augmentation = self.hparams.augmentation
    self.permutation = self.hparams.permutation
    self.discrete_dist = self.hparams.discrete_dist
    self.noise_scale =  self.hparams.noise_aug
    self.average_reward = False

    if self.no_embedding:
        self.latent_dim = self.hparams.context_dim
    else:
        if self.discrete_dist:
            self.category_size = self.hparams.category_size
            self.class_size = self.hparams.class_size
            self.latent_dim = self.category_size * self.class_size
        else:
            self.latent_dim = self.hparams.layers_size[-1]

    self.param_dim=self.latent_dim
    self.context_dim = self.hparams.context_dim
    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior
    self.device = device
    self.fragments = 1
    self.kld_coeff = self.hparams.kld_coeff
    self.decoder_coeff = self.hparams.dec_coeff
    self.horizon = hparams.horizon
    self.dtype = torch.float32
    self.ucb_coeff = hparams.ucb_coeff
    self.state_based_value = hparams.state_based_value
    self.step_size =  hparams.latent_step_size
    self.sigma2_s = 10.0

    self.mu = [torch.zeros(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]
    self.f  = [torch.zeros(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]
    self.yy = [0 for _ in range(self.fragments)]

    # self.cov = [(1.0 / self.lambda_prior) * torch.eye(self.param_dim, device=device) for _ in range(self.fragments)]

    self.precision = [self.lambda_prior * torch.eye(self.param_dim, device=device, dtype=self.dtype) for _ in range(self.fragments)]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.fragments)]
    self.b = [self._b0 for _ in range(self.fragments)]

    # Regression and NN Update Frequency
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.training_steps = 0

    self.data_h = storage

    self.method = hparams.method
    self.batch_data_number = 100

    self.ucb = 0
    self.ls_values = 0
    self.R = 0

    #Model learning
    # self.loss_fn = torch.nn.MSELoss()
    self.loss_fn = torch.nn.L1Loss()
    self.mse_loss = torch.nn.MSELoss()
    self.lr = hparams.initial_lr
    self.batch_size = hparams.batch_size
    self.training_iter = hparams.training_iter
    self.device = device
    self.lr_decay_rate = hparams.lr_decay_rate
    self.lr_step_size = hparams.lr_step_size
    self.max_grad_norm = hparams.max_grad_norm
    self.optimizer_name = hparams.optimizer
    self.gamma = hparams.discount
    self.gae_lambda = hparams.gae_lambda
    self.target_model_update = hparams.target_model_update
    self.soft_target_tau = 1e-1
    self.num_unroll_steps = hparams.num_unroll_steps
    self.stacked_observations = 1

    self.soft_bandit_update = hparams.soft_bandit_update
    self.init_model()


    # if self.optimizer_name == 'Adam':
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
    # elif self.optimizer_name == 'AdamW':
    #     self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-6)
    # elif self.optimizer_name == 'RMSprop':
    #     self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
    # else:
    #     raise ValueError('optimizer name is unkown')
    # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_decay_rate)

  def init_model(self):
      if self.discrete_dist:
          self.model = NeuralBanditModelVAEDiscrete(self.hparams).to(self.device)
          self.target_model = NeuralBanditModelVAEDiscrete(self.hparams).to(self.device)
      else:
          self.model = NeuralBanditModelVAEGaussian(self.hparams).to(self.device)
          self.target_model = NeuralBanditModelVAEGaussian(self.hparams).to(self.device)
      self.target_model.load_state_dict(self.model.state_dict())
      for param in self.target_model.parameters():
          param.requires_grad = False
      self.target_model.eval()
      if self.hparams.optimizer == 'Adam':
          self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
      elif self.hparams.optimizer == 'AdamW':
          self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.hparams.weight_decay)
      elif self.hparams.optimizer == 'RMSprop':
          self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
      else:
          raise ValueError('optimizer name is unkown')
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size,
                                                       gamma=self.lr_decay_rate)



  def add_bandit_fragment(self):
      self.mu.append(torch.zeros(self.param_dim, device=self.device, dtype=self.dtype))
      self.f.append(torch.zeros(self.param_dim, device=self.device, dtype=self.dtype))
      self.yy.append(0)

      # self.cov.append((1.0 / self.lambda_prior) * torch.eye(self.param_dim, device=self.device))
      self.precision.append(self.lambda_prior * torch.eye(self.param_dim, device=self.device, dtype=self.dtype))

      self.a.append(self._a0)
      self.b.append(self._b0)


  def soft_update_from_to(self, source, target):
      for target_param, param in zip(target.parameters(), source.parameters()):
          target_param.data.copy_(
              target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
          )

  def find_parameters(self, initial_point, latent, obs, fixed_parameters=None, cal_R=False):
      self.model.eval()
      if fixed_parameters is None:
          x = initial_point.detach()
      else:
        x = initial_point[:,:-fixed_parameters].detach()
        y = initial_point[:,:fixed_parameters].detach()
        y.requires_grad = False
      for k in range(256):
        x.requires_grad = True
        if fixed_parameters is None:
            # phi_hat = self.model.sample(obs,x)
            phi_hat,_ = self.model.encode(obs,x)
        else:
            phi_hat = self.model.encode(obs, torch.cat([x,y],dim=-1))
        loss = self.mse_loss(phi_hat,latent)
        self.model.zero_grad()
        loss.backward()
        grad = x.grad.data.detach()
        grad = torch.clip(grad,-0.5,0.5)
        x = x.detach() - 0.001 * grad
        if cal_R:
            R = x - initial_point
            self.R = R.norm(dim=-1).mean().item()
        print(f' inverse step: {k}/{32}. loss={loss.item()}', end="\r")

      print(f' inverse loss: ={loss.item()}')
      if fixed_parameters is None:
          return x
      else:
          return torch.cat([x,y],dim=-1)

  def reverse_action(self, policy, obs, deltas, fragment):
      self.model.eval()
      with torch.no_grad():
          # network_values, decison_set_latent = self.model(obs, decison_set)
          # decison_set_latent = self.model.encode(obs, decison_set)
          est_vals, latent = self.model(obs, policy)
          self.est_vals = est_vals
          decison_set_latent_pos = latent + deltas
          decison_set_latent_pos = decison_set_latent_pos.to(self.dtype)
          if self.soft_bandit_update:
              decison_set_latent_neg = latent - deltas
              decison_set_latent_neg = decison_set_latent_neg.to(self.dtype)

          if self.method == 'ucb':
              d = self.latent_dim
              self.ucb = torch.sqrt(
                  torch.sum(torch.linalg.solve(self.precision[fragment], decison_set_latent_pos.T).T * decison_set_latent_pos,
                            dim=1))
              self.ls_values = decison_set_latent_pos @ self.mu[fragment]
              values_pos = self.ls_values + self.ucb_coeff * self.ucb
              if self.soft_bandit_update:
                  ucb_neg = torch.sqrt(torch.sum(torch.linalg.solve(self.precision[fragment],
                                                                    decison_set_latent_neg.T).T * decison_set_latent_neg,
                                                 dim=1))
                  ls_values_neg = decison_set_latent_neg @ self.mu[fragment]
                  values_neg = ls_values_neg + self.ucb_coeff * ucb_neg

          elif self.method == 'ts':
              if self.b[fragment] > 0:
                  self.sigma2_s = self.b[fragment] * invgamma.rvs(self.a[fragment])
              else:
                  print('Warning: parameter b is negative!')
                  self.sigma2_s = 10.0
              try:
                  w_dist = MultivariateNormal(self.mu[fragment],
                                              precision_matrix=(1 / self.sigma2_s) * self.precision[fragment])
                  w = w_dist.sample()
              except:
                  # Sampling could fail if covariance is not positive definite
                  d = self.param_dim
                  w_dist = MultivariateNormal(torch.zeros(d, device=self.device, dtype=self.dtype),
                                              torch.eye(d, device=self.device, dtype=self.dtype))
                  w = w_dist.sample()
              # decison_set_latent = decison_set_latent[R <= self.decison_set_radius]
              values_pos = torch.matmul(decison_set_latent_pos, w)
              if self.soft_bandit_update:
                values_neg = torch.matmul(decison_set_latent_neg, w)
          else:
              raise ValueError('method is unknown')


      if self.soft_bandit_update:
          V = torch.stack([values_pos.unsqueeze(0), values_neg.unsqueeze(0)], dim=0)
          V = V #/ V.std()
          g_hat = (V[0].float() - V[1].float()) @ deltas
          g_hat /= len(deltas)
          best_arm = latent + self.step_size * g_hat
          # best_arm = decison_set_latent[values.argmax()].float()
      else:
          best_arm = decison_set_latent_pos[values_pos.argmax()]

      # if self.hparams.filter == 'MeanStdFilter':
      #     best_parameters = self.find_parameters(policy, best_arm, obs)# , self.hparams.obs_dim * 2)
      # else:
      best_parameters = self.find_parameters(policy, best_arm, obs)
      # with torch.no_grad():
      #   best_parameters = self.model.decoder(best_arm)
      best_parameters = best_parameters.cpu().numpy()
      best_arm_index = values_pos.argmax()
      # best_value = values.max()

      return best_parameters, best_arm_index, values_pos, values_pos.std()



  def sample_ts(self,fragment=0):
      if self.b[fragment] > 0:
          self.sigma2_s = self.b[fragment] * invgamma.rvs(self.a[fragment])
      else:
          print('Warning: parameter b is negative!')
          self.sigma2_s = 10.0
      w_dist = MultivariateNormal(self.mu[fragment], precision_matrix=(1 / self.sigma2_s) * self.precision[fragment])
      # w = w_dist.sample()
      self.w = w_dist.sample().float()
      if torch.isnan(self.w.mean()):
          print('Cov matrix is not positive definite')
          self.w = self.model.get_last_layer_weights()

  def action(self, decison_set, obs, fragment, ref_point=None):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    # if self.t < self.hparams.initial_pulls:
    #   return  torch.randn(self.context_dim).to(self.device), torch.zeros([]), torch.zeros([])

    self.model.eval()
    with torch.no_grad():
      # est_vals, decison_set_latent = self.model(obs, decison_set)
      # decison_set_latent = self.model.encode(obs, decison_set)
      if self.discrete_dist:
          est_vals, decison_set_latent,  _ = self.model.forward_sample(obs, decison_set)
      else:
        est_vals, decison_set_latent, _, _ = self.model.forward_sample(obs, decison_set)
      self.est_vals = est_vals
      decison_set_latent = decison_set_latent.to(self.dtype)
      if ref_point is not None:
        if self.discrete_dist:
            # ref_latent = self.model.encode(obs[:1], ref_point.unsqueeze(0))
            ref_latent = self.model.sample(obs[:1], ref_point.unsqueeze(0))
        else:
            # ref_latent, _ = self.model.encode(obs[:1], ref_point.unsqueeze(0))
            ref_latent = self.model.sample(obs[:1], ref_point.unsqueeze(0))
        R = decison_set_latent - ref_latent
        self.R = R.norm(dim=-1).mean().item()



    if self.method == 'ucb':
        d = self.latent_dim
        self.ucb = torch.sqrt(torch.sum(torch.linalg.solve(self.precision[fragment],decison_set_latent.T).T * decison_set_latent, dim=1))
        self.ls_values = decison_set_latent @ self.mu[fragment]
        values = self.ls_values + self.ucb_coeff * self.ucb
        if torch.isnan(values).sum() > 0:
          print('cov is not PSD.. using default setting')
          ucb_default = torch.sqrt(torch.sum(torch.linalg.solve(torch.eye(d, device=self.device, dtype=self.dtype),decison_set_latent.T).T * decison_set_latent, dim=1))
          values = self.ls_values + self.ucb_coeff * ucb_default
    elif self.method == 'ts':
        values =  torch.matmul(decison_set_latent,self.w)

    elif self.method == 'network':
        values = est_vals.squeeze()
    else:
        raise ValueError('method is unknown')

    best_arm = decison_set[values.argmax()]
    best_arm_index = values.argmax()
    # best_value = values.max()

    return best_arm,  best_arm_index, values, values.std()



  def compute_returns_and_advantage(self, masks, rewards, values) -> None:
      """
      Post-processing step: compute the lambda-return (TD(lambda) estimate)
      and GAE(lambda) advantage.

      Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
      to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
      where R is the sum of discounted reward with value bootstrap
      (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

      The TD(lambda) estimator has also two special cases:
      - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
      - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

      For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

      :param last_values: state value estimation for the last step (one for each env)
      :param dones: if the last step was a terminal step (one bool for each env).
      """
      # flag = (masks[:,-1] == 0)
      values = values.clone().float()
      # values[torch.where(flag)[0],-1] = rewards[torch.where(flag)[0],-1].float()

      masks = masks[:,:-1]
      rewards = rewards[:,:-1]
      advantages = torch.zeros_like(rewards)
      traj_len = rewards.shape[-1]

      last_gae_lam = 0
      for step in reversed(range(traj_len)):
          next_non_terminal = masks[:,step]
          next_values = values[:,step + 1]
          delta = rewards[:,step] + self.gamma * next_values * next_non_terminal - values[:,step]
          last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
          advantages[:,step] = last_gae_lam
      # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
      # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
      returns = advantages + values[:,:-1]
      return returns

  def extract_features(self):
      data = self.storage.get_data()
      i = 0
      features = []
      labels = []
      policy_vector = []
      for j in range(len(data[::5])):

          exp, context = data[j]
          with torch.no_grad():
              states =  exp.obs
              policy_vector.append(context)
              context = torch.tensor(context).float()
              context = context.unsqueeze(0).to(self.device)
              #Sample from traj because there is corelation between in-traj samples
              idx = 0 #np.random.randint(len(states))# if self.state_based_value else 0
              # idx = h * self.horizon
              if not self.state_based_value:
                  returns = np.sum(exp.reward[idx: idx + self.num_unroll_steps] * self.gamma ** np.arange(
                      len(exp.reward[idx:idx + self.num_unroll_steps])))
                  # returns = torch.tensor(returns, device=self.device, dtype=self.dtype)
                  # rewards = exp.reward[idx: idx + self.num_unroll_steps]
                  # returns = 0
                  # running_return = 0
                  # for r in reversed(rewards):
                  #     running_return = r + self.gamma * running_return
                  #     returns += running_return
                  # returns = returns / len(rewards) * (1 - self.gamma)
              else:
                  states = torch.tensor(np.array(exp.obs[idx:idx + self.num_unroll_steps])).to(self.device).float()
                  target_values, _ = self.target_model(states, context)
                  target_values = target_values[:, 0].unsqueeze(0)
                  returns = self.compute_returns_and_advantage(
                      torch.tensor(exp.mask[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                      torch.tensor(exp.reward[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                      target_values)
                  returns = returns[0, 0].cpu().float().numpy()



              labels.append(returns)


              first_states = torch.tensor(states[idx:idx+1]).float().to(self.device)
              new_z = self.model.sample(first_states, context)
              features.append(new_z.cpu().float().numpy())

      return np.concatenate(features,axis=0), np.stack(policy_vector), np.array(labels)

  def update(self, exp, context, train=True):
    """Updates the posterior using linear bayesian regression formula."""

    # rewards = self.normalize_reward(rewards)
    self.model.eval()
    self.t += 1
    exp.ret = np.sum( exp.reward * self.gamma ** np.arange(
                          len(exp.reward)))
    context = context.astype(np.float64)
    self.storage.save_exp(exp, context)

    # fragment_max = len(exp.reward) // self.horizon
    # while fragment_max >= self.fragments:
    #     self.add_bandit_fragment()
    #     self.fragments += 1


    if self.t % self.update_freq_nn == 0 and self.t >= self.batch_size and train:

      data = self.storage.get_data()
      # data = data[-self.hparams.n_directions:]
      # tic = time.time()
      mean_loss = self.train(data)
      # toc = time.time()
      # print(f'Training time: {toc-tic}')
      self.model.eval()
      # print('model has been updated!')
      # Update the latent representation of every datapoint collected so far
      # print(f'representaion has been updated with mean loss: {mean_loss}')
      for i in range(self.fragments):
          self.precision[i] = self.lambda_prior * torch.eye(self.param_dim, device=self.device, dtype=self.dtype)
          self.f[i] = torch.zeros(self.param_dim, device=self.device, dtype=self.dtype)

      self.a = [self._a0 for _ in range(self.fragments)]
      self.b = [self._b0 for _ in range(self.fragments)]
      self.yy = [0 for _ in range(self.fragments)]
      # self.f[i] =self.precision[i] @ self.model.get_last_layer_weights().to(self.dtype)

      i = 0
      for j in range(len(data)):

          self.a[i] += 0.5
          exp, context = data[j]
          step_idx = exp.step[:-1]
          with torch.no_grad():
              # states = torch.tensor(np.array(exp.obs)).float()
              states =  exp.obs
              context = torch.tensor(context).float()
              # context1 =  torch.stack([context for _ in range(len(states))])
              # target_values, _ = self.target_model(states.to(self.device), context1.to(self.device))
              # target_values = target_values[:, 0].unsqueeze(0)
              # returns = self.compute_returns_and_advantage(torch.tensor(exp.mask).unsqueeze(0),
              #                                              torch.tensor(exp.reward).unsqueeze(0),
              #                                              target_values.cpu())  # returns is 1 shorter than H
              # returns = returns[0].to(self.dtype)
              # context1 = context1[:-1]
              # states   = states[:-1]


          # for i in range(self.fragments):
              # obs_idx = i * self.horizon

              #Sample from traj because there is corelation between in-traj samples
              for h in range(len(states) // self.horizon + 1):
                  idx = 0 #np.random.randint(len(states))# if self.state_based_value else 0
                  # idx = h * self.horizon
                  returns = np.sum(
                      exp.reward[idx:self.num_unroll_steps + idx] * self.gamma ** np.arange(
                          len(exp.reward[idx:idx + self.num_unroll_steps])))
                  if len(exp.reward[idx:]) > self.num_unroll_steps:
                      target_values, _ = self.target_model(states.to(self.device), context1.to(self.device))
                      target_values = target_values[:, 0].unsqueeze(0)
                      returns += exp.mask[idx + self.num_unroll_steps] * target_values[
                          0, idx + self.num_unroll_steps] * self.gamma ** self.num_unroll_steps
                      returns = torch.tensor(returns, device=self.device)
                  returns = torch.tensor(returns, device=self.device, dtype=self.dtype)
                  values = returns.unsqueeze(0)  # torch.tensor(returns, device=self.device)

                  self.yy[i] += values[0] ** 2
                  # contexts =  context1[idx:idx+1].to(self.device)
                  contexts = context.unsqueeze(0).to(self.device)
                  first_states = [] #states[idx:idx+1].float().to(self.device)
                  # values  =returns[idx:idx+1].to(self.device)

                  # if self.discrete_dist:
                  #     new_z = self.model.encode(first_states, contexts)
                  # else:
                  #     new_z, _ = self.model.encode(first_states, contexts)
                  new_z = self.model.sample(first_states, contexts)
                  new_z = new_z.to(self.dtype)

                  # The algorithm could be improved with sequential formulas (cheaper)
                  self.precision[i] += torch.matmul(new_z.T, new_z)
                  self.f[i] += torch.matmul(values, new_z)

      self.mu[i] = torch.linalg.solve(self.precision[i], self.f[i])
      b_upd = 0.5 * (self.yy[i] - torch.matmul(self.mu[i], torch.matmul(self.precision[i], self.mu[i])))
      self.b[i] = self.b0 + b_upd

    else:
        mean_loss = 0
        with torch.no_grad():

            step_idx = exp.step[:-1]
            # states = torch.tensor(np.array(exp.obs)).to(self.device).float()
            states = exp.obs
            context = torch.tensor(context).float().unsqueeze(0).to(self.device)
            # contexts = torch.stack([context for _ in range(len(states))])
            # contexts = contexts.to(self.device)
            # target_values, _ = self.target_model(states, contexts)
            # target_values = target_values[:,0].unsqueeze(0)
            # returns = self.compute_returns_and_advantage(torch.tensor(exp.mask).unsqueeze(0).to(self.device), torch.tensor(exp.reward).unsqueeze(0).to(self.device), target_values)
            # returns = returns[0].to(self.device)
            # states = states[:-1]
            # contexts = contexts[:-1]
            for h in range(len(states) // self.horizon  + 1):
                # idx = h * self.horizon
                idx = 0 #np.random.randint(len(states))# if self.state_based_value else 0
                returns = np.sum(exp.reward[idx: idx + self.num_unroll_steps] * self.gamma ** np.arange(
                    len(exp.reward[idx:idx + self.num_unroll_steps])))
                if len(exp.reward[idx:]) > self.num_unroll_steps:
                    target_values, _ = self.target_model(states, contexts)
                    target_values = target_values[:, 0].unsqueeze(0)
                    returns +=  exp.mask[idx + self.num_unroll_steps] * target_values[0,idx + self.num_unroll_steps] * self.gamma ** self.num_unroll_steps
                returns = torch.tensor(returns, device=self.device, dtype=self.dtype)
                returns = returns.unsqueeze(0) #torch.tensor(returns,device=self.device)


                i = 0

                # returns = returns[idx:idx+1]
                states = states#[idx:idx+1]
                contexts = context#[idx:idx+1]

                # if self.discrete_dist:
                #     phi = self.model.encode(states, contexts)
                # else:
                #     phi, _ = self.model.encode(states, contexts)
                phi = self.model.sample(states, contexts)
                phi = phi.to(self.dtype)
                # Retrain the network on the original data (data_h)
                self.precision[i] += torch.matmul(phi.T, phi)
                self.f[i] += torch.matmul(returns, phi)
                self.yy[i] += returns[0] ** 2
            # self.cov[i] = torch.linalg.inv(self.precision[i])
            self.mu[i] = torch.linalg.solve(self.precision[i], self.f[i])
            # self.mu[i] = torch.matmul(self.cov[i], self.f[i])

            # Inverse Gamma posterior update
            self.a[i] += 0.5
            b_upd = 0.5 * (self.yy[i] - torch.matmul(self.mu[i], torch.matmul(self.precision[i], self.mu[i])))
            self.b[i] = self.b0 + b_upd
    return mean_loss


  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior

  def train(self, data):
        self.model.train()
        dataset = ReprlDatasetVae(data, self.num_unroll_steps, self.stacked_observations, self.horizon, self.state_based_value,
                               augmentation=self.augmentation,
                               permutation=self.permutation,
                               discount=self.gamma,
                               noise_scale=self.noise_scale,
                               average_reward=self.average_reward)

        smplr = torch.utils.data.RandomSampler(dataset,
                                               replacement=True,
                                               num_samples=self.training_iter * self.batch_size)
        dataloader = DataLoader(dataset,
                           batch_size= self.batch_size,
                           sampler=smplr,
                           shuffle=False,
                           pin_memory=True,
                           num_workers=8,)
                           # collate_fn=my_collate)


        mean_loss = 0
        # num_epoch = 20
        # for ep in range(num_epoch):
        num_iter = 0
        for i, sample in enumerate(dataloader):
            self.training_steps += 1
            if self.training_steps % self.target_model_update == 0:
                # self.soft_update_from_to(self.model, self.target_model)
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()
            loss = self.train_step(sample)
            num_iter += 1
            mean_loss += loss
            print( f' Training step: {num_iter}/{self.training_iter}. loss={loss}',end="\r")
            # print( f'Epoch {ep}/{num_epoch}, Training step: {num_iter}/{len(dataloader)}. loss={loss}',end="\r")

        mean_loss = mean_loss / num_iter
        print('Training mean loss:', mean_loss)
        return mean_loss


  def train_step(self, sample):

      sample = sample.to(self.device)
      context = sample.policy

      obs = sample.obs

      if self.state_based_value:
          mask = sample.mask
          reward = sample.reward
          states = sample.obs #BxTxC
          B,T,_ = states.shape
          # states = states.reshape(B * T, -1) #BTxC
          # weights, biases = context
          # weights = [w.unsqueeze(1).expand(-1,T,-1,-1,-1).flatten(start_dim=0,end_dim=1) for w in weights]
          # biases = [b.unsqueeze(1).expand(-1,T,-1,-1).flatten(start_dim=0,end_dim=1) for b in biases]
          # context = [weights,biases]

          with torch.no_grad():
              target_values, _ = self.target_model(states, context, seq_data=True)
              returns = self.compute_returns_and_advantage(mask,reward,target_values)
          value_hat, z,_,_ = self.model.forward_sample(states, context, seq_data= True)
          value_hat = value_hat[:,:-1]
          likelihood = self.mse_loss(value_hat, returns)
          loss = likelihood

      else:
            returns = sample.ret
            value_hat, z,_,_ = self.model.forward_sample(obs, context)
            likelihood = self.mse_loss(value_hat[:, 0], returns)

            if self.discrete_dist:
                value_hat, z, logits = self.model.forward_sample(obs, context)
                shape = logits.shape
                logits = torch.reshape(logits, shape=(*shape[:-1], self.category_size, self.class_size))
                prior_logits = torch.ones_like(logits)
                posterior = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
                prior = td.Independent(td.OneHotCategoricalStraightThrough(logits=prior_logits), 1)
                kld_loss = torch.distributions.kl.kl_divergence(posterior, prior)
                # kld_loss = torch.clip(kld_loss, 0.0, 1.0)
                kld_loss = torch.mean(kld_loss)

            else:
                value_hat, z, mu, log_var  = self.model.forward_sample(obs, context)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            context_hat = self.model.decoder(z)
            decoder_loss = self.mse_loss(context_hat, context)
            # log_sigma = ((context_hat - context) ** 2).mean().sqrt().log()
            # log_sigma = -6 + F.softplus(log_sigma + 6)
            # likelihood_p = 0.5 * torch.pow((context_hat - context) / log_sigma.exp(), 2) + log_sigma #+ 0.5 * np.log(2 * np.pi)
            # likelihood_p = likelihood_p.mean()

            likelihood = self.mse_loss(value_hat[:, 0], returns)

            loss = likelihood + self.kld_coeff * kld_loss + self.decoder_coeff * decoder_loss
            # loss = likelihood + kld_loss




      total_loss = loss.item()
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
      self.optimizer.step()
      # self.scheduler.step()

      return total_loss

#
#   def train(self, data):
#         self.model.train()
#         dataset = NeuralLinearDataset(data, self.num_unroll_steps, self.stacked_observations, self.horizon, self.state_based_value)
#
#         smplr = torch.utils.data.RandomSampler(dataset,
#                                                replacement=True,
#                                                num_samples=self.training_iter * self.batch_size)
#         dataloader = DataLoader(dataset,
#                            batch_size= self.batch_size,
#                            sampler=smplr,
#                            shuffle=False,
#                            pin_memory=True,
#                            num_workers=0,
#                            collate_fn=my_collate)
#
#
#         num_iter = 0
#         mean_loss = 0
#         for i, sample in enumerate(dataloader):
#             self.training_steps += 1
#             if self.training_steps % self.target_model_update == 0:
#                 self.soft_update_from_to(self.model, self.target_model)
#                 self.target_model.eval()
#             loss = self.train_step(sample)
#             num_iter += 1
#             mean_loss += loss
#             print( f' Training step: {num_iter}/{self.training_iter}. loss={loss}',end="\r")
#
#         mean_loss = mean_loss / num_iter
#         print('Training mean loss:', mean_loss)
#         return mean_loss
#
#
#
#
#   def train_step(self, sample):
#       context, obs, _, reward, mask, steps, ret = sample
#       # context = torch.tile(context,[1,obs.shape[1],1])
#       B = context.shape[0]
#
#       cnt = 0
#       obs_arr = []
#       return_arr = []
#       for b in range(B):
#           obs1 = obs[b].to(self.device)
#           H = obs1.shape[0]
#           if H == 1:
#               return_arr.append(reward[b].to(self.device))
#           else:
#
#               cnt += 1
#               context1 = torch.stack([context[b] for _ in range(H)])  #  H x C
#
#               reward1 = reward[b].to(self.device)
#               mask1 = mask[b].to(self.device)
#               context1 = context1.to(self.device)
#
#               if self.state_based_value:
#                   with torch.no_grad():
#                       # target_value, _ = self.target_model(obs1, context1)
#                       # returns = self.compute_returns_and_advantage(mask1.unsqueeze(0), reward1.unsqueeze(0), target_value[:,0].unsqueeze(0))
#                       returns = torch.sum(
#                           reward1[:self.num_unroll_steps] * self.gamma ** torch.arange(len(reward1[:self.num_unroll_steps]), device=self.device))
#                       if len(reward1) > self.num_unroll_steps:
#                           target_value, _ = self.target_model(obs1, context1)
#                           returns +=  mask1[self.num_unroll_steps] * target_value[self.num_unroll_steps, 0] * self.gamma ** self.num_unroll_steps
#                       # returns = torch.tensor(returns, device=self.device)
#               else:
#                   returns = ret[b].to(self.device)
#
#               return_arr.append(returns.unsqueeze(0))
#               # return_arr.append(returns[:,0])
#           obs_arr.append(obs1[0])
#
#       context = context.to(self.device)
#       obs1 = torch.stack(obs_arr)
#       returns = torch.cat(return_arr)
#
#
#       if self.discrete_dist:
#           value_hat, z, logits = self.model.forward_sample(obs1, context)
#           shape = logits.shape
#           logits = torch.reshape(logits, shape=(*shape[:-1], self.category_size, self.class_size))
#           prior_logits = torch.ones_like(logits)
#           posterior = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
#           prior = td.Independent(td.OneHotCategoricalStraightThrough(logits=prior_logits), 1)
#           kld_loss = torch.distributions.kl.kl_divergence(posterior, prior)
#           # kld_loss = torch.clip(kld_loss, 0.0, 1.0)
#           kld_loss = torch.mean(kld_loss)
#
#       else:
#           value_hat, z, mu, log_var  = self.model.forward_sample(obs1, context)
#           kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
#
#       context_hat = self.model.decoder(z)
#       decoder_loss = self.mse_loss(context_hat, context)
#       # log_sigma = ((context_hat - context) ** 2).mean().sqrt().log()
#       # log_sigma = -6 + F.softplus(log_sigma + 6)
#       # likelihood_p = 0.5 * torch.pow((context_hat - context) / log_sigma.exp(), 2) + log_sigma #+ 0.5 * np.log(2 * np.pi)
#       # likelihood_p = likelihood_p.mean()
#
#       likelihood = self.mse_loss(value_hat[:, 0], returns)
#
#       loss = likelihood + self.kld_coeff * kld_loss + self.decoder_coeff * decoder_loss
#       # loss = likelihood + kld_loss
#
#       total_loss = loss.item()
#       self.optimizer.zero_grad()
#       loss.backward()
#       torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
#       self.optimizer.step()
#       self.scheduler.step()
#
#       return total_loss
#


#
# class NeuralLinearDataset(Dataset):
#
#   def __init__(self, data, num_unroll, stacked_observations, horizon, state_based):
#     self.data = data
#     self.num_unroll = num_unroll
#     self.stacked_observations = stacked_observations
#     self.horizon = horizon
#     self.state_based = state_based
#
#   def __len__(self):
#     return len(self.data)
#
#   def __getitem__(self, item):
#     exp = self.data[item][0]
#     traj_len = exp.reward.shape[0]
#     t = 0 #np.random.randint(traj_len)# if self.state_based else 0
#     policy_vector =  self.data[item][1]
#     policy_vector = torch.tensor(policy_vector).float()
#     # t = 0
#     # obss = [torch.tensor(str_to_arr(o)) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations-1]]
#     obss = [torch.tensor(o) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations]]
#     obss = torch.stack(obss,dim=0) # T x H x W x C
#     actions = exp.action[t:t+self.num_unroll + 1]
#     steps = exp.step[t:t+self.num_unroll + 1]
#     rewards = exp.reward[t:t+self.num_unroll + 1]
#     # rewards = exp.reward
#     masks = exp.mask[t:t+self.num_unroll + 1]
#     ret = torch.tensor(exp.ret).float()
#     actions, rewards, masks, steps = torch.tensor(actions).float(), torch.tensor(rewards).float(), torch.tensor(masks).float(), torch.tensor(steps).float()
#     # if t >= traj_len - self.num_unroll:
#     #     obss_pad = torch.zeros([self.num_unroll - traj_len + t] + list(obss.shape[1:]))
#     #     rewards_pad = torch.zeros([self.num_unroll - traj_len + t])
#     #     actions_pad = torch.zeros([self.num_unroll - traj_len + t] + list(actions.shape[1:]))
#     #     masks_pad = torch.zeros([self.num_unroll - traj_len + t])
#     #     obss = torch.cat([obss, obss_pad],dim=0)
#     #     rewards = torch.cat([rewards, rewards_pad], dim=0)
#     #     actions = torch.cat([actions, actions_pad], dim=0)
#     #     masks = torch.cat([masks, masks_pad], dim=0)
#
#     obss = obss.float() #/ 255.0 #uint8 --> float32
#
#     return policy_vector, obss, actions, rewards, masks, steps, ret
#
#
# # def my_collate(batch):
# #   policy_vec = [item[0] for item in batch]
# #   obs = torch.stack([item[1] for item in batch])
# #   action = torch.stack([item[2] for item in batch])
# #   reward = torch.stack([item[3] for item in batch])
# #   mask = torch.stack([item[4] for item in batch])
# #   return [policy_vec, obs, action, reward, mask]
#
# def my_collate(batch):
#   policy_vec = torch.stack([item[0] for item in batch])
#   obs = [item[1] for item in batch]
#   action = [item[2] for item in batch]
#   reward = [item[3] for item in batch]
#   mask = [item[4] for item in batch]
#   step = [item[5] for item in batch]
#   ret = [item[6] for item in batch]
#   return [policy_vec, obs, action, reward, mask, step, ret]