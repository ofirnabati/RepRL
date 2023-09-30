"""Thompson Sampling with linear posterior over a learnt deep representation."""

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from neural_bandit_model_dws import NeuralBanditModelDWS as NeuralBanditModelDWS
from exp_dws.data import ReprlDataset
from scipy.stats import invgamma

class NeuralLinearPosteriorSampling:
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, storage,  device, hparams):

    self.hparams = hparams
    self.storage = storage
    self.no_embedding = self.hparams.no_embedding
    self.augmentation = self.hparams.augmentation
    self.permutation = self.hparams.permutation
    self.discrete_dist = self.hparams.discrete_dist
    self.noise_scale =  self.hparams.noise_aug
    self.average_reward = self.hparams.average_reward

    self._lambda_prior = self.hparams.lambda_prior
    self.device = device



    self.kld_coeff = self.hparams.kld_coeff
    self.decoder_coeff = self.hparams.dec_coeff
    self.horizon = hparams.horizon
    self.dtype = torch.float64
    self.ucb_coeff = hparams.ucb_coeff
    self.state_based_value = False
    self.step_size =  hparams.latent_step_size
    self.sigma2_s = 10.0



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
    self.latent_dim = self.model.latent_dim
    self.param_dim = self.latent_dim

    self.mu = torch.zeros(self.param_dim, device=device, dtype=self.dtype)
    self.f  = torch.zeros(self.param_dim, device=device, dtype=self.dtype)
    self.yy = 0


    self.precision = self.lambda_prior * torch.eye(self.param_dim, device=device, dtype=self.dtype)

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = self._a0
    self.b = self._b0



  def init_model(self):
      self.model = NeuralBanditModelDWS(self.hparams).to(self.device)
      self.target_model = NeuralBanditModelDWS(self.hparams).to(self.device)
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

  def sample_ts(self,fragment=0):
      if self.b > 0:
          self.sigma2_s = self.b * invgamma.rvs(self.a)
      else:
          print('Warning: parameter b is negative!')
          self.sigma2_s = 10.0
      w_dist = MultivariateNormal(self.mu, precision_matrix=(1 / self.sigma2_s) * self.precision)
      # w = w_dist.sample()
      self.w = w_dist.sample().float()
      if torch.isnan(self.w.mean()):
          print('Cov matrix is not positive definite')
          self.w = self.model.get_last_layer_weights()

  def action(self, decison_set,  encode_policy=True):

    """Samples beta's from posterior, and chooses best action accordingly."""

    self.model.eval()
    W1, biases1 = decison_set[0]
    if encode_policy:
        W_batch = [torch.stack([W[i].unsqueeze(-1).float().to(self.device) for W,biases in decison_set]) for i in range(len(W1))]
        biases_batch = [torch.stack([biases[i].unsqueeze(-1).float().to(self.device) for W,biases in decison_set]) for i in range(len(biases1))]
        self.decison_set_4_network = [W_batch, biases_batch]

    with torch.no_grad():
      if self.discrete_dist:
          est_vals, decison_set_latent,  _ = self.model.forward_sample(self.decison_set_4_network, encode_policy=encode_policy)
      else:
        est_vals, decison_set_latent, _, _ = self.model.forward_sample(self.decison_set_4_network, encode_policy=encode_policy)
      self.est_vals = est_vals



    if self.method == 'ucb':
        decison_set_latent = decison_set_latent.to(self.dtype)
        d = self.latent_dim
        self.ucb = torch.sqrt(torch.sum(torch.linalg.solve(self.precision,decison_set_latent.T).T * decison_set_latent, dim=1))
        self.ls_values = decison_set_latent @ self.mu
        values = self.ls_values + self.ucb_coeff * self.ucb
        if torch.isnan(values).sum() > 0:
          print('cov is not positive definite.. using default setting')
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
              policy_vector.append(self.model.concat_weights(context,start_dim=0).cpu().float().numpy())
              weights, biases = context
              weights = tuple([w.unsqueeze(-1).unsqueeze(0).float().to(self.device) for w in weights])
              biases = tuple([b.unsqueeze(-1).unsqueeze(0).float().to(self.device) for b in biases])
              contexts = [weights, biases]
              #Sample from traj because there is corelation between in-traj samples
              idx = 0 #np.random.randint(len(states))# if self.state_based_value else 0
              # idx = h * self.horizon
              if not self.state_based_value:
                  if self.average_reward:
                      rewards = exp.reward[idx: idx + self.num_unroll_steps]
                      returns = 0
                      running_return = 0
                      for r in reversed(rewards):
                          running_return = r + self.gamma * running_return
                          returns += running_return
                      returns = returns / len(rewards)
                  else:
                      returns = np.sum(exp.reward[idx: idx + self.num_unroll_steps] * self.gamma ** np.arange(
                          len(exp.reward[idx:idx + self.num_unroll_steps])))
                  first_states = torch.tensor(np.array(states[idx:idx+1])).float().to(self.device)

              else:
                  states = torch.tensor(np.array(exp.obs[idx:idx + self.num_unroll_steps])).to(self.device).float()
                  target_values, _ = self.target_model(states, contexts)
                  target_values = target_values[:, 0].unsqueeze(0)
                  returns = self.compute_returns_and_advantage(
                      torch.tensor(exp.mask[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                      torch.tensor(exp.reward[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                      target_values)
                  returns = returns[0, 0].cpu().float().numpy()
                  first_states = states[0:1]



              labels.append(returns)


              new_z = self.model.sample(first_states, contexts)
              features.append(new_z.cpu().float().numpy())

      return np.concatenate(features,axis=0), np.stack(policy_vector), np.array(labels)

  def update(self, exp, context, train=True):
    """Updates the posterior using linear bayesian regression formula."""

    # rewards = self.normalize_reward(rewards)
    self.model.eval()




    if self.t % self.update_freq_nn == 0 and self.t >= self.batch_size and train:

          data = self.storage.get_data()
          mean_loss = self.train(data)
          self.model.eval()
          self.precision = self.lambda_prior * torch.eye(self.param_dim, device=self.device, dtype=self.dtype)
          self.f = torch.zeros(self.param_dim, device=self.device, dtype=self.dtype)

          self.a = self._a0
          self.b = self._b0
          self.yy = 0

          for j in range(len(data)):

              self.a += 0.5
              exp, context = data[j]
              with torch.no_grad():
                  states =  exp.obs

                  #Sample from traj because there is corelation between in-traj samples
                  # for h in range(len(states) // self.horizon + 1):

                  weights, biases = context
                  weights = tuple([w.unsqueeze(-1).unsqueeze(0).float().to(self.device) for w in weights])
                  biases = tuple([b.unsqueeze(-1).unsqueeze(0).float().to(self.device) for b in biases])
                  contexts= [weights, biases]

                  idx = np.random.randint(len(states) - self.num_unroll_steps) if self.state_based_value else 0

                  if not self.state_based_value:
                      if self.average_reward:
                          rewards = exp.reward[idx: idx + self.num_unroll_steps]
                          returns = 0
                          running_return = 0
                          for r in reversed(rewards):
                              running_return = r + self.gamma * running_return
                              returns += running_return
                          returns = returns / len(rewards)
                      else:
                          returns = np.sum(exp.reward[idx: idx + self.num_unroll_steps] * self.gamma ** np.arange(
                              len(exp.reward[idx:idx + self.num_unroll_steps])))
                      returns = torch.tensor(returns, device=self.device, dtype=self.dtype)

                  else:
                      states = torch.tensor(np.array(exp.obs[idx:idx + self.num_unroll_steps])).to(self.device).float()
                      target_values, _ = self.target_model(states, contexts)
                      target_values = target_values[:, 0].unsqueeze(0)
                      returns = self.compute_returns_and_advantage(
                          torch.tensor(exp.mask[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                          torch.tensor(exp.reward[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device),
                          target_values)
                      returns = returns[0, 0].to(self.device).to(self.dtype)


                  values = returns.unsqueeze(0)  # torch.tensor(returns, device=self.device)

                  self.yy += values[0] ** 2

                  first_states = torch.tensor(np.array(exp.obs[idx:idx+1])).float().to(self.device)
                  new_z = self.model.sample(first_states, contexts)
                  new_z = new_z.to(self.dtype)

                  # The algorithm could be improved with sequential formulas (cheaper)
                  self.precision += torch.matmul(new_z.T, new_z)
                  self.f += torch.matmul(values, new_z)

          self.mu = torch.linalg.solve(self.precision, self.f)
          b_upd = 0.5 * (self.yy - torch.matmul(self.mu, torch.matmul(self.precision, self.mu)))
          self.b = self.b0 + b_upd

    else:
        mean_loss = 0
        with torch.no_grad():

            # step_idx = exp.step[:-1]
            states = exp.obs
            weights, biases = context
            weights = tuple([w.unsqueeze(-1).unsqueeze(0).float().to(self.device) for w in weights])
            biases = tuple([b.unsqueeze(-1).unsqueeze(0).float().to(self.device) for b in biases])
            contexts = [weights, biases]

            # for h in range(len(states) // self.horizon  + 1):
            # idx = h * self.horizon
            idx = np.random.randint(len(states) - self.num_unroll_steps) if self.state_based_value else np.random.randint(len(states)) #0
            obs = torch.tensor( np.array(exp.obs[idx:idx + 1])).float().to(self.device)



            if not self.state_based_value:
                if self.average_reward:
                    rewards = exp.reward[idx: idx + self.num_unroll_steps]
                    returns = 0
                    running_return = 0
                    for r in reversed(rewards):
                        running_return = r + self.gamma * running_return
                        returns += running_return
                    returns = returns / len(rewards)
                else:
                    returns = np.sum(exp.reward[idx: idx + self.num_unroll_steps] * self.gamma ** np.arange(
                        len(exp.reward[idx:idx + self.num_unroll_steps])))
                returns = torch.tensor(returns, device=self.device, dtype=self.dtype)
            else:
                states = torch.tensor(np.array(exp.obs[idx:idx + self.num_unroll_steps])).to(self.device).float()
                target_values, _ = self.target_model(states, contexts)
                target_values = target_values[:, 0].unsqueeze(0)
                returns = self.compute_returns_and_advantage(torch.tensor(exp.mask[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device), torch.tensor(exp.reward[idx:idx + self.num_unroll_steps]).unsqueeze(0).to(self.device), target_values)
                returns = returns[0,0].to(self.device).to(self.dtype)
            returns = returns.unsqueeze(0) #torch.tensor(returns,device=self.device)



            phi = self.model.sample(obs, contexts)
            phi = phi.to(self.dtype)
            # Retrain the network on the original data (data_h)
            self.precision += torch.matmul(phi.T, phi)
            self.f += torch.matmul(returns, phi)
            self.yy += returns[0] ** 2
            self.mu = torch.linalg.solve(self.precision, self.f)

            # Inverse Gamma posterior update
            self.a += 0.5
            b_upd = 0.5 * (self.yy - torch.matmul(self.mu, torch.matmul(self.precision, self.mu)))
            self.b = self.b0 + b_upd
    return mean_loss

  def train(self, data):
        self.model.train()
        dataset = ReprlDataset(data, self.num_unroll_steps, self.stacked_observations, self.horizon, self.state_based_value,
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
      context = (sample.weights, sample.biases)

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
          # value_hat, z,_,_ = self.model.forward_sample(states, context, seq_data= True)
          value_hat, z,_,_ = self.model.forward_sample(states[:,0], context)
          value_hat = value_hat #[:,:-1]
          likelihood = self.mse_loss(value_hat, returns[:,0])

      else:
            returns = sample.ret
            value_hat, z,_,_ = self.model.forward_sample(obs, context)
            likelihood = self.mse_loss(value_hat[:, 0], returns)

      loss = likelihood

      total_loss = loss.item()
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
      self.optimizer.step()
      # self.scheduler.step()

      return total_loss

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
