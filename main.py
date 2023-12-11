
import torch
import random
import time
import os
import pickle
import numpy as np
import gym
import ray
import utils as utils
from utils import str2bool
import optimizers as optimizers
from policies import *
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shared_noise import SharedNoiseTableSet, create_shared_noise
from neural_linear import NeuralLinearPosteriorSampling

from sparseMuJoCo.envs.mujoco.hopper_v0 import SparseHopperV0 as SparseHopperV0
from sparseMuJoCo.envs.mujoco.humanoid_v0 import SparseHumanoidV0 as SparseHumanoidV0
from sparseMuJoCo.envs.mujoco.half_cheetah_v0 import SparseHalfCheetahV0 as SparseHalfCheetahV0
from sparseMuJoCo.envs.mujoco.ant_v0 import SparseAntV0 as SparseAntV0
from sparseMuJoCo.envs.mujoco.walker2d_v0 import SparseWalker2dV0 as SparseWalker2dV0
from gridworlds.envs.gridworld_gauss_reward import GaussGridWorld

from storage import Storage
import wandb


# register(id = 'SparseHalfCheetah-v0',entry_point = 'half_cheetah_sparse:SparseHalfCheetahDirEnv', max_episode_steps=1000)

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 data_rollout_length=1000,
                 delta_std=0.02,
                 samples_per_policy=1,
                 state_based_value=False,
                 ):


        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(env_seed)

        # initialize OpenAI environment for each worker
        if env_name == 'SparseHalfCheetah-v0':
            self.env = SparseHalfCheetahV0()
        elif env_name == 'SparseHopper-v0':
            self.env = SparseHopperV0()
        elif env_name == 'SparseHumanoid-v0':
            self.env = SparseHumanoidV0()
        elif env_name == 'SparseAnt-v0':
            self.env = SparseAntV0()
        elif env_name == 'SparseWalker2d-V0':
            self.env = SparseWalker2dV0()
        elif env_name == 'GaussGridWorld':
            self.env = GaussGridWorld()
        else:
            self.env = gym.make(env_name)
        self.env.reset(seed=env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.

        self.policy_params = policy_params

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'discrete':
            self.policy = DiscretePolicy(policy_params)
        elif policy_params['type'] == 'nn':
            self.policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'],
                                          policy_params['ob_filter'],
                                          device='cpu:0')
        else:
            raise NotImplementedError
        self.deltas = SharedNoiseTableSet(deltas,  self.policy.get_weights(), env_seed + 7)
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.data_rollout_length = data_rollout_length
        self.samples_per_policy = samples_per_policy
        self.state_based_value = state_based_value
        if self.state_based_value:
            self.ob = self.env.reset()

        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        # assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    
    def get_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        # assert self.policy_params['type'] == 'linear'
        return self.policy.get_stats()

    def rollout(self, shift = 0., rollout_length = None, evaluate=False):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """

        obss    = []
        masks   = []
        actions = []
        rewards = []
        step_id   = []


        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        if self.state_based_value and not evaluate:
            ob = self.ob
        else:
            ob = self.env.reset()
        for i in range(rollout_length):
            if len(ob.shape) > 1:
                ob = ob.flatten()
            obss.append(ob)
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            masks.append(1 - done)
            actions.append(action)
            rewards.append(reward - shift)
            step_id.append(i)

            if done:
                if self.state_based_value and not evaluate:
                    ob = self.env.reset()
                else:
                    break

        if self.state_based_value and not evaluate:
            self.ob = ob



        return total_reward, steps, obss, np.stack(actions), np.array(rewards), np.array(masks), np.array(step_id)

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False, idxes=None):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx, obss_arr, actions_arr, rewards_arr, masks_arr, step_id_arr= [], [], [], [], [], [], []
        steps = 0
        if idxes is not None:
            num_rollouts = len(idxes)

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                reward, r_steps, obss, actions, rewards, masks, step_id = self.rollout(shift = 0., rollout_length=self.rollout_length,evaluate=evaluate)
                rollout_rewards.append(reward)
                obss_arr.append(obss)
                actions_arr.append(actions)
                rewards_arr.append(rewards)
                masks_arr.append(masks)
                step_id_arr.append(step_id)

            else:
                idx, delta = self.deltas.get_delta()

                for _ in range(self.samples_per_policy):
                    # delta = (self.delta_std * delta).reshape(w_policy.shape)
                    deltas_idx.append(idx)

                    # set to true so that state statistics are updated
                    self.policy.update_filter = True

                    # compute reward and number of timesteps used for positive perturbation rollout
                    W,bias = w_policy
                    W_noise, bias_noise = delta
                    noisy_weights = [w + W_noise[k] * self.delta_std for k,w in enumerate(W)]
                    noisy_bias = [b + bias_noise[k] * self.delta_std for k,b in enumerate(bias)]

                    self.policy.update_weights([noisy_weights, noisy_bias])
                    # self.policy.update_weights(w_policy)
                    pos_reward, pos_steps, pos_obss, pos_actions, pos_rewards, pos_masks, pos_step_id  = self.rollout(shift = shift, rollout_length=self.data_rollout_length)

                    # compute reward and number of timesteps used for negative pertubation rollout
                    noisy_weights = [w - W_noise[k] * self.delta_std for k,w in enumerate(W)]
                    noisy_bias = [b - bias_noise[k] * self.delta_std for k,b in enumerate(bias)]
                    self.policy.update_weights([noisy_weights, noisy_bias])
                    neg_reward, neg_steps, neg_obss, neg_actions, neg_rewards, neg_masks, neg_step_id  = self.rollout(shift = shift, rollout_length=self.data_rollout_length)
                    steps += pos_steps + neg_steps

                    rollout_rewards.append([pos_reward, neg_reward])
                    obss_arr.append([pos_obss, neg_obss])
                    actions_arr.append([pos_actions, neg_actions])
                    rewards_arr.append([pos_rewards, neg_rewards])
                    masks_arr.append([pos_masks, neg_masks])
                    step_id_arr.append([pos_step_id, neg_step_id])

        return {'deltas_idx': deltas_idx,
                'rollout_rewards': rollout_rewards,
                'steps' : steps,
                "obss_arr": obss_arr,
                "actions_arr": actions_arr,
                "rewards_arr": rewards_arr,
                "masks_arr": masks_arr,
                "step_id_arr": step_id_arr}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class Learner(object):

    def __init__(self,
                 env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=640,
                 num_bandit_deltas=320,
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 data_rollout_length=1000,
                 step_size=0.01,
                 shift=0,
                 params=None,
                 seed=123,
                 storage=None,
                 bandit_algo=None,
                 device='cpu',
                 save_exp=False,
                 soft_bandit_update=True):


        env = gym.make(env_name)

        self.env_name = env_name
        self.timesteps = 0
        self.rollouts = 0
        self.save_exp = save_exp
        if save_exp:
            self.history = []

        if type(env.observation_space) is gym.spaces.box.Box:
            self.ob_size = np.prod(env.observation_space.shape)
        else:
            self.ob_size = env.observation_space.n
        if type(env.action_space) is gym.spaces.box.Box:
            self.action_size = np.prod(env.action_space.shape)
        else:
            self.action_size = env.action_space.n

        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.num_bandit_deltas = num_bandit_deltas
        self.rollout_length = rollout_length
        self.data_rollout_length = data_rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.delta_bandit_std = args.delta_bandit_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.device = device

        self.bandit_algo = bandit_algo
        print('latent dim:', bandit_algo.latent_dim)
        self.storage = storage
        self.horizon = params['horizon']
        self.discount = params['discount']
        self.eval_freq = params['eval_freq']
        self.max_timesteps = params['max_timesteps']
        self.explore_coeff = params['explore_coeff']
        self.soft_bandit_update = soft_bandit_update
        self.samples_per_policy = params['samples_per_policy']
        self.do_tsne = params['do_tsne']
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()

        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      data_rollout_length=data_rollout_length,
                                      delta_std=delta_std,
                                      samples_per_policy=self.samples_per_policy,
                                      ) for i in range(num_workers)]


        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'discrete':
            self.policy = DiscretePolicy(policy_params)
        elif policy_params['type'] == 'nn':
            self.policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'],
                                          policy_params['ob_filter'],
                                          device='cpu:0')
        else:
            raise NotImplementedError

        self.w_policy = self.policy.get_weights()
        self.deltas = SharedNoiseTableSet(ray.get(deltas_id), self.w_policy, seed=seed + 3)
        self.policy_history_set = [self.w_policy for _ in range(params['policy_history_set_size'])]
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        if num_rollouts > 0:
            rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts = num_rollouts,
                                                     shift = self.shift,
                                                     evaluate=evaluate) for worker in self.workers]
        else:
            rollout_ids_one = []
            results_one = []

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results
        if num_rollouts > 0:
            results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []


        loss = None
        rollout_len = 0
        for result in results_one:
            if not evaluate:
                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                for j in range(num_rollouts * self.samples_per_policy):
                    delta = self.deltas.get(result['deltas_idx'][j])
                    W, bias = self.w_policy
                    W_noise, bias_noise = delta
                    noisy_weights_plus = tuple([w + W_noise[k] * self.delta_std for k, w in enumerate(W)])
                    noisy_bias_plus = tuple([b + bias_noise[k] * self.delta_std for k, b in enumerate(bias)])
                    noisy_weights_neg = tuple([w - W_noise[k] * self.delta_std for k, w in enumerate(W)])
                    noisy_bias_neg = tuple([b - bias_noise[k] * self.delta_std for k, b in enumerate(bias)])


                    exp_pos = utils.DictListObject()
                    exp_pos.obs    = result['obss_arr'][j][0]
                    exp_pos.action = result['actions_arr'][j][0]
                    exp_pos.reward = result['rewards_arr'][j][0]
                    exp_pos.mask   = result['masks_arr'][j][0]
                    exp_pos.step   = result['step_id_arr'][j][0]
                    if self.params['filter'] == 'MeanStdFilter':
                        mu, std_matrix = ray.get(self.workers[0].get_stats.remote())
                        noisy_weights_plus = std_matrix + noisy_weights_plus
                        noisy_bias_plus = mu + noisy_bias_plus

                    update_par = [noisy_weights_plus, noisy_bias_plus]
                    loss_pos = self.bandit_algo.update(exp_pos, update_par)
                    # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                    if loss_pos is not None:
                        loss = loss_pos


                    exp_neg = utils.DictListObject()
                    exp_neg.obs    = result['obss_arr'][j][1]
                    exp_neg.action = result['actions_arr'][j][1]
                    exp_neg.reward = result['rewards_arr'][j][1]
                    exp_neg.mask   = result['masks_arr'][j][1]
                    exp_neg.step   = result['step_id_arr'][j][1]
                    if self.params['filter'] == 'MeanStdFilter':
                        mu, std_matrix = ray.get(self.workers[0].get_stats.remote())
                        noisy_weights_neg = std_matrix + noisy_weights_neg
                        noisy_bias_neg = mu + noisy_bias_neg
                    update_par = [noisy_weights_neg, noisy_bias_neg]
                    loss_neg = self.bandit_algo.update(exp_neg, update_par)
                    # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                    if loss_neg is not None:
                        loss = loss_neg

            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']


        for result in results_two:
            if not evaluate:

                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                for j in range(self.samples_per_policy):
                    delta = self.deltas.get(result['deltas_idx'][j])
                    W, bias = self.w_policy
                    W_noise, bias_noise = delta
                    noisy_weights_plus = tuple([w + W_noise[k] * self.delta_std for k, w in enumerate(W)])
                    noisy_bias_plus = tuple([b + bias_noise[k] * self.delta_std for k, b in enumerate(bias)])
                    noisy_weights_neg = tuple([w - W_noise[k] * self.delta_std for k, w in enumerate(W)])
                    noisy_bias_neg = tuple([b - bias_noise[k] * self.delta_std for k, b in enumerate(bias)])
                    exp_pos = utils.DictListObject()
                    exp_pos.obs = result['obss_arr'][j][0]
                    exp_pos.action = result['actions_arr'][j][0]
                    exp_pos.reward = result['rewards_arr'][j][0]
                    exp_pos.mask = result['masks_arr'][j][0]
                    exp_pos.step = result['step_id_arr'][j][0]
                    if self.params['filter'] == 'MeanStdFilter':
                        mu, std_matrix = ray.get(self.workers[0].get_stats.remote())
                        noisy_weights_plus = std_matrix + noisy_weights_plus
                        noisy_bias_plus = mu + noisy_bias_plus
                    update_par = [noisy_weights_plus, noisy_bias_plus]
                    loss_pos = self.bandit_algo.update(exp_pos, update_par)
                    # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                    if loss_pos is not None:
                        loss = loss_pos

                    exp_neg = utils.DictListObject()
                    exp_neg.obs = result['obss_arr'][j][1]
                    exp_neg.action = result['actions_arr'][j][1]
                    exp_neg.reward = result['rewards_arr'][j][1]
                    exp_neg.mask = result['masks_arr'][j][1]
                    exp_neg.step = result['step_id_arr'][j][1]
                    if self.params['filter'] == 'MeanStdFilter':
                        mu, std_matrix = ray.get(self.workers[0].get_stats.remote())
                        noisy_weights_neg = std_matrix + noisy_weights_neg
                        noisy_bias_neg = mu + noisy_bias_neg
                    update_par = [noisy_weights_neg, noisy_bias_neg]
                    loss_neg = self.bandit_algo.update(exp_neg, update_par)
                    # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                    if loss_neg is not None:
                        loss = loss_neg

            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)

        self.timesteps += rollout_len
        if not evaluate:
            print(f'Average rollout length:{rollout_len / (num_deltas * 2 * self.samples_per_policy)}')
            print('Average reward of collected rollouts:', rollout_rewards.mean())
        else:
            print('EVAL: Average reward of collected rollouts:', rollout_rewards.mean())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards, results_one

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        if np.std(rollout_rewards) > 0:
            rollout_rewards /= np.std(rollout_rewards)


        # aggregate rollouts to form g_hat, the zero-order gradient
        g_hat  = utils.batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  [self.deltas.get(idx)
                                                   for idx in deltas_idx])

        ## Bandits from here
        t1 = time.time()
        W, bias = self.w_policy
        W = [torch.zeros_like(w) for w in W]
        bias = [torch.zeros_like(b) for b in bias]
        G = [tuple(W), tuple(bias)]
        with torch.no_grad():
                if self.bandit_algo.method == 'ts':
                    self.bandit_algo.sample_ts()
                if self.params['filter'] == 'MeanStdFilter':
                    mu, std_matrix = ray.get(self.workers[0].get_stats.remote())
                decison_set = []
                values = []
                idxes, deltas = self.deltas.get_deltas(self.num_bandit_deltas * len(self.policy_history_set))
                values_p = []
                for i,p in enumerate(self.policy_history_set):
                    decison_set_pos= []
                    decison_set_pos1, decison_set_neg1 = [], []

                    W, bias  = p
                    for delta in deltas[i*self.num_bandit_deltas : (i+1) * self.num_bandit_deltas]:
                        W_noise, bias_noise = delta
                        noisy_weights_plus = tuple([w + W_noise[k] * self.delta_std for k, w in enumerate(W)])
                        noisy_bias_plus = tuple([b + bias_noise[k] * self.delta_std for k, b in enumerate(bias)])
                        if self.params['filter'] == 'MeanStdFilter':
                            noisy_weights_plus1 = std_matrix + noisy_weights_plus
                            noisy_bias_plus1 = mu + noisy_bias_plus
                        else:
                            noisy_weights_plus1 =  noisy_weights_plus
                            noisy_bias_plus1 = noisy_bias_plus
                        decison_set_pos.append([noisy_weights_plus, noisy_bias_plus])
                        decison_set_pos1.append([noisy_weights_plus1, noisy_bias_plus1])
                        if self.soft_bandit_update:
                            noisy_weights_neg = tuple([w - W_noise[k] * self.delta_std for k, w in enumerate(W)])
                            noisy_bias_neg = tuple([b - bias_noise[k] * self.delta_std for k, b in enumerate(bias)])
                            if self.params['filter'] == 'MeanStdFilter':
                                noisy_weights_neg1 = std_matrix + noisy_weights_neg
                                noisy_bias_neg1 = mu + noisy_bias_neg
                            else:
                                noisy_weights_neg1 = noisy_weights_neg
                                noisy_bias_neg1 = noisy_bias_neg
                            decison_set_neg1.append([noisy_weights_neg1, noisy_bias_neg1])


                    curr_step_size = 0
                    p1, best_idx_plus, values_plus, _ = self.bandit_algo.action(decison_set_pos1)
                    decison_set += decison_set_pos
                    values_p.append(values_plus)
                    if self.soft_bandit_update:
                        p1, best_idx_minus, values_minus, _ = self.bandit_algo.action(decison_set_neg1)

                        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
                        V = torch.stack([values_plus, values_minus], dim=1)
                        V = V.cpu().numpy()
                        V_std = np.std(V)
                        if V_std == 0:
                            print('Poor representation.. std is zero')
                            bandit_rollout_rewards = V
                        else:
                            bandit_rollout_rewards = V / V_std

                        G1 = utils.batched_weighted_sum(bandit_rollout_rewards[:,0] - bandit_rollout_rewards[:,1],
                                                                  [self.deltas.get(idx)
                                                                   for idx in idxes])

                        W_G1, bias_G1 = G1
                        W_G, bias_G = G
                        WW = [w +  w1 for w, w1 in zip(W_G, W_G1)]
                        bbias = [b +  b1 for b, b1 in zip(bias_G, bias_G1)]
                        G = [tuple(WW),tuple(bbias)]

                values_p = torch.cat(values_p, -1)
                values.append(values_p)
        if  self.soft_bandit_update:
            W, bias = self.w_policy
            W, bias = list(W), list(bias)
            W_g, bias_g = g_hat
            W_G, bias_G = G
            for idx, w in enumerate(W):
                overall_g = (1 - self.explore_coeff) * W_g[idx] + self.explore_coeff * W_G[
                    idx]
                W[idx] -= self.optimizer._compute_step(overall_g)
            for idx, b in enumerate(bias):
                overall_g = (1 - self.explore_coeff) * bias_g[idx] + self.explore_coeff * bias_G[
                    idx]
                bias[idx] -= self.optimizer._compute_step(overall_g)
            self.w_policy = [tuple(W), tuple(bias)]
        else:
            values = torch.stack(values)
            values_max_idx = values.argmax(-1)
            best_policy_idx, _ = torch.mode(values_max_idx)
            self.w_policy = decison_set[best_policy_idx]
        self.policy_history_set.pop(0)
        self.policy_history_set.append(self.w_policy)
        t2 = time.time()
        print(f'Curr step size:{curr_step_size}')
        print('time to compute gradient', t2 - t1)

        return G, loss


    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        g_hat, loss = self.aggregate_rollouts()
        return loss

    def train(self, num_iter):

        loss = 0
        i = 0
        while self.timesteps <= self.max_timesteps:


            if (i % self.eval_freq == 0):
                rewards, results_one = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                if self.save_exp:
                    traj = []
                    num_rollouts = int(100 / self.num_workers)
                    for result in results_one:
                            # self.timesteps += result["steps"]
                            for j in range(num_rollouts):
                               traj.append(result['obss_arr'][j])

                
                print(sorted(self.params.items()))

                log = {"iteration": i + 1,
                       "memory size": self.storage.size(),
                       "value": np.mean(rewards),
                       "value std": np.std(rewards),
                       "value max": np.max(rewards),
                       "value min": np.min(rewards),
                       "global_step": self.timesteps,
                       "rollout_num": self.rollouts,
                       "mean_loss": loss,
                       "sigma": self.bandit_algo.sigma2_s,
                       "b": self.bandit_algo.b,
                       "ls_bandit_val": 0 if self.params[
                                                 'method'] != 'ucb' or self.bandit_algo.t == 0 else self.bandit_algo.ls_values.mean().item(),
                       "network_val": 0 if self.params[
                                               'method'] != 'ucb' or self.bandit_algo.t == 0 else self.bandit_algo.est_vals.mean().item(),
                       "ucb mean": 0 if self.params[
                                            'method'] != 'ucb' or self.bandit_algo.t == 0 else self.bandit_algo.ucb.mean().item(),
                       "ucb std": 0 if self.params[
                                           'method'] != 'ucb' or self.bandit_algo.t == 0 else self.bandit_algo.ucb.std().item(),
                       "overall ucb": 0 if self.params[
                                               'method'] != 'ucb' or self.bandit_algo.t == 0 else self.bandit_algo.ls_values.mean().item() +
                                                                                                 self.params[
                                                                                                     'ucb_coeff'] * self.bandit_algo.ucb.mean().item(),
                       "radius": self.bandit_algo.R
                       }
                if self.bandit_algo.t > 0 and (i % (self.eval_freq * 20) == 0) and not self.bandit_algo.no_embedding and self.do_tsne:
                    features, policy_vectors, labels = self.bandit_algo.extract_features()
                    D = {'features': features, 'policy_vectors':policy_vectors, 'labels':labels}
                    with open('saved_tsne_data_' + self.env_name +'.pkl', 'wb') as f:
                        pickle.dump(D, f)
                    if self.bandit_algo.latent_dim == 2:
                        low_dim_features = features
                    else:
                        low_dim_features = TSNE(
                            n_components=2, random_state=42, perplexity=50, n_jobs=1
                        ).fit_transform(features)
                    data = [
                        [*list(x), y]
                        for (x, y) in zip(low_dim_features, labels)
                    ]
                    df = pd.DataFrame(data, columns=["f1", "f2", "value"])
                    fig, ax = plt.subplots()
                    extra_params = dict(
                        palette="RdBu"
                    )  # sns.cubehelix_palette(as_cmap=True))
                    sns.scatterplot(
                        data=df,
                        x="f1",
                        y="f2",
                        hue="value",
                        ax=ax,
                        **extra_params,
                    )
                    log["test/scatter_latent"] = wandb.Image(plt)
                    plt.close(fig)

                    low_dim_features = TSNE(
                            n_components=2, random_state=42, perplexity=50, n_jobs=1
                        ).fit_transform(policy_vectors)
                    data = [
                        [*list(x), y]
                        for (x, y) in zip(low_dim_features, labels)
                    ]
                    df = pd.DataFrame(data, columns=["f1", "f2", "value"])
                    fig, ax = plt.subplots()
                    extra_params = dict(
                        palette="RdBu"
                    )  # sns.cubehelix_palette(as_cmap=True))
                    sns.scatterplot(
                        data=df,
                        x="f1",
                        y="f2",
                        hue="value",
                        ax=ax,
                        **extra_params,
                    )
                    log["test/scatter_policy_vec"] = wandb.Image(plt)
                    plt.close(fig)


                wandb.log(log)

            t1 = time.time()
            loss1 = self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i, ' done')
            self.rollouts += self.num_bandit_deltas * 2

            if loss1 is not None:
                loss = loss1

            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
            i += 1

        if self.save_exp:
            id = 0
            name = './gridworld'
            while os.path.exists(name+str(id)+'.npy'):
                id += 1
            np.save(name+'_'+wandb.run.id+'.npy',self.history)

        return

def run_exp(args):

    params = vars(args)
    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])


    if type(env.observation_space) is gym.spaces.box.Box:
        ob_dim = np.prod(env.observation_space.shape)
    else:
        ob_dim = env.observation_space.n
    if type(env.action_space) is gym.spaces.box.Box:
        ac_dim = np.prod(env.action_space.shape)
    else:
        ac_dim = env.action_space.n


    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type': params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    if policy_params['type'] == 'linear':
        policy = LinearPolicy(policy_params)
    elif policy_params['type'] == 'discrete':
        policy = DiscretePolicy(policy_params)
    elif policy_params['type'] == 'nn':
        policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'],
                                           policy_params['ob_filter'],
                                           device='cpu:0')
    else:
        raise NotImplementedError

    W, bias = policy.get_weights()
    if params['filter'] == 'MeanStdFilter':
        mu, std_matrix = policy.get_stats()
        W = std_matrix + W
        if len(bias) > 0:
            bias = mu + bias
        else:
            bias = mu


    weight_shapes = tuple(w.shape[:2] for w in W)
    bias_shapes = tuple(b.shape[:1] for b in bias)

    args.weight_shapes = weight_shapes
    args.bias_shapes = bias_shapes
    args.obs_dim = ob_dim

    wandb.init(project="PolicySpaceOptimization_new", entity="ofirnabati", config=args.__dict__)
    wandb.run.name = args.env_name + '_' + 'neural_es_dws'
    if args.policy_type == 'nn':
        wandb.run.name = wandb.run.name + '_nn'
    wandb.run.save()

    storage = Storage(config=args)
    bandit_algo = NeuralLinearPosteriorSampling(storage, device, args)

    Learmer_model = Learner(env_name=params['env_name'],
                  policy_params=policy_params,
                  num_workers=params['n_workers'],
                  num_deltas=params['n_directions'],
                  num_bandit_deltas = params['n_bandit_directions'],
                  deltas_used=params['deltas_used'],
                  step_size=params['step_size'],
                  delta_std=params['delta_std'],
                  logdir=logdir,
                  rollout_length=params['rollout_length'],
                  data_rollout_length=params['data_rollout_length'],
                  shift=params['shift'],
                  params=params,
                  seed = params['seed'],
                  bandit_algo=bandit_algo,
                  storage=storage,
                  device=params['device'],
                  save_exp=params['save_exp'],
                  soft_bandit_update=params['soft_bandit_update'])
        
    Learmer_model.train(params['n_iter'])
       
    return 




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--max_timesteps', '-max_t', type=int, default=5e6)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--n_bandit_directions', type=int, default=512)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--latent_step_size',  type=float, default=1.0)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--delta_bandit_std',  type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--explore_coeff', type=float, default=0.1)
    parser.add_argument('--do_tsne', type=str2bool, default=False)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_cpus', type=int, default=80)
    parser.add_argument('--device', type=int, default=0)

    #bandits args
    parser.add_argument("--discount", type=float, default=0.995,
                        help="discount factor (default: 0.9996)")
    parser.add_argument("--gae-lambda", type=float, default=0.97,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--state_latent_dim", type=int, default=64)
    parser.add_argument("--use_target_network", type=str2bool, default=True)
    parser.add_argument("--policy_history_set_size", type=int, default=1)
    parser.add_argument("--num_unroll_steps", type=int, default=64)
    parser.add_argument("--method", type=str, default='ts')
    parser.add_argument("--target_model_update", type=int, default=500)
    parser.add_argument("--a0", type=float, default=6)
    parser.add_argument("--b0", type=float, default=10)
    parser.add_argument("--ucb_coeff", type=float, default=10.0)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--memory_size", type=int, default=5000000)
    parser.add_argument("--layers_size", type=int, default=[4096])
    parser.add_argument("--lr_step_size", type=int, default=1000000000)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--training_freq_network", type=int, default=50)
    parser.add_argument("--training_iter", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--kld_coeff", type=float, default=0.1)
    parser.add_argument("--dec_coeff", type=float, default=0.1)
    parser.add_argument("--noise_aug", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument("--state_based_value", type=str2bool, default=False)
    parser.add_argument("--average_reward", type=str2bool, default=False)
    parser.add_argument("--save_exp", type=str2bool, default=False)
    parser.add_argument("--no_embedding", type=str2bool, default=False)
    parser.add_argument("--discrete_dist", type=str2bool, default=False)
    parser.add_argument("--category_size", type=int, default=64)
    parser.add_argument("--class_size", type=int, default=64)
    parser.add_argument("--soft_bandit_update", type=str2bool, default=False)
    parser.add_argument("--samples_per_policy", type=int, default=1)
    parser.add_argument("--data_rollout_length", type=int, default=1000)
    parser.add_argument("--train_iter_num_mult", type=int, default=10)
    parser.add_argument('--filter', type=str, default='MeanStdFilter')


    #DWS
    parser.add_argument("--dim_hidden", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--output_features", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--reduction", type=str, default="max", choices=["mean", "sum", "max"])
    parser.add_argument("--n_fc_layers", type=int, default=1)
    parser.add_argument("--set_layer", type=str, default='sab', choices=["ds", "sab"])
    parser.add_argument("--n_out_fc", type=int, default=1)
    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--add_bn", type=str2bool, default=False)
    parser.add_argument("--use_invariant_layer", type=str2bool, default=False)
    parser.add_argument("--add_skip", type=str2bool, default=False)
    parser.add_argument("--add_layer_skip", type=str2bool, default=False)
    parser.add_argument("--permutation", type=str2bool, default=False)
    parser.add_argument("--augmentation", type=str2bool, default=False)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--clf_layers", type=int, default=1)



    args = parser.parse_args()

    if args.no_embedding:
        args.training_freq_network = 10000000000
    else:
        args.training_freq_network = args.n_directions * 2 * 5
    args.training_iter = args.n_directions * 2 * args.train_iter_num_mult

    args.target_model_update = args.training_iter
    args.delta_bandit_std = args.delta_std
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.soft_bandit_update:
        args.policy_history_set_size = 1
    if not args.state_based_value:
        args.data_rollout_length = args.rollout_length
        args.num_unroll_steps = args.rollout_length
    if args.explore_coeff == 0.0:
        args.training_freq_network = 100000000000
    if args.policy_type != 'linear':
        args.dec_coeff = 0.0
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)




    args.device = device
    run_exp(args)
