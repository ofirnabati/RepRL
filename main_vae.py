'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''


import random
import time
import os
import gym
import ipdb
import torch
import numpy as np
import ray
import utils as utils
import optimizers as optimizers
from policies_old import *
from sklearn.manifold import TSNE
import pandas as pd
from utils import str2bool
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from shared_noise import SharedNoiseTable, create_shared_noise
from neural_linear_gae_vae import NeuralLinearPosteriorSampling


from sparseMuJoCo.envs.mujoco.hopper_v0 import SparseHopperV0 as SparseHopperV0
from sparseMuJoCo.envs.mujoco.humanoid_v0 import SparseHumanoidV0 as SparseHumanoidV0
from sparseMuJoCo.envs.mujoco.half_cheetah_v0 import SparseHalfCheetahV0 as SparseHalfCheetahV0
from sparseMuJoCo.envs.mujoco.ant_v0 import SparseAntV0 as SparseAntV0
from sparseMuJoCo.envs.mujoco.walker2d_v0 import SparseWalker2dV0 as SparseWalker2dV0
from gridworlds.envs.gridworld_gauss_reward import GaussGridWorld

# import sparseMuJoCo
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
                 state_based_value=False):


        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(env_seed)


        # initialize OpenAI environment for each worker
        if env_name == 'SparseHalfCheetah-v1':
            self.env = SparseHalfCheetah()
        elif env_name == 'SparseHopper-v1':
            self.env = SparseHopper()
        elif env_name == 'SparseHalfCheetah-v1':
            self.env = SparseHalfCheetah()
        elif env_name == 'SparseAnt-v1':
            self.env = SparseAnt()
        elif env_name == 'SparseSwimmer-v1':
            self.env = SparseSwimmer()
        elif env_name == 'SparseWalker2d-v1':
            self.env = SparseWalker2d()
        elif env_name == 'SparseHumanoid-v1':
            self.env = SparseHumanoid()
        elif env_name == 'SparseHalfCheetah-v0':
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
        self.seed = env_seed
        # self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
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

        # exps = utils.DictListObject()
        # exps.obs = obss
        # exps.action = np.concatenate(actions)
        # exps.reward = np.array(rewards)
        # exps.mask = np.array(masks)
        # exps.step = np.array(step_id)

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

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                # reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                reward, r_steps, obss, actions, rewards, masks, step_id = self.rollout(shift = 0., rollout_length=self.rollout_length,evaluate=evaluate)
                rollout_rewards.append(reward)
                obss_arr.append(obss)
                actions_arr.append(actions)
                rewards_arr.append(rewards)
                masks_arr.append(masks)
                step_id_arr.append(step_id)

            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                for _ in range(self.samples_per_policy):
                    deltas_idx.append(idx)

                    # set to true so that state statistics are updated
                    self.policy.update_filter = True

                    # compute reward and number of timesteps used for positive perturbation rollout
                    self.policy.update_weights(w_policy + delta)
                    # self.policy.update_weights(w_policy)
                    pos_reward, pos_steps, pos_obss, pos_actions, pos_rewards, pos_masks, pos_step_id  = self.rollout(shift = shift)

                    # compute reward and number of timesteps used for negative pertubation rollout
                    self.policy.update_weights(w_policy - delta)
                    # self.policy.update_weights(w_policy)
                    neg_reward, neg_steps, neg_obss, neg_actions, neg_rewards, neg_masks, neg_step_id  = self.rollout(shift = shift)
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


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
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
                 shift='constant zero',
                 params=None,
                 seed=123,
                 storage=None,
                 bandit_algo=None,
                 device='cpu',
                 save_exp=False,
                 soft_bandit_update=True):

        # logz.configure_output_dir(logdir)
        # logz.save_params(params)

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
        self.average_first_state = params['average_first_state']
        self.horizon = params['horizon']
        self.discount = params['discount']
        self.eval_freq = params['eval_freq']
        self.max_timesteps = params['max_timesteps']
        self.explore_coeff = params['explore_coeff']
        self.samples_per_policy = params['samples_per_policy']
        self.do_tsne = params['do_tsne']
        self.soft_bandit_update = soft_bandit_update
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
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
                                      state_based_value=params['state_based_value']) for i in range(num_workers)]


        # initialize policy
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'discrete':
            self.policy = DiscretePolicy(policy_params)
        elif policy_params['type'] == 'nn':
            self.policy = StableBaselinePolicy(policy_params['ob_dim'], policy_params['ac_dim'], policy_params['ob_filter'], device='cpu:0')
        else:
            raise NotImplementedError

        self.w_policy = self.policy.get_weights()
        self.policy_history_set = [self.w_policy for _ in range(params['policy_history_set_size'])]
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

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


        loss = []
        rollout_len = 0
        for result in results_one:
            if not evaluate:
                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                for j in range(num_rollouts):
                    delta = self.deltas.get(result['deltas_idx'][j], self.w_policy.size)
                    delta = self.delta_std * delta
                    exp_pos = utils.DictListObject()
                    exp_pos.obs    = result['obss_arr'][j][0]
                    exp_pos.action = result['actions_arr'][j][0]
                    exp_pos.reward = result['rewards_arr'][j][0]
                    exp_pos.mask   = result['masks_arr'][j][0]
                    exp_pos.step   = result['step_id_arr'][j][0]
                    if self.params['filter'] == 'MeanStdFilter':
                        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                        update_par = np.concatenate([self.w_policy.reshape(-1) + delta, w[1], w[2]])
                    else:
                        update_par = self.w_policy.reshape(-1) + delta
                    loss_pos = self.bandit_algo.update(exp_pos, update_par)
                    # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                    if loss_pos is not None:
                        loss.append(loss_pos)


                    exp_neg = utils.DictListObject()
                    exp_neg.obs    = result['obss_arr'][j][1]
                    exp_neg.action = result['actions_arr'][j][1]
                    exp_neg.reward = result['rewards_arr'][j][1]
                    exp_neg.mask   = result['masks_arr'][j][1]
                    exp_neg.step   = result['step_id_arr'][j][1]
                    if self.params['filter'] == 'MeanStdFilter':
                        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                        update_par = np.concatenate([self.w_policy.reshape(-1) - delta, w[1], w[2]])
                    else:
                        update_par = self.w_policy.reshape(-1) - delta
                    loss_neg = self.bandit_algo.update(exp_neg, update_par)
                    # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                    if loss_neg is not None:
                        loss.append(loss_neg)

            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']


        for result in results_two:
            if not evaluate:

                # self.timesteps += result["steps"]
                rollout_len += result["steps"]
                delta = self.deltas.get(result['deltas_idx'][0], self.w_policy.size)
                delta = self.delta_std * delta
                exp_pos = utils.DictListObject()
                exp_pos.obs = result['obss_arr'][0][0]
                exp_pos.action = result['actions_arr'][0][0]
                exp_pos.reward = result['rewards_arr'][0][0]
                exp_pos.mask = result['masks_arr'][0][0]
                exp_pos.step = result['step_id_arr'][0][0]
                if self.params['filter'] == 'MeanStdFilter':
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                    update_par = np.concatenate([self.w_policy.reshape(-1) + delta, w[1], w[2]])
                else:
                    update_par = self.w_policy.reshape(-1) + delta
                loss_pos = self.bandit_algo.update(exp_pos, update_par)
                # loss_pos = self.bandit_algo.update(exp_pos, self.w_policy.reshape(-1))
                if loss_pos is not None:
                    loss.append(loss_pos)

                exp_neg = utils.DictListObject()
                exp_neg.obs = result['obss_arr'][0][1]
                exp_neg.action = result['actions_arr'][0][1]
                exp_neg.reward = result['rewards_arr'][0][1]
                exp_neg.mask = result['masks_arr'][0][1]
                exp_neg.step = result['step_id_arr'][0][1]
                if self.params['filter'] == 'MeanStdFilter':
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                    update_par = np.concatenate([self.w_policy.reshape(-1) - delta, w[1], w[2]])
                else:
                    update_par = self.w_policy.reshape(-1) - delta
                loss_neg = self.bandit_algo.update(exp_neg, update_par)
                # loss_neg = self.bandit_algo.update(exp_neg, self.w_policy.reshape(-1))
                if loss_neg is not None:
                    loss.append(loss_neg)

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


        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum_old(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        # self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)

        ## Bandits from here
        t1 = time.time()
        G = 0
        max_val = -1e10
        max_policy = self.policy_history_set[0]
        with torch.no_grad():
                if self.bandit_algo.method == 'ts':
                    self.bandit_algo.sample_ts()
                # if self.params['filter'] == 'MeanStdFilter':
                #     _, mu, std_matrix = ray.get(self.workers[0].get_weight_plus_stats.remote())
                decison_set = []
                values = []
                idxes, deltas = self.deltas.get_deltas(self.w_policy.size, self.num_bandit_deltas * len(self.policy_history_set))
                first_state_visted = False
                for _ in range(args.average_first_state):
                        decison_set_pos = np.concatenate([p.reshape(-1) + deltas[i*self.num_bandit_deltas : (i+1)*self.num_bandit_deltas] for i,p in enumerate(self.policy_history_set)])
                        decison_set_pos = torch.tensor(decison_set_pos.reshape(len(idxes), -1))
                        if self.soft_bandit_update:
                            decison_set_neg = np.concatenate([p.reshape(-1) - deltas[i*self.num_bandit_deltas : (i+1)*self.num_bandit_deltas] for i,p in enumerate(self.policy_history_set)])
                            decison_set_neg = torch.tensor(decison_set_neg.reshape(len(idxes), -1))

                        if self.params['filter'] == 'MeanStdFilter':
                            w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                            w = torch.tensor(np.concatenate([w[1],w[2]]))
                            w = torch.tile(w.unsqueeze(0),[len(idxes),1])
                            decison_set_pos1 = torch.cat([decison_set_pos, w], dim=-1)
                            if self.soft_bandit_update:
                                decison_set_neg1 = torch.cat([decison_set_neg, w], dim=-1)
                        else:
                            decison_set_pos1 = decison_set_pos
                            if self.soft_bandit_update:
                                decison_set_neg1 = decison_set_neg
                        decison_set_pos1 = decison_set_pos1.float().to(self.device)
                        if self.soft_bandit_update:
                            decison_set_neg1 = decison_set_neg1.float().to(self.device)

                        # Vpos, Vneg, emprical_return = 0,0,0
                        curr_step_size = 0
                        Vpos = 0
                        Vneg = 0
                        # for frag_idx in range(self.bandit_algo.fragments):
                        first_state, frag_idx = self.storage.sample_random_state_and_fragment(self.horizon, self.bandit_algo.fragments, self.num_deltas * 2)
                        first_state = torch.tensor(first_state, dtype=torch.float).to(device)
                        first_state = torch.stack([first_state for _ in range(decison_set_pos.shape[0])])
                        first_state = first_state.float()
                        if self.params['filter'] == 'MeanStdFilter':
                            w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                            ref_point = np.concatenate([self.w_policy.reshape(-1), w[1], w[2]])
                            ref_point = torch.tensor(ref_point, device=self.device, dtype=torch.float)
                        else:
                           ref_point =  torch.tensor(self.w_policy.reshape(-1), device=self.device, dtype=torch.float)

                        p1, best_idx_plus, values_plus, _ = self.bandit_algo.action(decison_set_pos1, first_state, fragment=frag_idx, ref_point = ref_point)
                        # p1, best_idx, values_plus, _ = self.bandit_algo.action(decison_set_pos, first_state, fragment=frag_idx)
                        if self.soft_bandit_update:
                            p1, best_idx_minus, values_minus, _ = self.bandit_algo.action(decison_set_neg1, first_state, fragment=frag_idx)




                        if self.soft_bandit_update:
                            Vpos = values_plus
                            Vneg = values_minus
                            # aggregate rollouts to form g_hat, the gradient used to compute SGD step
                            V = torch.stack([Vpos, Vneg], dim=1)
                            # emprical_return = emprical_return / 10.0
                            V_max, V_max_idx_1 = torch.max(V, dim=1)
                            V = V.cpu().numpy()

                            V_std = np.std(V)
                            if V_std == 0:
                                print('Poor representation.. std is zero')
                                bandit_rollout_rewards = V
                            else:
                                bandit_rollout_rewards = V / V_std

                            g_hat_bandits, count = utils.batched_weighted_sum_old(bandit_rollout_rewards[:,0] - bandit_rollout_rewards[:,1],
                                                                      (self.deltas.get(idx, self.w_policy.size)
                                                                       for idx in idxes),
                                                                      batch_size = 500)
                            g_hat_bandits /= idxes.size
                            # g_hat_bandits = np.clip(g_hat_bandits,-1.0,1.0)
                            G += g_hat_bandits
                            # self.w_policy -= self.optimizer._compute_step(g_hat_bandits).reshape(self.w_policy.shape)
                            curr_step_size += (bandit_rollout_rewards[:,0] - bandit_rollout_rewards[:,1]).mean().item()

        if not self.soft_bandit_update:
            self.w_policy = decison_set_pos[best_idx_plus].cpu().numpy().reshape(self.w_policy.shape)
        else:
            # G /= args.average_first_state
            overall_g = (1 - self.explore_coeff) * g_hat + self.explore_coeff * G
            self.w_policy -= self.optimizer._compute_step(overall_g).reshape(self.w_policy.shape)
        self.policy_history_set.pop(0)
        self.policy_history_set.append(self.w_policy)
        # self.policy_history_set.pop(0)
        # self.policy_history_set.append(self.w_policy)
        t2 = time.time()
        curr_step_size /= args.average_first_state
        print(f'Curr step size:{curr_step_size}')
        print('time to compute gradient', t2 - t1)
            # G /= args.average_first_state
            # self.w_policy -= self.optimizer._compute_step(G).reshape(self.w_policy.shape)
            # delta = self.deltas.get(idxes[V_max_idx_0], self.w_policy.size)
            # delta = (self.delta_std * delta).reshape(list(self.w_policy.shape))
            # if V_max_idx_1[V_max_idx_0] == 0:
            #     self.w_policy += delta
            # else:
            #     self.w_policy -= delta

        return G, loss


    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        g_hat, loss = self.aggregate_rollouts()
        # print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        # self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return loss

    def train(self, num_iter):

        loss = 0
        start = time.time()
        # for i in range(num_iter):
        i = 0
        while self.timesteps <= self.max_timesteps:


            # record statistics every 10 iterations
            if (i % self.eval_freq == 0):
                rewards, results_one = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                if self.save_exp:
                    traj = []
                    num_rollouts = int(100 / self.num_workers)
                    for result in results_one:
                            # self.timesteps += result["steps"]
                            for j in range(num_rollouts):
                               traj.append(result['obss_arr'][j])

                    traj = np.array(traj)
                    self.history.append(traj)







                # w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                # np.savez(self.logdir + "/lin_policy_plus", w)

                print(sorted(self.params.items()))
                # logz.log_tabular("Time", time.time() - start)
                # logz.log_tabular("Iteration", i + 1)
                # logz.log_tabular("AverageReward", np.mean(rewards))
                # logz.log_tabular("StdRewards", np.std(rewards))
                # logz.log_tabular("MaxRewardRollout", np.max(rewards))
                # logz.log_tabular("MinRewardRollout", np.min(rewards))
                # logz.log_tabular("timesteps", self.timesteps)
                # logz.log_tabular("rollout_num", self.rollouts)
                # logz.dump_tabular()
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
                       "b": self.bandit_algo.b[0],
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



                wandb.log(log)
                # if self.bandit_algo.t > 0 and (i % (self.eval_freq * 10) == 0) and not self.bandit_algo.no_embedding:
                #     plt.close(fig)


            t1 = time.time()
            loss1 = self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i, ' done')
            self.rollouts += self.num_bandit_deltas * 2

            if len(loss1) > 0:
                loss = np.mean(loss1)

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

def run_ars(args):

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

    if params['filter'] == 'MeanStdFilter':
        w = policy.get_weights_plus_stats()
        context_dim = np.concatenate([w[0].reshape(-1), w[1], w[2]]).size
    else:
        context_dim = policy.get_weights().size

    # args.layers_size = [context_dim * 2, context_dim * 4]
    args.context_dim = context_dim
    print(context_dim)
    args.obs_dim = ob_dim

    wandb.init(project="PolicySpaceOptimization_new", entity="ofirnabati", config=args.__dict__)
    wandb.run.name = args.env_name
    if args.policy_type == 'nn':
        wandb.run.name = wandb.run.name + '_nn'
    wandb.run.save()

    storage = Storage(config=args)
    bandit_algo = NeuralLinearPosteriorSampling(storage, device, args)

    ARS = ARSLearner(env_name=params['env_name'],
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


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ARS.train(params['n_iter'])

    return




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--max_timesteps', '-max_t', type=int, default=5e6)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--n_bandit_directions', type=int, default=512)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--latent_step_size', type=float, default=1.0)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--delta_bandit_std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--average_first_state', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--explore_coeff', type=float, default=0.1)
    parser.add_argument('--do_tsne', type=str2bool, default=True)

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

    # bandits args
    parser.add_argument("--discount", type=float, default=0.995,
                        help="discount factor (default: 0.9996)")
    parser.add_argument("--gae-lambda", type=float, default=0.97,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
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
    parser.add_argument("--lr_step_size", type=int, default=100000)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--training_freq_network", type=int, default=50)
    parser.add_argument("--training_iter", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--noise_aug", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--kld_coeff", type=float, default=0.1)
    parser.add_argument("--dec_coeff", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--state_based_value", type=str2bool, default=False)
    parser.add_argument("--save_exp", type=str2bool, default=False)
    parser.add_argument("--no_embedding", type=str2bool, default=False)
    parser.add_argument("--discrete_dist", type=str2bool, default=False)
    parser.add_argument("--category_size", type=int, default=64)
    parser.add_argument("--class_size", type=int, default=64)
    parser.add_argument("--soft_bandit_update", type=str2bool, default=False)
    parser.add_argument("--samples_per_policy", type=int, default=1)
    parser.add_argument("--data_rollout_length", type=int, default=1000)
    parser.add_argument("--train_iter_num_mult", type=int, default=10)
    parser.add_argument("--permutation", type=str2bool, default=False)
    parser.add_argument("--augmentation", type=str2bool, default=False)
    parser.add_argument("--use_bn", type=str2bool, default=False)
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(_redis_address= local_ip + ':6379')
    # ray.init()

    args = parser.parse_args()
    # if args.n_directions > 50:
    #     args.training_freq_network = args.n_directions * 2
    #     args.training_iter = args.n_directions * 2 * 5
    # else:
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
    if args.state_based_value:
        args.training_iter = args.training_iter * 10
    else:
        args.average_first_state = 1
        args.data_rollout_length = args.rollout_length
        args.num_unroll_steps = args.rollout_length
        args.discount = 1.0
    # args.fragments = args.rollout_length // args.horizon
    # if args.rollout_length % args.horizon > 0:
    #     args.fragments += 1
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)




    args.device = device
    run_ars(args)