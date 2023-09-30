import numpy as np
import torch


class Storage(object):
    def __init__(self, config=None):
        self.config = config
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.max_traj_len = np.array([])

        self.priorities = []

        self._eps_collected = 0
        self.base_idx = 0
        self.max_samples = config.memory_size
        self.clear_time = 0
        self.num_of_usage = np.zeros(self.max_samples)
        self.new_traj = True

    def save_batch(self, pools):
        # save a list of game histories
        rewards_batch, policy_batch = pools
        batch_size = rewards_batch.shape[0]
        for i in range(batch_size):
            self.save_exp(rewards_batch[i], policy_batch[i])


    def save_exp(self, exp, policy ):

        current_size = self.size()
        if current_size < self.max_samples:
            self.buffer.append([exp, policy ])
            self.max_traj_len= np.append(self.max_traj_len, exp.step.max())
        else:
            self.max_traj_len = np.delete(self.max_traj_len, 0)
            del self.buffer[0]
            self.buffer.append([exp, policy])
            self.max_traj_len = np.append(self.max_traj_len, exp.step.max())

    def get_data(self):
        return self.buffer

    def prepare_batch(self):

        total = self.size()

        indices_lst = np.random.choice(total, self.batch_size, replace=False)

        # lf_batch = []
        rewards_batch = []
        value_batch = []
        policy_batch = []

        for idx in indices_lst:
            rewards, value, policy = self.buffer[idx]
            rewards_batch.append(rewards)
            value_batch.append(value)
            policy_batch.append(policy)
            self.num_of_usage[idx] += 1



        return torch.stack(rewards_batch), torch.stack(value_batch), torch.stack(policy_batch)


    def sample_random_state_and_fragment(self, horizon, max_frag, last_ind):
        if len(self.buffer) == 0:
            return None, None
        traj = self.sample_traj()
        idx = np.random.randint(len(traj))
        return traj[idx], 0

    def sample_state(self, step):
        rel_samples = np.where(self.max_traj_len >= step)[0]
        if len(rel_samples) == 0:
            return None
        idx = int(np.random.choice(rel_samples))
        return self.buffer[idx][0].obs[step]

    def sample_traj(self):
        idx = np.random.randint(len(self.buffer))
        return self.buffer[idx][0].obs

    def sample_last_traj(self, last_ind):
        idx = np.random.randint(len(self.buffer[-last_ind:]))
        return self.buffer[-last_ind+idx][0].obs

    def get_random_obs(self, step):
        relevant_arr = np.where(self.max_traj_len > step)[0]
        if len(relevant_arr) > 0:
            idx = np.random.choice(relevant_arr)
            return self.buffer[idx][0].obs[step]
        else:
            return None

    def return_idx(self, idx):
        return self.buffer[idx]


    def clear_buffer(self):
        del self.buffer[:]

    def size(self):
        # number of games
        return len(self.buffer)

    def episodes_collected(self):
        # number of collected histories
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def empty(self):
        return self.size() == 0


