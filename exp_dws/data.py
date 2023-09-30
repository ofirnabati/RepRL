import json
import random
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# from exp_dws.utils import make_coordinates
# from dwsnet.inr import INR


class BatchStateBased(NamedTuple):
    weights: Tuple
    biases: Tuple
    reward: torch.Tensor
    mask: torch.Tensor
    obs: torch.Tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            obs=self.obs.to(device),
            reward=self.reward.to(device),
            mask=self.mask.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    # actions: torch.Tensor
    # reward: torch.Tensor
    # mask: torch.Tensor
    # steps: torch.Tensor
    ret: torch.Tensor
    obs: torch.Tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            ret=self.ret.to(device),
            obs=self.obs.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class BatchVAE(NamedTuple):
    policy: torch.Tensor
    # actions: torch.Tensor
    # reward: torch.Tensor
    # mask: torch.Tensor
    # steps: torch.Tensor
    ret: torch.Tensor
    obs: torch.Tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            policy=self.policy.to(device),
            ret=self.ret.to(device),
            obs=self.obs.to(device),
        )

    def __len__(self):
        return len(self.policy)

class ImageBatch(NamedTuple):
    image: torch.Tensor
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.image)


class ReprlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        num_unroll,
        stacked_observations,
        horizon,
        state_based,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-2,
        drop_rate=0.0,
        resize_scale=0.0,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
        average_reward=False,
        discount = 1.0,
    ):
        # assert split in ["test", "train"]
        self.split = split
        self.dataset = data
        self.num_unroll = num_unroll
        self.stacked_observations = stacked_observations
        self.horizon = horizon
        self.state_based = state_based
        self.augmentation = augmentation
        self.permutation = permutation
        self.average_reward = average_reward
        # self.normalize = normalize
        # if self.normalize:
        #     self.stats = torch.load(statistics_path, map_location="cpu")

        self.translation_scale = translation_scale
        self.rotation_degree = rotation_degree
        self.noise_scale = noise_scale
        self.drop_rate = drop_rate
        self.resize_scale = resize_scale
        self.pos_scale = pos_scale
        self.quantile_dropout = quantile_dropout
        self.discount = discount

        # if class_mapping is not None:
        #     self.class_mapping = class_mapping
        #     self.dataset["label"] = [
        #         self.class_mapping[l] for l in self.dataset["label"]
        #     ]

    def __len__(self):
        return len(self.dataset)

    # def _normalize(self, weights, biases):
    #     wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
    #     bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]
    #
    #     weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
    #     biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))
    #
    #     return weights, biases

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # # translation
        # translation = torch.empty(weights[0].shape[0]).uniform_(
        #     -self.translation_scale, self.translation_scale
        # )
        # order = random.sample(range(1, len(weights)), 1)[0]
        # bias_res = translation
        # i = 0
        # for i in range(order):
        #     bias_res = bias_res @ weights[i]
        #
        # new_biases[i] += bias_res

        # # rotation
        # if new_weights[0].shape[0] == 2:
        #     rot_mat = self.rotation_mat(self.rotation_degree)
        #     new_weights[0] = rot_mat @ new_weights[0]

        # noise
        # new_weights = [w + w.std() * self.noise_scale for w in new_weights]
        # new_biases = [
        #     b + b.std() * self.noise_scale if b.shape[0] > 1 else b for b in new_biases
        # ]
        new_weights = [w + torch.randn_like(w) * self.noise_scale for w in new_weights]
        new_biases = [
            b + torch.randn_like(b) * self.noise_scale  for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # scale
        if self.resize_scale > 0:
            # todo: can also apply to deeper layers
            rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
            new_weights[0] = new_weights[0] * rand_scale

        # positive scale
        if self.pos_scale > 0:
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = new_weights[i] * scale
                new_biases[i] = new_biases[i] * scale
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T

        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return new_weights, new_biases

    def __getitem__(self, item):
        exp = self.dataset[item][0]
        traj_len = exp.reward.shape[0]
        t =  np.random.randint(traj_len - self.num_unroll) if self.state_based else np.random.randint(traj_len)#0
        weights,biases = self.dataset[item][1]
        # t = 0
        # obss = [torch.tensor(str_to_arr(o)) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations-1]]
        # obss = [torch.tensor(o) for o in exp.obs[t:t + self.num_unroll + self.stacked_observations]]
        # obss = torch.stack(obss, dim=0)  # T x H x W x C
        # actions = exp.action[t:t + self.num_unroll + 1]
        # steps = exp.step[t:t + self.num_unroll + 1]
        obs = exp.obs[t:t+self.num_unroll]
        rewards = exp.reward[t:t + self.num_unroll]
        masks = exp.mask[t:t + self.num_unroll]
        if self.state_based:
            obs = torch.stack([torch.tensor(o).float() for o in obs])
            masks = torch.tensor(masks).float()
            rewards = torch.tensor(rewards).float()
        else:
            if self.average_reward:
                ret = 0
                running_return = 0
                for r in reversed(rewards):
                    running_return = r + self.discount * running_return
                    ret += running_return
                ret = ret / len(rewards)
            else:
                ret = np.sum( rewards * self.discount ** np.arange(
                          len(rewards)))


            ret = torch.tensor(ret).float()
            obs = torch.tensor(exp.obs[t]).float()
        # # rewards = exp.reward
        # ret = torch.tensor(exp.ret).float()
        # actions, rewards, masks, steps = torch.tensor(actions).float(), torch.tensor(rewards).float(), torch.tensor(
        #     masks).float(), torch.tensor(steps).float()
        # obss = obss.float()  # / 255.0 #uint8 --> float32

        weights = tuple([w.float() for w in weights])
        biases = tuple([b.float() for b in biases])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1).float() for w in weights])
        biases = tuple([b.unsqueeze(-1).float() for b in biases])

        # if self.normalize:
        #     weights, biases = self._normalize(weights, biases)

        if self.permutation:
              weights, biases = self._permute(weights, biases)


        # return Batch(weights=weights, biases=biases, obs=obss, actions=actions, reward=rewards, mask=masks, steps=steps, ret=ret)
        if self.state_based:
            return BatchStateBased(weights=weights, biases=biases, obs=obs, reward=rewards, mask=masks)
        else:
            return Batch(weights=weights, biases=biases, ret=ret, obs=obs)




class ReprlDatasetVae(ReprlDataset):

    def __getitem__(self, item):
        exp = self.dataset[item][0]
        traj_len = exp.reward.shape[0]
        t =  np.random.randint(traj_len - self.num_unroll) if self.state_based else 0 #np.random.randint(traj_len)#0
        policy = self.dataset[item][1]
        policy = torch.tensor(policy).float()
        # t = 0
        # obss = [torch.tensor(str_to_arr(o)) for o in exp.obs[t:t+self.num_unroll+self.stacked_observations-1]]
        # obss = [torch.tensor(o) for o in exp.obs[t:t + self.num_unroll + self.stacked_observations]]
        # obss = torch.stack(obss, dim=0)  # T x H x W x C
        # actions = exp.action[t:t + self.num_unroll + 1]
        # steps = exp.step[t:t + self.num_unroll + 1]
        obs = exp.obs[t:t+self.num_unroll]
        rewards = exp.reward[t:t + self.num_unroll]
        masks = exp.mask[t:t + self.num_unroll]
        if self.state_based:
            obs = torch.stack([torch.tensor(o).float() for o in obs])
            masks = torch.tensor(masks).float()
            rewards = torch.tensor(rewards).float()
        else:
            if self.average_reward:
                ret = 0
                running_return = 0
                for r in reversed(rewards):
                    running_return = r + self.discount * running_return
                    ret += running_return
                ret = ret / len(rewards)
            else:
                ret = np.sum( rewards * self.discount ** np.arange(
                          len(rewards)))


            ret = torch.tensor(ret).float()
            obs = torch.tensor(exp.obs[t]).float()
        # # rewards = exp.reward
        # ret = torch.tensor(exp.ret).float()
        # actions, rewards, masks, steps = torch.tensor(actions).float(), torch.tensor(rewards).float(), torch.tensor(
        #     masks).float(), torch.tensor(steps).float()
        # obss = obss.float()  # / 255.0 #uint8 --> float32


        # if self.augmentation:
        #     weights, biases = self._augment(weights, biases)

        # add feature dim

        # if self.normalize:
        #     weights, biases = self._normalize(weights, biases)



        # return Batch(weights=weights, biases=biases, obs=obss, actions=actions, reward=rewards, mask=masks, steps=steps, ret=ret)
        if self.state_based:
            return BatchStateBased(weights=weights, biases=biases, obs=obs, reward=rewards, mask=masks)
        else:
            return BatchVAE(policy=policy, ret=ret, obs=obs)