"""
This file defines A2C and PPO rollout buffer.

You need to implement the compute_returns function.

-----
*2020-2021 Term 1, IERG 5350: Reinforcement Learning. Department of Information Engineering, The Chinese University of
Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao, SUN Hao, ZHAN Xiaohang.*
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class PPORolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, act_dim, device,
                 use_gae=True, gae_lambda=0.95):
        def zeros(*shapes, dtype=None):
            return torch.zeros(shapes, dtype=dtype).to(device)

        self.observations = zeros(num_steps + 1, num_processes, *obs_shape, dtype=torch.uint8)
        self.rewards = zeros(num_steps, num_processes, 1)
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        self.actions = zeros(num_steps, num_processes, act_dim).to(torch.float)
        self.masks = torch.ones(num_steps + 1, num_processes, 1, dtype=torch.bool).to(device)

        self.num_steps = num_steps
        self.step = 0

        self.gae = use_gae
        self.gae_lambda = gae_lambda

    def feed_forward_generator(self, advantages, mini_batch_size):
        """A generator to provide samples for PPO. PPO run SGD for multiple
        times so we need more efforts to prepare data for it."""
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(
                -1, *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma):
        if self.gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                # [TODO] Implement GAE advantage computing here.
                # Hint:
                #  1. The return at timestep t should be (advantage_t + value_t)
                #  2. You should use reward, values, mask to compute TD error
                #   delta. Then combine TD error of timestep t with advantage
                #   of timestep t+1 to get the advantage of timestep t.
                #  3. The variable `gae` represents the advantage
                #  4. The for-loop is in a reverse order.

                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

        else:
            # Ignore this part
            raise NotImplementedError("Not for this assignment.")

    def insert(self, current_obs, action, action_log_prob, value_pred, reward, mask):
        if isinstance(current_obs, np.ndarray):
            current_obs = torch.from_numpy(current_obs.astype(np.uint8))
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def before_update(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        self.observations[0].copy_(obs)


# Rollout Storage for Sensor Fus
class MyPPORolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shapes, act_dim, device,
                 use_gae=True, gae_lambda=0.95):
        self.device = device

        def zeros(*shapes, dtype=None):
            return torch.zeros(shapes, dtype=dtype).to(device)
            # return torch.zeros(shapes, dtype=dtype).to("cpu")

        def zeros_cpu(*shapes, dtype=None):
            # return torch.zeros(shapes, dtype=dtype).to(device)
            return torch.zeros(shapes, dtype=dtype).to("cpu")

        self.observations = [
            zeros_cpu(num_steps + 1, num_processes, *obs_shapes[0], dtype=torch.uint8),
            zeros_cpu(num_steps + 1, num_processes, *obs_shapes[1], dtype=torch.float),
            zeros_cpu(num_steps + 1, num_processes, *obs_shapes[2], dtype=torch.float),
        ]
        self.rewards = zeros(num_steps, num_processes, 1)
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        self.actions = zeros(num_steps, num_processes, act_dim).to(torch.float)
        self.masks = torch.ones(num_steps + 1, num_processes, 1, dtype=torch.bool).to(device)

        self.num_steps = num_steps
        self.step = 0

        self.gae = use_gae
        self.gae_lambda = gae_lambda

    def feed_forward_generator(self, advantages, mini_batch_size):
        """A generator to provide samples for PPO. PPO run SGD for multiple
        times so we need more efforts to prepare data for it."""
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = []
            for i in range(3):
                observations_batch.append(self.observations[i][:-1].view(
                    -1, *(self.observations[i].size()[2:]))[indices].to(self.device))

            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices].to(self.device)
            return_batch = self.returns[:-1].view(-1, 1)[indices].to(self.device)
            masks_batch = self.masks[:-1].view(-1, 1)[indices].to(self.device)
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices].to(self.device)
            adv_targ = advantages.view(-1, 1)[indices].to(self.device)

            yield observations_batch, actions_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma):
        if self.gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                # [TODO] Implement GAE advantage computing here.
                # Hint:
                #  1. The return at timestep t should be (advantage_t + value_t)
                #  2. You should use reward, values, mask to compute TD error
                #   delta. Then combine TD error of timestep t with advantage
                #   of timestep t+1 to get the advantage of timestep t.
                #  3. The variable `gae` represents the advantage
                #  4. The for-loop is in a reverse order.

                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1].float() - self.value_preds[step]
                gae = delta + gamma * self.gae_lambda * self.masks[step + 1].float() * gae
                self.returns[step] = gae + self.value_preds[step]

        else:
            # Ignore this part
            raise NotImplementedError("Not for this assignment.")

    def insert(self, current_obs, action, action_log_prob, value_pred, reward, mask):
        # self.observations[self.step + 1].copy_(current_obs)
        self.insert_observation(current_obs)

        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def insert_observation(self, obs):
        assert isinstance(obs, tuple), "[MyPPORolloutStorage]: Error observation input in observation insertion."
        assert len(obs) == 4, "[MyPPORolloutStorage]: Error input length, the lenght must be 4. Got {}.".format(len(obs))

        obs_list = []
        for i in range(3):
            if isinstance(obs[i], np.ndarray):
                obs_list.append(torch.from_numpy(obs[i]))

        self.observations[0][self.step + 1].copy_(obs_list[0])
        self.observations[1][self.step + 1].copy_(obs_list[1].unsqueeze(2))
        self.observations[2][self.step + 1].copy_(obs_list[2].unsqueeze(1))

    def after_update(self):
        for i in range(3):
            self.observations[i][0].copy_(self.observations[i][-1])
        self.masks[0].copy_(self.masks[-1])

    def before_update(self, obs):
        obs_list = []
        for i in range(3):
            if isinstance(obs[i], np.ndarray):
                obs_list.append(torch.from_numpy(obs[i]))

        self.observations[0][0].copy_(obs_list[0])
        self.observations[1][0].copy_(obs_list[1].unsqueeze(2))
        self.observations[2][0].copy_(obs_list[2].unsqueeze(1))

    def get_observation(self, index):
        color = self.observations[0][index].to(self.device)
        depth = self.observations[1][index].to(self.device)
        force = self.observations[2][index].to(self.device)
        return color, depth, force
