"""
This file implement a base trainer class.

You should finish `compute_action` and run this file to verify the base trainer is implement correctly.

-----
*2020-2021 Term 1, IERG 5350: Reinforcement Learning. Department of Information Engineering, The Chinese University of
Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao, SUN Hao, ZHAN Xiaohang.*
"""
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from multimodal.models.sensor_fusion import SensorFusionEncoder


class BaseTrainer:
    def __init__(self, env, config, _test=False):
        self.device = config.device
        self.config = config
        self.lr = config.lr
        self.num_envs = config.num_envs
        self.gamma = config.gamma
        self.num_steps = config.num_steps
        # self.grad_norm_max = config.grad_norm_max

        if isinstance(env.action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete = False
        else:
            self.discrete = True

        if isinstance(env.observation_space, gym.spaces.Tuple):
            num_feats = env.observation_space[0].shape
            self.num_actions = env.action_space[0].n
        elif isinstance(env.observation_space, list):
            if self.discrete:
                self.num_actions = env.action_space.n
            else:
                self.num_actions = env.action_space.shape[0]
            num_feats = self.config.state_dim
        else:
            num_feats = env.observation_space.shape
            if self.discrete:
                self.num_actions = env.action_space.n
            else:
                self.num_actions = env.action_space.shape[0]
        self.num_feats = num_feats  # (num channel, width, height)

        if _test:  # Used for CartPole-v0
            self.model = MLP(num_feats[0], self.num_actions)
        else:
            self.model = ActorCritic(self.num_feats,
                                     self.num_actions,
                                     self.discrete,
                                     device=self.device,
                                     encoder_ckpt=self.config.encoder_ckpt)

        self.model = self.model.to(self.device)
        self.model.train()

        self.setup_optimizer()
        self.setup_rollouts()

    def setup_optimizer(self):
        raise NotImplementedError()

    def setup_rollouts(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts):
        raise NotImplementedError()

    def update(self, rollout):
        raise NotImplementedError()

    def process_obs(self, obs_in):
        # Change to tensor, change type, add batch dimension for observation.
        obs_out = []
        for obs in obs_in:
            if not isinstance(obs, torch.Tensor):
                obs = np.asarray(obs)
                obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            obs = obs.float()
            # if len(obs.shape) == 1 or len(obs.shape) == 3:  # Add additional batch dimension.
            #     obs = obs.view(1, *obs.shape)
            obs_out.append(obs)

        return obs_out

    def compute_action(self, obs, deterministic=False):
        obs = self.process_obs(obs)

        # [TODO] Get the actions and the log probability of the action from the output of neural network.
        #  Hint: Use proper distribution to help you

        if self.discrete:  # Please use categorical distribution.
            logits, values = self.model(obs)
            dist = Categorical(logits=logits)
            if deterministic:
                actions = dist.probs.argmax(dim=-1, keepdim=True)
            else:
                actions = dist.sample()
            action_log_probs = dist.log_prob(actions.view(-1))
            actions = actions.view(-1, 1)  # In discrete case only return the chosen action.

        else:  # Please use normal distribution. You should
            means, log_std, values = self.model(obs)
            dist = Normal(means, torch.exp(log_std))
            actions = dist.sample()

            actions = actions.view(-1, self.num_actions)
            action_log_probs = dist.log_prob(actions)

        action_log_probs = action_log_probs.view(-1).sum()
        values = values.view(-1, 1)

        # print(actions)
        return values, actions, action_log_probs

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""

        obs = self.process_obs(obs)

        if self.discrete:
            logits, values = self.model(obs)
            pass
            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(act.view(-1)).view(-1, 1)
            dist_entropy = dist.entropy().mean()
        else:
            means, log_std, values = self.model(obs)
            pass
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(means, action_std)
            # print(type(act.double()), act.shape)
            action_log_probs = dist.log_prob(act).sum(dim=1).view(-1, 1)
            dist_entropy = dist.entropy().mean()

        assert dist_entropy.shape == ()

        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        return values, action_log_probs, dist_entropy

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        obs = self.process_obs(obs)
        if self.discrete:
            _, values = self.model(obs)
        else:
            _, _, values = self.model(obs)
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            print("Successfully load weights from {}!".format(save_path))
            return True
        else:
            print("Failed to load weights from {}!".format(save_path))
            return False


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions, discrete, device, encoder_ckpt=None):
        super(ActorCritic, self).__init__()
        # init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
        #                                   lambda x: nn.init.constant_(x, 0),
        #                                   nn.init.calculate_gain('relu'))

        # Setup the log std output for continuous action space
        self.discrete = discrete
        if discrete:
            self.actor_logstd = None
        else:
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

        # The network structure is designed for 42X42 observation.
        # if input_shape[1:] == (42, 42):
        #     self.conv1 = init_(nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2))
        #     self.conv2 = init_(nn.Conv2d(16, 32, kernel_size=4, stride=2))
        #     self.conv3 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        #     self.conv4 = init_(nn.Conv2d(64, 128, kernel_size=2, stride=2))
        #
        # elif input_shape[1:] == (96, 96):  # For cCarRacing-v0
        #     self.conv1 = init_(nn.Conv2d(4, 16, kernel_size=4, stride=2))  # (-1, 16, 47, 47)
        #     self.conv2 = init_(nn.Conv2d(16, 32, kernel_size=4, stride=2))  # (-1, 32, 22, 22)
        #     self.conv3 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))  # (-1, 64, 10, 10)
        #     self.conv4 = init_(nn.Conv2d(64, 128, kernel_size=4, stride=2))  # (-1, 128, 4, 4)
        #     self.conv5 = init_(nn.Conv2d(128, 256, kernel_size=4, stride=2))  # (-1, 256, 1, 1)
        #     self.hidden = nn.Linear(256, 100)
        # else:
        #     raise ValueError("We only support input shape (42, 42), or (96, 96) right now.")
        self.fusion_encoder = SensorFusionEncoder(device=device, z_dim=input_shape, deterministic=True)
        if encoder_ckpt:
            self.fusion_encoder.load_state_dict(torch.load(encoder_ckpt))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(input_shape, 1))

        init_ = lambda m: self.layer_init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )
        self.actor_linear = init_(nn.Linear(input_shape, num_actions))

        self.train()

    def forward(self, inputs):
        _, _, _, _, z = self.fusion_encoder(*inputs)

        value = self.critic_linear(z)
        logits = self.actor_linear(z)

        if self.discrete:
            return logits, value
        else:
            return logits, self.actor_logstd, value

    def feature_size(self, input_shape):
        return self.fusion_encoder(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.policy = nn.Linear(100, output_size)
        self.value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.policy(x)
        value = self.value(x)
        return action, value


# def test_base_trainer():
#     # from competitive_rl import make_envs
#
#     class FakeConfig:
#         def __init__(self):
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.num_envs = 1
#             self.num_steps = 200
#             self.gamma = 0.99
#             self.lr = 5e-4
#
#     class FakeTrainer(BaseTrainer):
#         def setup_optimizer(self):
#             pass
#
#         def setup_rollouts(self):
#             pass
#
#     # ===== Discrete case =====
#     env = make_envs("cPong-v0", asynchronous=False, num_envs=3)
#     trainer = FakeTrainer(env, FakeConfig())
#     obs = env.reset()
#     # Input single observation
#     values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
#     new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
#     assert actions.shape == (1, 1), actions.shape
#     assert values.shape == (1, 1), values.shape
#     assert action_log_probs.shape == (1, 1), action_log_probs.shape
#     assert dist_entropy.shape == ()
#     assert (values == new_values).all()
#     assert (action_log_probs == new_action_log_probs).all()
#     assert dist_entropy.shape == ()
#
#     # Input multiple observations
#     values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
#     new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
#     assert actions.shape == (3, 1), actions.shape
#     assert values.shape == (3, 1), values.shape
#     assert action_log_probs.shape == (3, 1), action_log_probs.shape
#     assert dist_entropy.shape == ()
#     assert (values == new_values).all()
#     assert (action_log_probs == new_action_log_probs).all()
#     assert dist_entropy.shape == ()
#
#     print("Base trainer discrete case test passed!")
#     env.close()
#
#     # ===== Continuous case =====
#     env = make_envs("cCarRacing-v0", asynchronous=False, num_envs=3)
#     trainer = FakeTrainer(env, FakeConfig())
#     obs = env.reset()
#     # Input single observation
#     values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
#     new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
#     assert actions.shape == (1, 2), actions.shape
#     assert values.shape == (1, 1), values.shape
#     assert action_log_probs.shape == (1, 1), action_log_probs.shape
#     assert dist_entropy.shape == ()
#     assert (values == new_values).all()
#     assert (action_log_probs == new_action_log_probs).all()
#     assert dist_entropy.shape == ()
#
#     # Input multiple observations
#     values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
#     new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
#     assert actions.shape == (3, 2), actions.shape
#     assert values.shape == (3, 1), values.shape
#     assert action_log_probs.shape == (3, 1), action_log_probs.shape
#     assert dist_entropy.shape == ()
#     assert (values == new_values).all()
#     assert (action_log_probs == new_action_log_probs).all()
#     assert dist_entropy.shape == ()
#
#     print("Base trainer continuous case test passed!")
#     env.close()


if __name__ == '__main__':
    # test_base_trainer()
    print("Base trainer test passed!")
