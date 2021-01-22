import time

import torch
import gym
import torch.nn.functional as F
from RLUtils import create_network_with_nn,ReplayBuffer,create_network_end_activation

from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

'''
Current implementation will not work with CNN's as Qnetwork or Pnetwork
Mistakes:
1) no disabling cg construction while taking steps
2) not taking care of dims while getting targets for Qnetwork
3) zeroing grads for networks still not used resulting in error of Nonetype has no data
4) not using np.clip()
5) not freezing target networks before starting the update cycle.

Experiments and analysis:
1) I spent a lot of time on this algo. Turns out, keep giving noise in actions forever and the scale 
of noise is really important for the algo to work. Also the kind of noise, I was using Normal(0,1) for noise
but np.random.rand() works better, i don't know why??
2) adding a lot of random start actions helped
3) how many times you update the network is also important. Like update after 50 steps but for 50 times.
4) adding tanh() in the end of policy networks helped
'''


class QNetwork:
    def __init__(self, qnetwork_mid_dims, action_space, observation_space):
        qnetwork_mid_dims.append(1)
        qnetwork_mid_dims.insert(0, action_space + observation_space)
        self.target_network = create_network_with_nn(qnetwork_mid_dims)
        self.current_network = create_network_with_nn(qnetwork_mid_dims)

    def __call__(self, network, input_):
        if network == "target":
            return self.target_network(input_)
        if network == "current":
            return self.current_network(input_)


class PNetwork:
    def __init__(self, pnetwork_mid_dims, action_space, action_space_high, action_space_low, observation_space):
        self.action_space = action_space
        self.action_high = torch.as_tensor(action_space_high[0])
        self.action_low = torch.as_tensor(action_space_low)
        self.observation_space = observation_space
        pnetwork_mid_dims.append(action_space)
        pnetwork_mid_dims.insert(0, observation_space)
        self.PNetwork_target = create_network_end_activation(pnetwork_mid_dims)
        self.PNetwork_current = create_network_end_activation(pnetwork_mid_dims)

    def take_action(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation)
            action = self.PNetwork_current(observation)
            action = action*self.action_high
            noise = 0.1 * np.random.randn(self.action_space)
            action += noise

            # return self.clip_action(action)
            return np.clip(action,self.action_low,self.action_high)

    def clip_action(self, action):
        if action < self.action_low: action = self.action_low
        if action > self.action_high: action = self.action_high

        return action

    def __call__(self, input_, network):
        if network == "current":
            return self.PNetwork_current(input_)
        elif network == "target":
            return self.PNetwork_target(input_)
        else:
            raise ValueError


class DDPG:
    def __init__(self, pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, discount_factor, lr):
        self.PNetwork_ = PNetwork(pnetwork_mid_dims, action_space, action_space_high, action_space_low,
                                  observation_space)
        self.QNetwork = QNetwork(qnetwork_mid_dims, action_space, observation_space)
        self.ReplayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.polyak = polyak
        self.discount_factor = discount_factor
        self.lr = lr
        self.QNetwork_current_optim = optim.Adam(self.QNetwork.current_network.parameters(),lr=lr)
        self.PNetwork_current_optim = optim.Adam(self.PNetwork_.PNetwork_current.parameters(),lr=lr)

    def UpdateQ(self, batch):
        s, a, r, s_, d = batch
        self.QNetwork_current_optim.zero_grad()

        # Compute targets
        with torch.no_grad():
            a_targets = self.PNetwork_(s_, "target")
            q_targets = self.QNetwork("target", torch.cat((s_, a_targets), -1))
            targets = r[:, None] + self.discount_factor * (1 - d)[:, None] * q_targets

        logits = self.QNetwork("current", torch.cat((s, a[:,None]), -1))
        loss_ = F.mse_loss(logits, targets)
        loss_.backward()
        self.QNetwork_current_optim.step()

    def UpdateP(self, batch):
        s, _, _, _, _ = batch
        s = s.detach()

        # Setting self.QNetwork.current_network.eval() still allows gradient accumulation
        for i in self.QNetwork.current_network.parameters():
            i.requires_grad = False

        self.PNetwork_current_optim.zero_grad()
        a = self.PNetwork_(s,"current")
        cost_func_for_policy = -self.QNetwork("current", torch.cat((s, a), -1)).mean()
        cost_func_for_policy.backward()
        self.PNetwork_current_optim.step()

        for i in self.QNetwork.current_network.parameters():
            i.requires_grad = True

    def UpdateNetworks(self):
        with torch.no_grad():
            for i, j in zip(self.PNetwork_target.parameters(), self.PNetwork_current.parameters()):
                assert (i.shape == j.shape)
                i.data.mul_(self.polyak)
                i.data.add_((1-self.polyak)*j.data)

            for i, j in zip(self.QNetwork.target_network.parameters(),self.QNetwork.current_network.parameters()):
                assert (i.shape == j.shape)
                i.data.mul_(self.polyak)
                i.data.add_((1 - self.polyak) * j.data)

    def freeze_target_networks(self):
        # the target networks dont need gradient. Update them only with polyak
        for i in self.PNetwork_.PNetwork_target.parameters():
            i.requires_grad = False
        for i in self.QNetwork.target_network.parameters():
            i.requires_grad = False

    def __getattr__(self, item):
        # if hasattr(self,item):
        #     return getattr(self,item)
        if hasattr(self.PNetwork_, item):
            return getattr(self.PNetwork_, item)
        elif hasattr(self.QNetwork, item):
            return getattr(self.QNetwork, item)
        else:
            raise AttributeError


def main():
    # arguments
    epochs = 1500
    max_steps_per_episode = 200
    random_actions_till = 10000
    update_every = 50
    update_after = 1000
    batch_size = 100
    buffer_size = 10000
    polyak = 0.995
    pnetwork_mid_dims = [256,128,64]
    qnetwork_mid_dims = [256,128,64]
    add_noise_till = 10000
    discount_factor = 0.9
    lr = 0.0001
    no_of_updates = 50
    test_epochs = min(int(epochs*0.01),20)
    test_steps = 200
    action_noise = 0.1

    # Environment
    env_name = "Pendulum-v0" # "MountainCarContinuous-v0"
    env = gym.make(env_name)
    action_space = env.action_space.shape[0]
    action_space_high = env.action_space.high
    action_space_low = env.action_space.low
    observation_space = env.observation_space.shape[0]

    # Agent
    agent = DDPG(pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, discount_factor, lr)

    agent.freeze_target_networks()

    total_steps = 0
    rewards_list = []
    score_deque = deque(maxlen=100)
    for i in range(epochs):
        observation = env.reset()
        done = False
        j = 0
        game_reward = []
        while (not done) and j < max_steps_per_episode:
            if total_steps > random_actions_till:
                action = agent.take_action(observation)
            else:
                action = torch.FloatTensor(env.action_space.sample())
            next_observation, reward, done, _ = env.step(action)
            game_reward.append(reward)
            score_deque.append(reward)
            agent.ReplayBuffer(observation, action, reward, next_observation, done)
            observation = next_observation

            j += 1
            total_steps += 1

        if total_steps > update_after and total_steps % update_every == 0:
            for k in range(no_of_updates):
                batch = agent.ReplayBuffer.sample(batch_size)
                agent.UpdateQ(batch)
                agent.UpdateP(batch)
                agent.UpdateNetworks()

        avg_reward_this_game = sum(game_reward) / len(game_reward)
        game_reward = []
        rewards_list.append(avg_reward_this_game)
        print(f'For game number {i},avg reward this game {avg_reward_this_game}, mean of last 100 rewards = {sum(score_deque) / 100}')
        env.close()

    # Plotting avg rewards per game
    plt.figure(figsize=(8, 6))
    plt.title("Average reward of DDPG agent on"+env_name+"for each game")
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig("figures/DDPG_" + env_name + "_rewards.png")
    plt.show()

    for i_ in range(test_epochs):
        with torch.no_grad():
            observation = env.reset()
            done = False
            j_ = 0
            while not (done or j_ > test_steps):
                env.render()
                time.sleep(1e-3)
                action = agent.take_action(observation)
                observation, _, done, _ = env.step(action)
                j_ += 1
            env.close()


if __name__ == "__main__":
    torch.manual_seed(171)
    main()
