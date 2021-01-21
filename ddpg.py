import time

import torch
import gym
import torch.nn.functional as F
from torch.distributions.normal import Normal
from RLUtils import freeze_network,unfreeze_network,create_network,forward,ReplayBuffer

from collections import deque
import matplotlib.pyplot as plt

'''
Current implementation will not work with CNN's as Qnetwork or Pnetwork
Mistakes:
1) no disabling cg construction while taking steps
2) not taking care of dims while getting targets for Qnetwork
3) zeroing grads for networks still not used resulting in error of Nonetype has no data
'''


class QNetwork:
    def __init__(self, qnetwork_mid_dims, action_space, observation_space):
        qnetwork_mid_dims.append(1)
        qnetwork_mid_dims.insert(0, action_space + observation_space)
        self.target_network = create_network(qnetwork_mid_dims)
        self.current_network = create_network(qnetwork_mid_dims)

    def __call__(self, network, input_):
        if network == "target":
            return forward(self.target_network, input_)
        if network == "current":
            return forward(self.current_network, input_)


class PNetwork:
    def __init__(self, pnetwork_mid_dims, action_space, action_space_high, action_space_low, observation_space,
                 add_noise_till):
        self.action_space = action_space
        self.action_high = torch.FloatTensor(action_space_high)
        self.action_low = torch.FloatTensor(action_space_low)
        self.observation_space = observation_space
        pnetwork_mid_dims.append(action_space)
        pnetwork_mid_dims.insert(0, observation_space)
        self.PNetwork_target = create_network(pnetwork_mid_dims)
        self.PNetwork_current = create_network(pnetwork_mid_dims)
        self.noise = Normal(0, 1)
        self.add_noise_till = add_noise_till

    def take_action(self, observation, total_steps):
        with torch.no_grad():
            observation = torch.FloatTensor(observation)
            action = forward(self.PNetwork_current, observation)
            if total_steps <= self.add_noise_till:
                noise = self.noise.sample()
                action += noise

            return self.clip_action(action)

    def clip_action(self, action):
        if action < self.action_low: action = self.action_low
        if action > self.action_high: action = self.action_high

        return action

    def __call__(self, input_, network="current"):
        if network == "current":
            return forward(self.PNetwork_current, input_)
        if network == "target":
            return forward(self.PNetwork_target, input_)


class DDPG:
    def __init__(self, pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, add_noise_till, discount_factor, lr):
        self.PNetwork_ = PNetwork(pnetwork_mid_dims, action_space, action_space_high, action_space_low,
                                  observation_space, add_noise_till)
        self.QNetwork = QNetwork(qnetwork_mid_dims, action_space, observation_space)
        self.ReplayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.polyak = polyak
        self.discount_factor = discount_factor
        self.lr = lr

    def UpdateQ(self, batch):
        s, a, r, s_, d = batch
        # Compute targets
        with torch.no_grad():
            a_targets = self.PNetwork_(s_, "target")
            q_targets = self.QNetwork("target", torch.cat((s_, a_targets), -1))
            targets = r[:, None] + self.discount_factor * (1 - d)[:, None] * q_targets
        logits = self.QNetwork("current", torch.cat((s, a[:, None]), -1))
        loss_ = F.mse_loss(logits, targets)
        loss_.backward()
        self.optim_step(self.QNetwork.current_network)
        self.zero_grad([self.QNetwork.current_network])

    def optim_step(self, network, max=False):
        params_ = []
        for i in network:
            if torch.is_tensor(i) and i.requires_grad == True:
                params_.append(i)
        with torch.no_grad():
            for i in params_:
                if max:
                    i.data += self.lr * i.grad
                else:
                    i.data -= self.lr * i.grad

    def zero_grad(self, networks):
        # Ideally I shud have model.get_trainable_params here
        for j in networks:
            for i in j:
                if torch.is_tensor(i) and i.requires_grad == True:
                    i.grad.data.zero_()

    def UpdateP(self, batch):
        s, _, _, _, _ = batch
        a = self.PNetwork_(s)
        freeze_network(self.QNetwork.current_network)
        '''
        The thing is u don't need to freeze and unfreeze coz:
        1) backward() will remove the CG graph associated with calculating gradients and wont let Pnetwork.current get gradients from QNetwork updates
        2) we were zeroing gradients for self.QNetwork.current_network
        
        Freezing stops wastage of creating graph nodes and later zeroing the gradients
        '''
        cost_func_for_policy = torch.sum(self.QNetwork("current", torch.cat((s, a), -1))) / len(s)
        cost_func_for_policy.backward()
        self.optim_step(self.PNetwork_current, max=True)
        networks = [self.PNetwork_current]
        unfreeze_network(self.QNetwork.current_network)
        self.zero_grad(networks)

    def UpdateNetworks(self):
        with torch.no_grad():
            for i, j in zip(self.PNetwork_target, self.PNetwork_current):
                if torch.is_tensor(i) and torch.is_tensor(j) and i.requires_grad == True and j.requires_grad == True:
                    i = self.polyak * i + (1 - self.polyak) * j

            for i, j in zip(self.QNetwork.current_network, self.QNetwork.target_network):
                if torch.is_tensor(i) and torch.is_tensor(j) and i.requires_grad == True and j.requires_grad == True:
                    i = self.polyak * i + (1 - self.polyak) * j

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
    epochs = 100
    max_steps_per_episode = 1500
    random_actions_till = 10000
    update_every = 50
    update_after = 1000
    batch_size = 100
    buffer_size = 10000
    polyak = 0.995
    pnetwork_mid_dims = [100, 100]
    qnetwork_mid_dims = [150, 100]
    add_noise_till = 30000
    discount_factor = 0.9
    lr = 0.001
    no_of_updates = 5
    test_epochs = 1
    test_steps = 200

    # Environment
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)
    action_space = env.action_space.shape[0]
    action_space_high = env.action_space.high
    action_space_low = env.action_space.low
    observation_space = env.observation_space.shape[0]

    # Agent
    agent = DDPG(pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, add_noise_till, discount_factor, lr)

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
                action = agent.take_action(observation, total_steps)
            else:
                action = torch.FloatTensor(env.action_space.sample())
            next_observation, reward, done, _ = env.step(action)
            game_reward.append(reward)
            score_deque.append(reward)
            agent.ReplayBuffer(observation, action, reward, next_observation, done)
            observation = next_observation

            if total_steps % update_every == 0 and total_steps > update_after:
                for k in range(no_of_updates):
                    batch = agent.ReplayBuffer.sample(batch_size)
                    agent.UpdateQ(batch)
                    agent.UpdateP(batch)
                    agent.UpdateNetworks()

            j += 1
            total_steps += 1
        avg_reward_this_game = sum(game_reward) / len(game_reward)
        game_reward = []
        rewards_list.append(avg_reward_this_game)
        print(f'For game number {i}, mean of last 100 rewards = {sum(score_deque) / 100}')
        env.close()

    # Plotting avg rewards per game
    plt.figure(figsize=(8, 6))
    plt.title("Average reward of DDPG agent on Pendulum-v0 for each game")
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
                action = agent.take_action(observation, add_noise_till + 1)
                observation, _, done, _ = env.step(action)
                j_ += 1
            env.close()


if __name__ == "__main__":
    main()
