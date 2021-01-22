import time
from collections import deque

import gym
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

'''
Note: Right now my library doesnt have right backward gradient function..using
vjp is too much work. Will use pytorch here.
Idea:
Simple vanilla gradient. Will implement it for 3 envs, Cart Pole, Mountain Car and Lunar Lander

Proposed structure:
class vpg():
    policy network()
    get action()
    value_function_network()
    Advantage_function()
        reward at that time step - state value function for that state
        reward to go - state value function for that state
    Rewards_to_go()
    policy gradients():
        expectation of policy gradients
        update step()
    value function gradients():
        update step()

main()
    run D trajectories
    save rewards and derivatives of log of policy actions and value functions at each timestep
    plotting results

Questions:
How am I choosing an action? Its obviously not max of policy logits

Versions and analysis:
I am not all using policy gradient theorem here.
I AM SERIOUSLY DISCOURAGED. I thought of implementing policy gradients using derivative of log probs
but didn't understand how to do that. So I looked around different implementations and found that they
have converted the problem to a supervised one. Maximizing the log likelihoods  instead of proper gradients
I WAS WRONG ABOVE. I DIDN'T understand it right. The derivative of line 105 is 
..math::
 \nabla_\theta \ln_\pi(s,a)*\text{advantage_fucntion} 
which is the true policy gradient.
v1: discrete action space
v2: calculate actual gradients
v3: with sequentail networks and advantage function removed
'''


class Vpg:
    def __init__(self, policy_network_dims, value_function_network_dims,lr, states_type="discrete"):
        end = None
        if states_type == "discrete":
            end = "softmax"
        elif states_type == "continuous":
            end = "gaussian"
        self.policy_network = self.create_network(policy_network_dims, end)
        self.value_fn_network = self.create_network(value_function_network_dims)
        self.lr = lr
        self.policy_optim = optim.Adam(self.policy_network.parameters(),lr=lr)
        self.value_optim = optim.Adam(self.value_fn_network.parameters(),lr=lr)

    def create_network(self, dims, end=None):
        network = []
        for i, j in enumerate(dims):
            if i != len(dims) - 1:
                network.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    network.append(nn.ReLU())
        if end:
            if end == "softmax":
                network.append(torch.nn.Softmax())
            if end == "gaussian":
                raise NotImplementedError
        return nn.Sequential(*network)

    def forward(self, input_: torch.Tensor, network: nn.Module):
        return network(input_)

    def get_policy(self, observation):
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation,dtype=torch.float32)
        logits = self.forward(observation, self.policy_network)
        return Categorical(logits)

    def get_action(self, observation):
        distribution = self.get_policy(observation)
        return distribution.sample().item()

    def get_value_function(self, observation):
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32)
        value_fn = self.forward(observation, self.value_fn_network)
        return value_fn

    def compute_policy_loss(self, observation, action, rewards):
        policy = self.get_policy(observation)
        with torch.no_grad():
            value_fn = self.get_value_function(observation)
        action = torch.FloatTensor(action)
        rewards = torch.FloatTensor(rewards)
        log_probabilities = policy.log_prob(action)
        advantage_fn = rewards - value_fn
        return -(log_probabilities * advantage_fn).mean()

    def compute_value_loss(self, observation, rewards):
        value_fn = self.get_value_function(observation)
        rewards = torch.FloatTensor(rewards)[:,None]
        return F.mse_loss(value_fn, rewards)



def get_rewards_to_go(rewards):
    length = len(rewards)
    total_rewards = sum(rewards)
    a = 0
    rewards_to_go = []
    for i in rewards:
        a = a + i
        rewards_to_go.append(total_rewards - a)
    return rewards_to_go


def main():
    games = 5000
    test_games = 10
    D = 3
    policy_update_iters = 3
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = Vpg([input_dims, 256,128,64, action_space], [input_dims, 256,128,64, 1], 0.0001,states_type="discrete")

    observations_ = []
    rewards_ = []
    actions_ = []

    avg_reward_per_game = []
    last_100_rewards = deque(maxlen=100)
    d = 0
    for i in range(games):
        observation = env.reset()
        done = False

        episodic_reward = []
        episodic_observation = []
        episodic_actions = []

        rewards_this_game = []
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, _ = env.step(action)
            episodic_reward.append(reward)
            episodic_observation.append(observation)
            episodic_actions.append(action)
            observation = next_observation
            rewards_this_game.append(reward)
            last_100_rewards.append(reward)

        observations_.append(episodic_observation)
        actions_.append(episodic_actions)
        rewards_to_go = get_rewards_to_go(episodic_reward)
        rewards_.append(rewards_to_go)
        d += 1
        if d == D:
            # flattening lists
            observation = [k for episode in observations_ for k in episode]
            action = [k for episode in actions_ for k in episode]
            rewards = [k for episode in rewards_ for k in episode]

            # get policy gradients
            for klm in range(policy_update_iters):
                agent.value_fn_network.eval()
                policy_loss = agent.compute_policy_loss(observation,action,rewards)
                agent.policy_optim.zero_grad()
                policy_loss.backward()
                agent.policy_optim.step()
                agent.value_fn_network.train()
                # get value func gradients
                agent.value_optim.zero_grad()
                value_fn_loss = agent.compute_value_loss(observation,rewards)
                value_fn_loss.backward()
                agent.value_optim.step()


            observations_.clear()
            actions_.clear()
            rewards_.clear()
            d = 0

        avg_reward_this_game = sum(rewards_this_game)/len(rewards_this_game)
        avg_reward_per_game.append(avg_reward_this_game)
        print(f'For game {i}, avg reward {avg_reward_this_game}, avg last 100 rewards {sum(last_100_rewards)/100}')

    # Plotting avg rewards per game
    plt.figure(figsize=(8, 6))
    plt.title("Average reward of VPG agent on" + env_name + " for each game")
    plt.plot(range(len(avg_reward_per_game)), avg_reward_per_game)
    plt.savefig("figures/VPG_" + env_name + "_rewards.png")
    plt.show()

    # test policy
    observation = env.reset()
    for i in range(test_games):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            time.sleep(1e-3)
            action = agent.get_action(observation)
            observation, r, done, _ = env.step(action)
            # env.close()


if __name__ == '__main__':
    main()
