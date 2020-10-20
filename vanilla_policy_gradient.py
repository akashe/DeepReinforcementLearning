import time

import gym
import torch
from TorchFunctions.dataInitialization import kaiming_initialization
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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
v1: discrete action space
v2: calculate actual gradients
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

    def create_network(self, dims, end=None):
        network = []
        for i, j in enumerate(dims):
            if i != len(dims) - 1:
                network.append(kaiming_initialization(torch.randn([dims[i], dims[i + 1]])))
                if i < len(dims) - 2:
                    network.append(F.relu_)
        if end:
            if end == "softmax":
                network.append(torch.nn.functional.softmax)
            if end == "gaussian":
                raise NotImplementedError
        return network

    def forward(self, input_: torch.Tensor, network: []):
        for i in network:
            if torch.is_tensor(i):
                input_ = input_.matmul(i)
            else:
                input_ = i(input_)
        return input_

    def get_policy(self, observation):
        if not torch.is_tensor(observation):
            observation = torch.FloatTensor(observation)
        logits = self.forward(observation, self.policy_network)
        return Categorical(logits)

    def get_action(self, observation):
        distribution = self.get_policy(observation)
        return distribution.sample().item()

    def get_value_function(self, observation):
        if not torch.is_tensor(observation):
            observation = torch.Tensor(observation, dtype=torch.float)
        value_fn = self.forward(observation, self.value_fn_network)
        return value_fn

    def compute_policy_loss(self, observation, action, rewards):
        policy = self.get_policy(observation)
        value_fn = self.get_value_function(observation)
        action = torch.FloatTensor(action)
        rewards = torch.FloatTensor(rewards)
        log_probabilities = policy.log_prob(action)
        advantage_fn = rewards - value_fn
        return -log_probabilities * advantage_fn  # taking negative log likelihood rather than policy gradients

    def compute_value_loss(self, observation, rewards):
        value_fn = self.get_value_function(observation)
        rewards = torch.FloatTensor(rewards)
        return F.mse_loss(value_fn, rewards)

    def update_params(self,network):
        with torch.no_grad():
            for i in network:
                if torch.is_tensor(i):
                    i -= self.lr*i.grad
                    i.grad.zero_()


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
    games = 2000
    test_games = 10
    D = 5
    env = gym.make('LunarLander-v2')
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = Vpg([input_dims, 20, 10, action_space], [input_dims, 20, 1], 0.001,states_type="discrete")

    observations_ = []
    rewards_ = []
    actions_ = []

    for i in range(games):
        observation = env.reset()
        done = False
        d = 0
        episodic_reward = []
        episodic_observation = []
        episodic_actions = []
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, _ = env.step(action)
            episodic_reward.append(reward)
            episodic_observation.append(observation)
            episodic_actions.append(action)
            observation = next_observation

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
            policy_loss = agent.compute_policy_loss(observation,action,rewards)
            policy_loss.backward()
            agent.update_params(agent.policy_network)

            # get value func gradients
            value_fn_loss = agent.compute_value_loss(observation,rewards)
            value_fn_loss.backward()
            agent.update_params(agent.value_fn_network)

            observations_.clear()
            actions_.clear()
            rewards_.clear()
            d = 0

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


if __name__ == '__main__':
    main()
