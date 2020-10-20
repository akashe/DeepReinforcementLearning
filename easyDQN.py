import collections

import torch
from CostFunctions.MeanSquarredError import MeanSquarredError
from Optimizers.SGD import SGD
from TorchFunctions.dataModifications import appendOnes
import gym
import random
import time
import math
import torch.nn.functional as F
from torch.autograd.functional import vjp
import matplotlib.pyplot as plt

'''
Idea:
The idea is to build a simple dqn algo with only linear transformation and updating the parameters with SGD
I will use simple LunarLander gym.
The idea is to get a feel of the algo not to extend functionality. So the implementation will be bare minimum.
For v2, the implementation uses vjp not traditional torch.backwards

Questions: 
should error signals from target network be broadcasted..no I did mistake..I ignored actions
in Q(s,a) I was subtracting Q(s) but I have to subtract for the action taken Q(s,'a')

Versions and analysis:
v1: simple linear transform
Result: model underfitting..though with 5000 games and 50 episodes for switch the lander did pretty good
v2: add non linearity and appropriate gradients..
Actually both linear and non linear models converge but they need to be trained for 
high number of games
'''


class DQNAgentv2:

    def __init__(self, discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                 batch_size, learning_rate, max_memory_size, layer1_dim=50, layer2_dim=10):
        '''

        :param discounted_factor: discount factor for future rewards
        :param epsilon_start: starting value of epsilon in e-greedy exploration
        :param epsilon_decay: rate of epsilon decay
        :param epsilon_end: ending value of epsilon
        :param observation_size: state size or observation size or input size
        :param no_of_actions: total number of discrete actions
        :param batch_size: batch size for experience replay
        :param max_memory_size: total size of memory buffer
        '''
        self.d = discounted_factor
        self.e = epsilon_start
        self.e_decay = epsilon_decay
        self.e_end = epsilon_end
        self.o_size = observation_size
        self.n_actions = no_of_actions
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=max_memory_size)
        self.lr = learning_rate
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        # Initializing two sets of network parameters
        # Target network
        self.Wt1 = torch.randn(self.o_size + 1, self.layer1_dim, dtype=torch.float) * math.sqrt(
            2 / (self.o_size + 1))
        self.Wt2 = torch.randn(self.layer1_dim + 1, self.layer2_dim, dtype=torch.float) * math.sqrt(
            2 / (self.layer1_dim + 1))
        self.Wt3 = torch.randn(self.layer2_dim + 1, self.n_actions, dtype=torch.float) * math.sqrt(
            2 / (self.layer2_dim + 1))

        self.target_network = [self.Wt1, self.Wt2, self.Wt3]

        # Current network
        self.Wc1 = torch.randn(self.o_size + 1, self.layer1_dim, dtype=torch.float) * math.sqrt(
            2 / (self.o_size + 1))
        self.Wc2 = torch.randn(self.layer1_dim + 1, self.layer2_dim, dtype=torch.float) * math.sqrt(
            2 / (self.layer1_dim + 1))
        self.Wc3 = torch.randn(self.layer2_dim + 1, self.n_actions, dtype=torch.float) * math.sqrt(
            2 / (self.layer2_dim + 1))

        self.current_network = [self.Wc1, self.Wc2, self.Wc3]

        self.loss = []
        self.step_counter = 0
        self.learn_counter = 0

    def store_experience(self, experience):
        # observation, action, reward, next_observation, done = experience
        self.memory.appendleft(experience)

    def sample_experience(self):
        # don't forget to put check for len(deque)< batch_size
        return random.sample(self.memory, self.batch_size)

    def switch_networks(self):
        # Switch target and current network after every 10 games
        for i, j in enumerate(self.current_network):
            self.target_network[i] = j.clone()
            # will this lead to memory waste?

    def reduce_epsilon(self):
        # Putting this in take_action rather than learn
        self.e = self.e - self.e_decay if self.e > self.e_end else self.e_end

    def take_action(self, observation):
        # don't forget to decay epsilon
        c = random.random()
        self.reduce_epsilon()
        if c > self.e:
            observation = torch.FloatTensor(observation)[None,]
            actions = self.forward(self.current_network, observation)
            action = torch.argmax(actions).item()
            return action
        else:
            return random.randrange(0, self.n_actions)

    def forward(self, network, input_):
        for count, i_ in enumerate(network):
            input_ = appendOnes(input_)
            z = input_.mm(i_)
            if count != (len(network) - 1):
                x = F.relu_(z)
                input_ = x
        return z

    def forward_for_gradients(self, W1, W2, W3, input_, target_,actions):
        # I know this is bloated implementation resulting in lower efficiency
        layers = [W1, W2, W3]
        for count, i_ in enumerate(layers):
            input_ = appendOnes(input_)
            z = input_.mm(i_)
            if count != (len(layers) - 1):
                x = F.relu_(z)
                # x = z
                input_ = x

        z = z.gather(1, actions.unsqueeze(1).long())
        # loss = MeanSquarredError(labels=target_.unsqueeze(-1), targets=z, batch_size=self.batch_size)
        # try F.mse_loss and validate gradients
        loss = F.mse_loss(z,target_.unsqueeze(-1))

        return loss

    def learn(self):
        replay_experience = self.sample_experience()
        # I dont have to specifically train only current network in the beginning it wud be just
        # as random as the target network
        replay_experience = list(zip(*replay_experience))
        observations = torch.FloatTensor(replay_experience[0])
        actions = torch.FloatTensor(replay_experience[1])
        rewards = torch.FloatTensor(replay_experience[2])
        next_observations = torch.FloatTensor(replay_experience[3])
        non_terminal_state = 1 - (torch.BoolTensor(replay_experience[4])).type(torch.float)

        # Loss = { target_value - actual_value}^2
        # Calculating target value using target_network

        with torch.no_grad():
            future_rewards = self.d * torch.max(self.forward(self.target_network, next_observations), dim=1)[
                0] * non_terminal_state
            target_value = rewards + future_rewards

        # Calculating actual values
        # Now to get gradients I will have to pass each of current networks parameter
        # to get gradients from vjp

        inputs_for_gradients = (self.current_network[0], self.current_network[1], self.current_network[2], observations, target_value,actions)
        loss, gradients = vjp(self.forward_for_gradients, inputs_for_gradients)

        print(" Loss for step " + str(self.step_counter) + " = " + str(loss.item()))
        self.loss.append(loss)

        for count, i__ in enumerate(gradients[0:3]):
            self.current_network[count] = SGD(parameters=self.current_network[count], gradients=i__,
                                              learning_rate=self.lr)


if __name__ == "__main__":
    # define system parameters
    (discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions, batch_size,
     learning_rate,
     max_memory_size) = \
        (0.9, 1.0, 0.0001, 0.001, 8, 4, 64, 0.003, 50000)
    # define env
    env = gym.make('LunarLander-v2')
    # define agent
    agent = DQNAgentv2(discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                       batch_size, learning_rate,
                       max_memory_size)
    # train a policy
    total_games = 1000
    scores = []
    # train loop
    for i in range(total_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.take_action(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.store_experience((observation, action, reward, next_observation, done))
            score += reward
            agent.step_counter += 1
            observation = next_observation
            if agent.step_counter < batch_size:
                continue
            else:
                agent.learn()
                agent.learn_counter += 1
                # Switch networks after every 100 learn steps
                if agent.learn_counter == 100:
                    print("Switching networks")
                    agent.switch_networks()
                    agent.learn_counter = 0

        scores.append(score)
        print(" Game ", i, 'score %.2f' % score, )

    plt.plot(range(len(scores)),scores)
    plt.show()
    # test policy
    observation = env.reset()
    test_episodes = 20
    for i in range(test_episodes):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            time.sleep(1e-3)
            action = agent.take_action(observation)
            observation, r, done, _ = env.step(action)


class DQNAgent:

    def __init__(self, discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                 batch_size, learning_rate, max_memory_size):
        '''

        :param discounted_factor: discount factor for future rewards
        :param epsilon_start: starting value of epsilon in e-greedy exploration
        :param epsilon_decay: rate of epsilon decay
        :param epsilon_end: ending value of epsilon
        :param observation_size: state size or observation size or input size
        :param no_of_actions: total number of discrete actions
        :param batch_size: batch size for experience replay
        :param max_memory_size: total size of memory buffer
        '''
        self.d = discounted_factor
        self.e = epsilon_start
        self.e_decay = epsilon_decay
        self.e_end = epsilon_end
        self.o_size = observation_size
        self.n_actions = no_of_actions
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=max_memory_size)
        self.lr = learning_rate
        # Initializing two networks with simple (Wx + b)..no non-linearity
        self.target_network = torch.randn(self.o_size + 1, self.n_actions, dtype=torch.float) * math.sqrt(
            2 / (self.o_size + 1))
        self.current_network = torch.randn(self.o_size + 1, self.n_actions, dtype=torch.float) * math.sqrt(
            2 / (self.o_size + 1))

        self.loss = []
        self.step_counter = 0
        self.learn_counter = 0

    def store_experience(self, experience):
        # observation, action, reward, next_observation, done = experience
        self.memory.appendleft(experience)

    def sample_experience(self):
        # don't forget to put check for len(deque)< batch_size
        return random.sample(self.memory, self.batch_size)

    def switch_networks(self):
        # Switch target and current network after every 10 games
        self.target_network.data = self.current_network.clone()

    def reduce_epsilon(self):
        # Putting this in take_action rather than learn
        self.e = self.e - self.e_decay if self.e > self.e_end else self.e_end

    def take_action(self, observation):
        # don't forget to decay epsilon
        c = random.random()
        self.reduce_epsilon()
        if c > self.e:
            observation = appendOnes(torch.FloatTensor(observation))[None,]
            actions = observation.mm(self.current_network)
            action = torch.argmax(actions).item()
            return action
        else:
            return random.randrange(0, self.n_actions)

    def learn(self):
        replay_experience = self.sample_experience()
        # I dont have to specifically train only current network in the beginning it wud be just
        # as random as the target network
        replay_experience = list(zip(*replay_experience))
        observations = torch.FloatTensor(replay_experience[0])
        actions = torch.FloatTensor(replay_experience[1])
        rewards = torch.FloatTensor(replay_experience[2])
        next_observations = torch.FloatTensor(replay_experience[3])
        non_terminal_state = 1 - (torch.BoolTensor(replay_experience[4])).type(torch.float)

        # Appending observations for bias
        observations = appendOnes(observations)
        next_observations = appendOnes(next_observations)

        # Loss = { target_value - actual_value}^2
        # Calculating target value using target_network

        future_rewards = self.d * torch.max(next_observations.mm(self.target_network), dim=1)[0] * non_terminal_state
        target_value = rewards + future_rewards

        # Calculating actual values
        # Question: are we broadcasting target value to full action space/Q values and subtracting them
        # error??
        actual_value = observations.mm(self.current_network)

        loss = MeanSquarredError(labels=target_value.unsqueeze(-1), targets=actual_value, batch_size=self.batch_size)
        print(" Loss for step " + str(self.step_counter) + " = " + str(loss.item()))
        self.loss.append(loss)

        # gradients transpose of observation(batch_size* input dim+1)* (target- actual)(batch*size* output dim)
        gradients = -(2 / self.batch_size) * observations.t().mm(
            (target_value.unsqueeze(-1) - actual_value))

        # apply gradients
        self.current_network = SGD(parameters=self.current_network, gradients=gradients, learning_rate=self.lr)
