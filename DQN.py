import collections

import torch

from RLUtils import MeanSquarredError
from RLUtils import SGD
from RLUtils import appendOnes
import gym
import random
import time
import math
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.functional import vjp
import matplotlib.pyplot as plt
from RLUtils import create_network_with_nn
from copy import deepcopy

random.seed(7)
torch.manual_seed(1334)
'''
Idea:
The idea is to build a simple dqn algo with only linear transformation and updating the parameters with SGD
I will use simple LunarLander gym.
The idea is to get a feel of the algo not to extend functionality. So the implementation will be bare minimum.
For v2, the implementation uses vjp not traditional torch.backwards
For v3, it will have proper networks and gradient clipping and proper documentation.

Versions and analysis:
v1: simple linear transform
Result: model underfitting..though with 5000 games and 50 episodes for switch the lander did pretty good
v2: add non linearity and appropriate gradients..
Mistake in v2 that I used only 1 non-linearity. The average rewards was negative for v2 and loss also 
not decreasing. 
v3:
gradient clipping is good. The algo works better and faster with bigger model capacity like for lunar lander [256,128,64]
rather than [50,10]. A very small Replaybuffer doesn't work well. Learning and switching the networks together
speeds up getting good rewards.
'''


class DQNAgentv3:

    def __init__(self, discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                 batch_size, learning_rate, max_memory_size, mid_layers=[256,128,64], gradient_clip=1.,polyak=0.995):
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
        # Initializing two sets of network parameters
        # Target network
        self.target_network = create_network_with_nn([observation_size, *mid_layers, no_of_actions])
        # Current network
        self.current_network = create_network_with_nn([observation_size, *mid_layers, no_of_actions])
        self.loss = []
        self.step_counter = 0
        self.learn_counter = 0
        self.optim = optim.Adam(self.current_network.parameters(), lr=self.lr)
        self.gradient_clip = gradient_clip
        self.polyak = polyak

    def store_experience(self, experience):
        # observation, action, reward, next_observation, done = experience
        self.memory.appendleft(experience)

    def sample_experience(self):
        # sample from the replay experience of the length batch_size
        return random.sample(self.memory, self.batch_size)

    def switch_networks(self):
        # Switch target and current network after every fixed number of games
        # Replacing deep copy with polyak averaging
        # self.target_network = deepcopy(self.current_network)
        for cur_params, tar_prams in zip(self.current_network.parameters(), self.target_network.parameters()):
            tar_prams.data.mul_(self.polyak)
            tar_prams.data.add_((1 - self.polyak) * cur_params.data)

    def reduce_epsilon(self):
        # Reducing episolon value over time
        # self.e = self.e - self.e_decay if self.e > self.e_end else self.e_end
        self.e = max(self.e_end,self.e*self.e_decay)

    def take_action(self, observation):
        c = random.random()
        self.reduce_epsilon()
        if c > self.e:
            observation = torch.FloatTensor(observation)[None,]
            with torch.no_grad():
                q_values = self.current_network(observation)
            action = torch.argmax(q_values).item()
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

        # Loss = { target_value - actual_value}^2
        # Calculating target value using target_network

        with torch.no_grad():
            future_rewards = self.d * torch.max(self.target_network(next_observations).detach(), dim=1)[
                0] * non_terminal_state
            target_value = rewards + future_rewards
        target_value = target_value.unsqueeze(-1)

        self.optim.zero_grad()

        z = self.current_network(observations).gather(1, actions.unsqueeze(1).long())

        loss = F.mse_loss(z, target_value)

        loss.backward()

        # clipping gradients..clipping helps in faster rewards
        torch.nn.utils.clip_grad_norm(self.current_network.parameters(), self.gradient_clip)

        self.optim.step()

        self.loss.append(loss)

        return loss.item()


if __name__ == "__main__":
    # define system parameters
    (discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions, batch_size,
     learning_rate,
     max_memory_size, max_steps) = \
        (0.99, 1.0, 0.995, 0.01, 8, 4, 64, 5e-4, 100, 1000)
    # define env
    env = gym.make('LunarLander-v2')
    # define agent
    agent = DQNAgentv3(discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                       batch_size, learning_rate,
                       max_memory_size)
    # train the Q network
    total_games = 350
    scores = []
    avg_score = []
    avg_loss = []
    game_loss = []
    # train loop
    for i in range(total_games):
        observation = env.reset()
        done = False
        score = 0
        t = 0
        while not done and t < max_steps:
            action = agent.take_action(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.store_experience((observation, action, reward, next_observation, done))
            score += reward
            agent.step_counter += 1
            observation = next_observation
            t += 1
            if agent.step_counter < batch_size:
                continue
            else:
                loss = agent.learn()
                game_loss.append(loss)
                if agent.learn_counter == 4:
                    agent.switch_networks()
                    agent.learn_counter = 0
                agent.learn_counter += 1
        print(f' Total reward for game {i} is {score}')
        scores.append(score)
        if len(game_loss) > 0:
            avg_loss.append(sum(game_loss) / len(game_loss))
        game_loss = []

    # Plotting average reward
    plt.figure(figsize=(8, 6))
    plt.savefig("figures/DQN_Lunar_lander_rewards.png")
    plt.title("Reward of DQN agent on LunarLander-v2 for each game")
    plt.plot(range(len(scores)), scores)
    plt.show()
    # Plotting average loss

    # its not necessary that the loss converges because it is not in sync with actual training
    # every time we observe loss, we are calculating for instances from replaybuffer
    plt.figure(figsize=(8, 6))
    plt.title("Average Loss of DQN agent on LunarLander-v2")
    plt.plot(range(len(avg_loss)), avg_loss)
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
    """
    Doesn't converge.
    """

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


class DQNAgentv2:
    """
    Doesn't converge.
    """

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

    def forward_for_gradients(self, W1, W2, W3, input_, target_, actions):
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
        loss = F.mse_loss(z, target_.unsqueeze(-1))

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

        inputs_for_gradients = (
        self.current_network[0], self.current_network[1], self.current_network[2], observations, target_value, actions)
        # Documentation of vjp: https://pytorch.org/docs/stable/autograd.html#torch.autograd.functional.vjp
        # it gives gradient of inputs wrt to the function we pass to the vjp. I was experimenting with while trying to
        # understand autograd
        loss, gradients = vjp(self.forward_for_gradients, inputs_for_gradients)

        print(" Loss for step " + str(self.step_counter) + " = " + str(loss.item()))
        self.loss.append(loss)

        for count, i__ in enumerate(gradients[0:3]):
            self.current_network[count] = SGD(parameters=self.current_network[count], gradients=i__,
                                              learning_rate=self.lr)
