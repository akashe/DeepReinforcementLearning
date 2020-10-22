from abc import ABC

import torch

from RLUtils import create_network_with_nn, ReplayBuffer
from agents import Agent
from torch import nn
from copy import deepcopy
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn.functional as F

# TODO: make a new abstraction for PolicyNetwork and QValueNetwork as they are doing the same thing

class PolicyNetwork(nn.Module):
    def __init__(self, PolicyNetworkDims):
        super().__init__()
        self.Pnetwork = create_network_with_nn(PolicyNetworkDims)

    def forward(self, input_):
        return self.Pnetwork(torch.FloatTensor(input_))


class QValueNetwork(nn.Module):
    def __init__(self, QNetworkDims):
        super().__init__()
        self.QNetwork = create_network_with_nn(QNetworkDims)

    def forward(self, input_):
        return self.QNetwork(torch.FloatTensor(input_))


class TD3Agent(Agent, ABC):
    '''
    A td3 Agent has 2 Q-value functions each with current amd target network and 1 policy function
    with current and target network.
    Key differences from ddpg:
    1. using 2 Q-value functions and taking the minimum of both q-value approximations. Reason being, in ddpg
        a single Q-value function has a tendency to approximate high Q-values
    2. Policy updates are delayed. In ddpg policy updates happen every time Q-value updates happens. In
        TD3 it it delayed by 'policy_delay'
    3. Adding a noise to selected action. Why? As in point 1, Q-value functions will approximate high Q-values
        and policy function will exploit this to get maximize policy updates.
    '''

    def __init__(self, PolicyNetworkDims, QNetworkDims, action_space_high, action_space_low, buffer_size,
                 polyak, add_noise_till, discount_factor, q_lr, p_lr,noise_clip):
        self.q1_current = QValueNetwork(QNetworkDims)  # First q-value function
        self.q2_current = QValueNetwork(QNetworkDims)  # Second q-value function
        self.q1_target = deepcopy(self.q1_current)  # target network of first q-value function
        self.q2_target = deepcopy(self.q2_current)  # target network of second q-value function
        self.p_current = PolicyNetwork(PolicyNetworkDims)  # policy function
        self.p_target = deepcopy(self.p_current)  # target policy function
        #TODO: make better class docs ..move all the info to @params
        self.ReplayBuffer = ReplayBuffer(buffer_size)
        self.polyak = polyak # Averaging factor
        self.discount_factor = discount_factor # discount factor for future rewards

        self.q1_optim = Adam(lr=q_lr, params=self.q1_current.parameters())
        self.q2_optim = Adam(lr=q_lr, params=self.q2_current.parameters())
        self.p_optim = Adam(lr=p_lr, params=self.p_current.parameters())

        self.noise = Normal(0,1) # zero-mean,sd one gaussian noise to add to actions
        #TODO: make standard deviation learnable
        self.action_space_high = torch.FloatTensor(action_space_high)
        self.action_space_low = torch.FloatTensor(action_space_low)
        self.add_noise_till = add_noise_till # no of steps after which we stop adding noise
        self.noise_clip = noise_clip # max and min value of noise to be added to action while computing targets
        # for Q-value updates

    def take_action(self,observation,steps):
        with torch.no_grad():
            action = self.p_current.forward(observation)
            noise = self.noise.sample()
            if steps < self.add_noise_till:
                action += noise
            return self.clip_action(action)

    def clip_action(self,action):
        if action > self.action_space_high:
            return self.action_space_high
        if action < self.action_space_low:
            return self.action_space_low

        return action

    def clip_noise(self,noise):
        if noise > self.noise_clip:
            return self.noise_clip
        elif noise < (-self.noise_clip):
            return -self.noise_clip
        else:
            return noise

    def clip_action_batch(self,action):
        a = (action > self.action_space_high[:,None]).float()
        action = a*self.action_space_high[:,None] + (1-a)*action
        b = (action < self.action_space_low[:None]).float()
        action = b*self.action_space_low[:,None] + (1-b)*action

        return action


    def action_for_Q(self,observation):
        action = self.p_target.forward(observation)
        noise = self.clip_noise(self.noise.sample())
        action += noise
        return self.clip_action_batch(action)

    def updateQ(self,batch):
        s, a, r, s_,d = batch #Do s,a,r get gradients?? No right
        with torch.no_grad():
            action_for_next_observation = self.action_for_Q(s_)
            q_value1_for_next_observation = self.q1_target.forward(torch.cat((s_,action_for_next_observation),dim=-1))
            q_value2_for_next_observation = self.q2_target.forward(torch.cat((s_,action_for_next_observation),dim=-1))
            min_ = (q_value1_for_next_observation > q_value2_for_next_observation).float()
            minimum_Q = (1-min_)*q_value1_for_next_observation + min_*q_value2_for_next_observation
            targets = r[:,None] + self.discount_factor*(1-d)[:,None]*minimum_Q

        #Compute losses
        q1_loss_input = self.q1_current.forward(torch.cat((s,a[:,None]),dim =-1))
        q2_loss_input = self.q2_current.forward(torch.cat((s,a[:,None]),dim=-1))

        # Update self.q1_current
        loss_q1 = F.mse_loss(target=targets,input=q1_loss_input)
        loss_q1.backward()
        self.q1_optim.step()
        self.q1_optim.zero_grad()

        # Update self.q2_current
        loss_q2 = F.mse_loss(target=targets,input=q2_loss_input)
        loss_q2.backward()
        self.q2_optim.step()
        self.q2_optim.zero_grad()

    def updateP(self,batch):
        s, a, r, s_, d = batch
        target_actions = self.p_current.forward(s)

        # freezing Q network params:
        for i in self.q1_current.parameters():
            i.requires_grad = False

        q_predictions = self.q1_current.forward(torch.cat((s,target_actions),dim=-1))
        policy_loss = - q_predictions.mean()

        policy_loss.backward()
        self.p_optim.step()
        self.p_optim.zero_grad()

        # Unfreeze Q network params:
        for i in self.q1_current.parameters():
            i.requires_grad = True

    def updateNetworks(self):
        networks = [(self.q1_current,self.q1_target),(self.q2_current,self.q2_target),(self.p_current,self.p_target)]
        for (i,j) in networks:
            for cur_params,tar_prams in zip(i.parameters(),j.parameters()):
                tar_prams.data.mul_(self.polyak)
                tar_prams.data.add_((1-self.polyak)*cur_params.data)
