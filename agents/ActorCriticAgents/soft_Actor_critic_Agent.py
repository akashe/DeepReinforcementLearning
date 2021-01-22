from copy import deepcopy

import torch
import torch.nn as nn
from agents import Agent
from RLUtils import create_network_with_nn,ReplayBuffer
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam


class NormalPolicyNetwork(nn.Module):
    def __init__(self,dims,act_limit):
        super().__init__()
        assert len(dims) >= 3
        self.p = create_network_with_nn(dims[:-1]) # this would not have activations for the second last layers
        self.mu = nn.Linear(dims[-2],dims[-1])  # mean of the distribution
        self.std = nn.Linear(dims[-2],dims[-1]) # standard deviation of the distribution

        self.act_limit = act_limit

    def forward(self,o,deterministic=False,log_probs = True):
        # log_probs is the term we use for entropy while updating P and Q
        o = torch.FloatTensor(o)
        o_ = self.p(o)
        mu = self.mu(o_)
        clamped_std = torch.clamp(self.std(o_),-20,2) # clamping in a way that std doesnt have too huge value
        std = torch.exp(clamped_std)
        distribution = Normal(mu, std) # Distribution to sample actions from

        if deterministic:
            action = mu
        else:
            action = distribution.rsample()
        if log_probs:
            log_probs_ = distribution.log_prob(action).sum(axis=-1)
            log_probs_ -= (2*(np.log(2)+action -F.softplus(2*action))).sum(axis=1)

            # This method of calculating log_probs is better coz it wont involve squaring
            # of small tanh values. Original idea for this reformulation from spinning up,

        else:
            log_probs_ = None

        clamped_action = torch.tanh(action) # this results in clamping the action values while having all possible values from [-1,1]
        scaled_action = self.act_limit*clamped_action

        return scaled_action,log_probs_


class QNetwork(nn.Module):
    def __init__(self,dims):
        super(QNetwork, self).__init__()
        self.network = create_network_with_nn(dims)

    def forward(self,input_):
        return self.network.forward(input_)


class SAC_Agent(Agent):
    '''
    This version of Soft actor Critic Agent has
    1) constant entropy co-efficient
    2) no clipping for actions as tanh will itself clip it

    SAC agents have entropy regularization. Instead of reducing entropy, agent is trying to maximize it.
    Which enables greater probabilities to all outcomes and stops agent from exploiting peaks in Q functions.

    Visually you can see the presence of entropy terms between V(s) and Q(s,a) in a MDP as shown below.
        V(s)
            -
             -
              - entropy terms.. Entropy = Expectation(\pi)[-log(\pi(s)]
             - -
     Q(s,a')-   - Q(s,a)
           -     -
    R(s,a')-      -R(s,a)
          -        -
    V(s'')-         - V(s')

    You can use the above graph to formulate definitions of V_\pi(s) and Q_\pi(s,a) in entropy
    maximization context.

    This changes the goal of a policy. Now policy has to maximize both reward and the entropy.

    '''
    def __init__(self,PolicyNetworkDims,QNetworkDims,buffer_size,polyak,discount_factor,q_lr,p_lr,entropy_constant,act_limit):
        super(SAC_Agent, self).__init__()
        # An SAC agent has 1 policy network(no target policy network) and 2 q-value networks
        self.policy = NormalPolicyNetwork(PolicyNetworkDims,act_limit)
        self.q1 = QNetwork(QNetworkDims)
        self.q2 = deepcopy(self.q1)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q1)

        # Buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size)

        # params
        self.polyak = polyak
        self.discount_factor = discount_factor
        self.entropy_constant = entropy_constant

        # optims:
        self.p_optim = Adam(self.policy.parameters(),p_lr)
        self.q1_optim = Adam(self.q1.parameters(),q_lr)
        self.q2_optim = Adam(self.q2.parameters(),q_lr)

    def take_action(self,observation,deterministic=False,log_probs=False):
        with torch.no_grad():
            return self.policy.forward(observation,deterministic,log_probs)

    def batch_min(self,t1,t2):
        a = (t1 > t2).float()
        return a*t2 + (1-a)*t1

    def updateQ(self,batch):
        s,a,r,s_,d = batch

        with torch.no_grad():
            future_a,entropy_sample = self.policy.forward(s_,log_probs=True)    # policy action for s_
            # the reason for naming this entropy sample coz its a sample of total entropy of policy for that state
            future_o_plus_a = torch.cat((s_,future_a),dim=-1)
            future_q1 = self.q1_target.forward(future_o_plus_a) # Estimate of Q-value of next state and future action by q1 target network
            future_q2 = self.q2_target.forward(future_o_plus_a) # Q(s_,future_a) by q2 target network

            future_q = self.batch_min(future_q1,future_q2)
            future_rewards = future_q - self.entropy_constant*entropy_sample[:,None]

            r,d,a = r[:,None],d[:,None],a[:,None] # increasing dimensions of rewards, done and action
            target = r + self.discount_factor*(1-d)*future_rewards

        # Updating q1
        self.q1_optim.zero_grad()
        q1_prediction = self.q1.forward(torch.cat((s,a),dim=-1))
        q1_loss = F.mse_loss(target,q1_prediction)
        q1_loss.backward()
        self.q1_optim.step()


        # Updating q2
        q2_prediction = self.q2.forward(torch.cat((s,a),dim=-1))
        q2_loss = F.mse_loss(target,q2_prediction)
        q2_loss.backward()
        self.q2_optim.step()
        self.q2_optim.zero_grad()

    def updateP(self,batch):
        s,_,_,_,_ = batch
        # while updating policy we take the minimum of predictions of q1 and q2 and not (q1_target,q2_target)

        for i in self.q1.parameters():
            i.requires_grad = False
        for i in self.q2.parameters():
            i.requires_grad = False

        self.p_optim.zero_grad()
        action,entropy = self.policy.forward(s,log_probs=True)
        future_s_a = torch.cat((s,action),dim=-1)

        q1_pred = self.q1.forward(future_s_a)
        q2_pred = self.q2.forward(future_s_a)

        q_pred = self.batch_min(q1_pred,q2_pred)

        p_loss = - (q_pred-self.entropy_constant*entropy).mean()
        p_loss.backward()
        self.p_optim.step()

        for i in self.q1.parameters():
            i.requires_grad = True
        for i in self.q2.parameters():
            i.requires_grad = True

    def freeze_target_networks(self):
        # target networks are updated using polyak and not gradients
        for i in self.q1_target.parameters():
            i.requires_grad = False
        for i in self.q2_target.parameters():
            i.requires_grad = False

    def updateNetworks(self):
        networks = [(self.q1,self.q1_target),(self.q2,self.q2_target)]
        with torch.no_grad():
            for i,j in networks:
                for cur_params,tar_prams in zip(i.parameters(),j.parameters()):
                    tar_prams.data.mul_(self.polyak)
                    tar_prams.data.add_((1-self.polyak)*cur_params.data)
