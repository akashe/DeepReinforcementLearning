from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from agents import Agent
from RLUtils import create_network_with_nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from gym.spaces import Box, Discrete

'''
The implementation is heavily influenced by implementation here:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
'''


class PPO_buffer:
    """
    This buffer saves information the trajectory in each game/epoch.
    Can't use a deque here because after each game, we need all the rewards and value functions
    to get future discounted rewards with generalized advantage estimation.
    """
    def __init__(self,observation_size,action_size,buffer_size,discount_factor,lambda_):
        # making observation sizes generic so that buffer can accomodate multi dims observations
        if not isinstance(observation_size,list):
            observation_size = [observation_size]
        if not isinstance(action_size,list):
            action_size = [action_size]
        self.obs_buf = np.zeros((buffer_size,*observation_size),dtype=np.float32)   # observation values buffer
        self.act_buf = np.zeros((buffer_size,*action_size),dtype=np.float32)    # action values buffer
        self.adv_buf = np.zeros(buffer_size,dtype=np.float32) # advantage function values buffer
        self.rew_buf = np.zeros(buffer_size,dtype=np.float32)   # rewards buffer
        self.ret_buf = np.zeros(buffer_size,dtype=np.float32)   # reward to go buffer
        self.val_buf = np.zeros(buffer_size,dtype=np.float32)   # value function buffer
        self.logp_buff = np.zeros(buffer_size,dtype=np.float32) # logp values buffer
        self.gamma = discount_factor
        self.lam = lambda_
        self.ptr,self.path_start_idx,self.max_size = 0,0,buffer_size

    def store(self,obs,act,rew,val,lopgp):
        # Unlikely but for cases when no of allowed steps in an epoch > buffer size we need to raise an error.
        # Because we use the entire trajectory for policy improvement
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buff[self.ptr] = lopgp
        self.ptr += 1

    def finish_path(self,last_val):
        '''
        This function creates the future return buffer.
        Based on the predicted V estimates, it calculates the expected future rewards for Value function error
        and advantage function values for policy updates after discounting them and applying GAE-lambda value which
        helps in smoothing.
        :param last_val: final v value. Could be non zero when the epoch timeouts or zero when 'done'.
        :return: None
        '''

        path_slice = slice(self.path_start_idx,self.ptr)
        rewards = np.append(self.rew_buf[path_slice],last_val) # final reward estimate or 0 for completion of trajectory
        values = np.append(self.val_buf[path_slice],last_val)

        # # Advantage estimates for each step of the game
        #
        # To include future advantage values and future rewards along with discounted factor
        discounted_advantage_values = []
        gae = 0
        for i in reversed(range(len(rewards)-1)):
            delta = rewards[i] + self.gamma*values[i+1]-values[i]
            gae = delta + self.gamma*self.lam*gae
            discounted_advantage_values.insert(0,gae)

        discounted_advantage_values = np.array(discounted_advantage_values)
        # check if you are getting same values as
        # advantage_estimates = rewards[:-1] + self.gamma*values[1:] - values[:-1]
        # scipy.signal.lfilter([1], [1, float(-self.gamma*self.lam)], advantage_estimates[::-1], axis=0)[::-1]

        # Normalizing advantage values
        discounted_advantage_values_mean,discounted_advantage_values_std = discounted_advantage_values.mean(),discounted_advantage_values.std()
        normalized_discounted_advantage_values = discounted_advantage_values-discounted_advantage_values_mean/(1e-8+ discounted_advantage_values_std)
        self.adv_buf[path_slice] = normalized_discounted_advantage_values

        discounted_rewards = []
        future_reward = 0
        for i in reversed(range(len(rewards))):
            reward_ = rewards[i]
            future_reward = reward_ + self.gamma*future_reward
            discounted_rewards.insert(0,future_reward)

        self.ret_buf[path_slice] = np.array(discounted_rewards[:-1])
        # check if correct values with
        # scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr,self.path_start_idx = 0,0
        data = dict(obs=self.obs_buf,act=self.act_buf,ret=self.ret_buf,
                    adv=self.adv_buf,logp=self.logp_buff)
        return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in data.items()}


class CategoricalPolicy(nn.Module):
    def __init__(self,network_dims):
        super(CategoricalPolicy, self).__init__()
        self.net = create_network_with_nn(network_dims,activation=nn.Tanh())

    def forward(self,observation):
        logits = F.softmax(self.net(observation)[:,None],dim=0).squeeze(-1)
        distribution = Categorical(logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        return action, log_probs

    def distribution_and_logprobs(self, observation, action):
        # this could be done better
        # returns the distribution for observation and
        logits = F.softmax(self.net(observation),dim=-1)
        distribution = Categorical(logits)
        log_probs_a = distribution.log_prob(action)

        return distribution, log_probs_a



class NormalPolicy(nn.Module):
    def __init__(self,network_dims):
        super(NormalPolicy, self).__init__()
        action_dims = network_dims[-1]
        self.log_std = -0.5*torch.ones(action_dims)
        self.mu = create_network_with_nn(network_dims,activation=nn.Tanh())

    def forward(self,observation):
        mu = self.mu(observation)
        std = torch.exp(self.log_std)
        distribution = Normal(mu,std)
        action = distribution.rsample()
        log_probs = distribution.log_prob(action).sum(axis=-1) # sum for torch normal distributions

        return action,log_probs

    def distribution_and_logprobs(self,observation,action):
        # this could be done better
        # returns the distribution for observation and
        mu = self.mu(observation)
        std = torch.exp(self.log_std)
        distribution = Normal(mu, std)
        log_probs_a = distribution.log_prob(action)

        return distribution,log_probs_a

class VNetwork(nn.Module):
    def __init__(self,network_dims):
        super(VNetwork, self).__init__()
        self.net = create_network_with_nn(network_dims)

    def forward(self,observation):
        return self.net(observation)


class PPO_clip_agent(Agent, ABC):
    def __init__(self,observation_space,action_space,PolicyNetworkDims,VNetworkDims,buffer_size,discount_factor=0.99,lambda_=0.97,v_lr=0.001,p_lr=0.0001,train_pi_iters=80,train_v_iters=80,clip_ratio=0.2,target_kl=0.01):
        super(PPO_clip_agent, self).__init__()
        self.observation_size = observation_space.shape[0]
        if isinstance(action_space,Box):
            self.pi = NormalPolicy(PolicyNetworkDims)
            self.action_size = action_space.shape[0]
            self.buffer = PPO_buffer(self.observation_size, self.action_size, buffer_size, discount_factor, lambda_)
        if isinstance(action_space,Discrete):
            self.pi = CategoricalPolicy(PolicyNetworkDims)
            self.action_size = action_space.n
            self.buffer = PPO_buffer(self.observation_size, 1, buffer_size, discount_factor, lambda_)

        self.v = VNetwork(VNetworkDims)



        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters # number of iterations for training policy
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        # optimizers
        self.pi_optimizer = Adam(self.pi.parameters(),lr=p_lr)
        self.v_optimizer = Adam(self.v.parameters(),lr=v_lr)


    def take_action(self,obs):
        with torch.no_grad():
            action, log_probs = self.pi.forward(obs)
            v = self.v.forward(obs)

        return action.numpy(),v.numpy(),log_probs.numpy()

    def compute_loss_pi(self,data):
        observation,action,advantage_values,logp_old = data['obs'],data['act'],data['adv'],data['logp']

        # Policy loss
        pi,logp = self.pi.distribution_and_logprobs(observation,action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*advantage_values
        loss_pi = -(torch.min(ratio*advantage_values,clip_adv)).mean()

        # additional KL info
        # approximation of how far the old policy is from new one..P(x) -> old,Q(x)-> new
        approx_kl = (logp_old-logp).mean().item() # missing the multiplicative 'P(x)' term P(x)*(log(P(x)-log(Q(x))
        ent = pi.entropy().mean().item() # entropy of the current policy
        '''
        the policy is clipped if:
        1) when advantage is positive: then the new policy will try to be more like old policy(i.e. increase 
        prob values). We restrict the policy jump size by allowing (1+clip_ratio) jump size by enforcing a min bw them.
        2) when advantage is negative: then the new policy will try to go away from old policy(i.e. decrease
        prob values). We restrict the maximum amount the policy can decrease by (1-clip_ratio) by enforcing a max bw them.
        So policy was clipped whenever it was more than (1+clip ratio) and was less than (1-clip_ratio)  
        '''
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped,dtype=torch.float32).mean().item() # how many times clipping happened
        pi_info = dict(kl=approx_kl,ent=ent,cf=clipfrac)

        return loss_pi,pi_info


    def compute_loss_v(self,data):
        obs,ret = data['obs'],data['ret']
        return ((self.v.forward(obs)-ret.unsqueeze(-1))**2).mean()

    def update(self):
        data = self.buffer.get()

        # These set as benchmark for the old policy while we get a new policy self.train_pi_iters times
        pi_l_old,pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi,pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping because of excess KL difference at policy train iteration {i}")
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.v_optimizer.step()


