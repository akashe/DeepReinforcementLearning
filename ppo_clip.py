from collections import deque

import gym
import torch
import time
import matplotlib.pyplot as plt
from agents.ActorCriticAgents.PPO_clip_agent import PPO_clip_agent
import numpy as np
from gym.spaces import Box, Discrete

'''
Idea: 
To implement PPO algorithm, specifically ppo-clip with early stopping.
PPO is an on-policy algorithm which means it learns as it performs. In off-policy algos like
ddpg,td3 the policy is updated from previous data that we save in a Replay buffer.
PPO can work for both continuous and discrete environments.

Ingredients:
an agent with 1 policy networks, one value function estimator, funcs to:
1) act
2) update policy
3) update value function
A special buffer with:
1) func to return discounted rewards
2) func to get data
3) func to store data

Experiment and Results:
1) changing activations from Relu to tanh works better for pendulum-v0
2) even with tanh the overall rewards dont increase in pendulum-v0
'''


def ppo_clip(seed=1003):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setting up Environment
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    action_space = env.action_space
    observation_space = env.observation_space
    observation_dim = observation_space.shape[0]
    if isinstance(action_space, Box): # TODO: I am doing this twice, second time in PPO agent,update later
        action_dim = action_space.shape[0]
    if isinstance(action_space, Discrete):
        action_dim = action_space.n


    # Training parameters
    epochs = 1000
    steps_per_epoch = 1000 # Number of steps in 1 epoch
    max_ep_len = 4000 # total steps allowed..for cases when steps per epoch can be very high
    buffer_size = steps_per_epoch # size of PPO buffer
    discount_factor = 0.99 # discount factor for future rewards
    lambda_ = 0.97 # lambda for GAE lambda
    v_lr = 0.0001 # Value function learning rate
    p_lr = 0.0001 # policy learning rate
    train_pi_iters = 80
    train_v_iters = 80
    clip_ratio = 0.2
    target_kl = 0.01
    PolicyNetworkDims = [observation_dim, 16,16, action_dim] # policy network dims for policy network
    VNetworkDims = [observation_dim, 16,16, 1] # network dims for value function network
    test_epochs = int(epochs*0.01)
    test_steps = 1500

    agent = PPO_clip_agent(observation_space,action_space,PolicyNetworkDims,VNetworkDims,buffer_size,discount_factor,lambda_,v_lr,p_lr,train_pi_iters,train_v_iters,clip_ratio,target_kl)

    avg_rewards = []
    last_100_reward = deque(maxlen=100)
    for epoch in range(epochs):
        o = env.reset()
        j = 0 # epoch length
        this_epoch_rewards = []
        for t in range(steps_per_epoch):
            # get action from agent
            action, value_estimate, logp = agent.take_action(torch.as_tensor(o,dtype=torch.float32))

            # take the action
            next_o,reward,done,_ = env.step(action)
            this_epoch_rewards.append(reward)
            last_100_reward.append(reward)
            j += 1

            agent.buffer.store(o,action,reward,value_estimate,logp)
            o = next_o

            timeout = j == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                if timeout or epoch_ended: # Getting a final estimate of V since trajectory didn't finish
                    _,v,_ = agent.take_action(torch.as_tensor(o,dtype=torch.float32))
                else:
                    v = 0
                agent.buffer.finish_path(v)
        avg_reward_this_epoch = sum(this_epoch_rewards)/len(this_epoch_rewards)
        print(f'For epoch {epoch}, avg reward = {avg_reward_this_epoch}, last 100 rewards = {sum(last_100_reward)}')
        avg_rewards.append(avg_reward_this_epoch)
        agent.update()

    # Plotting avg rewards per game
    plt.figure(figsize=(8, 6))
    plt.title("Average reward of PPO agent on" + env_name + "for each game")
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.savefig("figures/PPO_" + env_name + "_rewards.png")
    plt.show()

    for i_ in range(test_epochs):
        with torch.no_grad():
            observation = env.reset()
            done = False
            j_ = 0
            while not (done or j_ > test_steps):
                env.render()
                time.sleep(1e-3)
                action, _, _ = agent.take_action(torch.as_tensor(observation,dtype=torch.float32))
                observation, _, done, _ = env.step(action)
                j_ += 1
            env.close()


ppo_clip()