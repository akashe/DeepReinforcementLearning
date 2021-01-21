import time

import gym
from agents import TD3Agent
import torch
from collections import deque
import matplotlib.pyplot as plt

'''
Idea: to implement td3: twin delayed deep deterministic policy gradient using only pytorch and not
my library
Experiment and result:
1) there is a clear difference in speed bw nn.Sequential and my create_networks()
2) getting the right hyperparams is tough
'''


def td3():
    # Setting up environment
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    action_space = env.action_space.shape[0]
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    observation_space = env.observation_space.shape[0]
    noise_clip = 0.5  # this variable should change with different envs
    # 0.5 here since a_low and a_high in MountainCarContinuous is -1 and +1
    print(f' action space high {action_space_high}, action space low {action_space_low}, observation space {observation_space}')

    # Training variables
    epochs = 1000
    max_steps_per_episode = 1000
    random_actions_till = 15000
    policy_delay = 3
    update_every = 50
    update_after = 15000
    batch_size = 100
    buffer_size = 10000
    polyak = 0.995
    # polyak = 0.8
    PolicyNetworkDims = [observation_space, 256,128,64, action_space]
    QNetworkDims = [observation_space + action_space, 256,128,64, 1]
    add_noise_till = 10000000
    discount_factor = 0.9
    q_lr = 0.0001
    p_lr = 0.0001
    no_of_updates = 6
    test_epochs = 1
    test_steps = 1500
    test_after = int(epochs*0.99)

    agent = TD3Agent(PolicyNetworkDims, QNetworkDims, action_space_high, action_space_low, buffer_size,
                     polyak, add_noise_till, discount_factor, q_lr, p_lr, noise_clip)

    total_steps = 0
    rewards_list = []
    score_deque = deque(maxlen=100)
    for i in range(epochs):
        observation = env.reset()
        j = 0
        done = False
        game_reward = []
        while (not done) and j < max_steps_per_episode:
            if total_steps < random_actions_till:
                action = torch.FloatTensor(env.action_space.sample())  # For starting episodes take random actions
            else:
                action = agent.take_action(observation, total_steps)
            new_observation, reward, done, _ = env.step(action)
            game_reward.append(reward)
            score_deque.append(reward)
            agent.ReplayBuffer(observation, action, reward, new_observation, done)  # Save the experience
            observation = new_observation

            if  total_steps > update_after and total_steps % update_every == 0:  # update parameters after agent has explored enough by taking random actions
                for k in range(no_of_updates):
                    batch = agent.ReplayBuffer.sample(batch_size)  # sample from replay memory
                    agent.updateQ(batch)
                    if k % policy_delay == 0:  # In td3, policy updates are delayed
                        agent.updateP(batch)
                        agent.updateNetworks()

            j += 1
            total_steps += 1
        avg_reward_this_game = sum(game_reward)/len(game_reward)
        game_reward = []
        rewards_list.append(avg_reward_this_game)
        print(f'For game number {i}, mean of last 100 rewards = {sum(score_deque)/100}')
        env.close()

        if i > test_after:  # Test agent after certain number of training epochs
            for i_ in range(test_epochs):  # Run test by visualizing TODO: use reward instead of visualization
                with torch.no_grad():
                    observation_ = env.reset()
                    done_ = False
                    j_ = 0
                    while (not done_) and j_ < test_steps:
                        env.render()
                        # time.sleep(1e-3)
                        action = agent.take_action(observation_, add_noise_till + 1)
                        o_, _, done_, _ = env.step(action)
                        observation_ = o_
                        j += 1
                    env.close()

    # Plotting avg rewards per game
    plt.figure(figsize=(8, 6))
    plt.title("Average reward of TD3 agent on"+env_name+" for each game")
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig("figures/TD3_"+env_name+"_rewards.png")
    plt.show()


if __name__ == '__main__':
    td3()
