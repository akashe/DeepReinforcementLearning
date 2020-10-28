import gym
import torch

from agents import SAC_Agent

'''
Idea: Implement a SAC agent for continuous task using pytorch.

Results & Experiments:

'''


def sac():
    # Environment setup:
    env = gym.make('MountainCarContinuous-v0')
    action_space = env.action_space.shape[0]
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    observation_space = env.observation_space.shape[0]

    # Hyperparams
    epochs = 4000
    max_steps_per_episode = 400
    random_actions_till = 100000
    update_every = 50
    update_after = 125000
    batch_size = 200
    buffer_size = 100000
    polyak = 0.995
    discount_factor = 0.1 # Favouring immediate reward for this experiment
    q_lr = 0.05
    p_lr = 0.05
    no_of_updates = 5
    test_epochs = 1
    test_steps = max_steps_per_episode
    test_after = int(epochs*0.95)
    entropy_constant = 0.5

    # Network Dims
    PolicyNetworkDims = [observation_space,20,20,action_space]
    QNetworkDims = [observation_space+action_space,20,20,1]

    agent = SAC_Agent(PolicyNetworkDims,QNetworkDims,buffer_size,polyak,discount_factor,q_lr,p_lr,entropy_constant)

    total_steps = 0
    for i in range(epochs):
        observation = env.reset()
        done = False
        j = 0
        print("Training epoch {}".format(i))
        while (not done) and j < max_steps_per_episode:
            if total_steps > random_actions_till:
                action,_ = agent.take_action(observation)
            else:
                action = torch.FloatTensor(env.action_space.sample())
            new_observation,reward,done,_ = env.step(action)
            agent.ReplayBuffer(observation,action,reward,new_observation,done)
            observation = new_observation

            if total_steps > update_after and total_steps% update_every==0:
                for k in range(no_of_updates):
                    batch = agent.ReplayBuffer.sample(batch_size)
                    agent.updateQ(batch)
                    agent.updateP(batch)
                    agent.updateNetworks()
            j += 1
            total_steps += 1
        env.close()

        if i > test_after:
            for i_ in range(test_epochs):
                obs_ = env.reset()
                done = False
                j_ = 0
                while (not done) and j_<test_steps:
                    env.render()
                    action = agent.take_action(obs_,deterministic=True)
                    obs_,_,done,_ = env.step(action)
                    j += 1
                env.close()


if __name__=="__main__":
    sac()
