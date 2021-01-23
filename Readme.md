### Deep RL algorithms implemented using Pytorch

#### Algo list:
1. [DQN](https://github.com/akashe/DeepReinforcementLearning/blob/main/DQN.py) 
2. [Vanilla policy Gradient](https://github.com/akashe/DeepReinforcementLearning/blob/main/vanilla_policy_gradient.py)
3. [Deep Deterministic Policy Gradient](https://github.com/akashe/DeepReinforcementLearning/blob/main/ddpg.py)
4. [Twin Delayed Deep Deterministic Policy Gradient](https://github.com/akashe/DeepReinforcementLearning/blob/main/td3.py)
5. [Soft Actor Critic](https://github.com/akashe/DeepReinforcementLearning/blob/main/SoftActorCritic.py)
6. [Proximal Policy Optimization - CLIP](https://github.com/akashe/DeepReinforcementLearning/blob/main/ppo_clip.py)

###### Article on deeper Look into [policy gradients](https://akashe.io/blog/2020/10/14/policy-gradient-methods/) 

#### Experimental Results:

|Algorithm| Discrete Env: LunarLander-v2 | Continuous Env: Pendulum-v0 |
| :---: | :---: | :---: |
| DQN | ![LunnarLander-DQN](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/DQN_Lunar_lander_rewards.png) | - |
| VPG | ![LunarLander-VPG](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/VPG_LunarLander-v2_rewards.png) | - |
| DDPG | - | ![Pendulum-DDPG](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/DDPG_Pendulum-v0_rewards.png)| 
| TD3 | - | ![Pendulum-TD3](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/TD3_Pendulum_rewards.png) |
| SAC | - | ![Pendulum-SAC](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/SAC_Pendulum-v0_rewards.png) |
| PPO | - | ![Pendulum-PPO](https://raw.githubusercontent.com/akashe/DeepReinforcementLearning/main/figures/PPO_Pendulum-v0_rewards.png) |

#### Usage:
Just run the file/algorithm directly. There is no common structures between algorithms as I implemented them as I learnt them. 
Different algorithms are inspired from different sources.

#### Resources:
1. [RL course by David Silver](https://www.youtube.com/watch?v=KHZVXao4qXs&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=7)
2. [Lecture slides for above course](https://www.davidsilver.uk/teaching/)
3. [Spinning up by OpenAI](https://spinningup.openai.com)
3. [More exhaustive RL guide by Deeny Britz](https://github.com/dennybritz/reinforcement-learning)

#### Future projects:
1. If time available I will add a simple program for elevator using RL.
2. Better graphs