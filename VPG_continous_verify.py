"""
Use OpenAI gym - Mountaincair to test
the functionality of the algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
# from SISO_env import SISO_Channel
from env.SISO import SISO_Channel
import gym
import wandb

env = gym.make('MountainCarContinuous-v0')
wandb.init("VPG_verify")

# env = SISO_Channel(K=10)
# env = gym.make('CartPole-v0')

class Actor(nn.Module):
    def __init__(self, input=2, hidden=128, output=2) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden,1)
        self.sigma_head = nn.Linear(hidden,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, input=2, hidden=64, output=1) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)
    
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def choose_action(state) -> list:
    state = torch.tensor(state, dtype=torch.float32)
    mu, sigma = actor(state)
    distribution = Normal(mu,sigma)
    # distributions = Categorical(actor(state))
    action = distribution.sample()
    return action

def compute_advantage(rewards, state_values):
    r2g = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        r2g[i] = rewards[i] + (r2g[i+1] if i + 1 < len(rewards) else 0)
    
    advantages = r2g - np.array(state_values)

    r2g = torch.as_tensor(r2g, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    return advantages, r2g

EPISODE_NUM = 1000000
GAMMA = 0.98
LEARNING_RATE = 0.001

actor = Actor()
critic = Critic()
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
critic_loss_func = nn.MSELoss(reduce='sum')

for i in range(EPISODE_NUM):

    # for k in range(env.pair_num):
    states = []
    rewards = []
    actions = []
    next_states = []
    states_values = []
    state = env.reset()
    while True:
        action  = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        states_values.append(critic(torch.as_tensor(state,dtype=torch.float32)).detach().item())

        state = next_state
        if done:
            break
    
    # optimize actor
    states = torch.tensor(states,dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)


    actor_optimizer.zero_grad()
    advantages = compute_advantage(rewards, states_values)[0]

    mus, sigmas = actor(states)
    mu, sigma = actor(states[0])
    distributions = Normal(mus,sigmas)
    distribution = Normal(mu, sigma)
    # distributions = Categorical(actor(states))
    actions = torch.unsqueeze(actions, 1)
    log_probs = distributions.log_prob(actions)
    log_prob = distribution.log_prob(actions[0])

    advantages = torch.unsqueeze(advantages, 1)
    actor_loss =  - torch.sum(torch.mul(log_probs, advantages))

    actor_loss.backward()
    actor_optimizer.step()

    # opyimize critic
    critic_optimizer.zero_grad()
    rtg = compute_advantage(rewards, states_values)[1]
    estimate_v = critic(states).squeeze()
    critic_loss = critic_loss_func(rtg, estimate_v)
    critic_loss.backward()
    critic_optimizer.step()


    result = np.sum(rewards)
    print("episode: {};  average_reward:{}".format(i, result))
    wandb.log({"average_reward": result})

    







