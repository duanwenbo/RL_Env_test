"""
# One Sub-action each iteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from env.SISO_env_V1 import SISO_Channel
import wandb


wandb.init(project="03-14V2")


env = SISO_Channel(K=4)
# env = gym.make('CartPole-v0')

class Actor(nn.Module):
    def __init__(self, input=len(env.state), hidden=128, output=2) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden,1)
        self.sigma_head = nn.Linear(hidden,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, input=len(env.state), hidden=64, output=1) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)
    
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Actor(nn.Module):
#     def __init__(self, input=4, hidden=64, output=2) -> None:
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(input, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, output)
    
#     def forward(self, input):
#         x = F.relu(self.fc1(input))
#         x = F.relu(self.fc2(x))
#         x = F.softmax(self.fc3(x))
#         return x

# class Critic(nn.Module):
#     def __init__(self, input=4, hidden=64, output=1) -> None:
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(input, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, output)
    
#     def forward(self, input):
#         x = F.relu(self.fc1(input))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


def choose_action(state):
    state = torch.as_tensor(state, dtype=torch.float32)
    # perdict a normal distribution
    mu, sigma = actor(state)
    distribution = Normal(mu,sigma)
    action = distribution.sample()
    return action.item()

def compute_advantage(rewards, state_values):
    r2g = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        r2g[i] = rewards[i] + (r2g[i+1] if i + 1 < len(rewards) else 0)
    
    advantages = r2g - np.array(state_values)

    r2g = torch.as_tensor(r2g, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    return advantages, r2g

EPISODE_NUM = 80000
GAMMA = 0.98
LEARNING_RATE = 0.00001

actor = Actor()
critic = Critic()
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
critic_loss_func = nn.MSELoss(reduce='sum')


def train():
    for i in range(EPISODE_NUM):
        result = []
        for k in range(env.pair_num):
            states = []
            rewards = []
            actions = []
            next_states = []
            states_values = []
            # k: the index of the Transmitter(Tx)
            # i: the index of the episode
            state = env.reset(k,i)
            while True:
                action = choose_action(state)
                next_state, reward, done, _ = env.step(Tx_No=k, action=action)
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
            mus, sigmas = actor(states)
            distributions = Normal(mus,sigmas)
            # distributions = Categorical(actor(states))
            actions = torch.unsqueeze(actions, 1)
            log_probs = distributions.log_prob(actions)
            advantages = compute_advantage(rewards, states_values)[0]
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

            sub_result = np.sum(rewards) / len(rewards)
            result.append(sub_result)

        result = np.sum(result)/env.pair_num
        wandb.log({"reward":result})
        print(result, i)

    # 状态空间要加上Tx位置!!!

if __name__ == "__main__":
    train()




    







