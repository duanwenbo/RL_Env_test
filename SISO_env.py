"""
env for VPG_V1
"""

from gym import Env
from gym.spaces import Box
import numpy as np

def sum_rate(pair_num, state_info, Pmax=1, var_noise=1):
    # first version, without SSL
    # suppose all var are one-dimentional
    state_info = state_info[:-1]
    power = state_info[pair_num**2:]
    assert len(power) == pair_num, "check check length"
    H = state_info[:pair_num**2].reshape(pair_num, pair_num)
    rate = 0 
    for i in range(pair_num):
        S = power[i] * Pmax * (H[i, i] ** 2)
        I = var_noise + np.sum(power * Pmax * (H[i, :] ** 2)) - S
        rate += np.log2(1 + S / I)
    return rate

class SISO_Channel(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, K, seed = 100) -> None:
        self.seed = seed
        np.random.seed(self.seed)
        # N : number of transmitter-receiver pairs
        super(SISO_Channel, self).__init__()
        self.pair_num = K
        # power control ratio [0,1]
        self.action_space = Box(low=np.array([-1]), high=np.array([1]))
        # total bit of the state info: CSI + power control of each Tx + the position index of Tx
        num_state = K**2 + K + 1 
        self.observation_space = Box(low=np.array([0]*num_state), high=np.array([1]*num_state))
        # state space = Channel state information + Power control ratio
        self.CSI = abs(1/np.sqrt(2)*(np.random.randn(1,K**2)+1j*np.random.randn(1,K**2)))
        self.power_ratio = np.random.rand(1,K)
        self.Tx_position = np.array([[1.]])
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, self.Tx_position)
        # the time of a single epoch
        self.time = 200
    
    def step(self, Tx_No, action):
        """
        Only update one Tx per unit time
        Tx_No: which transmiter to be optimized
        action: increase / decrease the power
        """
        # the postion of the optimized Tx in the state info
        Tx_index = self.pair_num**2 + Tx_No
        self.time -= 1
        self.state[Tx_index] += action

        if self.state[Tx_index] > 1:
            done = True
            reward = - self.state[Tx_index]
        elif self.state[Tx_index] < 0:
            done = True
            reward = self.state[Tx_index] - 1
        elif self.time < 0:
            done = True
            reward = sum_rate(self.pair_num, self.state)
        else:
            reward = sum_rate(self.pair_num, self.state)
            done = False

        # to solve the prolem when the trajectory is only 1 step length.
        # We assume one more step will not affect the result as the rewards have been adjusted
        if self.time == 99:
            done = False

        info = {}
        return self.state, reward, done, info
    
    def reset(self, k, epoch):
        # k: the position index of Tx
        # np.random.seed(self.seed)
        np.random.seed(208)
        self.CSI = abs(1/np.sqrt(2)*(np.random.randn(1,self.pair_num**2)+1j*np.random.randn(1,self.pair_num**2)))
        # self.power_ratio = np.random.rand(1,self.pair_num)
        self.power_ratio = np.array([0.5]*self.pair_num) # fixed initial power ratio
        self.Tx_position = np.array([k])
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, self.Tx_position)
        self.time = 100
        return self.state
    
    def render(self):
        pass

if __name__ == "__main__":
    for _ in range(3):
        env = SISO_Channel(K=3)
        env.reset()
        print(env.state)
        print("/n/n")

        


        
        