"""
env for VPG_V2
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
        # total bit of the state info: CSI + power control of each Tx 
        num_state = K**2 + K 
        self.observation_space = Box(low=np.array([0]*num_state), high=np.array([1]*num_state))
        # state space = Channel state information + Power control ratio
        self.CSI = abs(1/np.sqrt(2)*(np.random.randn(1,K**2)+1j*np.random.randn(1,K**2)))
        self.power_ratio = np.random.rand(1,K)
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, 0)
        # the time of a single epoch
        self.time = 100

    def check_channel(self, states):
        power_list = states[self.pair_num**2:-1]
        for i, power in enumerate(power_list):
            if power > 1:
                done = True
                reward =  -power
                break
            elif power < 0:
                done =True
                reward = power - 1
                break
            elif i == len(power_list) - 1:
                done = False
                reward = sum_rate(self.pair_num, self.state)
        return done, reward
    
    def step(self, action:list):
        """
        Only update one Tx per unit time
        Tx_No: which transmiter to be optimized
        action: increase / decrease the power
        """
        assert len(action) == self.pair_num, "check your action size"
        # the postion of the optimized Tx in the state info
        self.time -= 1
        assert len(self.state[self.pair_num**2:-1]) == self.pair_num, "check size"
        for i in range(self.pair_num):
            self.state[self.pair_num**2+i] += action[i]

        # check if terminate in terms of channel state
        done, reward = self.check_channel(self.state)

        # check if terminate in terms of time
        if self.time == 99:
            done = False
        elif self.time < 0:
            done = True

        info = {}
        return self.state, reward, done, info
    
    def reset(self):
        # k: the position index of Tx
        np.random.seed(self.seed)
        # np.random.seed(epoch)
        self.CSI = abs(1/np.sqrt(2)*(np.random.randn(1,self.pair_num**2)+1j*np.random.randn(1,self.pair_num**2)))
        # self.power_ratio = np.random.rand(1,self.pair_num)
        self.power_ratio = np.array([0.5]*self.pair_num) # fixed initial power ratio
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, 0)
        self.time = 100
        return self.state
    
    def render(self):
        pass



        
        