import numpy as np
import math
from env.SISO import SISO_Channel, sum_rate



def WMMSE_sum_rate(p_int, H, Pmax=1, var_noise=1):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-5:
            break
            

    p_opt = np.square(b) / Pmax
    return p_opt

def baseline():
    K=6
    env = SISO_Channel(K)
    # env = gym.make('CartPole-v0')
    # baseline
    env.reset()
    channel = env.CSI
    channel = np.reshape(channel, (env.pair_num, env.pair_num))
    p_init = np.ones(env.pair_num)
    baseline_p = WMMSE_sum_rate(p_init, channel)
    channel = np.squeeze(env.CSI)
    state = np.concatenate((channel, baseline_p), axis=0)
    state = np.append(state, 0)
    baseline_capacity = sum_rate(env.pair_num, state)
    print("K={}; WMMES result:{}".format(K, baseline_capacity))

baseline()
