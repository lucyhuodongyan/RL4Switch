#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:13:38 2022

@author: huodongyan
"""
import os
import datetime

import numpy as np
import torch


from network_testNN import SwitchNetwork
from policy_testNN import Policy
from utils import Scaler


N = 8
rho = 0.9
arrival = "diagonal"
sn = SwitchNetwork(N, rho, arrival)


kl_targ = 0.05
hid1_mult = 10
policy = Policy(N, kl_targ, hid1_mult = hid1_mult, alt_formulation = True) # Policy Neural Network initialization

cwd = os.getcwd()
initial_weights = torch.load(cwd + '/weights/weights_80.pth')
policy.set_weights(initial_weights)
    
    
obs_dim = sn.buffers_num
scaler = Scaler(obs_dim) # normilizer initialization
scale, offset = np.load(cwd + '/weights/scaler_1.npy')
scaler.vars=(1/scale-0.1)**2
    
episode_length = 10**6
start_time = datetime.datetime.now()
trajectories, total_steps, action_distr = policy.run_episode(sn, scaler, episode_length, np.zeros(obs_dim))
end_time = datetime.datetime.now()
print('simulation time: {:.3f}...'.format(int(((end_time - start_time).total_seconds() / 60) * 100) / 100.), 'minutes')

states = np.sum(trajectories['unscaled_obs'], axis = 1)

batch_num = 50
states_batch = np.array_split(states, batch_num)
mean_lst = [np.mean(states_batch[i]) for i in range(batch_num)]
avg_rwd = np.mean(mean_lst)
ci = 1.96 * np.std(mean_lst) / np.sqrt(batch_num)
print("mean: ", avg_rwd)
print("ci width: ", ci)
