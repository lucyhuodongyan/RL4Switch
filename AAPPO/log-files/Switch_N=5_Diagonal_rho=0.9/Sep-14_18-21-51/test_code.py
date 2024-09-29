#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 23:23:34 2022

@author: huodongyan
"""

import ray

ray.shutdown()
ray.init()

from network_testNN import SwitchNetwork


import train_testNN_parallel as train

N = 5
rho = 0.9
arrival = "diagonal"
sn = SwitchNetwork(N, rho, arrival)
network_id = ray.put(sn)

kl_targ = 0.05
time_steps = 1 * 10 ** 4
gamma = 0.998
lam = 0.99
hid1_mult = 10
burn_in = 10 ** 2
clipping_parameter = 0.2

train.main(network_id, 100, gamma, lam, kl_targ, 50, hid1_mult, time_steps, burn_in, clipping_parameter, time_steps * 10, 1,
           False, True, False)
