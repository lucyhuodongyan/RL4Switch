#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:18:20 2020

@author: huodongyan
"""

#%%
import ray  # package for distributed computations
ray.shutdown()

import switchNetwork as sn
import switchTrain as train

import argparse


# ray.shutdown()
ray.init()

#%%

parser = argparse.ArgumentParser(description=('Train policy for a switch network '
                                              'using Proximal Policy Optimizer'))
parser.add_argument('-I', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                    default = 100)
parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                    default = 0.998)
parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                    default = 0.99)
parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                    default = 0.05)
parser.add_argument('-Q', '--number_of_actors', type=int, help='Number of episodes per training batch',
                    default = 50)
parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                    default = 10)
parser.add_argument('-N', '--episode_length', type=int, help='Number of time-steps per an episode',
                    default = 10**4)
parser.add_argument('-L', '--burn_in', type=int, help='burn-in length',
                    default = 10**2)
parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                    default = 0.2)
parser.add_argument('-E', '--episode_length_evaluation', type=int, help='Number of arrivals in an episode used for'
                                                                        ' policy evaluation after the algorithm termination',
                    default = 10**6  )
parser.add_argument('-T', '--target_iter', type=int, help='Number of target iterations per policy update',
                    default = 1 )
parser.add_argument('-p', '--policy_model', type=bool, help='Whether to incorporate policy information when compuating Q-target',
                    default = False )
parser.add_argument('-q', '--q_model', type=bool, help='Whether to incorporate model information when compuating Q-target',
                    default = False )
parser.add_argument('-im', '--policy_import', type=bool, help='Whether to import previously trained policy weights',
                    default = False )

# if incorporate policy info, then Expected SARSA
# if further incorporate model info, then one more expectatio  than Expected SARSA

args, unknown = parser.parse_known_args()


#%%
rho_vector = [0.9]
N_vector = [5]
ti_vector = [1]

pi_vector = [True]
# above corresponds to SARSA

for N in N_vector:
    for r in rho_vector:
        network = sn.SwitchNetwork(N,r,"skewed_bot") # queuing network declaration
        print(network.network_name)
        network_id = ray.put(network)
        
        for ti in ti_vector:
            args.target_iter = ti
            
            for pi in range(len(pi_vector)):
                args.policy_import = pi_vector[pi]
                train.main(network_id,  **vars(args))


