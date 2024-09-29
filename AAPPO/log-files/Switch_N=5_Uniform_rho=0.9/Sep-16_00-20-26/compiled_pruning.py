#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn.utils.prune as prune


# In[2]:


import ray
ray.shutdown()
ray.init()


# In[3]:


MAX_ACTORS = 50


# In[4]:


from network_testNN import SwitchNetwork
from policy_testNN import Policy
from utils import Scaler


# In[5]:


N_vec = [3, 4, 5, 6, 7, 8]
arrival_vec = ["Uniform", "Diagonal", "Skewed_bot"]
rho = 0.9


# In[6]:


path_dict = {}
path_dict[(3, "Uniform")] = 'Jul-26_02-46-24'
path_dict[(3, "Diagonal")] = 'Jul-28_14-14-48'
path_dict[(3, "Skewed_bot")] = 'Jul-28_19-05-57'

path_dict[(4, "Uniform")] = 'Jul-26_12-59-35'
path_dict[(4, "Diagonal")] = 'Jul-28_13-59-44'
path_dict[(4, "Skewed_bot")] = 'Jul-29_01-22-36'

path_dict[(5, "Uniform")] = 'Jul-26_02-20-47'
path_dict[(5, "Diagonal")] = 'Sep-14_18-21-51'
path_dict[(5, "Skewed_bot")] = 'Jul-29_01-23-11'

path_dict[(6, "Uniform")] = 'Jul-26_13-52-09'
path_dict[(6, "Diagonal")] = 'Aug-01_07-14-45'
path_dict[(6, "Skewed_bot")] = 'Jul-30_03-58-27'

path_dict[(7, "Uniform")] = 'Jul-26_21-12-44'
path_dict[(7, "Diagonal")] = 'Aug-03_18-04-13'
path_dict[(7, "Skewed_bot")] = 'Jul-29_11-06-31'

path_dict[(8, "Uniform")] = 'Jul-27_01-49-50'
path_dict[(8, "Diagonal")] = 'Aug-02_11-48-52'
path_dict[(8, "Skewed_bot")] = 'Jul-29_01-24-08'


# In[7]:


iter_dict = {}
iter_dict[(3, "Uniform")] = 100
iter_dict[(3, "Diagonal")] = 100
iter_dict[(3, "Skewed_bot")] = 100

iter_dict[(4, "Uniform")] = 100
iter_dict[(4, "Diagonal")] = 100
iter_dict[(4, "Skewed_bot")] = 100

iter_dict[(5, "Uniform")] = 100
iter_dict[(5, "Diagonal")] = 100
iter_dict[(5, "Skewed_bot")] = 100

iter_dict[(6, "Uniform")] = 100
iter_dict[(6, "Diagonal")] = 200
iter_dict[(6, "Skewed_bot")] = 200

iter_dict[(7, "Uniform")] = 100
iter_dict[(7, "Diagonal")] = 100
iter_dict[(7, "Skewed_bot")] = 100

iter_dict[(8, "Uniform")] = 200
iter_dict[(8, "Diagonal")] = 200
iter_dict[(8, "Skewed_bot")] = 100


# In[8]:


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


# In[9]:


@ray.remote
def prune_pol(N, arrival, path_dict, iter_dict, time_steps, kl_targ, hid1_mult, pol_formulation):
    
    if (N, arrival) not in path_dict:
        return
        
    network = SwitchNetwork(N, rho, arrival)
    # print(network.network_name)
    
    output_dict = {}

    folder = path_dict[(N, arrival)]
    count = str(iter_dict[(N, arrival)])
    path = 'log-files/' +  network.network_name + '/'+ folder + '/weights/'

    policy_path = path + 'weights_' + count + '.pth'
    scaler_path = path + 'scaler_' + count +'.npy'

    obs_dim = N**2
    scaler = Scaler(obs_dim)
    scale, offset = np.load(scaler_path)
    scaler.vars=(1/scale-0.1)**2

    policy = Policy(N, kl_targ, hid1_mult = hid1_mult, alt_formulation = pol_formulation) 
    weights = torch.load(policy_path)
    policy.set_weights(weights)

    testNN = policy.policynetwork
    layer_one = testNN.fc1

    average_performance, id, ci = policy.policy_performance(network, scaler, time_steps, np.zeros(N**2), count)
    num_zeros, num_elements, sparsity = measure_module_sparsity(layer_one, use_mask=False)
    output_dict[(network, sparsity)] = [average_performance, ci]

    while sparsity < 0.9:

        prune.l1_unstructured(layer_one, name="weight", amount= N**4)
        average_performance, id, ci = policy.policy_performance(network, scaler, time_steps, np.zeros(N**2), count)
        num_zeros, num_elements, sparsity = measure_module_sparsity(layer_one, use_mask=True)
        output_dict[(network, sparsity)] = [average_performance, ci]
        
    return output_dict


# In[12]:


def run_pruning(N_vec, arrival_vec, rho, time_steps):
    
    kl_targ = 0.05
    hid1_mult = 10
    pol_formulation = True

    N_vec_repeat = np.repeat(N_vec, len(arrival_vec))
    arrival_vec_repeat = np.tile(arrival_vec, len(N_vec))
    
    actors_per_run = len(N_vec_repeat) // MAX_ACTORS
    remainder = len(N_vec_repeat) - actors_per_run * MAX_ACTORS

    dict_output_vec = []

    simulation_output_vec = []
    for i in range(actors_per_run):
        simulation_output_vec.extend([prune_pol.remote(N_vec_repeat[i * MAX_ACTORS + j], arrival_vec_repeat[i * MAX_ACTORS + j], 
                                                       path_dict, iter_dict, time_steps, kl_targ, hid1_mult, pol_formulation) for j in range(MAX_ACTORS)])

    if remainder > 0:
        simulation_output_vec.extend([prune_pol.remote(N_vec_repeat[actors_per_run * MAX_ACTORS + j], arrival_vec_repeat[actors_per_run * MAX_ACTORS + j], 
                                                       path_dict, iter_dict, time_steps, kl_targ, hid1_mult, pol_formulation) for j in range(remainder)])

    for i in range(len(simulation_output_vec)):

        simulation_output = ray.get(simulation_output_vec[i])
        dict_output_vec.append(simulation_output)

    return dict_output_vec


# In[13]:


dict_output_vec = run_pruning(N_vec, arrival_vec, rho, 10**6)


# In[14]:


output_table = pd.DataFrame(columns = ['N', 'rho', 'Arrival', 'Sparsity', 'Mean_Est', 'CI_Width'])


for dict_output in dict_output_vec:

    for key in dict_output:
        
        N = key[0].N
        rho = key[0].rho
        arrival = key[0].arrival_type
        sparsity = key[1]
        
        mean_est = dict_output[key][0]
        ci_width = dict_output[key][1]

        i = output_table.shape[0]
        output_table.loc[i] = [N, rho, arrival, sparsity, mean_est, ci_width]


output_table.to_csv('Pruning Result Compilation.csv')

