#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 23:12:52 2022

@author: huodongyan
"""


import ray  # package for distributed computations (v1.3 for pytorch env)
import numpy as np
import torch

import os
import random
import datetime
import copy
#import matplotlib.pyplot as plt

from policy_testNN import Policy
from value_testNN import NNValueFunction
from utils import Logger, Scaler


MAX_ACTORS = 50 # max number of parallel simulations

def run_weights(network_id, num_policy_iterations, policy, scaler, logger, time_steps, pol_formulation):
    #policy evaluation
    """
    :param network_id: switch network structure and first-order info
    :param policy: switch network policy
    :param scaler: normalization values
    :param num_policy_iterations: given the PI rounds, evaluate every 10th iteration
    :param time_steps: max time steps in an episode
    :return: long-run average cost, CIs
    """


    ######## simulation #######
    iter_eval = np.hstack([np.array(range(0,num_policy_iterations,10))+1,num_policy_iterations])
    episodes = len(iter_eval) # currently assume the episodes here < MAX_ACTORS
    
    # initial state selection
    if scaler.initial_states is None:
        initial_state_0 = np.zeros((ray.get(network_id).buffers_num, episodes))
    else:
        initial_state_0 = scaler.sample_initial(episodes)
        

    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(ray.get(network_id).N, policy.kl_targ, hid1_mult = policy.hid1_mult, alt_formulation =pol_formulation) for _ in range(episodes)]
    res = []
    ray.get([s.set_weights.remote(torch.load(logger.path_weights+'/weights_'+str(iter_eval[i])+'.pth')) for i, s in enumerate(simulators)])

    scaler_id = ray.put(scaler)
    
    res.extend(ray.get([simulators[i].policy_performance.remote(network_id, scaler_id, time_steps, initial_state_0[i], i)
                      for i in range(episodes)]))

    print('simulation is done')
    #########################

    average_cost_set = np.zeros(episodes)
    ci_set = np.zeros(episodes)

    for i in range(episodes):

        average_cost_set[res[i][1]] = res[i][0]
        ci_set[res[i][1]] = res[i][2]

    print('Average cost: ', average_cost_set)
    print('CI: ', ci_set)

    return average_cost_set, ci_set


def run_policy(network_info, policy, simulators, scaler, logger, gamma, policy_iter_num, episodes, time_steps):
    """
    Run given policy and collect data
    :param network_id: switch network structure and first-order info
    :param policy: queuing network policy
    :param scaler: normalization values
    :param logger: metadata accumulator
    :param gamma: discount factor
    :param policy_iter_num: policy iteration, only used for printing purpose
    :param episodes: number of parallel simulations (episodes)
    :param time_steps: max time steps in an episode
    :return: trajectories = (states, actions, rewards)
    """

    total_steps = 0

    #### declare actors for distributed simulations of a current policy#####
    
    actors_per_run = episodes // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = episodes - actors_per_run * MAX_ACTORS
    weights = policy.get_weights()  # get neural network parameters
    for s in simulators:
        s.set_weights.remote(weights) # assign the neural network weights to all actors
    ######################################################

    ######### save neural network parameters to file ###########
    file_weights = os.path.join(logger.path_weights, 'weights_'+str(policy_iter_num)+'.pth')
    torch.save(weights, file_weights)
    ##################

    scaler_id = ray.put(scaler)
    scale, offset = scaler.get()
    
    initial_state = scaler.sample_initial(episodes)

    ######### policy simulation ########################
    accum_res = []  # results accumulator from all actors
    trajectories = []  # list of trajectories
    for j in range(actors_per_run):
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_info, scaler_id, time_steps, initial_state[j*MAX_ACTORS + i]) for i in range(MAX_ACTORS)]))
        
    if remainder>0:
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_info, scaler_id, time_steps, initial_state[actors_per_run * MAX_ACTORS + i]) for i in range(remainder)]))
    print('simulation is done')

    for i in range(len(accum_res)):
        trajectories.append(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps
        
    #################################################

    average_reward = np.mean([t['rewards'] for t in trajectories])
    
    next_initial = np.concatenate([t['unscaled_obs'][-100:] for t in trajectories])
    scaler.update_initial(next_initial)
            
    # step_reward_scale = np.repeat([2*np.arange(ray.get(network_info).N, 0, -1) - 1], time_steps, axis = 0)
    # adj_step_reward_scale = step_reward_scale.reshape(-1)

    #### normalization of the states in data ####################
    for t in trajectories:
                 
        z = t['rewards'] - average_reward #approximate r_x_star using long-run-average
        t['rewards'] = z
        
        z_step = t['rewards_by_step'] - average_reward / (ray.get(network_info).N) # average_reward * step_reward_scale / (ray.get(network_info).N **2)
        t['rewards_by_step'] = z_step
        
        z_step_reshape = t['adj_rewards_step'] - average_reward / (ray.get(network_info).N) # average_reward * adj_step_reward_scale / (ray.get(network_info).N **2)
        t['adj_rewards_step'] = z_step_reshape
    

    ########## results report ##########################
    print('Average cost: ',  -average_reward)

    logger.log({'_AverageReward': -average_reward,
                'Steps': total_steps,
    })
    
    avg_cost = -average_reward
    ####################################################
    return trajectories, avg_cost



def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array


@ray.remote
def add_disc_sum_rew_parallel(trajectory, policy, network, val_func, gamma, lam, scaler, iteration, burn_in):
    """
    compute value function for further training of Value Neural Network
    #modification for switch network: instead of compute expectation explicitly, use approximation
    
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """

    unscaled_obs = trajectory['adj_unscaled_obs']
    
    if iteration!=1:

        next_state_val = np.append(trajectory['adj_v_values'][1:], [trajectory['adj_v_values'][-1]], axis=0)
        v_values = trajectory['adj_v_values']
        
        
        #############################################################################################################
        
        # td-error computing
        tds_pi = trajectory['adj_rewards_step'][:,np.newaxis] - v_values + gamma * next_state_val #GAE

        # value function computing for futher neural network training
        disc_sum_rew = relative_af(unscaled_obs, td_pi=tds_pi, lam=gamma*lam) + v_values
        
    else:
        disc_sum_rew = relative_af(unscaled_obs, td_pi=trajectory['adj_rewards_step'], lam=gamma*lam)

    
    return disc_sum_rew


def compute_target(trajectories, policy, network, val_func, gamma, lam, scaler, iteration, burn_in):
    """
    compute value function for further training of Value Neural Network
    #modification for switch network: instead of compute expectation explicitly, use approximation
    
    :param trajectories: list structure, list of trajectories (simulated data)
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    actors_per_run = len(trajectories) // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = len(trajectories) - actors_per_run * MAX_ACTORS
    disc_sum_rew_output = []
    disc_sum_rew_output_list = []
    
    for j in range(actors_per_run):
        disc_sum_rew_output.extend([add_disc_sum_rew_parallel.remote(t, policy, network, val_func, gamma, lam, scaler, iteration, burn_in) for t in trajectories[(j * MAX_ACTORS):((j+1) * MAX_ACTORS)]])
    if remainder>0:
        disc_sum_rew_output.extend([add_disc_sum_rew_parallel.remote(t, policy, network, val_func, gamma, lam, scaler, iteration, burn_in) for t in trajectories[(actors_per_run * MAX_ACTORS): (actors_per_run * MAX_ACTORS + remainder)]])
        
    for i in range(len(disc_sum_rew_output)):
        disc_sum_rew_output_list.append(ray.get(disc_sum_rew_output[i]))
        trajectories[i]['adj_disc_sum_rew'] = ray.get(disc_sum_rew_output[i])
        
    disc_sum_rew = np.concatenate([t[:-burn_in] for t in disc_sum_rew_output_list])

    ######### normalization ########################
    if iteration == 1:
        scaler.update_val(disc_sum_rew)
    
    scale, offset = scaler.get()    
    
    state_observes = np.concatenate([t['adj_state_obs'][:-burn_in * network.N] for t in trajectories])
    port_observes = np.concatenate([t['adj_availability_mat'][:-burn_in * network.N] for t in trajectories])
    
    disc_sum_rew = np.concatenate([t['adj_disc_sum_rew'][:-burn_in * network.N] for t in trajectories])
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    ##################################################

    return state_observes, port_observes, disc_sum_rew_norm



def relative_af(unscaled_obs, td_pi,  lam):
    # return advantage function
    disc_array = np.copy(td_pi)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        
        # if np.sum(unscaled_obs[i+1]) != 0:
        #     sum_tds = td_pi[i+1] + lam * sum_tds
        # else:
        #     sum_tds = 0
        
        sum_tds = td_pi[i+1] + lam * sum_tds
        disc_array[i] += sum_tds

    return disc_array

@ray.remote
def add_value_parallel(trajectory, network, val_func, policy, scaler):
    
    values = val_func.predict(trajectory['adj_state_obs'], trajectory['adj_availability_mat']) #predicted val is normalized, need to convert to unnormalized val
    return values
    

def add_value(trajectories, network, val_func, policy, scaler, gamma, lam):
    
    actors_per_run = len(trajectories) // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = len(trajectories) - actors_per_run * MAX_ACTORS
    value_output = []
    
    scale, offset = scaler.get()
    
    for j in range(actors_per_run):
        value_output.extend([add_value_parallel.remote(t, network, val_func, policy, scaler) for t in trajectories[(j * MAX_ACTORS):((j+1) * MAX_ACTORS)]])
    if remainder>0:
        value_output.extend([add_value_parallel.remote(t, network, val_func, policy, scaler) for t in trajectories[(actors_per_run * MAX_ACTORS): (actors_per_run * MAX_ACTORS + remainder)]])
        
    for i in range(len(value_output)):
        trajectories[i]['adj_v_values'] = ray.get(value_output[i])/ scale[-1] + offset[-1]
        trajectories[i]['adj_advantages'] = trajectories[i]['adj_rewards_step'][:, np.newaxis] + gamma * np.append(trajectories[i]['adj_v_values'][1:], [trajectories[i]['adj_v_values'][-1]], axis=0) - trajectories[i]['adj_v_values']
        # trajectories[i]['adj_gae'] = relative_af(trajectories[i]['adj_unscaled_obs'], td_pi=trajectories[i]['adj_advantages'], lam=gamma*lam)



def build_train_set(trajectories, burn_in, network, value_func): #added network as an input
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :param policy: added in policy as a new input parameter for switch network, needed to approximate expectation
    :return: data for further Policy and Value neural networks training
    """

    disc_sum_rew = np.concatenate([t['adj_disc_sum_rew'][:-burn_in * network.N] for t in trajectories])

    state_obs = np.concatenate([t['adj_state_obs'][:-burn_in * network.N] for t in trajectories])
    port_avail = np.concatenate([t['adj_availability_mat'][:-burn_in * network.N] for t in trajectories])
    actions_glob = np.concatenate([t['adj_act_index'][:-burn_in * network.N] for t in trajectories])
    advantages = np.concatenate([t['adj_advantages'][:-burn_in * network.N] for t in trajectories])
    # advantages = np.concatenate([t['adj_gae'][:-burn_in * network.N] for t in trajectories])
    
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages
    disc_sum_rew = disc_sum_rew  / (disc_sum_rew.std() + 1e-6) # normalize advantages

    return state_obs, port_avail, actions_glob, disc_sum_rew, advantages


def log_batch_stats(observes, actions_glob, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions_glob),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })

def roll_back(previous_stable, logger, policy, val_func):
    
    pol_weights_path = os.path.join(logger.path_weights, 'weights_' + str(previous_stable) + '.pth')
    pol_weights = torch.load(pol_weights_path)
    policy.set_weights(pol_weights)
    
    val_weights_path = os.path.join(logger.path_weights, 'val_' + str(previous_stable) + '.pth')
    val_weights = torch.load(val_weights_path)
    val_func.set_weights(val_weights)

# TODO: check shadow name
def main(network_id, num_policy_iterations, gamma, lam, kl_targ, number_of_actors, hid1_mult, episode_length,
         burn_in, clipping_parameter, episode_length_evaluation, target_iter, val_formulation, pol_formulation, pol_import):
    """
    # Main training loop
    :param: see ArgumentParser below
    """
    N = ray.get(network_id).N
    obs_dim = ray.get(network_id).buffers_num
    

    ######### create directories to save the results and log-files############
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)
    ###################################

    scaler = Scaler(obs_dim) # normilizer initialization
    
    
    
    val_func = NNValueFunction(N, hid1_mult = hid1_mult, alt_formulation = val_formulation) # Value Neural Network initialization
    policy = Policy(N, kl_targ, hid1_mult = hid1_mult, alt_formulation = pol_formulation) # Policy Neural Network initialization
    
    #set up remote simulator (once and for all iterations)
    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(N, kl_targ, hid1_mult = hid1_mult, alt_formulation = pol_formulation ) for _ in range(MAX_ACTORS)]

    ############ initialize the algorithm with the proportionally randomized policy ###################
    
    if pol_import:
        cwd = os.getcwd()
        scale, offset = np.load(cwd + '/N='+ str(N) +'_initial'+ '/scaler.npy')
        scaler.vars=(1/scale-0.1)**2
    
        initial_weight_dir = cwd + '/N='+ str(N) +'_initial'+'/weights.pth'
        initial_weights = torch.load(initial_weight_dir)
        policy.set_weights(initial_weights)
        
        initial_states_set = np.zeros((1,obs_dim))
        trajectories, total_steps, action_distr = \
            policy.run_episode(ray.get(network_id), scaler, episode_length * 10, initial_states_set[0])
    
        unscaled_obs = trajectories['unscaled_obs']
        scaler.update_initial(unscaled_obs[-100:])
    
    else:
        
        initial_states_set = np.zeros(obs_dim)
        trajectories, total_steps, action_distr = \
            policy.run_episode(ray.get(network_id), scaler, episode_length * 100, initial_states_set, rpp=True)
    
        unscaled_obs = trajectories['unscaled_obs']
        scaler.update_initial(unscaled_obs[-100:])
    
        adj_unscaled_obs = trajectories['adj_unscaled_obs']
        scaler.update_state(adj_unscaled_obs)
        
        scale, offset = scaler.get()
        adj_observes = (adj_unscaled_obs - offset[:-1]) * scale[:-1]
        
        adj_availability_mat = trajectories['adj_availability_mat']
        action_prob_reshape = np.reshape(action_distr, (-1, obs_dim))
        policy.initialize_rpp(adj_observes, adj_availability_mat, action_prob_reshape)
        init_kl = policy.compute_kl(adj_observes, adj_availability_mat, action_prob_reshape)
        print("initial kl:  ", init_kl)
    
    ma_cost = np.repeat(10**5, 5) # initially *2, just to avoid cut the alpha rate in the first round
    previous_cost = np.mean(ma_cost)
    #####################################################

    iteration = 0  # count of policy iterations

    alpha = 1
    benchmark =0.01
    previous_stable = 0
    
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        # alpha = 1. - (iteration-1) / num_policy_iterations
        policy.clipping_range = max(0.005, alpha*clipping_parameter)
        policy.lr_multiplier = max(0.005/clipping_parameter, alpha)

        print('Clipping range is ', policy.clipping_range)

            
        simulation_start = datetime.datetime.now()

        trajectories, avg_cost = run_policy(network_id, policy, simulators, scaler, logger, gamma, iteration, 
                                            episodes=number_of_actors, time_steps=episode_length) #simulation
        
        if avg_cost > (1.1 * ma_cost[-1]):
            roll_back(int(previous_stable), logger, policy, val_func)
            print("Current iteration diverged. Roll back to previous iteration.")
            continue
            
        
        previous_stable = iteration 
        
        ma_cost = np.append(ma_cost[1:],avg_cost)
        current_cost = avg_cost
        print('simulation time: {:.3f}...'.format(int(((datetime.datetime.now() - simulation_start).
                                                            total_seconds() / 60) * 100) / 100.), 'minutes')


        
        if np.abs(current_cost-previous_cost)<benchmark:
            alpha = alpha * 0.5
            ma_cost = np.repeat(current_cost * 2,5)
            # benchmark = max(0.0001, benchmark * 0.5)

        previous_cost = np.mean(ma_cost)
        
        ti_iter = 0
        while ti_iter<target_iter:
            
            add_value(trajectories, ray.get(network_id), val_func, policy, scaler, gamma, lam)
            target_computation_start = datetime.datetime.now()
            state_observes, port_observes, disc_sum_rew_norm = compute_target(trajectories, policy, ray.get(network_id), val_func, gamma, lam, scaler,iteration, burn_in)  # calculate values from data
            print('target computation time: {:.3f}...'.format(int(((datetime.datetime.now() - target_computation_start).
                                                            total_seconds() / 60) * 100) / 100.), 'minutes')
            
            if len(np.shape(disc_sum_rew_norm))==1:
                disc_sum_rew_norm = disc_sum_rew_norm[:, np.newaxis]
            
            val_func.fit(state_observes, port_observes,disc_sum_rew_norm, logger)  # update value function
            
            ti_iter = ti_iter+1
        
        if iteration == 1:
            file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
            scale, offset = scaler.get()
            np.save(file_scaler, np.asarray([scale, offset]))
        
        value_add_start = datetime.datetime.now()
        add_value(trajectories, ray.get(network_id), val_func, policy, scaler, gamma, lam)
        print('add value time: {:.3f}...'.format(int(((datetime.datetime.now() - value_add_start).
                                                            total_seconds() / 60) * 100) / 100.), 'minutes')
        
        val_weights = val_func.get_weights()
        file_val = os.path.join(logger.path_weights, 'val_' + str(iteration) + '.pth')
        torch.save(val_weights, file_val)
    
        build_start = datetime.datetime.now()
        state_obs, port_avail, actions_glob, disc_sum_rew, advantages = build_train_set(trajectories, burn_in, ray.get(network_id), val_func)
        print('build time: {:.3f}...'.format(int(((datetime.datetime.now() - build_start).
                                                            total_seconds() / 60) * 100) / 100.), 'minutes')
        
        log_batch_stats(state_obs, actions_glob, disc_sum_rew, logger, iteration)  # add various stats
        policy.update( state_obs, port_avail, actions_glob, np.squeeze(advantages), logger)  # update policy


        logger.write(display=True)  # write logger results to file


    ############## save weights of the policy NN and normilizer  #####################
    
    weights = policy.get_weights()

    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.pth')
    torch.save(weights, file_weights)

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    
    ###################################################################################
    

    ########## performance estimation of the saved policies #########################################
    #  time_steps -- is the total length (all batches) used to evaluate performance of a policy

    performance_evolution_all, ci_all = run_weights(network_id, num_policy_iterations, policy, scaler, logger,  episode_length_evaluation, pol_formulation)


    file_res = os.path.join(logger.path_weights, 'average_' + str(performance_evolution_all[-1]) + '+-' +str(ci_all[-1]) + '.txt')
    file = open(file_res, "w")
    for i in range(len(ci_all)):
        file.write(str(performance_evolution_all[i])+'\n')
    file.write('\n')
    for i in range(len(ci_all)):
        file.write(str(ci_all[i])+'\n')


    logger.close()

