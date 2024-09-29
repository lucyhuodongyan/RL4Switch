import ray  # package for distributed computations (v1.3 for pytorch env)
import numpy as np
import torch

import os
import random
import datetime
import copy
#import matplotlib.pyplot as plt

from switchNetwork import SwitchNetwork
from switchPolicy import Policy
from switchValue import NNValueFunction
from switchUtils import Logger, Scaler




#MAX_ACTORS = 25  # max number of parallel simulations
MAX_ACTORS = 50

def run_weights(network_id, num_policy_iterations, policy, scaler, logger, time_steps):
    #policy evaluation
    """
    :param network_id: switch network structure and first-order info 
    :param policy: switch network policy
    :param scaler: normalization values
    :param num_policy_iterations: given the PI rounds, evaluate every 10th iteration
    :param time_steps: max time steps in an episode
    :return: long-run average cost, CIs
    """

    # initial state selection
    initial_state_0 = np.zeros(policy.get_obs_dim())

    ######## simulation #######
    iter_eval = np.hstack([np.array(range(0,num_policy_iterations,10))+1,num_policy_iterations])
    episodes = len(iter_eval)

    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(policy.get_obs_dim(), policy.get_act_dim(), policy.get_kl_targ()) for _ in range(episodes)]
    res = []
    ray.get([s.set_weights.remote(torch.load(logger.path_weights+'/weights_'+str(iter_eval[i])+'.pth')) for i, s in enumerate(simulators)])

    scaler_id = ray.put(scaler)
    scaler_new = Scaler(policy.get_obs_dim())
    scaler_new_id = ray.put(scaler_new)
    
    res.extend(ray.get([simulators[0].policy_performance.remote(network_id, scaler_new_id, time_steps, initial_state_0, 0)]))
    res.extend(ray.get([simulators[i].policy_performance.remote(network_id, scaler_id, time_steps, initial_state_0, i)
                      for i in range(1, episodes)]))

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



def run_policy(network_id, policy, simulators, scaler, logger, gamma,
               policy_iter_num, episodes, time_steps):
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
    
    # initial_states_set = np.zeros(policy.get_obs_dim())  #always start from zero
    initial_states_set = scaler.sample_initial(episodes)

    ######### policy simulation ########################
    accum_res = []  # results accumulator from all actors
    trajectories = []  # list of trajectories
    for j in range(actors_per_run):
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps,
                                 initial_states_set[j*MAX_ACTORS + i]) for i in range(MAX_ACTORS)]))
    if remainder>0:
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps,
                                 initial_states_set[actors_per_run * MAX_ACTORS + i]) for i in range(remainder)]))
    print('simulation is done')

    for i in range(len(accum_res)):
        trajectories.append(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps
        
    #################################################

    average_reward = np.mean([t['rewards'] for t in trajectories])
    
    ####### computation of the r_x_star value #############
    # av_value = 0
    # av_times = 0
    # for trajectory in trajectories:
    #     reward_vector = trajectory['rewards']

    #     disc_vec = discount(reward_vector, gamma, 0)
    #     for i in range(len(disc_vec)):
    #         if reward_vector[i]==0:
    #             av_value += disc_vec[i]
    #             av_times += 1

    # r_x_star = (1-gamma) * av_value / av_times

    ########################################################
    last_tenpercent_index = time_steps//100
    if policy_iter_num == 1:
        initial_states = np.concatenate([t['unscaled_obs'][-last_tenpercent_index:] for t in trajectories])
        scaler.update_initial(initial_states)

    #### normalization of the states in data ####################
    for t in trajectories:
             t['pol_observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
             t['val_observes'] = (t['val_input_mat'] - offset[:-1]) * scale[:-1]
                          
             # z = t['rewards'] - r_x_star 
             z = t['rewards'] - average_reward #approximate r_x_star using long-run-average
             t['rewards'] = z
    

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


def compute_val(trajectory, network, val_func, policy, scaler):
    
    time_step = len(trajectory['unscaled_obs'])
    
    all_action_matrix = np.array(list(network.dict_absolute_to_matrix_action.values()))
    action_matrix_traj_adj = np.repeat(all_action_matrix[:, :, np.newaxis], time_step, axis=2)
    action_matrix_traj_adj = np.transpose(action_matrix_traj_adj, (2,1,0))
    
    unscaled_obs = trajectory['unscaled_obs']
    unscaled_obs_act_adj = np.repeat(unscaled_obs[:, :, np.newaxis], network.action_size, axis=2)
    val_input_mat = np.maximum(unscaled_obs_act_adj - action_matrix_traj_adj, 0)
    
    
    val_input_mat_transpose = val_input_mat.transpose(0,2,1)
    val_input_mat_for_val_nn = val_input_mat_transpose.reshape((-1, network.buffers_num))
    
    scale, offset = scaler.get()
    val_input_scaled = (val_input_mat_for_val_nn - offset[:-1]) * scale[:-1]
        
    # state_val = val_func.predict(val_input_mat_for_val_nn)
    state_val = val_func.predict(val_input_scaled)
    
    
    state_val_unnorm = state_val / scale[-1] + offset[-1]
    state_val_shape_back = state_val_unnorm.reshape((-1,network.action_size))
    
    pol_observes = trajectory['pol_observes']
    pol_array = policy.sample(pol_observes)
    # pol_array = policy.sample(unscaled_obs)
    
    state_val_output = np.sum(state_val_shape_back * pol_array, axis = 1, keepdims = True)
    
    # trajectory['v_values'] = state_val_output
    
    return state_val_output

def compute_next_val_model(trajectory, network, val_func, policy, scaler):
    
    N = network.N
    lambda_matrix = network.lambda_matrix
    all_arrival=((np.arange(2**(N*N)).reshape(-1,1) & (2**np.arange(N*N)))!= 0).astype(int)[:,::-1]
    all_arrival_prob = np.prod(lambda_matrix * all_arrival, where = (lambda_matrix * all_arrival>0), axis=1) * np.prod((1-lambda_matrix) * (1-all_arrival), where = ((1-lambda_matrix) * (1-all_arrival)>0), axis=1)
    
    scale, offset = scaler.get()
    
    ## to compute expected next state val given current state and action
    unscaled_obs = trajectory['unscaled_obs']

    time_steps = len(unscaled_obs)
    
    unscaled_obs_arrival_adj = np.repeat(unscaled_obs[:, :, np.newaxis], 2**network.buffers_num, axis=2)
    arrival_state_adj = np.repeat(all_arrival.T[np.newaxis,:, :], time_steps, axis = 0)
    prob_state_adj = np.repeat(all_arrival_prob[np.newaxis,:], time_steps, axis = 0)
    
    all_action_matrix = np.array(list(network.dict_absolute_to_matrix_action.values()))
    action_matrix_traj_adj = np.repeat(all_action_matrix[:, :, np.newaxis], time_steps * (2**network.buffers_num), axis=2)
    action_matrix_traj_adj = np.transpose(action_matrix_traj_adj, (2,1,0))
    
    next_state_val_given_act = []
    for act in range(network.action_size):
        actions_mat = np.repeat([network.dict_absolute_to_matrix_action[act]], time_steps, axis=0) 
        action_state_arrival_adj = np.repeat(actions_mat[:, :, np.newaxis], 2**network.buffers_num, axis=2)
        all_next_state = (np.maximum(unscaled_obs_arrival_adj - action_state_arrival_adj, 0) + arrival_state_adj).astype(int)
    
        transposed_next_state = np.transpose(all_next_state, (2, 0, 1))
        next_state = transposed_next_state.reshape((-1,network.buffers_num))
        
        unscaled_obs_act_adj = np.repeat(next_state[:, :, np.newaxis], network.action_size, axis=2)
        input_mat = np.maximum(unscaled_obs_act_adj - action_matrix_traj_adj, 0)
        
        input_mat_transpose = input_mat.transpose(0,2,1)
        input_mat_for_val_nn = input_mat_transpose.reshape((-1, network.buffers_num))
        # state_val = val_func.predict(input_mat_for_val_nn)
        
        
        observes_for_val_nn = (input_mat_for_val_nn - offset[:-1]) * scale[:-1]
        state_val = val_func.predict(observes_for_val_nn) 
        
        state_val_unnorm = state_val / scale[-1] + offset[-1]
        state_val_shape_back = state_val_unnorm.reshape((-1, network.action_size))
        
        next_state_scale = (next_state - offset[:-1]) * scale[:-1]
        pol_array = policy.sample(next_state_scale)
        # pol_array = policy.sample(next_state)
        
        state_val_output = np.sum(state_val_shape_back * pol_array, axis = 1, keepdims = True)
        next_state_val_shape_back = state_val_output.reshape((2**network.buffers_num, -1)).transpose()
        
        if act ==0:
            next_state_val_given_act = np.sum(next_state_val_shape_back * prob_state_adj, axis = 1, keepdims = True)
        else:
            next_state_val_given_act = np.hstack((next_state_val_given_act, np.sum(next_state_val_shape_back * prob_state_adj, axis = 1, keepdims = True)))
    
    pol_observes = trajectory['pol_observes']
    pol_distr = policy.sample(pol_observes)
    # pol_distr = policy.sample(unscaled_obs)
    
    next_state_expectation_all_action = np.sum(next_state_val_given_act.squeeze() * pol_distr, axis = 1, keepdims = True)
    
    return next_state_expectation_all_action
    


def compute_target(trajectories, policy, network, val_func, gamma, lam, scaler, iteration, burn_in, policy_model = True, q_model = False):
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
    
    for trajectory in trajectories:
            
        if iteration!=1:
            
            if policy_model:
                
                if q_model:
                    next_state_val = compute_next_val_model(trajectory, network, val_func, policy, scaler)
                else:
                    next_state_val = np.append(trajectory['v_values'][1:], [trajectory['v_values'][-1]], axis=0)
                
            else:
                next_state_val = np.append(trajectory['q_values'][1:], [trajectory['q_values'][-1]], axis=0)
            

            q_values = trajectory['q_values']
            unscaled_obs = trajectory['unscaled_obs']
            
            #############################################################################################################
            
            # td-error computing
            tds_pi = trajectory['rewards'] - q_values + gamma * next_state_val #GAE

            # value function computing for futher neural network training
            disc_sum_rew = relative_af(unscaled_obs, td_pi=tds_pi, lam=gamma*lam) + q_values
            
        else:
            disc_sum_rew = relative_af(trajectory['unscaled_obs'], td_pi=trajectory['rewards'], lam=gamma*lam)
            

        trajectory['disc_sum_rew'] = disc_sum_rew

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn_in] for t in trajectories])
    val_input_mat = np.concatenate([t['val_input_mat'][:-burn_in] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn_in] for t in trajectories])


    ######### normalization ########################
    if iteration == 1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew)))
        # scaler.update(np.hstack((val_input_mat, disc_sum_rew)))
    
    scale, offset = scaler.get()
    
    if iteration == 1:
        for t in trajectories:
            t['pol_observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
            t['val_observes'] = (t['val_input_mat'] - offset[:-1]) * scale[:-1]
    
    
    pol_observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    val_observes = (val_input_mat - offset[:-1]) * scale[:-1]
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    ##################################################

    return val_observes, pol_observes, disc_sum_rew_norm



def relative_af(unscaled_obs, td_pi,  lam):
    # return advantage function
    disc_array = np.copy(td_pi)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        
        sum_tds = td_pi[i+1] + lam * sum_tds
        disc_array[i] += sum_tds

    return disc_array


def add_value(trajectories, network, val_func, policy, scaler):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    :param possible_states: transitions that are possible for the queuing network
    """

    scale, offset = scaler.get()

    # approximate value function for trajectory_whole['unscaled_obs']
    for trajectory in trajectories:
        values = val_func.predict(trajectory['val_observes']) #predicted val is normalized, need to convert to unnormalized val
        trajectory['q_values'] = values / scale[-1] + offset[-1]
        
        trajectory['v_values'] = compute_val(trajectory, network, val_func, policy, scaler)


def build_train_set(trajectories, gamma, scaler, burn_in, network, value_func): #added network as an input
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :param policy: added in policy as a new input parameter for switch network, needed to approximate expectation
    :return: data for further Policy and Value neural networks training
    """

    for trajectory in trajectories:
        
        q_values = trajectory['q_values']
        v_values = trajectory['v_values']
        ##############################################################################################################

        advantages = q_values - v_values
        trajectory['advantages'] = np.asarray(advantages)

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn_in] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn_in] for t in trajectories])

    scale, offset = scaler.get()
    actions = np.concatenate([t['actions'][:-burn_in] for t in trajectories])
    actions_glob = np.concatenate([t['actions_glob'][:-burn_in] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn_in] for t in trajectories])
    pol_observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages
    disc_sum_rew = disc_sum_rew  / (disc_sum_rew.std() + 1e-6) # normalize advantages


    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################

    return pol_observes,  actions, actions_glob, disc_sum_rew, advantages


def log_batch_stats(observes, actions_glob, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions_glob),
                # '_mean_adv': np.mean(advantages),
                # '_min_adv': np.min(advantages),
                # '_max_adv': np.max(advantages),
                # '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })



# TODO: check shadow name
def main(network_id, num_policy_iterations, gamma, lam, kl_targ, number_of_actors, hid1_mult, episode_length,
         burn_in, clipping_parameter, episode_length_evaluation, target_iter, policy_model, q_model, policy_import):
    """
    # Main training loop
    :param: see ArgumentParser below
    """

    N = ray.get(network_id).N
    obs_dim = ray.get(network_id).buffers_num
    act_dim = ray.get(network_id).action_size
    

    ######### create directories to save the results and log-files############
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)
    ###################################

    scaler = Scaler(obs_dim) # normilizer initialization
    val_func = NNValueFunction(obs_dim, hid1_mult) # Value Neural Network initialization
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult) # Policy Neural Network initialization
    
    #set up remote simulator (once and for all iterations)
    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(policy.get_obs_dim(),policy.get_act_dim(),policy.get_kl_targ()) for _ in range(MAX_ACTORS)]

    ############ initialize the algorithm with the proportionally randomized policy ###################
    if policy_import is False:
        scale, offset = scaler.get()
        initial_states_set = np.zeros((1,obs_dim))
        #initial_states_set = random.sample(scaler.initial_states, k=1)
        trajectories, total_steps, action_distr = \
            policy.run_episode(ray.get(network_id), scaler, episode_length, initial_states_set[0],rpp=True)
        state_input = (trajectories['unscaled_obs'] - offset[:-1]) * scale[:-1]
        policy.initialize_rpp(state_input, action_distr)
    else:
        cwd = os.getcwd()
        scale, offset = np.load(cwd + '/N='+ str(N) +'_initial'+ '/scaler.npy')
        scaler.vars=(1/scale-0.1)**2
        
        initial_weight_dir = cwd + '/N='+ str(N) +'_initial'+'/weights.pth'
        initial_weights = torch.load(initial_weight_dir)
        policy.set_weights(initial_weights)
    
    ma_cost = np.repeat(episode_length,5) # initially *2, just to avoid cut the alpha rate in the first round
    previous_cost = np.mean(ma_cost)
    #####################################################

    iteration = 0  # count of policy iterations

    alpha = 1
    benchmark =0.01
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        # alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.0001, alpha*clipping_parameter)
        policy.lr_multiplier = max(0.0001/clipping_parameter, alpha)

        print('Clipping range is ', policy.clipping_range)

        # if iteration % 10 == 1:
            # plt.imshow(policy.get_weights()['h1/kernel'])
            # plt.colorbar()
            # #plt.show()
            # kernel_title = 'h1_kernel_' + str(iteration)+'.png'
            # plt.savefig(kernel_title)
            
            # plt.imshow(policy.get_weights()['act_prob/kernel'])
            # plt.colorbar()
            # #plt.show()
            # kernel_title = 'act_prob_kernel_' + str(iteration)+'.png'
            # plt.savefig(kernel_title)

        trajectories, avg_cost = run_policy(network_id, policy, simulators, scaler, logger, gamma, iteration, 
                                            episodes=number_of_actors, time_steps=episode_length) #simulation
        ma_cost = np.append(ma_cost[1:],avg_cost)
        current_cost = avg_cost
        
        if np.abs(current_cost-previous_cost)<benchmark:
            alpha = alpha * 0.5
            ma_cost = np.repeat(current_cost * 2,5)
            # benchmark = max(0.0001, benchmark * 0.5)

        previous_cost = np.mean(ma_cost)
        
        ti_iter = 0
        while ti_iter<target_iter:
            
            add_value(trajectories, ray.get(network_id), val_func, policy, scaler)
            val_observes, pol_observes, disc_sum_rew_norm = compute_target(trajectories, policy, ray.get(network_id), val_func, gamma, lam, scaler,iteration, burn_in, policy_model, q_model)  # calculate values from data
            val_func.fit(val_observes, disc_sum_rew_norm, logger)  # update value function
            
            ti_iter = ti_iter+1
        
        add_value(trajectories, ray.get(network_id), val_func, policy, scaler)
        
        pol_observes, actions, actions_glob, disc_sum_rew, advantages = build_train_set(trajectories, gamma, scaler, burn_in, ray.get(network_id), val_func)

        log_batch_stats(pol_observes, actions_glob, disc_sum_rew, logger, iteration)  # add various stats
        policy.update(pol_observes, actions_glob, np.squeeze(advantages), logger)  # update policy


        logger.write(display=True)  # write logger results to file


    ############## save weights of the policy NN and normilizer  #####################
    # plt.imshow(policy.get_weights()['h1/kernel'])
    # plt.colorbar()
    # #plt.show()
    # kernel_title = 'h1_kernel_' + str(iteration)+'.png'
    # plt.savefig(kernel_title)
    
    # plt.imshow(policy.get_weights()['act_prob/kernel'])
    # plt.colorbar()
    # #plt.show()
    # kernel_title = 'act_prob_kernel_' + str(iteration)+'.png'
    # plt.savefig(kernel_title)
    
    weights = policy.get_weights()

    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.pth')
    torch.save(weights, file_weights)
    
    file_value_weights = val_func.get_weights()
    file_value = os.path.join(logger.path_weights, 'value_' + str(iteration) + '.pth')
    torch.save(file_value_weights, file_value)

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    
    ###################################################################################
    

    ########## performance estimation of the saved policies #########################################
    #  time_steps -- is the total length (all batches) used to evaluate performance of a policy

    performance_evolution_all, ci_all = run_weights(network_id, num_policy_iterations, policy, scaler, logger, time_steps= episode_length_evaluation)
    #################################################################################################



    file_res = os.path.join(logger.path_weights, 'average_' + str(performance_evolution_all[-1]) + '+-' +str(ci_all[-1]) + '.txt')
    file = open(file_res, "w")
    for i in range(len(ci_all)):
        file.write(str(performance_evolution_all[i])+'\n')
    file.write('\n')
    for i in range(len(ci_all)):
        file.write(str(ci_all[i])+'\n')


    logger.close()

