#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:02:38 2022

@author: huodongyan
"""

import numpy as np
import copy
import datetime

import torch
# torch.autograd.set_detect_anomaly(True)

class PolicyNetwork(torch.nn.Module):
    """ Policy Neural Network (Nested)"""
    
    def __init__(self, obs_dim, act_dim, hid1_mult = 1, temp = 1.0):
        """
        This policy network would only work for switch network with atomic actions (not generally for any given network)
        Parameters
        ----------
        obs_dim : input state dimension, depends on how input states are encoded
        act_dim : action dimension (output dimension of the policy nn)

        """
        super(PolicyNetwork, self).__init__()                      
        # Inherited from the parent class nn.Module
        self.obs_dim = obs_dim
        self.hid_size = int(self.obs_dim * hid1_mult)
        self.act_dim = act_dim
        
        self.temp = temp
        
        self.fc1 = torch.nn.Linear(self.obs_dim, self.hid_size, bias = False)       
        self.fc2 = torch.nn.Linear(self.hid_size, self.act_dim, bias = False)  
        # self.fc3 = torch.nn.Linear(self.hid_size, self.hid_size, bias = False)  
        self.act_func = torch.nn.Tanh()   
        self.softmax = torch.nn.Softmax(dim=1)  


    def forward(self, pol_observes):    
                          
        # Forward pass: stacking each layer together
       
        out = self.fc1(pol_observes)
        out = self.act_func(out)
        # out = self.fc3(out)
        # out = self.act_func(out)
        out = self.fc2(out) 
        # out = self.act_func(out) # to "normalize" it within -1 and 1
        out = self.softmax(torch.div(out, self.temp))
        
        return out
        

class Policy(object):
    """ Policy class to hold functions for the nested PolicyNetwork """

    def __init__(self, N, kl_targ, clipping_range=0.2, hid1_mult = 5, temp=1.0, alt_formulation = False):
        """
        :param N: switch network N
        :param hid1_mult: size of first hidden layer, multiplier of obs_dim
        :param clipping_range:
        :param temp: temperature parameter (used in policy neural network softmax)
        """
        self.alt_formulation = alt_formulation
        self.obs_dim = (N ** 2) * 2 # convert input port to input state n^2 matrix, one n^2 for state, one n^2 for port availability
        
        if self.alt_formulation:
            self.obs_dim = N ** 2
        
        self.act_dim = N ** 2 # N^2 (rather than act_dim, should interpret as act_size, atomic action)
        self.temp = temp
        self.hid1_mult = hid1_mult
        self.policynetwork = PolicyNetwork(self.obs_dim, self.act_dim, hid1_mult = hid1_mult, temp = self.temp)
        
        self.kl_targ = kl_targ
        self.clipping_range = clipping_range

        self.epochs = 1 # 3
        
        self.optimizer = torch.optim.Adam(self.policynetwork.parameters())
        self.kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
    
    def get_pol_observes(self, state_obs, port_availability):
        
        if self.alt_formulation:
            pol_observes = np.array(state_obs) * np.array(port_availability) # - (1-np.array(port_availability))
        else:
            pol_observes = np.hstack((state_obs, port_availability))
            
        return pol_observes
    
    def get_gradient_norm(self, to_print = True):
        
        total_norm = 0
        parameters = [p for p in self.policynetwork.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if to_print:
            print("Total gradient 2-norm: ", total_norm)
        
        return total_norm

    def compute_entropy(self, state_obs, port_availability):
        """
        Calculate KL-divergence between old and new distributions
        """
        
        pol_observes = self.get_pol_observes(state_obs, port_availability)

        actions_probs = self.policynetwork(torch.Tensor(pol_observes)).detach()
        actions_probs_clip = torch.clamp(actions_probs, 1e-10, 1.0)
        
        #clip_by_value, to avoid the probability to be 0 or above 1
        entropy_vec = torch.sum(actions_probs_clip * torch.log(actions_probs_clip), axis=1)
        return -torch.mean(entropy_vec)
        
    def compute_kl(self,  state_obs, port_availability, actions_probs_old):
        """
        Calculate KL-divergence between old and new distributions
        """
        
        actions_probs_old_clip = torch.clamp(torch.Tensor(actions_probs_old), 1e-10, 1.0)

        pol_observes = self.get_pol_observes(state_obs, port_availability)
            
        actions_probs = self.policynetwork(torch.Tensor(pol_observes)).detach()
        actions_probs_clip = torch.clamp(actions_probs, 1e-10, 1.0)
        
        #clip_by_value, to avoid the probability to be 0 or above 1
        kl_vec = torch.sum(actions_probs_old_clip * torch.log(torch.div(actions_probs_old_clip,actions_probs_clip)), axis=1)
        return torch.mean(kl_vec)  
        
        
    def sample(self, state_obs, port_availability, stochastic=True):
        """
        :param obs: state (assume it to be one state only, not sample for multiple states simultaneously)
                    need to be scaled
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
                 vector of length N!
        """

        
        pol_observes = self.get_pol_observes(state_obs, port_availability)
        pr = self.policynetwork(torch.Tensor(pol_observes)).detach().numpy()

        if stochastic:
            return pr
        else:
            determ_prob = []
            
            inx = np.argmax(pr)
            ar = np.zeros(self.act_dim)
            ar[inx] = 1
            determ_prob.extend([ar[np.newaxis]])
            return determ_prob
        
    def run_episode(self, network, scaler, time_steps, initial_state, rpp = False, stochastic = True):
        """
        One episode simulation
        :param network: switch network (refer to SwitchNetwork object setup)
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param initial_state: initial state for the episode
        :return: collected data
        """

        total_steps = 0 # count steps
        
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        actions_mat = np.zeros((time_steps, network.action_size), 'int8') # store the complete matching matrix
        
        availability_mat = np.zeros((time_steps, network.N, network.buffers_num), 'int8') #tracks the availability of input-output port, 3d matrix
        
        actions_index = np.zeros((time_steps, network.N), 'int8') # store the N selected input-output port pair index
        
        array_actions = np.zeros((time_steps, network.N, network.action_size)) #store the policy distribution
        
        rewards = np.zeros((time_steps, network.N))        

        scale, offset = scaler.get()            
        state = np.asarray(initial_state, 'int32')

        ###############################################################

        t = 0
        while t < time_steps: # run until visit to the empty state (regenerative state)
        
            unscaled_obs[t] = state
            scaled_obs = (state - offset[:-1]) * scale[:-1]  # center and scale observations

            ###### compute action distribution according to Policy Neural Network for state###
            
            k = 0
            availability_track = np.ones(network.buffers_num) # 1 - denote available, 0 - denote infeasible matching
            while k < network.N :
                availability_mat[t, k] = availability_track

                if rpp:
                    # act_distr = np.squeeze(network.random_policy_distr(availability_track))
                    act_distr = np.squeeze(network.soft_greedy_pol_distr(state))
                else:
                    act_distr = np.squeeze(self.sample(np.array([scaled_obs]), np.array([availability_track]), stochastic = stochastic))
                
                array_actions[t, k] = act_distr
                
                act_distr_clean = np.nan_to_num(act_distr)
                # act_distr_clean = np.clip(act_distr_clean, 1e-10, 1)
                if np.sum(act_distr_clean * availability_track) > 0:
                    act_distr_normalized = (act_distr_clean * availability_track)/np.sum(act_distr_clean * availability_track)
                else:
                    act_distr_normalized = 1/np.sum(availability_track) * availability_track
                    
                index_sample = np.random.choice(network.action_size, 1, p=act_distr_normalized)
                actions_index[t, k] = index_sample
                
                input_sel, output_sel = network.return_tuple(index_sample)
                
                # update availability_track vector
                infeasible_index = np.unique(np.hstack((np.arange(network.N) * network.N + output_sel, np.arange(network.N) + network.N * input_sel)))
                
                
                infeasible_ports_induced = np.zeros(network.action_size)
                infeasible_ports_induced[infeasible_index] = 1
                rewards[t, k] = - np.sum((state * availability_track) * infeasible_ports_induced) #negative of queue size, in maximization scheme
                
                availability_track[infeasible_index] = 0

                k = k + 1
                
            ############################################
            complete_matching = network.matching_to_matrix(actions_index[t])
            actions_mat[t] = complete_matching
            
            state = network.next_state(state, complete_matching)

            t+=1

        total_steps += time_steps
        
        adj_unscaled_obs = np.repeat(unscaled_obs[:, :, np.newaxis], network.N, axis = 2).transpose(0,2,1) 
        adj_unscaled_obs = np.reshape(adj_unscaled_obs,(-1, network.buffers_num))
        adj_state_obs = (adj_unscaled_obs - offset[:-1]) * scale[:-1] 
        adj_availability_mat = np.reshape(availability_mat, (-1, network.buffers_num))
        
        adj_act_index = np.squeeze(np.reshape(actions_index, (-1,1)))
        adj_rewards_step = np.squeeze(np.reshape(rewards, (-1, 1)))
        
        # record simulation
        trajectory = {'unscaled_obs': unscaled_obs,
                      'adj_unscaled_obs': adj_unscaled_obs, # used for value/policy training, 2d matrix
                      'adj_state_obs': adj_state_obs,
                      
                      'availability_mat': availability_mat,
                      'adj_availability_mat': adj_availability_mat,
                      
                      'action_index': actions_index, 
                      'adj_act_index': adj_act_index,
                      
                      'actions_mat': actions_mat, # complete matching, corresponds to unscaled_obs
                      
                      'rewards_by_step': rewards, #2d matrix
                      'adj_rewards_step': adj_rewards_step, #reshape to 1d (vector), to be used for target computation
                      'rewards': np.sum(rewards,axis = 1),
                  }

        print('Network:', network.network_name + '.', 'Average cost:', -np.mean(trajectory['rewards']))

        return trajectory, total_steps, array_actions


    def initialize_rpp(self, state_obs, port_availability, action_distr, batch_size=2048):
        """
        training policy NN according to the PR policy
        :param observes: states, need to flatten to N^2 vector
        :param action_distr: distribution over actions under PR policy
        observes and action_distr must be in 2d
        :param batch_size: batch size in the NN training
        """
        print("Policy Initialization supervised-learning started.")
        pol_observes = self.get_pol_observes(state_obs, port_availability)


        x_train, y_train = torch.Tensor(pol_observes), torch.Tensor(action_distr)
        
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) 
        
        for e in range(10):
            
            for local_x, local_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.policynetwork(torch.Tensor(local_x))  # Forward pass
                loss_val = self.kl_loss(torch.log(pred_y), torch.Tensor(local_y)) # Compute Loss 
                                                                                  # equiv: torch.mean(torch.sum(local_y * torch.log(torch.div(local_y, pred_y)), axis=1))
                loss_val.backward()
                self.optimizer.step()

        
    def surr_loss(self, state_obs, port_availability, actions, old_prob, advantages):
        
        advantages = torch.Tensor(advantages) #just in case the input is not of torch.Tensor data type
        
        pol_observes = self.get_pol_observes(state_obs, port_availability)
        new_prob = self.policynetwork(torch.Tensor(pol_observes)) #the forward pass
        
        one_hot = torch.zeros(np.shape(actions)[0], self.act_dim)
        one_hot[torch.arange(np.shape(actions)[0]), actions.long()] = 1
        
        new_prob_act = torch.sum(torch.mul(new_prob, one_hot), axis=1)
        old_prob_act = torch.sum(torch.mul(old_prob, one_hot), axis=1)
        
        prob_ratio_act = torch.exp(torch.log(new_prob_act) - torch.log(old_prob_act))
        clip_ratio_act = torch.clamp(prob_ratio_act, 1-self.clipping_range, 1+self.clipping_range)
        
        # prob_ratio = torch.exp(torch.log(torch.clamp(new_prob, 1e-10,1)) - torch.log(torch.clamp(old_prob,1e-10,1)))
        # clip_ratio = torch.clamp(prob_ratio, 1-self.clipping_range, 1+self.clipping_range)
        
        # prob_ratio_act = torch.sum(torch.mul(prob_ratio, one_hot), axis=1)
        # clip_ratio_act = torch.sum(torch.mul(clip_ratio, one_hot), axis=1)
        
        clip_adv = torch.minimum(torch.mul(prob_ratio_act, advantages), 
                                 torch.mul(clip_ratio_act, advantages))
        
        # prob_ratio_act = torch.exp(torch.log(torch.clamp(torch.gather(new_prob, 1, torch.unsqueeze(actions.long(), -1)), 1e-10, 1)) - 
                                   # torch.log(torch.clamp(torch.gather(old_prob, 1, torch.unsqueeze(actions.long(), -1)), 1e-10, 1)))
        # clip_ratio_act = torch.clamp(prob_ratio_act, 1-self.clipping_range, 1 + self.clipping_range)
        
        # clip_adv = torch.minimum(torch.mul(torch.squeeze(prob_ratio_act), advantages), 
        #                          torch.mul(torch.squeeze(clip_ratio_act), advantages))
        
        surr_loss_val = -torch.mean(clip_adv)

        return surr_loss_val


    def update(self, state_obs, port_availability, actions, advantages, logger):
        # training of neural network
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions (action_ind only)
        again, observes and actions must be in 2d
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """
        print("Policy training started.")
        entropy = 0
        policy_training_start = datetime.datetime.now()
        
        pol_observes = self.get_pol_observes(state_obs, port_availability)
        
        obs_train, port_train, act_train, adv_train = torch.Tensor(state_obs), torch.Tensor(port_availability), torch.Tensor(actions), torch.Tensor(advantages)
        actions_prob_old = self.policynetwork(torch.Tensor(pol_observes)).detach()
        actions_prob_old_train = copy.deepcopy(torch.clamp(actions_prob_old, 1e-10,1))
        
        train_data = torch.utils.data.TensorDataset(obs_train, port_train, 
                                                    act_train, actions_prob_old_train, adv_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=2048, shuffle=True) 
        
        for e in range(self.epochs):
            
            for local_obs, local_port, local_act, local_prob, local_adv in train_loader:
                
                self.optimizer.zero_grad()
                loss_val = self.surr_loss(local_obs, local_port, local_act, local_prob, local_adv) # Compute Loss
                loss_val.backward()
                
                # self.get_gradient_norm()
                torch.nn.utils.clip_grad_norm_(self.policynetwork.parameters(), 0.25)
                # self.get_gradient_norm()
                
                self.optimizer.step()
                
                
                
            kl = self.compute_kl(state_obs, port_availability, actions_prob_old)
            if kl > self.kl_targ * 5:  # early stopping if D_KL diverges badly
                print('early stopping: D_KL diverges badly')
                break
            
        entropy = self.compute_entropy(state_obs, port_availability)

        one_hot = torch.zeros(np.shape(actions)[0], self.act_dim)
        one_hot[torch.arange(np.shape(actions)[0]), actions] = 1

        # actions probabilities w.r.t the new and old (current) policies
        actions_probs = self.policynetwork(torch.Tensor(pol_observes)).detach()
        actions_probs_act = torch.nansum(torch.mul(actions_probs, one_hot), axis=1)
        actions_probs_old_act = torch.nansum(torch.mul(actions_prob_old_train, one_hot), axis = 1)
        ratios_act = torch.exp(torch.log(torch.clamp(actions_probs_act, 1e-10, 1)) - torch.log(torch.clamp(actions_probs_old_act, 1e-10,1)))
        
        if self.clipping_range is not None:
            clipping_range = self.clipping_range
        else:
            clipping_range = 0

        logger.log({'Clipping': clipping_range,
                    'Max ratio': torch.max(ratios_act).numpy(),
                    'Min ratio': torch.min(ratios_act).numpy(),
                    'Mean ratio': torch.mean(ratios_act).numpy(),
                    'PolicyEntropy': entropy.numpy(),
                    'KL': kl.numpy(),})
        print('policy training time: {:.3f}...'.format(int(((datetime.datetime.now() - policy_training_start).
                                                            total_seconds() / 60) * 100) / 100.), 'minutes')

        

    def policy_performance(self, network, scaler, time_steps, initial_state, id, batch_num = 50, stochastic=True):
        #policy evaluation
        """
        :param: initial_state: initial state of the episode
        :param: id: episode id
        :param: batch_num: number of batched for the batch mean computation
        :return: average cost, CIs, id of the episode
        """

        average_performance_batch = np.zeros(batch_num)
        batch_size = time_steps//batch_num

        time_steps = batch_size * batch_num

        scale, offset = scaler.get()

        
        trajectory, _, _ = self.run_episode(network, scaler, time_steps, initial_state)
        unscaled_obs_sum = np.sum(trajectory['unscaled_obs'], axis = 1)
        
        average_performance_batch = np.mean(np.array(unscaled_obs_sum).reshape(-1, batch_size), axis=1)

        average_performance = np.mean(average_performance_batch)
        ci = np.std(average_performance_batch)*1.96 / np.sqrt(batch_num)


        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


    def get_weights(self):
        return self.policynetwork.state_dict()

    def set_weights(self, weights):
        # Set the weights in the network.
        self.policynetwork.load_state_dict(weights)



