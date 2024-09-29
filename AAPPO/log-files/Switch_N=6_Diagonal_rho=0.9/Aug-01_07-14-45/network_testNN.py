#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:58:40 2022

@author: huodongyan

"""

import numpy as np


class SwitchNetwork:
    # Switch Crossbar Network class
    # v2 takes state as 1*(N^2) row vector
    
    def __init__(self, N, rho, arrival_type):
        
        self.N = N # number of input/output ports
        
        self.buffers_num = self.N**2 # number of VOQs in switch network
        self.action_size = self.N**2 # total number of possible actions, atomic actions of size N^2
        self.action_set = np.identity(self.action_size) # one-hot vectors to indicate selected matching
        
        self.rho = rho # load
        self.arrival_type = str(arrival_type).lower()
        # initiate arrival rate (Bermoulli probability) lambda_matrix based on self.N and self.rho
        if self.arrival_type == "uniform":
            self.lambda_matrix = np.repeat(self.rho/self.N , self.N ** 2)
        elif self.arrival_type == "diagonal":
            self.lambda_matrix = np.array(self.rho*(2/3*np.identity(self.N) 
                                                    + np.vstack((np.hstack((np.zeros(self.N - 1)[:, np.newaxis], 1/3 * np.identity(self.N-1))), 
                                                                 np.hstack((1/3, np.zeros(self.N-1))))))).reshape(-1)
        elif self.arrival_type == "logdiagonal":
            self.lambda_matrix = np.array(np.exp2(self.N-(np.mod([np.asarray(range(self.N)) - y for y in range(self.N)],self.N) + 1)) 
                                          / (2**self.N-1) * self.rho).reshape(-1)
        elif self.arrival_type == "skewed_bot":
            self.lambda_matrix = np.repeat((self.rho/3)/(self.N-1), self.N**2)
            self.lambda_matrix[-1] = self.rho * 2/3

        elif self.arrival_type == "skewed_bot_v2":
            norm_constant = np.sum(np.arange(self.N + 1) ** 2)
            self.lambda_matrix = np.repeat([np.repeat([1/norm_constant], self.N)], self.N, axis = 0)
            for i in range(1, self.N):
                self.lambda_matrix[i:, i:] = (i + 1)**2 / norm_constant
            self.lambda_matrix = self.lambda_matrix.reshape(-1) * self.rho
        else:
            raise ValueError("unknwon arrival type")
        
        self.network_name = "Switch_N="+ str(self.N)+ "_" + self.arrival_type.capitalize() + "_rho=" +str(self.rho)
        
    def matching_to_matrix(self, individual_matching_list):
        """
        individual_matching_list(list of atomic actions) should be an array of size N with atomic act indices
        :return complete matching from N individual input-output port matchings
        
        """
        if len(individual_matching_list) != self.N:
            raise ValueError("incorrect input")
        
        complete_matching = np.zeros(self.buffers_num)
        complete_matching[individual_matching_list] = 1
        
        complete_matching_reshape = np.reshape(complete_matching, (self.N, self.N))
        if np.sum(np.sum(complete_matching_reshape, axis = 0) ==1) != self.N or np.sum(np.sum(complete_matching_reshape, axis = 0)==1) != self.N:
            raise ValueError("incorrect input")
        
        return complete_matching
            
    #may add additional functions later: all possible arrival patterns and corresponding probabilities, 
    def next_state(self, state, complete_matching):
        """
        generate the next state (second attempt)
        :param state: starting state, only allow one (N^2) vector
        :param complete_matching: full matching induced by atomic actions
        :return: next state, of same dimension as state input
        """
        
        arrival = np.random.binomial(1, self.lambda_matrix)        
        state_next = np.maximum(state - complete_matching, 0) + arrival
        
        return state_next

    #and a starting randomized stationary policy, which serves as intializing training set for policy neural network
    def random_policy_distr(self, port_availability):
        """
        Test - can only be used for uniform Bernoulli iid arrival (which can be stabilized by uniform random sampling)
        Return probability distribution of actions for each station based on uniform random policy
        :return: distribution of action according to random policy
        """
        # distr = np.array([1 / np.sum(port_availability) for _ in range(self.action_size)])
        # distr = distr * port_availability
        distr = np.array([1 / self.action_size for _ in range(self.action_size)])
        return distr
    
    def soft_greedy_pol_distr(self, state):
        
        temp = 0.4
        softmax_distr = np.exp(state/ temp) /np.sum(np.exp(state/temp))

        return softmax_distr
    
    def return_tuple(self, index_val):
        return index_val // self.N, index_val % self.N
    
