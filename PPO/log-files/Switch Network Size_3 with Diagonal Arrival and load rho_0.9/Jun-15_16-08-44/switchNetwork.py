import math
import numpy as np

from itertools import permutations

class SwitchNetwork:
    # Switch Crossbar Network class
    # v2 takes state as 1*(N^2) row vector
    
    def __init__(self, N, rho, arrival_type):
        
        self.N = N # number of input/output ports
        self.buffers_num = self.N**2 # number of VOQs in switch network
        self.rho = rho # load, also controls arrival rates （which is then reflected through the lambda matrix)
        
        self.action_set = list(permutations(range(self.N))) # get all actions (sigma/policies)
        self.action_size = math.factorial(self.N) # total number of possible actions
        
        self.arrival_type = str(arrival_type).lower()
        # initiate arrival rate (Bermoulli probability) lambda_matrix based on self.N and self.rho
        if self.arrival_type == "uniform":
            self.lambda_matrix = np.array([[self.rho/self.N for x in range(self.N)] for y in range(self.N)]).reshape((1,self.N**2))[0]
        elif self.arrival_type == "diagonal":
            self.lambda_matrix = np.array(self.rho*(2/3*np.identity(self.N) 
                                                    + np.vstack((np.concatenate((np.array([np.zeros(self.N-1)]).T, 1/3 * np.identity(self.N-1)), axis=1), 
                                                                 np.concatenate(([1/3], np.zeros(self.N-1))))))).reshape((1,self.N**2))[0]
        elif self.arrival_type == "logdiagonal":
            self.lambda_matrix = np.array(np.exp2(self.N-(np.mod([np.asarray(range(self.N)) - y for y in range(self.N)],self.N) + 1)) 
                                          / (2**self.N-1) * self.rho).reshape((1,self.N**2))[0]
        
        elif self.arrival_type == "skewed_bot":
            self.lambda_matrix = np.array([[(self.rho/3)/(self.N-1) for x in range(self.N)] for y in range(self.N)]).reshape((1,self.N**2))[0]
            self.lambda_matrix[-1] = self.rho * 2/3
            
        else:
            raise ValueError("unknwon arrival type")
        
        self.dict_absolute_to_matching_action, self.dict_absolute_to_matrix_action = self.absolute_to_matrix()
        self.network_name = "Switch Network Size_"+ str(self.N)+ " with " + self.arrival_type.capitalize() + " Arrival and load rho_" +str(self.rho)
        
    def absolute_to_matrix(self):
        """
        :return dict_absolute_to_matching_action: 
            keys are 'act_ind' action representation, 
            values are N vector representation of the matching of output port for each input
            e.g. [0,1,2] for N=3
        :return dict_absolute_to_matrix_action: 
            keys are 'act_ind' action representation, 
            values are N * N matrix representation of the matching
            e.g. [[1,0,0],[0,1,0],[0,0,1]] for [0,1,2] N=3 matching vector
        
        act_ind - all possible actions are numerated GLOBALLY as 0, 1, 2, ..., N!-1
        action_matrix - N*N matrix that have 1 per column and per row that represents matching of input to output ports
        """
        dict_absolute_to_matching_action = {} #{} represents dictionary format in Python
        dict_absolute_to_matrix_action = {}
        
        for l in range(self.action_size):
            matching_matrix = np.zeros((self.N,self.N))
            matching_matrix[list(range(self.N)),(self.action_set[l])] = 1
            
            dict_absolute_to_matching_action[l] = self.action_set[l]
            dict_absolute_to_matrix_action[l] =  np.array(matching_matrix).reshape((1,self.N**2))[0]
        
        return dict_absolute_to_matching_action, dict_absolute_to_matrix_action
            
    #may add additional functions later: all possible arrival patterns and corresponding probabilities, 
    def next_state(self, state, action_ind):
        """
        generate the next state (second attempt)
        :param state: starting state, need to be X * (N^2) matrix, 
                      flatten the square matrix structure, to enable matrix computation
        :param action_ind: action index, vector of dimension X
        :return: next state, of same dimension as state input
        """
        X = np.shape(action_ind)[0]
        
        lambda_reshape = np.tile(self.lambda_matrix,X).reshape((X,self.buffers_num))
        A = np.random.binomial(1,lambda_reshape)
        #A = np.array([np.random.binomial(1, self.lambda_matrix) for x in range(X)]) # generate arrival matrix based on lambda_matrix
        
        action = np.vstack([self.dict_absolute_to_matrix_action[ai] for ai in action_ind]) 
        
        state_next = np.maximum(state - action, 0) + A
        return state_next

    #and a starting randomized stationary policy, which serves as intializing training set for policy neural network
    def random_policy_distr(self):
        """
        Test - can only be used for uniform Bernoulli iid arrival (which can be stabilized by uniform random sampling)
        Test - to see whether the entire code can be pieced together
        Return probability distribution of actions for each station based on Random policy
        :param state: system state
        :return: distribution of action according to random policy
        """
        
        return np.array([1/self.action_size for i in range(self.action_size)])

    def get_N(self):
        return self.N
    
    def get_rho(self):
        return self.rho
    
    def get_arrival(self):
        return self.arrival_type