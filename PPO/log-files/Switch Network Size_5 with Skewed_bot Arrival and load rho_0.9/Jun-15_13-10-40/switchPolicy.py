import numpy as np
import copy
import datetime

import torch


class PolicyNetwork(torch.nn.Module):
    """ Policy Neural Network (Nested)"""
    
    def __init__(self, obs_dim, act_dim, temp = 1.0):
        """
        Parameters
        ----------
        obs_dim : int (=N)
            the number of input ports of the switch network.
        act_dim : int (=N!)
            the number of possible matchings for the switch network.

        """
        super(PolicyNetwork, self).__init__()                      
        # Inherited from the parent class nn.Module
        self.obs_dim = obs_dim
        self.hid_size = self.obs_dim
        self.act_dim = act_dim
        
        self.temp = temp
        
        self.fc1 = torch.nn.Linear(self.obs_dim, self.hid_size, bias = False)       
        self.fc2 = torch.nn.Linear(self.hid_size, self.act_dim, bias = False)  
        self.tanh = torch.nn.Tanh()   
        self.softmax = torch.nn.Softmax(dim=1)  
        
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
    
    def forward(self, x):                              
        # Forward pass: stacking each layer together
        
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.softmax(torch.div(out, self.temp))
        
        # out = self.logsoftmax(torch.div(out, self.temp))
        # out = torch.exp(out)
        
        return out

class Policy(object):
    """ Policy class to hold functions for the nested PolicyNetwork """

    def __init__(self, obs_dim, act_dim, kl_targ, clipping_range=0.2, temp=1.0):
        """
        v2, switch back to obs_dim and act_dim, input_n does not work properly when combining with ray
        :param obs_dim, act_dim: associated with switch network N, obs_dim = N^2, act_dim = N
        :param hid1_mult: size of first hidden layer, multiplier of obs_dim
        :param clipping_range:
        :param temp: temperature parameter (used in tensorflow's neural network calibration)
        """
        self.obs_dim = obs_dim # convert input port to input state n^2 matrix
        self.act_dim = act_dim # n! (rather than act_dim, should interpret as act_size)
        self.temp = temp
        self.policynetwork = PolicyNetwork(self.obs_dim, self.act_dim, temp = self.temp)
        
        self.kl_targ = kl_targ

        self.epochs = 1
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
                
        self.clipping_range = clipping_range
        
        self.optimizer = torch.optim.Adam(self.policynetwork.parameters())
        self.kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
    

    def compute_entropy(self, observes):
        """
        Calculate KL-divergence between old and new distributions
        """

        actions_probs = self.policynetwork(torch.Tensor(observes)).detach()
        actions_probs_clip = torch.clamp(actions_probs, 1e-10, 1.0)
        
        #clip_by_value, to avoid the probability to be 0 or above 1
        entropy_vec = torch.sum(actions_probs_clip * torch.log(actions_probs_clip), axis=1)
        return -torch.mean(entropy_vec)
        
    def compute_kl(self, observes, actions_probs_old):
        """
        Calculate KL-divergence between old and new distributions
        """
        
        actions_probs_old_clip = torch.clamp(torch.Tensor(actions_probs_old), 1e-10, 1.0)

        actions_probs = self.policynetwork(torch.Tensor(observes)).detach()
        actions_probs_clip = torch.clamp(actions_probs, 1e-10, 1.0)
        
        #clip_by_value, to avoid the probability to be 0 or above 1
        kl_vec = torch.sum(actions_probs_old_clip * torch.log(torch.div(actions_probs_old_clip,actions_probs_clip)), axis=1)
        return torch.mean(kl_vec)  
        
        
    def sample(self, obs, stochastic=True):
        """
        :param obs: state (assume it to be one state only, not sample for multiple states simultaneously)
                    need to be scaled
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
                 vector of length N!
        """

        if stochastic:
            obs_copy = np.array(obs)
            return self.policynetwork(torch.Tensor(obs_copy)).detach().numpy()
            # return self.policynetwork(torch.Tensor(copy.copy(obs))).detach().numpy()
        else:
            determ_prob = []
            
            pr = self.policynetwork(torch.Tensor(obs)).detach().numpy()
            inx = np.argmax(pr)
            ar = np.zeros(self.act_dim)
            ar[inx] = 1
            determ_prob.extend([ar[np.newaxis]])
            return determ_prob
        
    def run_episode(self, network, scaler, time_steps, initial_state, rpp = False, mw1 = False):
        """
        One episode simulation
        :param network: switch network (refer to SwitchNetwork object setup)
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param initial_state: initial state for the episode
        :return: collected data
        """

        policy_buffer = {} # save action disctribution of visited states, then no need to recalculate, only call the distribution here (in dictionary)

        total_steps = 0 # count steps
        
        observes = np.zeros((time_steps, network.buffers_num)) #observes store scaled states
        actions = np.zeros((time_steps, network.N), 'int8')
        actions_glob = np.zeros((time_steps,  ), 'int8') #index for action
        actions_mat = np.zeros((time_steps, network.buffers_num), 'int8')
        val_input_mat = np.zeros((time_steps, network.buffers_num), 'int8')
        
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        array_actions = np.zeros((time_steps, network.action_size))
        

        scale, offset = scaler.get()            
        state = np.asarray(initial_state, 'int32')

        ###############################################################

        t = 0
        while t < time_steps: # run until visit to the empty state (regenerative state)
            unscaled_obs[t] = state
            state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

            ###### compute action distribution according to Policy Neural Network for state###
            if tuple(state) not in policy_buffer:
                if rpp: #rpp stands for random proportional policy (from Mark's original code) 
                        #here it stands for following uniform random policy
                    act_distr = network.random_policy_distr()
                elif mw1:
                    act_distr = network.mw_alpha(state)
                    
                else:
                    act_distr = self.sample([state_input])[0]
                    
                policy_buffer[tuple(state)] = act_distr
                
            distr = policy_buffer[tuple(state)] # distribution for each station
            array_actions[t] = distr
                
            ############################################
            rewards[t] = -np.sum(state) #negative of queue size, in maximization scheme

            act_ind = np.random.choice(network.action_size, 1, p=distr)
            actions[t] = network.dict_absolute_to_matching_action[act_ind[0]]
            actions_mat[t] = network.dict_absolute_to_matrix_action[act_ind[0]]
            val_input_mat[t] = np.maximum(unscaled_obs[t] - actions_mat[t], 0)
            
            state = network.next_state(state.reshape(network.buffers_num), act_ind)[0]
            
            
            observes[t] = state_input
            actions_glob[t] = act_ind[0]

            t+=1

        total_steps += len(actions)
        
        # record simulation
        trajectory = {#'observes': observes,
                      'actions': actions,
                      'actions_glob': actions_glob,
                      'actions_mat': actions_mat,
                      'val_input_mat': val_input_mat,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs,
                  }

        print('Network:', network.network_name + '.', 'Average cost:', -np.mean(trajectory['rewards']))

        return trajectory, total_steps, array_actions


    def initialize_rpp(self, observes, action_distr, batch_size=4096):
        """
        training policy NN according to the PR policy
        :param observes: states, need to flatten to N^2 vector
        :param action_distr: distribution over actions under PR policy
        :param batch_size: batch size in the NN training
        """

        x_train, y_train = observes, action_distr
        n_size = np.shape(observes)[1]
        
        
        train_data = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) 
        
        # for e in range(n_size * 100 * 2):
        for e in range(100):
            
            for local_x, local_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.policynetwork(torch.Tensor(local_x))  # Forward pass
                loss_val = self.kl_loss(torch.log(pred_y), torch.Tensor(local_y)) # Compute Loss 
                                                                                  # equiv: torch.mean(torch.sum(local_y * torch.log(torch.div(local_y, pred_y)), axis=1))
                loss_val.backward()
                self.optimizer.step()

        
    def surr_loss(self, states, actions, old_prob, advantages):
        
        advantages = torch.Tensor(advantages) #just in case the input is not of torch.Tensor data type
        
        new_prob = self.policynetwork(torch.Tensor(states)) #the forward pass

        one_hot = torch.zeros(np.shape(actions)[0], self.act_dim)
        one_hot[torch.arange(np.shape(actions)[0]), actions.long()] = 1
        
        new_prob_act = torch.nansum(torch.mul(new_prob, one_hot), axis = 1)
        old_prob_act = torch.nansum(torch.mul(old_prob, one_hot), axis = 1)
        new_prob_act = torch.clamp(new_prob_act, 1e-10, 1)
        old_prob_act = torch.clamp(old_prob_act, 1e-10, 1)
        
        prob_ratio_act = torch.exp(torch.log(new_prob_act) - torch.log(old_prob_act))
        clip_ratio_act = torch.clamp(prob_ratio_act, 1-self.clipping_range, 1+self.clipping_range)

        # prob_ratio = torch.exp(torch.log(new_prob) - torch.log(old_prob))
        # clip_ratio = torch.clamp(prob_ratio, 1-self.clipping_range, 1+self.clipping_range)
        
        # one_hot = torch.zeros(np.shape(actions)[0], self.act_dim)
        # one_hot[torch.arange(np.shape(actions)[0]), actions.long()] = 1
        
        
        # prob_ratio_act = torch.sum(torch.mul(prob_ratio, one_hot), axis=1)
        # clip_ratio_act = torch.sum(torch.mul(clip_ratio, one_hot), axis=1)
        
        clip_adv = torch.minimum(torch.mul(prob_ratio_act, advantages), 
                                 torch.mul(clip_ratio_act, advantages))
        
        surr_loss_val = -torch.mean(clip_adv)

        return surr_loss_val


    def update(self, observes, actions, advantages, logger):
        # training of neural network
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions (action_ind only)
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """
        entropy = 0
        policy_training_start = datetime.datetime.now()
        
        obs_train, act_train, adv_train = observes, actions, advantages
        actions_prob_old = self.policynetwork(torch.Tensor(observes)).detach()
        actions_prob_old_train = copy.deepcopy(actions_prob_old)
        
        train_data = torch.utils.data.TensorDataset(torch.Tensor(obs_train), torch.Tensor(act_train), torch.Tensor(actions_prob_old_train), torch.Tensor(adv_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4096, shuffle=True) 
        
        for e in range(self.epochs):
            
            for local_obs, local_act, local_prob, local_adv in train_loader:
                
                self.optimizer.zero_grad()
                loss_val = self.surr_loss(local_obs, local_act, local_prob, local_adv) # Compute Loss
                loss_val.backward()
                self.optimizer.step()
                
            kl = self.compute_kl(observes, actions_prob_old)
            if kl > self.kl_targ * 5:  # early stopping if D_KL diverges badly
                print('early stopping: D_KL diverges badly')
                break
            
        entropy = self.compute_entropy(observes)

        # actions probabilities w.r.t the new and old (current) policies
        actions_probs = self.policynetwork(torch.Tensor(observes)).detach()
        ratios = torch.exp(np.log(actions_probs) - torch.log(actions_prob_old_train))
        
        one_hot = torch.zeros(np.shape(actions)[0], self.act_dim)
        one_hot[torch.arange(np.shape(actions)[0]), actions] = 1
        ratios_act = torch.nansum(torch.mul(torch.Tensor(ratios), one_hot), axis=1)
        
        if self.clipping_range is not None:
            clipping_range = self.clipping_range
        else:
            clipping_range = 0

        logger.log({'Clipping': clipping_range,
                    'Max ratio': torch.max(ratios_act).numpy(),
                    'Min ratio': torch.min(ratios_act).numpy(),
                    'Mean ratio': torch.mean(ratios_act).numpy(),
                    'PolicyEntropy': entropy.numpy(),
                    'KL': kl.numpy(),
                    '_lr_multiplier': self.lr_multiplier})
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
        policy_buffer = {} #python dictionary
        batch_size = time_steps//batch_num

        time_steps = batch_size * batch_num

        scale, offset = scaler.get()

        state = np.asarray(initial_state, 'int32')
        # print(state)


        batch = -1
        k = 0
        for t in range(time_steps):

            # evaluation progress
            if t % batch_size == 0:
                batch += 1
                # print(int(batch/batch_num *100), '%')
                k = -1
            k += 1

            state_input = (state - offset[:-1]) * scale [:-1] # center and scale observations

            if tuple(state) not in policy_buffer:
                act_distr = self.sample([state_input], stochastic)
                policy_buffer[tuple(state)] = act_distr
                
            distr = policy_buffer[tuple(state)][0]
            act_ind = np.random.choice(len(distr), 1, p=distr)

            average_performance_batch[batch] = 1/(k+1) * np.sum(state) + k / (k+1) * average_performance_batch[batch]

            state = network.next_state(state, act_ind)[0]

        average_performance = np.mean(average_performance_batch)
        ci = np.std(average_performance_batch)*1.96 / np.sqrt(batch_num)


        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci



    def get_obs_dim(self):
        return self.obs_dim

    def get_act_dim(self):
        return self.act_dim

    def get_weights(self):
        return self.policynetwork.state_dict()

    def set_weights(self, weights):
        # Set the weights in the network.
        self.policynetwork.load_state_dict(weights)
    
    def get_kl_targ(self):
        return self.kl_targ
    
    def get_policy(self, observes):
        return self.policynetwork(torch.Tensor(observes)).detach()


