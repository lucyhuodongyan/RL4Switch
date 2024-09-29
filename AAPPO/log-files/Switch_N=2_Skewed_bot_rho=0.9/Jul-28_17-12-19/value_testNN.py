#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:02:20 2022

@author: huodongyan
"""


"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import torch
import copy
import datetime

class ValueNetwork(torch.nn.Module):
    """ Policy Neural Network (Nested)"""
    
    def __init__(self, obs_dim, hid1_mult = 10):
        """
        Parameters
        ----------
        obs_dim : int (=N)
            the number of input ports of the switch network.
        hid1_mult : int (default to 10)
            controls how to scale the width of the value neural network
        act_dim : int (=N!)
            the number of possible matchings for the switch network.

        """
        super(ValueNetwork, self).__init__()                      
        # Inherited from the parent class nn.Module
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 
        self.hid3_size = hid1_mult
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        
        self.fc1 = torch.nn.Linear(self.obs_dim, self.hid1_size)      # new test: do not allow for bias in value nn
        self.fc2 = torch.nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = torch.nn.Linear(self.hid2_size, self.hid3_size)    
        self.fc4 = torch.nn.Linear(self.hid3_size, 1)  
        self.act_func = torch.nn.ReLU() # new test: try out tanh activation function (instead of relu)
    
    def forward(self, x):      
                        
        # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.act_func(out)
        out = self.fc2(out)
        out = self.act_func(out)
        out = self.fc3(out)
        out = self.act_func(out)
        out = self.fc4(out)
        
        return out
    
    

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, N, hid1_mult = 10, alt_formulation = False):
        """
        Args:
            N: switch network size (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.alt_formulation = alt_formulation
        self.obs_dim = (N ** 2) * 2 # include state, port availability and action
        
        if self.alt_formulation:
            self.obs_dim = N**2 #update obs_dim when using alternative formulation
            
        self.hid1_mult = hid1_mult
        self.valuenetwork = ValueNetwork(self.obs_dim, hid1_mult = self.hid1_mult)
        
        self.optimizer = torch.optim.Adam(self.valuenetwork.parameters())
        self.mse_loss = torch.nn.MSELoss()
        
        self.epochs = 1

    def get_val_observes(self, state_obs, port_availability):
        
        if self.alt_formulation:
            val_observes = np.array(state_obs) * np.array(port_availability) # - (1-np.array(port_availability))
        else:
            val_observes = np.hstack((state_obs, port_availability))
            
        return val_observes


    def fit(self, x1, x2, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x1: scaled states
            x2: port availability
            y: target
            logger: logger to save training loss and % explained variance
        """
        value_training_start = datetime.datetime.now()
        
        x = self.get_val_observes(x1, x2)
        
        x_train, y_train = torch.Tensor(x), torch.Tensor(y)
        
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=2048, shuffle=True) 
        
        for e in range(self.epochs):
            
            for local_x, local_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.valuenetwork(torch.Tensor(local_x))  # Forward pass
                loss_val = self.mse_loss(pred_y, torch.Tensor(local_y)) # Compute Loss 
                loss_val.backward()
                self.optimizer.step()

        logger.log({'ValFuncLoss': loss_val.detach().numpy()})
        print('ValFuncLoss', loss_val.detach().numpy())
        print('value training time: {:.3f}...'.format(int(((datetime.datetime.now() - value_training_start).total_seconds() / 60) * 100) / 100.), 'minutes')

    def predict(self, state_obs, port_availability):
        """ Predict method """
        
        val_observes = self.get_val_observes(state_obs, port_availability)
        y_hat = self.valuenetwork(torch.Tensor(val_observes)).detach().numpy()

        return y_hat

    def get_weights(self):
        return self.valuenetwork.state_dict()
    
    def set_weights(self, weights):
        # Set the weights in the network.
        self.valuenetwork.load_state_dict(weights)
