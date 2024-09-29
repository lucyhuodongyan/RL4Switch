"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import torch

class ValueNetwork(torch.nn.Module):
    """ Policy Neural Network (Nested)"""
    
    def __init__(self, obs_dim, hid1_mult):
        """
        Parameters
        ----------
        obs_dim : int (=N)
            the number of input ports of the switch network.
        act_dim : int (=N!)
            the number of possible matchings for the switch network.

        """
        super(ValueNetwork, self).__init__()                      
        # Inherited from the parent class nn.Module
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
        self.hid3_size = 10
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        
        self.fc1 = torch.nn.Linear(self.obs_dim, self.hid1_size)      
        self.fc2 = torch.nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = torch.nn.Linear(self.hid2_size, self.hid3_size)    
        self.fc4 = torch.nn.Linear(self.hid3_size, 1)  
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):                              
        # Forward pass: stacking each layer together
        
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        
        return out
    
    

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.valuenetwork = ValueNetwork(self.obs_dim, self.hid1_mult)
        
        self.optimizer = torch.optim.Adam(self.valuenetwork.parameters())
        self.mse_loss = torch.nn.MSELoss()
        
        self.epochs = 1



    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        
        
        x_train, y_train = x, y
        
        train_data = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4096, shuffle=True) 
        
        for e in range(self.epochs):
            
            for local_x, local_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.valuenetwork(torch.Tensor(local_x))  # Forward pass
                loss_val = self.mse_loss(pred_y, torch.Tensor(local_y)) # Compute Loss 
                loss_val.backward()
                self.optimizer.step()

        logger.log({'ValFuncLoss': loss_val.detach().numpy()})
        print('ValFuncLoss', loss_val.detach().numpy())

    def predict(self, x):
        """ Predict method """

        y_hat = self.valuenetwork(torch.Tensor(x)).detach().numpy()

        return y_hat

    def get_weights(self):
        return self.valuenetwork.state_dict()