"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
v2 update from original copy: want to get rid of the +1 obs_dim dimension change, and hence no more initial state procedure
"""
import numpy as np
import os
import shutil
import glob
import csv

class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.obs_dim = obs_dim
        self.vars = np.zeros(obs_dim +1) # include one more dimension for value function
        self.means = np.zeros(obs_dim +1)
        self.m = 0
        self.n = 0
        self.first_pass = True
        self.initial_states = None
        


    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """

        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False

        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n
    
    def update_initial(self, initial_states):
        
        #initial_states is np array
        
        if self.initial_states is None:
            self.initial_states = initial_states
    
    def sample_initial(self, count):
        
        if self.initial_states is None:
            return np.zeros((count, self.obs_dim))
        else:
            sample_index = np.random.choice(len(self.initial_states), size = count)
            return self.initial_states[sample_index, :]


    def get(self):
        """ returns 2-tuple: (scale, offset) """
        # return 1/(np.sqrt(self.vars) + 0.1)/3, self.means*0
        
        # remove scaling/centering for input states, only scale the value (dependent variable)
        return np.append(np.ones(self.obs_dim), [1/(np.sqrt(self.vars[-1]) + 0.1)]), self.means*0
        
        # have an update to this code, so that no normalization, i.e. std=1 and mean=0
        # return np.ones(self.obs_dim +1), np.zeros(self.obs_dim +1)


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now, time_start):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        self.time_start = time_start
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path = os.path.join(dirname, 'log-files', logname, now)
        os.makedirs(path)
        self.path_weights = os.path.join(path, 'weights')
        os.makedirs(self.path_weights)
        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)

        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method





    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Average Cost = {:.1f} *****'.format(log['_Episode'],
                                                               log['_AverageReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
