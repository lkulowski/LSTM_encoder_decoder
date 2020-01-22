# Author: Laura Kulowski

'''

Generate a synthetic dataset for our LSTM encoder-decoder
We will consider a noisy sinusoidal curve 

'''

import numpy as np 

def synthetic_data(Nt = 2000, tf = 80 * np.pi):
    
    '''
    create synthetic time series dataset
    : param Nt:       number of time steps (float)
    : param tf:       final time (float)
    : return t, y:    time, feature (arrays)
    '''
    
    t = np.linspace(0., tf, Nt)
    y = np.sin(2. * t) + 0.3 * np.sin(1. * t) + 0.5 * np.cos(t) + 1.2 * np.cos(0.5 * t + 0.2) + np.random.normal(0., 0.05, Nt)

    return t, y

def train_test_split(t, y, split = 0.8):

  '''
  
  split time series into train/test sets
  
  : param t:                      time (array) 
  : para y:                       feature (array)
  : para split:                   percent of data to include in training set (float)
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''
  
  indx_split = int(split * len(y))
  indx_train = np.arange(0, indx_split)
  indx_test = np.arange(indx_split, len(y))
  
  t_train = t[indx_train]
  y_train = y[indx_train]
  y_train = y_train.reshape(-1, 1)
  
  t_test = t[indx_test]
  y_test = y[indx_test]
  y_test = y_test.reshape(-1, 1)
  
  return t_train, y_train, t_test, y_test 
