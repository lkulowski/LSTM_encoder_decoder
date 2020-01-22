# Author: Laura Kulowski

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''

import numpy as np
import matplotlib.pyplot as plt

import generate_dataset

import matplotlib
matplotlib.rcParams.update({'font.size': 17})

# generate dataset for LSTM
t, y = generate_dataset.synthetic_data()
t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split = 0.8)

plt.figure(figsize = (18, 6))
plt.plot(t_train, y_train, color = '0.4', linewidth = 2, label = 'Train') 
plt.plot(np.concatenate([t_train[-2:-1], t_test]), np.concatenate([y_train[-2:-1], y_test]),
         color = (0.21, 0.47, 0.69), linewidth = 2, label = 'Test')
plt.xlim([t[0], t[-1]])
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Time Series Split into Train and Test Sets')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout
plt.savefig('plots/train_test_split.png')
