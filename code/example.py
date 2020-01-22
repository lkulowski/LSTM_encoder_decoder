# Author: Laura Kulowski

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''


import numpy as np
import matplotlib.pyplot as plt

import generate_dataset

# generate dataset for LSTM
t, y = generate_dataset.synthetic_data()
t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split = 0.8)

plt.figure(figsize = (17,4))
plt.plot(t_train, y_train, color = '0.4', label = 'Train') 
plt.plot(np.concatenate([t_train[-2:-1], t_test]), np.concatenate([y_train[-2:-1], y_test]),
         color = (0.21, 0.47, 0.69), label = 'Test')
plt.xlim([t[0], t[-1]])
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Splitting Time Series into Train and Test Sets')
plt.legend(bbox_to_anchor=(1, 1))
plt.savefig('plots/train_test_split.png')
plt.show()
