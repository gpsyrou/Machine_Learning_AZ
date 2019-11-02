"""
Reinforcement Learning - Upper Confidence Bound (UCB)
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')

# We have 10 versions of the same advertisement
# And each row its a different user, when 1/0 indicates if the user clicked
# on the add (1)or not (0)

###### 1 ######
# First lets find the total reward, assuming we were using a random selection

# Select a random add for 10000 rounds
# and compare it with the truth in the data
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = data.values[n, ad]
    total_reward = total_reward + reward

# Visualise the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.grid(True, alpha = 0.3)
plt.show()
# We get almost uniform distribution

######## UCB algorithm ###########

# Implement the UCB algorithm
N = 10000   
d = 10      # Number of ads

numbers_of_selections = [0] * d # create a vector of zeros of size d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.grid(True, alpha = 0.3)
plt.show()
