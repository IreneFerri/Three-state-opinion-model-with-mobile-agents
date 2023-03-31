import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
#
#####################################################
# DATA
##################################################
N0 = 155  # Random geometric graph from 'positions.txt'
N = 104  # Giant connected component - R = 1.00001*d_c
temp = 0.05
nsteps = N0*10
reps = 100
#
#####################################################
# TIME CONSENT
##################################################
velocities = np.arange(0.01, 0.50, 0.01)
print('Time Consent)')
print(velocities)
dim_vel = velocities.size
#######################################
alpha = 0.75
time_array = np.zeros(dim_vel)
sigma_time = np.zeros(dim_vel)
for v in range(dim_vel):
  print(velocities[v])
  filename = 'RW_vctt_elastic_a'+"%4.2f"%alpha+'_data/RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+"%4.2f"%velocities[v]+'.csv'
  time = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [1]).values.T.reshape(reps)
  time = time[~np.isnan(time)]  # ~ logical-not operator
  print(time)
  time_array[v] = np.mean(time)
  sigma_time[v] = np.std(time)
#



#####################################################
# PLOT 
##################################################
title_size = 12
label_size = 12
fig_title =  'N = '+str(N)+' - Steps = 10*155'+' - Reps = '+str(reps)
#
fig, ax = plt.subplots()
ax.text(0.05, 4000, r'$\alpha = $'+str(alpha), fontsize=label_size)
#
ax.errorbar(velocities, time_array, yerr = sigma_time)
plt.show()
