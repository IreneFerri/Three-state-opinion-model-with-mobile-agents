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
# velocities array
v_array = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])
#
print('Neighbors')
print(v_array)
dim_v = v_array.size
##################################################
P = np.zeros(dim_v)
sigma = np.zeros(dim_v)
for v in range(dim_v):
  filename = 'second_neigh_RW_N'+str(N0)+'_v'+"%5.3f"%v_array[v]+'.csv'
  print(v_array[v])
  probs = pd.read_csv(filename, delim_whitespace = True, header=None).values
  P[v] = np.mean(probs)
  sigma[v] = np.std(probs)


print(P)
print(sigma)

#####################################################
# TIME CONSENT
##################################################
velocities = np.arange(0.01, 0.50, 0.01)
print('Time Consent)')
print(velocities)
dim_vel = velocities.size
#######################################
alpha = 1.00
time_array = np.zeros(dim_vel)
sigma_time = np.zeros(dim_vel)
for v in range(dim_vel):
  print(velocities[v])
  filename = '../time_consent/RW_vctt_a'+str(alpha)+'_data/RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+"%4.2f"%velocities[v]+'.csv'
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
#
#ax.plot(v_array, P, yerr = sigma)
ax.set_xlabel('v', fontsize = label_size)
#ax.set_ylabel(r'$P(first_{t}/second_{t-1})$', fontsize = label_size)
ax.set_title(fig_title, fontsize = title_size)
#
ax.text(0.25, 6000, r'$\alpha = $'+str(alpha), fontsize=label_size)
#
rescaling = np.mean(time_array)*3
ax.errorbar(v_array, P*rescaling, yerr = sigma*rescaling, label = r'$P(first_{t}/second_{t-1})$')
ax.errorbar(velocities, time_array, yerr = sigma_time, label = 'consensus time')
#
plt.legend()
plt.tight_layout()
plt.show()

















