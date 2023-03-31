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
nsteps = N0*10
reps = 100
v_array = np.arange(0.01, 0.51, 0.05)  # velocities array
v_array = np.concatenate((np.array([0.0]), v_array))

v_array = np.array([0.0, 0.01, 0.03, 0.06, 0.08, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])
v_array = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])

v_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])

print(v_array)
dim_v = v_array.size
##################################################
P = np.zeros(dim_v)
sigma = np.zeros(dim_v)
log_slope = np.zeros(dim_v)
for v in range(dim_v):
  filename = 'second_neigh_RW_N'+str(N0)+'_v'+"%5.3f"%v_array[v]+'.csv'
  filename2  = '../agg_reach_elastic/reachab_normal_1_N'+str(N0)+'_a_v'+"%5.3f"%v_array[v]+'.csv'
  print(v_array[v])
  probs = pd.read_csv(filename, delim_whitespace = True, header=None).values
  P[v] = np.mean(probs)
  sigma[v] = np.std(probs)
###########
  steps_array = pd.read_csv(filename2, delim_whitespace = True, header=None ,  usecols = [0]).values
  steps_reach = steps_array.size
  steps_array = steps_array.reshape(steps_reach)
#
  reach = pd.read_csv(filename2, delim_whitespace = True, header=None ,  usecols = [1]).values.reshape(steps_reach)
  dim_reach = reach.size
  step_adjust = steps_array[1:7]
  adjust = reach[1:7]
#  print(step_adjust, adjust)
  print(np.polyfit(np.log(step_adjust), adjust, 1))
  log_slope[v] = np.polyfit(np.log(step_adjust), adjust, 1)[0]
#
print(P)
print(sigma)

#####################################################
# PLOT 
##################################################
title_size = 12
label_size = 12
fig_title =  'Elastic - N = '+str(N)+' - Reps = '+str(reps)
#
fig, ax = plt.subplots()
#
#ax.plot(v_array, P, yerr = sigma)
ax.set_xlabel('v', fontsize = label_size)
#ax.set_ylabel(r'$P(first_{t}/second_{t-1})$', fontsize = label_size)
ax.set_title(fig_title, fontsize = title_size)
ax.errorbar(v_array, P, yerr = sigma, label=r'$P(first_{t}/second_{t-1})$')
ax.plot(v_array, log_slope, label='Agg. reach. log slope')

plt.legend()
plt.tight_layout()
#
plt.show()

















