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
nsteps = 100
reps = 100
tol = 0.0000000001
#
v_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])

v_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014,  0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])


print(v_array)
dim_v = v_array.size
##################################################
satur_time = np.zeros(dim_v)
log_slope = np.zeros(dim_v)
#sigma = np.zeros(dim_v)
for v in range(dim_v):
  filename = 'reachab_normal_1_N'+str(N0)+'_a_v'+"%5.3f"%v_array[v]+'.csv'
#  print(v_array[v])
  steps_array = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [0]).values.reshape(nsteps)
  reach = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [1]).values.reshape(nsteps)
  dim_reach = reach.size
  for step in range(2, dim_reach-1):
    if (abs(reach[step] - reach[step+1]) < tol):
      satur_time[v] = step
#  P[v] = np.mean(probs)
#  sigma[v] = np.std(probs)
  step_adjust = steps_array[1:7] 
  adjust = reach[1:7]
#  print(step_adjust, adjust)
  print(np.polyfit(np.log(step_adjust), adjust, 1))
  log_slope[v] = np.polyfit(np.log(step_adjust), adjust, 1)[0] 
print(satur_time)

#####################################################
# PLOT 
##################################################
title_size = 12
label_size = 12
fig_title =  'N = '+str(N)+' - Steps = '+str(nsteps)+' - Reps = '+str(reps)
#
fig, ax = plt.subplots()
#
#ax.plot(v_array, P, yerr = sigma)
ax.set_xlabel('v', fontsize = label_size)

#ax.set_ylabel('sat. time (steps)', fontsize = label_size)
ax.set_ylabel('Agg. reach. log slope', fontsize = label_size)

ax.set_title(fig_title, fontsize = title_size)

#ax.plot(v_array, satur_time)
ax.plot(v_array, log_slope)
#ax.errorbar(v_array, P, yerr = sigma)
#
plt.show()








