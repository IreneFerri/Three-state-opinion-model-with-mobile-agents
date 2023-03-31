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
nsteps = N0*350
reps = 100
tol = 0.0000000001
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
  filename = '../../neighbors/second_neigh_RW_N'+str(N0)+'_v'+"%5.3f"%v_array[v]+'.csv'
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
  filename = 'RW_vctt_elastic_steps350N0_data/RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+"%4.2f"%velocities[v]+'.csv'
  time = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [1]).values.T.reshape(reps)
  time = time[~np.isnan(time)]  # ~ logical-not operator
  print(time)
  time_array[v] = np.mean(time)
  sigma_time[v] = np.std(time)
#

#####################################################
# Agg Reach
##################################################
nsteps_agg = 100

v_agg_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014,  0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])


v_agg_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])


print(v_agg_array)
dim_v = v_agg_array.size
##################################################
satur_time = np.zeros(dim_v)
satur_reach = np.zeros(dim_v)
log_slope = np.zeros(dim_v)
#sigma = np.zeros(dim_v)
for v in range(dim_v):
  filename = '../../agg_reach_elastic/reachab_normal_1_N'+str(N0)+'_a_v'+"%5.3f"%v_agg_array[v]+'.csv'
#  print(v_agg_array[v])
  steps_array = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [0]).values.reshape(nsteps_agg)
  reach = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [1]).values.reshape(nsteps_agg)
  dim_reach = reach.size
  for step in range(2, dim_reach-1):
    if (abs(reach[step] - reach[step+1]) < tol):
      satur_time[v] = step
  satur_reach[v] = reach[-1]
#  P[v] = np.mean(probs)
#  sigma[v] = np.std(probs)
  step_adjust = steps_array[1:7]
  adjust = reach[1:7]
#  print(step_adjust, adjust)
  print(np.polyfit(np.log(step_adjust), adjust, 1))
  log_slope[v] = np.polyfit(np.log(step_adjust), adjust, 1)[0]
print(satur_time)
print(satur_reach)



#####################################################
# PLOT 
##################################################
title_size = 12
label_size = 12
fig_title =  'N = '+str(N)+' - Steps = 350*155/10*155/100'+' - Reps = '+str(reps)
#
fig, ax = plt.subplots()
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
#
#ax.plot(v_array, P, yerr = sigma)
ax.set_xlabel('v', fontsize = label_size)
#ax.set_ylabel(r'$P(first_{t}/second_{t-1})$', fontsize = label_size)
ax.set_title(fig_title, fontsize = title_size)
#
ax.text(0.15, 3000, r'$\alpha = $'+str(alpha), fontsize=label_size)
#
ax.set_ylabel('consensus time', color = 'darkorange')
ax.tick_params(axis='y')
ax2.errorbar(v_array, P, yerr = sigma , color = 'black', label=r'$P(first_{t}/second_{t-1})$')
#ax2.plot(v_agg_array, log_slope, label='Agg. reach. log slope')
ax2.plot(v_agg_array, satur_reach, label='Agg. reach. saturation')
ax.errorbar(velocities, time_array, yerr = sigma_time, color = 'darkorange')
ax2.tick_params(axis='y')
#
line1_x = 0.0
for v in range(dim_v):
  if (satur_reach[v] > 0.999):
    line1_x = v_agg_array[v]
    break
#plt.axvline(line1_x, ls='--', color='grey')
#
#ax2.set_ylabel(r'$P(first_{t}/second_{t-1})$')
#
plt.legend()
plt.tight_layout()
plt.show()

















