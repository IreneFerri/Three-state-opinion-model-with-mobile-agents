import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
my_seed = 1957
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
alpha = 0.00
time_array = np.zeros(dim_vel)
sigma_time = np.zeros(dim_vel)
fails = np.zeros(dim_vel)
for v in range(dim_vel):
  print(velocities[v])
  filename = 'RW_vctt_elastic_steps350N0_data/RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+"%4.2f"%velocities[v]+'.csv'
  filename_fails = 'RW_vctt_elastic_steps350N0_data/fails_RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+"%4.2f"%velocities[v]+'.csv'
  time = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [1]).values.T.reshape(reps)
  time = time[~np.isnan(time)]  # ~ logical-not operator
  print(time)
  time_array[v] = np.mean(time)
  sigma_time[v] = np.std(time)
#
#
  if (os.stat(filename_fails).st_size == 0):
    fails[v] = 0
  else:
    fail = pd.read_csv(filename_fails, delim_whitespace = True, header=None,  usecols = [0]).values.T
    fails[v] = fail.size  
# threshold
threshold = 0
for v in range(dim_v):
  if (fails[v] < 0.1*reps): # if 90% of the runs survive
    threshold = v
    break

#####################################################
# Agg Reach
##################################################
nsteps_agg = 100

v_agg_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014,  0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])


v_agg_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48])
valors = [0.001, 0.005, 0.01,  0.05, 0.10, 0.21, 0.31]

print(v_agg_array)
dim_v = v_agg_array.size
##################################################
satur_time = np.zeros(dim_v)
satur_reach = np.zeros(dim_v)
log_slope = np.zeros(dim_v)
reachability = np.zeros((len(valors), nsteps_agg))
#sigma = np.zeros(dim_v)
val = 0
for v in range(dim_v):
  filename = '../../agg_reach_elastic/reachab_normal_1_N'+str(N0)+'_a_v'+"%5.3f"%v_agg_array[v]+'.csv'
  steps_array = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [0]).values.reshape(nsteps_agg)
  reach = pd.read_csv(filename, delim_whitespace = True, header=None ,  usecols = [1]).values.reshape(nsteps_agg)
  dim_reach = reach.size
  for step in range(2, dim_reach-1):
    if (abs(reach[step] - reach[step+1]) < tol):
      satur_time[v] = step
  satur_reach[v] = reach[-1]
# Reachability --------
  if (v_agg_array[v] in valors):
    reachability[val, :] = reach
    val = val + 1
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
# Populations

v = 0.1
filename = '../../dist_pairs_vs_time/distances_RW_N'+str(N0)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+str(v)+'_s'+str(my_seed)+'_elastic.csv'
num1 = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [3]).values.T
num0 = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [4]).values.T
num_1 = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [5]).values.T
nsteps = num1.size
steps = np.arange(nsteps)
num1 = num1.reshape(nsteps)
num0 = num0.reshape(nsteps)
num_1 = num_1.reshape(nsteps)

#####################################################
# PLOT 
##################################################
title_size = 10
label_size = 12
tick_size = 10
fig_title =  'N = '+str(N)+' - Steps = 350*155/10*155/100'+' - Reps = '+str(reps)
#
fig, ax = plt.subplots(1, 3, figsize=(14,3.7), constrained_layout=True)
#spec = fig.add_gridspec(ncols=3, nrows=1)

#
# Plot reachability log-lin  **********************************
#
ax[0].set_xscale('log')
ax[0].set_xlabel('Time (steps)', fontsize = label_size)
ax[0].set_ylabel('Agregated reachability', fontsize = label_size)
ax[0].tick_params(axis='both', labelsize = tick_size , right=True)
ax[0].text(2.0, 0.65, r'a)', fontsize=label_size)
for i in range(len(valors)):
 ax[0].plot(steps_array, reachability[i,:], label=valors[i]) 

ax[0].legend(fontsize=tick_size, loc='lower right')
# ***************************************************************************
#
# Plot Consensus time + satur agg. reach. + fails ***************
#
ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
#
#ax.plot(v_array, P, yerr = sigma)
ax[1].set_xlabel('v', fontsize = label_size)
#ax[1].set_ylabel(r'$P(first_{t}/second_{t-1})$', fontsize = label_size)
#ax[1].set_title(fig_title, fontsize = title_size)
#
ax[1].text(0.15, 3000, r'b)', fontsize=label_size)
#
#ax[1].set_ylabel('consensus time (steps)', color = 'darkorange', fontsize = label_size)
ax[1].set_ylabel('Consensus time (steps)', color = 'black', fontsize = label_size)
ax[1].tick_params(axis='y', labelsize = tick_size)
ax[1].set_ylim(0, 17000)
#ax2.errorbar(v_array, P, yerr = sigma , color = 'black', label=r'$P(first_{t}/second_{t-1})$')
#ax2.plot(v_agg_array, log_slope, label='Agg. reach. log slope')
ax2.plot(v_agg_array, satur_reach, label='Agg. reach. saturation')
ax[1].errorbar(velocities[threshold:], time_array[threshold:], yerr = sigma_time[threshold:], color = 'darkorange', label='consensus time')
ax2.tick_params(axis='both', labelsize = tick_size)
ax[1].tick_params(axis='both', labelsize = tick_size)
#
tick_labels = velocities.tolist()
fails = fails/reps
ax2.bar(velocities, fails, width=0.01, color='red', alpha = 0.5, label='fails')
#
#ax2.set_ylabel(r'$P(first_{t}/second_{t-1})$')
ax[1].legend(fontsize=tick_size, loc='center right')
ax2.legend(fontsize=tick_size, loc='upper right')
#ax2.legend(fontsize=tick_size, loc='best')
#
# ************************************************************
# Plot Populations v = 0.1, seed = 1957 
#
ax[2].set_xlabel('Time (steps)', fontsize = label_size)
ax[2].set_ylabel(r'Opinion state fraction', fontsize = label_size)
ax[2].tick_params(axis='both', labelsize = tick_size, right=True)
ax[2].text(0.15, 0.8, r'c)', fontsize=label_size)
#
ax[2].plot(steps, num1/N, label = 'rightists')
ax[2].plot(steps, num0/N, label = 'centrists')
ax[2].plot(steps, num_1/N, label = 'leftists')

ax[2].legend(fontsize=tick_size, loc='center right')
# ***************************************************************************

#fig.tight_layout(pad=20.0)


plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.8,
                    top=0.9,
                    wspace=0.9,
                    hspace=0.9)
#ax.update(wspace=0.5, hspace=0.5)
fig.tight_layout()
plt.show()

















