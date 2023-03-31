#
#
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import networkx as nx
import os
###############################################################
N = 155
temp = 0.05
reps = 100
length = reps 
################################################################
#################################################################
# alpha = 0.00
################################################################
alpha = 0.00
#
####################################################################################
#velocities = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.41, 0.45])
velocities = np.array([0.01, 0.05, 0.07, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
#velocities = np.array([0.07, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.45])
velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5])
v_dim = velocities.size
fails = np.zeros(v_dim)
for v in range (v_dim):
  filename = 'RW_vctt_a'+str(alpha)+'_data/fails_RW_N'+str(N)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+str(velocities[v])+'.csv'
  if (os.stat(filename).st_size == 0):
    fails[v] = 0
  else:
    fail = pd.read_csv(filename, delim_whitespace = True, header=None,  usecols = [0]).values.T
    fails[v] = fail.size

print(fails)

#################################################################
####################################################################################

###########################################################
#  PLOT
###########################################################
#columns = [ratio_a0_005 ,ratio_a0_01,ratio_a0_05,ratio_a0_1,ratio_a0_2]
tick_size = 12
label_size = 15
#columns = times 
fig, ax = plt.subplots()
#ax.boxplot(columns, meanline=True, showmeans=True, widths=0.75)
tick_labels = velocities.tolist()
ax.bar(velocities, fails, label=tick_labels, width=0.01)
#ax.scatter(velocities, fails)

plt.title('Reps = '+ str(reps)+r' - max. steps = 50·$N_0$'+r' - ($N_0$ = '+str(N)+')')
plt.ticklabel_format(axis="y", style="plain", scilimits=(0,0))
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
#plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.41, 0.45 ], fontsize=15)
tick_list = np.arange(1,v_dim+1).tolist()
#tick_labels = velocities.tolist()
#print(tick_list)
#print(tick_labels)
#plt.xticks(tick_list, tick_labels, fontsize=tick_size)
#plt.yticks(fontsize=tick_size)
plt.xlabel(r'$v$',fontsize=label_size)
plt.ylabel('number of non-consent runs',fontsize=label_size)

plt.ylim(0, 35)

#ax.text(1, 0.3, r'$N = 200$', fontsize=12)
ax.text(0.1, 27, r'$\alpha = $'+str(alpha), fontsize=label_size)
#ax.text(1, 0.1, r'$Reps. = 20$', fontsize=12)

#ax.legend()
plt.tight_layout()
plt.show()

##########################################################################
#  PRINT RESULTS IN AN EXTERNAL FILE FOR Gnuplot
##########################################################################

#file=open('plot_traj_N'+str(N).zfill(4)+'_d'+str(d)+'00_a'+str(alpha)+'.csv', 'w')
#file.write('# t_ref   pos x   pox y   color cluster' + '\n')
#for i in range (time_stab):
#  for j in range(N):
#    print(i, positions_x[j, i], positions_y[j,i], colors[j,i]+1)
#    file.write(str(i)+ ' '+str(positions_x[j, i])+' '+str(positions_y[j,i])+' '+str(colors[j,i]+1)+ '\n')
#
####################################################################################
####################################################################################
 ##  PLOT THE GRAPH ###################
 ##############################################

# Plot the orientation of the last config
#pos=nx.spring_layout(G0)
#for i in range (N):
#  if G.nodes[i]['orientation'] == -1:   # RED
#    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='#fc0202',node_size=5,width=0.1)
#  elif G.nodes[i]['orientation'] == 1:    # BLUE
#    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='#0100ff',node_size=5,width=0.1)
#  else:                           # BLACK
#    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='#000000',node_size=5,width=0.1) 
#nx.draw_networkx_nodes(G0,pos,node_size=20) 
#nx.draw_networkx_edges(G0,pos,width=0.5)
#plt.show()
##############################################
#pos=nx.spring_layout(G)
#nx.draw_networkx_nodes(G,pos,node_size=20)
#nx.draw_networkx_edges(G,pos,width=0.5)
#plt.show()




