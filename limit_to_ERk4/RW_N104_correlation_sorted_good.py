#
#
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import ticker

#################################
# MAIN ##########################
###############################################################
N0 = 155
N = 104  # Number of agents/oscillators
alpha = 1.00
alpha_str = '1.00'  # Step parameter (it only affects the dynamics speed)
alpha_name = '1.00'
#filename_temp = 'RW_res_T0.05_N'+str(N0)+'_a'+str(alpha)+'_glauber_imc.csv'
#temp = pd.read_csv(filename_temp, delim_whitespace = True, header=None).to_numpy()[0]
temp=0.243
temperatura=str(temp)
reps = 1000  # Number of repetitions for statistics
length = reps # Input array size
net_name1 = 'z15'
net_name2 = 'RW_N104_a0.0'
com_file = net_name2+'.net'
fig_name = '../../Desktop/correlation/cij_'+net_name2+'_N'+str(N)+'_a'+alpha_name+'_T'+temperatura+'.png'  # Output name
###########################################################
#
# Orientations
##########################################################
G = nx.read_edgelist(com_file,nodetype=int) # Reads the community from the input file
N = G.number_of_nodes()
filename = 'orientations_'+net_name1+'_N'+str(N).zfill(4)+'_alpha'+"%04.2f"%alpha+'_T'+temperatura+'.csv' # Input file
node_list = pd.read_csv(filename, delim_whitespace = True, header=None).to_numpy()
#print(node_list[2,0])
output_file = 'correlations_sorted_N'+str(N).zfill(4)+'_alpha'+"%04.2f"%alpha+'.csv' # Output
print(filename)
#################################################################
###############################################################
# CALCULATE CORRELATIONS 
###############################################################
correlations_matrix = np.zeros((N, N))
for rep in range(reps):
  last0 = node_list[0, rep]
  last_1 = node_list[1, rep]
#
#  print(rep)
  nodes = node_list[2:, rep]
  for i in range(last_1):
    node = nodes[i]
    G.nodes[node]['orientation'] = -1
#
  for i in range(last_1, last0):
    node = nodes[i]
    G.nodes[node]['orientation'] = 0
#
  for i in range(last0, N):
    node = nodes[i]
    G.nodes[node]['orientation'] = 1
#
# For every pair of links (includen repeated pairs, so it will be a symmetrix matrix)
  for i in range(1,N+1):  
    for j in range(1,N+1):
      if (G.nodes[i]['orientation'] == G.nodes[j]['orientation']):
        correlations_matrix[i-1, j-1] =  correlations_matrix[i-1, j-1]+ 1.0   # Sum 1 if the last configuration is the same for both i and j
      else:
        correlations_matrix[i-1, j-1] =  correlations_matrix[i-1, j-1]- 1.0   # Sume -1 otherwise
#        print(rep, i, j, correlations_matrix[i-1, j-1])
#print(correlations_matrix)
correlations_matrix = correlations_matrix/float(reps)   # Normalize by the number of repetitions
#print(correlations_matrix)

########################################################
# SORTING nodes
########################################################

#print(correlations_matrix[0,:])

#print(correlations_matrix)
inds = np.argsort(correlations_matrix[0,:])   # Sort the indexes so the values of the row 0 goes from minimum to maximum
correlations_matrix = correlations_matrix[inds][:,inds]   # Rearrange rows and columns accordingly --> this finds the largest community
last = N-1  
#  Ignore the largest community and repeat the sorting process in the remaining box until getting the last node (which will be obviously correlated to himself)
for iteration in range(N):
  maximum = np.where(correlations_matrix[last,:]==1.0)[0][0]  
#  print('max = ', maximum)
  if (maximum == 0):    # When you get the last node, break the loop
    break
  inds = np.argsort(correlations_matrix[maximum-1,0:maximum])
  rest_inds = np.arange(maximum, N, 1)
  inds = np.concatenate((inds, rest_inds))
  correlations_matrix = correlations_matrix[inds][:,inds]
  last  = maximum-1

file=open(output_file,'w')
for line in correlations_matrix:
  np.savetxt(file, line, fmt='%.6f')
file.close()
###############################################################
#  PLOT  HEATMAP SQUARE
###############################################################
fig_name = '../../Desktop/correlation/cij_'+net_name2+'_N'+str(N)+'_a'+alpha_name+'.png'  # Output name
#
label_fontsize = 25
interpolation = 100  # number of interpolation points

#                     vmax=zmax, vmin=zmin)
CS = plt.imshow(correlations_matrix,cmap='hot',interpolation='nearest', vmin=-1.0, vmax=1.0)
# COLORBAR ----------------------
cb = plt.colorbar(CS)
cb.set_label(label=r'$\langle c_{ij} \rangle$',size=25,rotation=90 )
cb.ax.tick_params(labelsize='25')
tick_locator = ticker.MaxNLocator(nbins=7)
cb.locator = tick_locator
cb.update_ticks()
#
#Title and axis ------------------------
#
title_name = r'$\alpha =$  '+str(alpha)
plt.title(title_name,fontsize=25)
plt.xlabel(r'$i$',fontsize=25)
plt.ylabel(r'$j$',fontsize=25)
plt.text(17, 20, '(d)', fontsize=20)
#
# Tics -----------------------------
plt.yticks(np.arange(0, 110, 20), fontsize=25)
plt.xticks(np.arange(0, 110, 20), fontsize=25)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)
#
plt.tight_layout()
plt.show()
plt.legend()
#plt.savefig(fig_name, format='png', dpi=1200)  # Save the figure




