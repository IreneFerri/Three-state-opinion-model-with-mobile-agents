#
from __future__ import division
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
#
# ------------------------------------------------------------------------------
#   DATA
# ------------------------------------------------------------------------------
#
start=time.time()      # Start measuring time
com_file = 'RW_N104.net'
N = 104
temp = 0.05
net_name1 = 'z15'
net_name2 = 'RW_N104_a0.0'
#com_file = 'RW_N104_new.net'
G = nx.read_edgelist(com_file,nodetype=int) # Reads the community from the input file and creates the graph 
G = nx.relabel_nodes(G, lambda x: x - 1)  # Python indexing
number_nodes = G.number_of_nodes()
N = number_nodes
num_links = G.number_of_edges()
ave_k = 2*num_links/N
print('N = ', N, 'L = ', num_links, 'ave k = ', ave_k)
# -----------------------------------------------------------------------------

#  PLOTTING -------------------------------------------------------------------
#
label_fontsize = 13
tick_fontsize = 13
legend_fontsize = 13
node_s = 9
w = 0.5
edge_w = 0.3
#
fig, ax = plt.subplots(2, 2, figsize=(10,10))
#


# -------------------------------
# ax 2 correlation matrix alpha = 0
# -------------------------------
alpha = 0.0
filename = '../../../TFM_fortran/bimodular/mobility/correlations_sorted_N'+str(N).zfill(4)+'_alpha'+"%04.2f"%alpha+'.csv'
correlations_matrix = np.genfromtxt(filename, delimiter=' ')
print(correlations_matrix)
print('alpha = ', alpha,type(correlations_matrix), correlations_matrix.size, correlations_matrix.shape)
CS = ax[0,1].imshow(correlations_matrix,cmap='hot',interpolation='nearest', vmin=-1.0, vmax=1.0)
# COLORBAR ----------------------
cb = plt.colorbar(CS, ax =ax[0,1], shrink = 0.8, aspect=15)
cb.set_label(label=r'$\langle c_{ij} \rangle$',size=label_fontsize,rotation=90 )
cb.ax.tick_params(labelsize=legend_fontsize)
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator
cb.update_ticks()
#
#Title and axis ------------------------
#
title_name = r'$\alpha =$  '+str(alpha)
#ax[0,1].set_title(title_name,fontsize=tick_fontsize)
ax[0,1].set_xlabel(r'$i$',fontsize=label_fontsize)
ax[0,1].set_ylabel(r'$j$',fontsize=label_fontsize)
ax[0,1].text(17, 20, 'b)', fontsize=legend_fontsize)
ax[0,1].text(70, 20, r'$\alpha = 0.0$', fontsize=legend_fontsize)
#
# Tics -----------------------------
ax[0,1].set_yticks(np.arange(0, 110, 20), fontsize=tick_fontsize)
ax[0,1].set_xticks(np.arange(0, 110, 20), fontsize=tick_fontsize)
#plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)
ax[0,1].tick_params(axis='both', which='major', labelsize=tick_fontsize)


plt.show()
