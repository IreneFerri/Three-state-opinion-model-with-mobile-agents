##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
###############################################################
###############################################################
reps =999
length = reps
N=155
alpha=1.0
alpha2 = alpha*alpha
#alpha = '{:.2f}'.format(alpha) 
print(alpha)
alpha_num = float(alpha)
Nb = 30  # number of bins for the histogram
###############################################################
# READ THE DATA
###############################################################
###############################################################
k = 4.865384615384615
temp = 0.05
#################################
filename = 'RW_res_T'+str(temp)+'_N'+str(N)+'_a'+"%03.1f"%alpha+'_glauber_imc.csv' 
print(filename)
###############################################################
number = pd.read_csv(filename, delim_whitespace = True,header=1,   usecols = [0]).values.T.reshape(length)
N = int(number[0])
links = pd.read_csv(filename, delim_whitespace = True,header=1,   usecols = [1]).values.T.reshape(length)
L = int(links[0])
ave_k = L/N
print(N, L, ave_k)
###############################################################
n1 = pd.read_csv(filename, delim_whitespace = True,header=1,   usecols = [4]).values.T.reshape(length)
n_1 = pd.read_csv(filename, delim_whitespace = True,header=1,   usecols = [6]).values.T.reshape(length)
magnet = abs(n1 - n_1)
print(magnet)
m_max = np.amax(magnet)
#
#################################
#################################

################################################################
#   PLOT LINES
################################################################
#
fig, ax = plt.subplots()
fig_name = '../../../Documents/Doc/Recerca/Three-state-mobility/images/RW_m_N'+str(N)+'_a'+str(alpha)+'_glauber.pdf'
plt.title(r'$\alpha = $'+str(alpha), fontsize=15)
plt.xlabel(r'$|m|$', fontsize=15)
plt.ylabel(r'P($|m|$)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0, 0.8, 0.05))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_ylim([-0.02, 120])
#ax.set_xlim([-0.002, 0.20])
plt.tick_params(right=True, top=True)
#
m = np.arange(m_max+2)
np.histogram(magnet, m)
#hist, bins = np.histogram(residual, bins= Nb, range=(0.0, E_res_max))
hist, bins = np.histogram(magnet, m)
hist= hist.astype(np.float32) 
hist = hist/reps
amplada_bin = (np.amax(bins)/Nb)  # Normalize by bin width
#hist = hist/amplada_bin
#E_res = np.arange(E_res_max+1)
m = m/(N*ave_k)
hist = hist * (N*ave_k) # Chain rule 
bins = bins/(N*ave_k) # Energy per particle and per link 
#rel_min = abs((ground/(N*ave_k)) +  abs(ground - alpha2*ground)/(N*ave_k))
#if (alpha_num > 1):
#  rel_min = abs(ground - ground/alpha2)/(N*ave_k)/1
#else:
#  rel_min = abs(ground - alpha2*ground)/(N*ave_k)/2
#plt.axvline(x=rel_min,color='grey', linestyle='--', label='Min. Diff.')  # Plot a vertical dashed line on the difference between 0 minimum and +1/-1 minimum
hist = np.append(hist,[0])
#E_res = np.append(E_res,[0])
plt.plot(m, hist,  label=r'$\langle k \rangle$ = 2')
#

#plt.text(0.40, 5, '(a)', fontsize=15)
#plt.legend(fontsize=12)
plt.savefig(fig_name,format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()



################################################################
# PLOT  HISTOGRAM
###############################################################
#
#E_res = np.arange(E_res_max)
#np.histogram(residual, E_res)
#hist, bins = np.histogram(residual, E_res)
#hist = hist/1000
#print(hist)
#print(bins)
#
#plt.hist(hist,bins)
#plt.title("histogram") 
#plt.show()




