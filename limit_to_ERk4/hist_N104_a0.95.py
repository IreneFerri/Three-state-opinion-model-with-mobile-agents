##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
###############################################################
###############################################################
reps = 10000
length = reps
N=155
alpha=0.95
alpha2 = alpha*alpha
#alpha = '{:.2f}'.format(alpha) 
print(alpha)
alpha_num = float(alpha)
Nb = 200  # number of bins for the histogram
###############################################################
# READ THE DATA
###############################################################
###############################################################
k = 4.865384615384615
temp = 0.05
#################################
filename = 'RW_res_T'+str(temp)+'_N'+str(N)+'_a'+str(alpha)+'_glauber_imc.csv' 
###############################################################
number = pd.read_csv(filename, delim_whitespace = True,header=None,   usecols = [0]).values.T.reshape(length)
N = int(number[0])
links = pd.read_csv(filename, delim_whitespace = True,header=None,   usecols = [1]).values.T.reshape(length)
L = int(links[0])
ave_k = L/N
print(N, L, ave_k)
###############################################################
energies = pd.read_csv(filename, delim_whitespace = True,header=None,   usecols = [3]).values.T.reshape(length)
#
ground = -(N*ave_k*0.5) 
if (alpha_num > 1):
  ground = -(N*ave_k*0.5)*alpha2
print(ground)
residual= abs(ground - energies)
#for i in range (length):
#  print(residual[i])
#
E_res_max = np.amax(residual)
#################################
#################################

################################################################
#   PLOT LINES
################################################################
#
fig, ax = plt.subplots()
fig_name = '../../../Documents/Doc/Recerca/Three-state-mobility/images/ER_res_N'+str(N)+'_a'+str(alpha)+'_glauber.pdf'
plt.title(r'$\alpha = $'+str(alpha), fontsize=15)
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel(r'P($\epsilon$)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0, 0.8, 0.05))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_ylim([-0.02, 120])
#ax.set_xlim([-0.002, 0.20])
plt.tick_params(right=True, top=True)
#
E_res = np.arange(E_res_max+2)
np.histogram(residual, E_res)
#hist, bins = np.histogram(residual, bins= Nb, range=(0.0, E_res_max))
hist, bins = np.histogram(residual, E_res)
hist= hist.astype(np.float32) 
hist = hist/reps
amplada_bin = (np.amax(bins)/Nb)  # Normalize by bin width
#hist = hist/amplada_bin
#E_res = np.arange(E_res_max+1)
E_res = E_res/(N*ave_k)
hist = hist * (N*ave_k) # Chain rule 
bins = bins/(N*ave_k) # Energy per particle and per link 
#rel_min = abs((ground/(N*ave_k)) +  abs(ground - alpha2*ground)/(N*ave_k))
if (alpha_num > 1):
  rel_min = abs(ground - ground/alpha2)/(N*ave_k)/1
else:
  rel_min = abs(ground - alpha2*ground)/(N*ave_k)/2
#plt.axvline(x=rel_min,color='grey', linestyle='--', label='Min. Diff.')  # Plot a vertical dashed line on the difference between 0 minimum and +1/-1 minimum
hist = np.append(hist,[0])
#E_res = np.append(E_res,[0])
plt.plot(E_res, hist,  label=r'$\langle k \rangle$ = 2')
#

#plt.text(0.35, 5, '(b)', fontsize=15)
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




