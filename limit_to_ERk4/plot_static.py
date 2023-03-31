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
# ax 1 - netwrork drawing 
# -------------------------------


# Plot the orientation of the last config
#pos=nx.spring_layout(G)
pos={0: [0.6899885746697244, 0.6594213953861203], 1: [0.4338338208377638, 0.8826984063479036], 2: [0.40121851077199766, 0.5044728756351057], 3: [0.5165303927782083, 0.4580929328683002], 4: [0.07834532347900214, 0.46406420940380455], 5: [0.10944554442433918, 0.39103343837877436], 6: [0.09344171833768078, 0.40916508666143947], 7: [0.17694610893959373, 0.29923450100690174], 8: [0.14689512930213489, 0.14458142814714847], 9: [0.050186087470357355, 0.5221310681162682], 10: [0.058695267578365184, 0.2724340744132072], 11: [0.38025524588818804, 0.4921309724107549], 12: [0.276880594529488, 0.19851858498826302], 13: [0.11634566227852394, 0.5495560317767753], 14: [0.13157154307140217, 0.6399975882098868], 15: [0.4531941018650012, 0.44998736486071067], 16: [0.9898794737729372, 0.45116363879266175], 17: [0.13480778718547348, 0.7857671853870744], 18: [0.17105079819005287, 0.556868018971894], 19: [0.029300625652421863, 0.3368710685146492], 20: [0.4195535245744233, 0.1505766434599105], 21: [0.23389458193037183, 0.7181156767147072], 22: [0.3543255265398769, 0.708972958913924], 23: [0.7725750491039994, 0.70654105205855], 24: [0.0908445583797104, 0.3123222019881512], 25: [0.1333341312706674, 0.7275301764249812], 26: [0.9839193282472933, 0.725881693240489], 27: [0.15561927960423494, 0.42255964167570903], 28: [0.03714511239408458, 0.5297977367127713], 29: [0.8839090394511335, 0.7165190639875629], 30: [0.23379403936207954, 0.9183884371678132], 31: [0.22965067000938522, 0.0850870339373142], 32: [0.07994186107170809, 0.3245315276570896], 33: [0.08083551043792181, 0.30497565383694103], 34: [0.22387749439817817, 0.32260496563354724], 35: [0.3429938896502862, 0.8056431685801883], 36: [0.7404093859232722, 0.621431119572763], 37: [0.24969159831569576, 0.10845939956863726], 38: [0.13065906055786936, 0.2988454373222823], 39: [0.7857321432259196, 0.6595966382048526], 40: [0.4501878680271926, 0.502223007997167], 41: [0.8547379828636096, 0.4051987898342213], 42: [0.3295944369403756, 0.007640669614132567], 43: [0.44521310339009434, 0.9974916654981703], 44: [0.3262960208666674, 0.9260039737858663], 45: [0.808363799057692, 0.7033456232306958], 46: [0.17402586528841457, 0.5330484818467183], 47: [0.376234096834575, 0.9127867597141172], 48: [0.3767508153754864, 0.18455386116944017], 49: [0.879934689380538, 0.47843402582947836], 50: [0.3917093893711536, 0.9304282990848128], 51: [0.6953471344117034, 0.5179231238817492], 52: [0.6287252452832873, 0.48715601950011644], 53: [0.1290545030369885, 0.6981003662090879], 54: [0.14888181242048426, 0.4513150457341928], 55: [0.3626795368744259, 0.5911184540900021], 56: [0.0940675666928168, 0.7304475639687024], 57: [0.3924615370646416, 0.9165403195324199], 58: [0.31866551358028117, 0.7973745094584416], 59: [0.8689248440502974, 0.49033405557152143], 60: [0.22457270896567627, 0.507109832830531], 61: [0.9294833560941327, 0.3303555315064245], 62: [0.36516072449656745, 0.10745395986897277], 63: [0.8304241799582199, 0.5951893901128505], 64: [0.19535035136565293, 0.7838400315014332], 65: [0.3919446072473599, 0.2504913302952053], 66: [0.9075954663265657, 0.3172256923536324], 67: [0.7246703728980807, 0.5673755044022836], 68: [0.7060120619306324, 0.6126379868268055], 69: [0.1521563675201969, 0.3938403880592718], 70: [0.17779207217626047, 0.7905700440110711], 71: [0.285669839529478, 0.10848789178892815], 72: [0.1256348970136898, 0.6034562157036829], 73: [0.32793324680984837, 0.49031877712393546], 74: [0.22795072822358442, 0.9461408135914849], 75: [0.22531452403584462, 0.40043634194996647], 76: [0.40780248866972835, 0.2707833226298343], 77: [0.3810377752224866, 0.6447917566006707], 78: [0.6360526086589385, 0.5934688947004967], 79: [0.3594599698910489, 0.7234010259888507], 80: [0.27575799776000165, 0.24152346698239424], 81: [0.9667279922233492, 0.33498701453863744], 82: [0.49349599964511237, 0.9728738230136885], 83: [0.9373081141579397, 0.7050675498781801], 84: [0.6242403690337855, 0.6048272133083524], 85: [0.9347970894422881, 0.4210469559257052], 86: [0.5563351571692577, 0.452940226149948], 87: [0.3027759651035019, 0.0440170687703062], 88: [0.22865191895989467, 0.15047965697445886], 89: [0.19703346492713047, 0.28784704929565663], 90: [0.3838569685791703, 0.3417620711814022], 91: [0.3961839847102432, 0.10127989317545827], 92: [0.8431806173877292, 0.5241768603356255], 93: [0.2501064968684964, 0.06211309010480992], 94: [0.2879499910078168, 0.14165321949743415], 95: [0.2180564968136225, 0.860126306968937], 96: [0.782593303014222, 0.6387869466185363], 97: [0.10995577987590499, 0.33921716903864185], 98: [0.6837082978312394, 0.6543216094537153], 99: [0.17076991663166763, 0.6015335694223616], 100: [0.023515322994652643, 0.3999866127060878], 101: [0.8040666326631853, 0.48318758299516507], 102: [0.22805437900980385, 0.7838503925240745], 103: [0.8686499705507889, 0.3743980581342642], 104: [0.8048086801992508, 0.2548901763790764]}
#

nodelist = np.arange(N).tolist()
for i in range (N):
  nx.draw_networkx_nodes(G,pos=pos, ax=ax[0,0],nodelist=[i], node_color='grey', node_size=node_s).set_edgecolor('black')

nx.draw_networkx_edges(G,pos,ax=ax[0,0] ,width=edge_w)
ax[0,0].text(0.85, 0.85, 'a)', fontsize=legend_fontsize)



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
# -------------------------------
# ax 3 - magnet, neutrals vs alpha 
# -------------------------------
filename = '../../../TFM_fortran/bimodular/mobility/magnet_neutrals_vs_alpha/RW_N'+str(N)+'_T'+str(temp)+'.csv'
#
alpha_array = pd.read_csv(filename, delim_whitespace = True, header=None, usecols=[2]).to_numpy()
a_dim = alpha_array.size
alpha_array = alpha_array.reshape(a_dim)
magne_array = pd.read_csv(filename, delim_whitespace = True, header=None, usecols=[8]).to_numpy().reshape(a_dim)
magne_var = pd.read_csv(filename, delim_whitespace = True, header=None, usecols=[12]).to_numpy().reshape(a_dim)
n0_array = pd.read_csv(filename, delim_whitespace = True, header=None, usecols=[5]).to_numpy().reshape(a_dim)
n0_var = pd.read_csv(filename, delim_whitespace = True, header=None, usecols=[9]).to_numpy().reshape(a_dim)


ax[1,0].errorbar(alpha_array, magne_array/N, (magne_var)**0.5/N, label=r'$\langle |m| \rangle$')
ax[1,0].errorbar(alpha_array, n0_array/N, (n0_var)**0.5/N, label=r'$\langle n_{0} \rangle$')

ax[1,0].set_xlabel(r'$\alpha$',fontsize=label_fontsize)
ax[1,0].set_xticks(np.arange(0, 2.1, 0.4), fontsize=tick_fontsize)
ax[1,0].set_yticks(np.arange(0, 1.1, 0.2), fontsize=tick_fontsize)
ax[1,0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

ax[1,0].text(0.1, 0.85, 'c)', fontsize=legend_fontsize)
ax[1,0].legend(fontsize=legend_fontsize, loc='center right')
# -------------------------------
# ax 4 correlation matrix alpha = 0.75
# -------------------------------

alpha = 0.80
filename = '../../../TFM_fortran/bimodular/mobility/correlations_sorted_N'+str(N).zfill(4)+'_alpha'+"%04.2f"%alpha+'.csv'
correlations_matrix2 = np.genfromtxt(filename, delimiter=' ')
print(correlations_matrix2)
print('alpha = ', alpha, type(correlations_matrix2), correlations_matrix2.size, correlations_matrix2.shape)
CS2 = ax[1,1].imshow(correlations_matrix2,cmap='hot',interpolation='nearest', vmin=-1.0, vmax=1.0)
# COLORBAR ----------------------
cb2 = plt.colorbar(CS2, ax =ax[1,1], shrink = 0.8, aspect=15)
cb2.set_label(label=r'$\langle c_{ij} \rangle$',size=label_fontsize,rotation=90 )
cb2.ax.tick_params(labelsize=legend_fontsize)
tick_locator = ticker.MaxNLocator(nbins=8)
cb2.locator = tick_locator
cb2.update_ticks()
#
#Title and axis ------------------------
#
title_name = r'$\alpha =$  '+str(alpha)
#ax[0,1].set_title(title_name,fontsize=tick_fontsize)
ax[1,1].set_xlabel(r'$i$',fontsize=label_fontsize)
ax[1,1].set_ylabel(r'$j$',fontsize=label_fontsize)
ax[1,1].text(20, 24, 'd)', fontsize=legend_fontsize)
ax[1,1].text(70, 24, r'$\alpha = $'+str(alpha), fontsize=legend_fontsize)
#
# Tics -----------------------------
ax[1,1].set_yticks(np.arange(0, 110, 20), fontsize=tick_fontsize)
ax[1,1].set_xticks(np.arange(0, 110, 20), fontsize=tick_fontsize)
ax[1,1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
#plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)



plt.show()
