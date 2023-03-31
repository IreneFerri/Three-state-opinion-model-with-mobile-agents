import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# DATA
alpha = 1.2
v = 0.1
filename_polar = 'time_polar_v'+str(v)+'_a'+str(alpha)+'.csv'
filename_neutral = 'time_neutral_v'+str(v)+'_a'+str(alpha)+'.csv'
filename_partial_polar = 'time_partial_polar_v'+str(v)+'_a'+str(alpha)+'.csv'
filename_partial_neutral = 'time_partial_neutral_v'+str(v)+'_a'+str(alpha)+'.csv'
#################################################

# polar
temps_polar = pd.read_csv(filename_polar, usecols=[0], header=None).values
polar_dim = temps_polar.size
print(polar_dim)
temps_polar = temps_polar.reshape(polar_dim).tolist()
#
# partial polar
#temps_partial_polar = pd.read_csv(filename_partial_polar, usecols=[0], header=None).values
#partial_polar_dim = temps_partial_polar.size
#print(partial_polar_dim)
#temps_partial_polar = temps_partial_polar.reshape(partial_polar_dim).tolist()

# partial neutral
temps_partial_neutral = pd.read_csv(filename_partial_neutral, usecols=[0], header=None).values
partial_neutral_dim = temps_partial_neutral.size
print(partial_neutral_dim)
temps_partial_neutral = temps_partial_neutral.reshape(partial_neutral_dim).tolist()


# neutral
temps_neutral = pd.read_csv(filename_neutral, usecols=[0], header=None).values
neutral_dim = temps_neutral.size
print(neutral_dim)
temps_neutral = temps_neutral.reshape(neutral_dim).tolist()

#temps_polar = []
temps_partial_polar = []
#temps_partial_neutral = []
#temps_neutral  = []


# PLOT

label_size = 25
tick_size = 25
legend_size = 22

inter_max = max(max(temps_partial_polar, default=0), max(temps_partial_neutral, default=0), max(temps_neutral, default=0), max(temps_polar, default=0))
print('inter_max = ', inter_max)

intervalos = range(0,  inter_max+ 2, 50) #calculamos los extremos de los intervalos

inter_ticks = range(0, inter_max + 2, 500) 

plt.rcParams["figure.figsize"] = (7, 6)
plt.hist(x=[temps_polar,temps_neutral, temps_partial_polar, temps_partial_neutral], bins=intervalos, rwidth=1, label = ['global polar', 'global neutral', 'local polar', 'local'])
#plot.title('Temps de consens (total o parcial) v = 0.16, a = 1.0. MODEL NO ACOMULATIU, F = 0')
plt.xlabel('Consensus time (steps)', fontsize=label_size)
plt.ylabel('Number of realizations', fontsize=label_size)
plt.xticks(inter_ticks, fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=legend_size)
plt.text(380, 36, 'd)', fontsize=label_size)
plt.text(1000, 20, r'$v = $'+str(v), fontsize=label_size)
plt.text(1000, 16, r'$\alpha = $'+str(alpha), fontsize=label_size)

plt.tight_layout()
plt.show() #dibujamos el histograma
