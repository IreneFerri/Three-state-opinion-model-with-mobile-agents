import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# DATA
alpha = 0.95
v = 0.1
filename_polar = 'time_polar_v'+str(v)+'_a'+str(alpha)+'.csv'
filename_neutral = 'time_neutral_v'+str(v)+'_a'+str(alpha)+'.csv'
filename_partial = 'time_partial_v'+str(v)+'_a'+str(alpha)+'.csv'
#################################################

# polar
temps_polar = pd.read_csv(filename_polar, usecols=[0], header=None).values
polar_dim = temps_polar.size
print(polar_dim)
temps_polar = temps_polar.reshape(polar_dim).tolist()
#
# partial
#temps_partial = pd.read_csv(filename_partial, usecols=[0], header=None).values
#partial_dim = temps_partial.size
#print(partial_dim)
#temps_partial = temps_partial.reshape(partial_dim).tolist()

# neutral
#temps_neutral = pd.read_csv(filename_neutral, usecols=[0], header=None).values
#neutral_dim = temps_neutral.size
#print(neutral_dim)
#temps_neutral = temps_neutral.reshape(neutral_dim).tolist()

#temps_polar = []
#temps_partial = []
temps_neutral  = []

print('POLAR TIMES = ', temps_polar)
print('NEUTRAL TIMES = ', temps_neutral)

# PLOT

label_size = 25
tick_size = 25
legend_size = 22

intervalos = range(0, max(temps_polar) + 2, 10) #calculamos los extremos de los intervalos

inter_ticks = range(0, max(temps_polar) + 2, 100) 

plt.rcParams["figure.figsize"] = (7, 6)
plt.hist(x=[temps_polar,temps_neutral], bins=intervalos, rwidth=1, label = ['global polar', 'global neutral', 'local'])
#plot.title('Temps de consens (total o parcial) v = 0.16, a = 1.0. MODEL NO ACOMULATIU, F = 0')
plt.xlabel('Consensus time (steps)', fontsize=label_size)
plt.ylabel('Number of realizations', fontsize=label_size)
plt.xticks(inter_ticks, fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=legend_size)
plt.text(100, 9, 'c)', fontsize=label_size)
plt.text(400, 7, r'$v = $'+str(v), fontsize=label_size)
plt.text(400, 6, r'$\alpha = $'+str(alpha), fontsize=label_size)

plt.tight_layout()
plt.show() #dibujamos el histograma
