# program to plot the heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

valors = np.linspace(0.9, 1.2, num=31)
valors = ["{:.2f}".format(x) for x in valors]
vs = np.linspace(0.05, 0.2875, num=20)
vs = ["{:.5f}".format(x) for x in vs]

d = {}

#PERCENTATGE DE NEUTRES

for alpha in valors:
    filename = 'res0.05_0.3_' + str(alpha) + '_f2.00.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        promig = np.mean(col)
        i += 1
        percent.append(promig)
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d)
print(df)


h= sns.heatmap(df, vmax = 1)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Fraction of neutral spins after consent (total or local), amb factor 2 ACUMULATIU')
plt.show()

#MAGNETITZACIO

for alpha in valors:
    filename = 'RW_N100_f2.00_a' + str(alpha)+ '_T0.05_vmin0.05_vmax0.3.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        promig = np.mean(col)
        i += 1
        percent.append(promig)
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d)
print(df)

h= sns.heatmap(df, vmax = 1)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Magnetization after consent (total or local), amb factor 2 ACUMULATIU')
plt.show()


#TAMANY CLUSTERS (NEUTRAL)


for alpha in valors:
    filename = 'size_neutral0.05_0.3_' + str(alpha) + '.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        try:
            promig = np.nanmean(col)
        except TypeError:
            promig = 0
        i += 1
        percent.append(promig)
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d)
print(df)

h= sns.heatmap(df)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Size of the biggest neutral cluster')
plt.show()


#TAMANY CLUSTERS (POLAR)


for alpha in valors:
    filename = 'size_polar0.05_0.3_' + str(alpha) + '.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        promig = np.nanmean(col)
        i += 1
        percent.append(promig)
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d)
print(df)

h= sns.heatmap(df)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Size of the biggest polar cluster')
plt.show()
