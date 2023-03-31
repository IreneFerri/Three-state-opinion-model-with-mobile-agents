# program to plot the heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d

valors = np.linspace(0.8, 0.95, num=16)
valors = ["{:.2f}".format(x) for x in valors]
vs = np.linspace(0.06375, 0.28875, num=21)
vs = ["{:.5f}".format(x) for x in vs]

d = {}


for alpha in valors:
    filename = 'res0.06375_0.3_'+str(alpha)+'.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        neutres = 0
        for element in col:
            if element == 0:
                neutres += 1
        i += 1
        percent.append(neutres/len(col))
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d, index=vs, columns=valors,)
print(df)


h= sns.heatmap(df)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Fraction of neutral final configurations')
plt.show()

fig, ax = plt.subplots()
df_smooth = gaussian_filter(df, sigma=3)
h = sns.heatmap(df_smooth, vmax=1)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.ylabel("Velocity ")
plt.xticks(rotation = 'vertical')
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Fraction of neutral final configurations')
plt.show()



fig, ax = plt.subplots()
hm = plt.imshow(df, cmap='Pastel1_r',interpolation="spline16")
plt.colorbar(hm)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xticks(rotation = 'vertical')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Fraction of neutral final configurations')
plt.show()
