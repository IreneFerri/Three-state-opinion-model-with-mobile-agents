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


for alpha in valors:
    print(alpha)
    filename = 'res0.05_0.3_'+str(alpha)+'.csv'
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

label_size = 12

#h= sns.heatmap(df, vmin = 0, vmax = 1)
h = plt.imshow(df, cmap='Pastel1_r',interpolation="bicubic", aspect = 1.5)
cbar = plt.colorbar(h)
cbar.set_label(r'$\langle n_0 \rangle$')
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.ylabel("Velocity ", fontsize = label_size)
plt.yticks(rotation = 'horizontal')
plt.xticks(rotation = 'vertical')
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.gca().invert_yaxis()
#plt.title('Fraction of neutral agents')

plt.tight_layout()
plt.show()

for alpha in valors:
    filename = 'RW_N100_a' + str(alpha)+ '_T0.05_vmin0.05_vmax0.3.csv'
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

#h= sns.heatmap(df, vmin = 0, vmax = 1)

#h = plt.imshow(df, cmap='Pastel1_r',interpolation="bicubic")
#h.set(xlabel='Alpha', ylabel='Velocity')
#plt.ylabel("Velocity ")
#plt.yticks(rotation = 'horizontal')
#plt.xlabel("Alpha ")
#plt.gca().invert_yaxis()
#plt.title('Magnetization after consent (partial or total), amb factor 0')
#plt.show()
