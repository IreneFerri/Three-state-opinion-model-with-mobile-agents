# program to plot the heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata
from matplotlib import ticker


valors = np.linspace(0.77, 1.05, num=24)
valors = ["{:.2f}".format(x) for x in valors]
vs = np.linspace(0.06375, 0.28875, num=21)
#vs = np.linspace(0.06375, 0.49125, num=39)
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

label_size = 20
tick_size = 20
title_size = 20
#

valors_array = np.array(valors).T.astype(float)
vs_array = np.array(vs).astype(float)
print(type(valors_array))
val_dim = valors_array.size
vs_dim = vs_array.size
dim = val_dim*vs_dim
valors_matrix = np.concatenate((valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array), axis=0).reshape(dim)
vs_matrix = np.zeros(dim)



for j in range (vs_dim):
  for i in range(j*val_dim, j*val_dim+val_dim):
    vs_matrix[i] = vs_array[j].T
print('dim  ', dim, 'size = ', df.shape, valors_matrix.shape, vs_matrix.shape)

df  = df.to_numpy().reshape(dim)
xi = np.linspace(valors_array.min(),valors_array.max(),1000)
yi = np.linspace(vs_array.min(),vs_array.max(),1000)
zi = griddata((valors_matrix, vs_matrix), df, (xi[None,:], yi[:,None]), method='linear')

zmin = df.min()
zmax = df.max()
interpolation = 100
CS = plt.contourf(xi, yi, zi, interpolation, cmap=plt.cm.seismic,
                     vmax=zmax, vmin=zmin)
#CS = plt.contourf(xi, yi, zi, interpolation, cmap='Dark2_r',
#                     vmax=zmax, vmin=zmin)
# COLORBAR ----------------------
cb = plt.colorbar(CS)
cb.set_label(label=r'$\langle n_0 \rangle$',size=label_size,rotation=90 )
cb.ax.tick_params(labelsize=str(label_size))
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Tics and labels -----------------------------
plt.yticks(fontsize=tick_size)
plt.xticks(fontsize=tick_size)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)
#
plt.ylabel("v", fontsize = label_size)
plt.yticks([0.1, 0.15, 0.2, 0.25], rotation = 'horizontal', fontsize = tick_size)
plt.xticks([0.8, 0.9, 1.0], rotation = 'horizontal', fontsize = tick_size)
plt.xlabel(r"$\alpha$", fontsize = label_size)
#plt.text(0.97, 0.225, 'a)', fontsize = label_size)

plt.tight_layout()
plt.show()
