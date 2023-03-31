# program to plot the heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
from matplotlib import ticker

valors = np.linspace(0.9, 1.20, num=31)   # alpha values
valors = ["{:.2f}".format(x) for x in valors]
vs = np.linspace(0.05, 0.2875, num=20)  # velocity values
vs = ["{:.5f}".format(x) for x in vs]

d = {}


for alpha in valors:
    filename = 'magne_N100_f2.00_a' + str(alpha) + '_T0.05_vmin0.05_vmax0.3.csv'
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


# PLOT ------------------------------
label_size = 27
tick_size = 27
title_size = 27
#

#
valors_array = np.array(valors).T.astype(float)
vs_array = np.array(vs).astype(float)
print(type(valors_array))
val_dim = valors_array.size
vs_dim = vs_array.size
dim = val_dim*vs_dim
valors_matrix = np.concatenate((valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array, valors_array), axis=0).reshape(dim)
vs_matrix = np.zeros(dim)


for j in range (vs_dim):
  for i in range(j*val_dim, j*val_dim+val_dim):
    vs_matrix[i] = vs_array[j].T
print('dim  ', dim, 'size = ', df.shape, valors_matrix.shape, vs_matrix.shape)

df  = df.to_numpy().reshape(dim)
xi = np.linspace(valors_array.min(),valors_array.max(),1000)
yi = np.linspace(vs_array.min(),vs_array.max(),1000)
zi = griddata((valors_matrix, vs_matrix), df, (xi[None,:], yi[:,None]), method='cubic')
#h= sns.heatmap(df, vmax = 1)
zmin = df.min()
#zmin = -0.96
print(df.max(), '*****+')
#for row in zi:
#  for value in row:
#    if (value.any() < 0.0):
#    print(value)
zmax = df.max()
interpolation = 100
CS = plt.contourf(xi, yi, zi, interpolation, cmap=plt.cm.seismic,
                     vmax=zmax, vmin=zmin)
##h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("v", fontsize = label_size)
plt.yticks(rotation = 'horizontal')
plt.xlabel(r"$\alpha$", fontsize = label_size)
#plt.gca().invert_yaxis()
#plt.title('Fraction of neutral agents')
# COLORBAR ----------------------
#plt.colorbar(label=r'$|\bar{m}|$',size=20)  
cb = plt.colorbar(CS)
cb.set_label(label=r'$\langle |m| \rangle$',size=label_size,rotation=90 )
cb.ax.tick_params(labelsize=str(label_size))
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#tick_locator = ticker.MaxNLocator(nbins=7)
#cb.locator = tick_locator
#cb.update_ticks()
# Tics -----------------------------
#plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=25)
plt.yticks(fontsize=tick_size)
plt.xticks([0.9, 1.0, 1.1, 1.2], fontsize=tick_size)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)
plt.text(1.1, 0.10, 'b)', fontsize = label_size)


plt.tight_layout()
plt.show()
