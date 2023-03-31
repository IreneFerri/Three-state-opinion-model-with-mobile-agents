# program to plot the heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
from matplotlib import ticker


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


###### plot ##########
label_size = 27
text_size = 27
tick_size = 27
cb_size = 22.1
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
zmin = df.min()
zmax = df.max()
interpolation = 100
CS = plt.contourf(xi, yi, zi, interpolation, cmap=plt.cm.seismic,
                     vmax=zmax, vmin=zmin)
plt.ylabel("v", fontsize = label_size)
plt.yticks(rotation = 'horizontal')
plt.xlabel(r"$\alpha$", fontsize = label_size)
# COLORBAR ----------------------
cb = plt.colorbar(CS)
cb.set_label(label=r'$\langle n_0 \rangle$',size=label_size,rotation=90 )
cb.ax.tick_params(labelsize=str(label_size))
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Tics -----------------------------
plt.yticks(fontsize=tick_size)
plt.xticks([0.9, 1.0, 1.1, 1.2], fontsize=tick_size)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)

plt.text(1.10, 0.23,'a)', fontsize = text_size)


plt.tight_layout()
plt.show()



