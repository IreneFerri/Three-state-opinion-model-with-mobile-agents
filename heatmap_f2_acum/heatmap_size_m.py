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

# SIZE OF THE BIGGEST polar CLUSTER

for alpha in valors:
    filename = 'size_polar0.05_0.3_' + str(alpha) + '.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None)
    i = 0
    percent = []
    for velocitat in vs:
        col = clust[i]
        promig = np.nanmean(col)
        if (promig == np.nan):
          print(alpha)
          print(col)
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
df[np.isnan(df)] = 0  # Change nan by 0
xi = np.linspace(valors_array.min(),valors_array.max(),1000)
yi = np.linspace(vs_array.min(),vs_array.max(),1000)
zi = griddata((valors_matrix, vs_matrix), df, (xi[None,:], yi[:,None]), method='linear')
zmin = df.min()
zmax = df.max()
print(zmin, zmax)
interpolation = 100
#print(zi)
CS = plt.contourf(xi, yi, zi, interpolation, cmap='terrain_r',
                     vmax=zmax, vmin=zmin)
plt.ylabel("v", fontsize = label_size)
plt.yticks(rotation = 'horizontal')
plt.xlabel(r"$\alpha$", fontsize = label_size)
# COLORBAR ----------------------
cb = plt.colorbar(CS)
cb.set_label(label=r'Biggest polar clust. size',size=cb_size,rotation=90 )
cb.ax.tick_params(labelsize=str(tick_size))
cb.set_ticks([0, 20, 40, 60, 80])
# Tics -----------------------------
plt.yticks(fontsize=tick_size)
plt.xticks([0.9, 1.0, 1.1, 1.2], fontsize=tick_size)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=True)

plt.text(0.95, 0.10,'d)', fontsize = text_size)


plt.tight_layout()
plt.show()



