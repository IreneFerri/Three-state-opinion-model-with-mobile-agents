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


#h= sns.heatmap(df)
#plt.ylabel("Velocity ")
#plt.yticks(rotation = 'horizontal')
#plt.xlabel("Alpha ")
#plt.gca().invert_yaxis()
#plt.title('Fraction of neutral final configurations')
#plt.show()

#fig, ax = plt.subplots()
#df_smooth = gaussian_filter(df, sigma=3)
#h = sns.heatmap(df_smooth, vmax=1)
#plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
#plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
#plt.ylabel("Velocity ")
#plt.xticks(rotation = 'vertical')
#plt.yticks(rotation = 'horizontal')
#plt.xlabel("Alpha ")
#plt.gca().invert_yaxis()
#plt.title('Fraction of neutral final configurations')
##plt.show()

label_size = 12
tick_size = 12
title_size = 12
my_cmap = ['grey', 'blue', 'green', 'orange', 'black', 'red', 'brown', 'purple']

#vs = ["{:.2f}".format(x) for x in vs]

fig, ax = plt.subplots()
hm = plt.imshow(df, cmap='Pastel1_r',interpolation="bicubic")
#hm = plt.imshow(df, cmap=my_cmap, interpolation="bicubic")

#plt.colorbar(hm,ScalarMappable(norm=BoundaryNorm(np.arange(0.0, 1.1, 0.1), ncolors=10)))
plt.colorbar(hm)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
#plt.yticks(np.arange(0.5, len(df.index), 2), label=['0.06', '0.09', '0.12', '0.15', '0.18', '0.21', '0.24', '0.27', '0.29', '0.32'])
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)

plt.ylabel("v", fontsize = label_size)
plt.yticks(rotation = 'horizontal')
plt.xticks(rotation = 'vertical')
plt.xlabel(r"$\alpha$", fontsize = label_size)
plt.gca().invert_yaxis()
plt.title('Fraction of final neutral consensus')

#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places


plt.tight_layout()
plt.show()
