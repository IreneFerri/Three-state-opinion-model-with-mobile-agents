#consent time distributions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


N0 = 155

#valors = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
valors = [0.01, 0.02, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.5, 0.9]

valors = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.18, 0.21, 0.23, 0.26, 0.28, 0.31, 0.33, 0.36, 0.38, 0.41, 0.43, 0.46, 0.48]

#valors = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
#valors = [0.001, 0.005, 0.01,  0.05, 0.10, 0.2, 0.3, 0.4]

fig, ax = plt.subplots()

for vel in valors:

    filename = 'reachab_normal_1_N'+str(N0)+'_a_v'+"%5.3f"%vel+'.csv' 
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None, names = None)
    #plt.errorbar(vs, df['mean'], yerr = df['std'], label = alpha)

    ax.plot(clust[0], clust[1], label = vel)

    adjust_clust = clust[1:7]
    print(np.polyfit(np.log(adjust_clust[0]), adjust_clust[1], 1))

#custom_ticks = np.linspace(0, 0.3, num=10)
plt.xticks(rotation = 45)
plt.xscale('log')
plt.xlabel("Steps")
plt.ylabel("Agregated reachabilty")
plt.title('Agreg. reach +1')
ax.legend()
plt.show()
