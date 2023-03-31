
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

valors = np.linspace(0.75, 1.05, num=31)
valors = ["{:.2f}".format(x) for x in valors]
vs = np.linspace(0.06375, 0.49125, num=39)
vs = ["{:.5f}".format(x) for x in vs]

d = {}


for alpha in valors:
    filename = 'res0.06375_0.3_'+str(alpha)+'.csv'
    clust =  pd.read_csv(filename, delim_whitespace = True, header=None, names=np.linspace(0,20,num=21))
    filename2 = 'res0.3_0.5025_'+str(alpha)+'.csv'
    clust2 =  pd.read_csv(filename2, delim_whitespace = True, header=None, names=np.linspace(21,39,num=19))
    horizontal_stack = pd.concat([clust, clust2], axis=1)
    i = 0
    percent = []
    for velocitat in vs:
        col = horizontal_stack[i]
        neutres = 0
        for element in col:
            if element == 0:
                neutres += 1
        i += 1
        percent.append(neutres/len(col))
    d.update({str(alpha): pd.Series(percent, index = vs)})

df=pd.DataFrame(d, index=vs, columns=valors,)

print(df)

h= sns.heatmap(df, vmax = 1)
#h.set(xlabel='Alpha', ylabel='Velocity')
plt.ylabel("Velocity ")
plt.yticks(rotation = 'horizontal')
plt.xlabel("Alpha ")
plt.gca().invert_yaxis()
plt.title('Fraction of neutral final configurations')
plt.show()



fig, ax = plt.subplots()
hm = plt.imshow(df, cmap='tab20c',interpolation="spline16")
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
