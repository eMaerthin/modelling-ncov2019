import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, LogNorm)
from numpy.ma import masked_array

path='experiments/figure5_experiment/figure5.csv'
df = pd.read_csv(path, sep=';', index_col=False)
df['c1'] = df.detection_rate
df['c2'] = df.fear
df_prev200 = df[['c1', 'c2', 'Prevalence_360days']].groupby(['c1', 'c2'])[['Prevalence_360days']].apply(np.mean).reset_index()
df_prev200.head()
df_avg_subcritical = df[['c1', 'c2', 'Subcritical']].groupby(['c1', 'c2'])[['Subcritical']].apply(np.mean).reset_index()
df_band_times = df[['c1', 'c2', 'Band_hit_time']].groupby(['c1', 'c2']).Band_hit_time.apply(set).reset_index()
x=sorted(list(set(df['detection_rate'].values)))
y=sorted(list(set(df['fear'].values)))

@np.vectorize
def try_value3(x_id, y_id, col='Prevalence_360days'):
    r = df_prev200[df_prev200.c1==x[x_id]][df_prev200.c2==y[y_id]]
    v = r[col].values
    if len(v)>0:
        #print(v)
        if v[0] is None:
            return float('inf')
        else:
            return v[0]/1000
    else:
        return -1.0

@np.vectorize
def try_value4(x_id, y_id):
    r = df_avg_subcritical[df_avg_subcritical.c1==x[x_id]][df_avg_subcritical.c2==y[y_id]]
    s = r['Subcritical'].values
    band_hit_times = df_band_times[df_band_times.c1==x[x_id]][df_band_times.c2==y[y_id]]
    v = band_hit_times['Band_hit_time'].values
    if len(v)>0:
        times = []
        for elem in v[0]:
            if str(elem) != 'None':
                times.append(float(elem))
        ret_str = ""
        if len(s)>0:
            if len(times) < 10:
                if s[0]<1:
                        return 300
            if len(times) > 0:
                if len(times) < 10:
                    return 300
                return np.array(times).mean()

        return 400
    else:
        return 0

X,Y = np.mgrid[:len(x), :len(y)]

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

fig,ax = plt.subplots(figsize=(15.2,14.6))
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
ax.set_title('Prevalence', fontsize=22)
Z = try_value3(X, Y) * 1000/636307
c = ax.imshow(Z, norm=LogNorm(vmin=1e-3, vmax=1), cmap='Reds')
t = [1e-3, 1e-2, 1e-1, 1]
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 3, 7, 10])
ax.set_yticklabels([0.0, 0.3, 0.7, 1.0])#, list(np.arange(20)))
ax.set_ylabel('Detection rate of mild cases (fraction of all mild cases)', fontsize=18, labelpad=10)
ax.set_xlabel('R* as fraction of referenced R*=3.16 (growth rate for Poland as of 21st March)', fontsize=18, labelpad=10)
fig.colorbar(c, ticks=t, format='$%.3f$', shrink=0.4)#)#, cax=axcolor, ticks=t, format='$%.3f$')#, shrink=0.2)
plt.tight_layout()
plt.savefig(f'figure5a.png')
plt.close(fig)

Z = try_value4(X, Y)
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import masked_array

cdict = {'red':   ((0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  1.0, 1.0))}


cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.01,  0.1, 0.1),
                   (0.3, 0.0, 0.0),
                   (1.0,  0.7, 0.7)),

         'green': ((0.0,  0.0, 0.0),
                   (0.01,  0.0, 0.0),
                   (0.3, 0.0, 0.0),
                   (1.0,  0.7, 0.7)),

         'blue':  ((0.0,  0.3, 0.3),
                   (0.1,  1.0, 1.0),
                   (1.0,  1.0, 1.0))}
from matplotlib.colors import LinearSegmentedColormap
xyz = LinearSegmentedColormap('Xyz', cdict)
plt.register_cmap(cmap=xyz)
xyzw = LinearSegmentedColormap('Xyzw', cdict2)
plt.register_cmap(cmap=xyzw)

v1a = masked_array(Z,Z<200)
v1b = masked_array(Z,Z>200)
fig,ax = plt.subplots(figsize=(15.2,14.6))
ax.set_title("Time until passing ICU threshold", fontsize=18)
pa = ax.imshow(v1a,interpolation='nearest',cmap=xyz)
pb = ax.imshow(v1b,interpolation='nearest',cmap=xyzw)
t=[20, 30, 40, 60, 80, 100]
cbb = plt.colorbar(pb,shrink=0.35, ticks=t, format='$%d$')
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 3, 7, 10])
ax.set_yticklabels([0.0, 0.3, 0.7, 1.0])
ax.set_ylabel('Detection rate of mild cases (fraction of all mild cases)', fontsize=18)
ax.set_xlabel('R* as fraction of referenced R*=3.16 (growth rate for Poland as of 21st March)', fontsize=18, labelpad=16)

plt.tight_layout()
plt.savefig(f'figure5b.png')
plt.close(fig)
