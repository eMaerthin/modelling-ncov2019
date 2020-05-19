import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, LogNorm)
from numpy.ma import masked_array

path='experiments/figure4_experiment/figure4.csv'
df = pd.read_csv(path, sep=';')
x=sorted(list(set(df['c_norm'].values)))
y=sorted(list(set(df['Init_people'].values)))

print(x)
print(y)
@np.vectorize
def try_value(x_id, y_id, column='Mean_Time'):
    r = df[df.c==x[x_id]][df['Init_people']==y[y_id]]
    v = r[column].values
    w = r['Mean_Time'].values
    if len(v)>0:
        if v[0] > 1000000:
            return 1000000.0001
        return v[0]
    else:
        return float('nan')

cdict2 = {
    'red':   (
        (0.0,  0.7, 0.7),
        (0.7, 0.0, 0.0),
        (0.99,  0.1, 0.1),
        (1.0,  0.0, 0.0)
    ),
    'green': (
        (0.0,  0.7, 0.7),
        (0.7, 0.0, 0.0),
        (0.99,  0.0, 0.0),
        (1.0,  0.0, 0.0)
    ),
    'blue':  (
        (0.0,  1.0, 1.0),
        (0.9,  1.0, 1.0),
        (1.0,  0.3, 0.3)
    )
}

cmap_a = LinearSegmentedColormap('cmap_a', cdict2)
plt.register_cmap(cmap=cmap_a)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

fig, ax = plt.subplots(figsize=(15.2,14.6))

X,Y = np.mgrid[:len(x), :len(y)]
Z = try_value(X, Y)
v1a = masked_array(Z,Z>=700.1)
c = ax.imshow(v1a, cmap=cmap_a, interpolation='nearest')

ax.set_title('Prevalence', fontsize=22)
ax.set_xticks(np.arange(0, len(y), 2))
ax.set_xticklabels([f'{elem/1000:.1f}k' for elem in y[::2]])
ax.set_yticks(np.arange(0, len(x), 2))
ax.set_yticklabels([f'{float(elem):.2f}' for elem in x[::2]])
ax.set_title("Average number of days until extinction of epidemics\n starting from N people for given R* value\n", fontsize=22)
ax.set_ylabel('R* value - average number of people infected \nby a single person (outside of the household)', fontsize=18)
ax.set_xlabel('people initially infected (at "day0") (in thousands)', fontsize=18)
ax.xaxis.set_label_position('top')

# Let the horizontal axes labeling appear on top.
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
fig.colorbar(c, shrink=0.5)
fig.tight_layout()
plt.savefig(f'experiments/figure4_experiment/figure4a.png', dpi=300)
plt.close(fig)

mean_affected = try_value(X, Y, column='Mean_Affected')
fig,ax = plt.subplots(figsize=(15.2,14.6))
ax.set_title('Prevalence', fontsize=22)
Z = mean_affected/1000 # prevalence in thousands
v1a = masked_array(Z,Z>=1000.1)
c = ax.imshow(v1a, cmap='Reds', interpolation='nearest', norm=LogNorm(vmin=1, vmax=1000))
ax.set_xticks(np.arange(0, len(y), 2))
ax.set_xticklabels([f'{elem/1000:.1f}k' for elem in y[::2]])
ax.set_yticks(np.arange(0, len(x), 2))
ax.set_yticklabels([f'{float(elem):.2f}' for elem in x[::2]])
ax.set_title("Prevalence in thousands from N people infectious\n till last active case for given R* value\n", fontsize=22)
ax.set_ylabel('R* value - average number of people infected \nby a single person (outside of the household)', fontsize=18)
ax.set_xlabel('people initially infected (at "day0") (in thousands)', fontsize=18)
ax.xaxis.set_label_position('top')
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
fig.colorbar(c, shrink=0.5)
fig.tight_layout()
plt.savefig('experiments/figure4_experiment/figure4b.png', dpi=300)
plt.close(fig)
