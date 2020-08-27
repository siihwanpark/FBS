import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(8, 1)

total_channels = 0
safely_removable_channels = 0

for i in range(8):
    df = pd.read_csv(f'conv{i}.tsv', header=None, sep='\t')
    prob_arr = df.values

    ind = prob_arr.sum(axis=0).argsort(axis=0)
    extended_ind = np.expand_dims(ind, axis=0).repeat(10, axis=0)
    prob_arr_sorted = np.take_along_axis(prob_arr, extended_ind, axis=1)

    exact_zero_mask = prob_arr_sorted.sum(axis=0) < 1e-15

    total_channels += prob_arr_sorted.shape[1]
    safely_removable_channels += exact_zero_mask.sum()

    exact_zero_mask = np.expand_dims(exact_zero_mask, axis=0).repeat(10, axis=0)

    prob_arr_sorted[exact_zero_mask] = 0.3

    if prob_arr_sorted.shape[1] < 192:
        pad = np.zeros((10, 192 - prob_arr_sorted.shape[1]))
        prob_arr_sorted = np.concatenate((prob_arr_sorted, pad), axis=1)

    c = ax[i].pcolor(prob_arr_sorted, cmap='Greys')
    ax[i].axis('off')

fig.tight_layout()
plt.savefig('fig3b.png', format='png', bbox_inches='tight')

print(f'Static pruning ratio: {100*safely_removable_channels/total_channels:.2f}%')