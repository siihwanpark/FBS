import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.tsv', header=None, sep='\t')

baseline_acc = df[2].values[0]
baseline_MACs = df[3].values[0] * 1e-8

fbs_acc_arr = df[2].values[1:]
fbs_MACs_arr = df[3].values[1:] * 1e-8

plt.figure()
plt.title('M-CifarNet accuracy / MACs trade-off')
plt.xlabel('MACs (1e8)')
plt.ylabel('Accuracy')

plt.scatter(baseline_MACs, baseline_acc, c='blue', label='baseline')
plt.plot(fbs_MACs_arr, fbs_acc_arr, c='orange', marker='o', label='FBS')

plt.legend()
plt.savefig('fig3a.png', format='png', bbox_inches='tight')