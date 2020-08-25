import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


with open('./models/traindata.json', 'r') as fr:
    data = json.load(fr)
data = sorted(data, key=lambda i: i['width'])

widths = []
elap_cpu, elap_gpu, acc_cpu, acc_gpu = [], [], [], []
for i in data:
    if i['device'] == 'cpu':
        widths.append(i['width'])
        elap_cpu.append(i['elapsed'])
        acc_cpu.append(i['overall'])
    else:
        elap_gpu.append(i['elapsed'])
        acc_gpu.append(i['overall'])


# fit
def f(x, a, b, c):
    return a * np.exp(-b * x) + c


avg_acc = [(i + j) / 2 for i, j in zip(acc_cpu, acc_gpu)]
popt, pcov = curve_fit(f, widths, avg_acc, p0=[0, 0, 55])


fig, ax = plt.subplots()
ax.plot(widths, elap_cpu, color='blue', label='cpu', marker='o', markersize=5)
ax.plot(widths, elap_gpu, color='red', label='gpu', marker='o', markersize=5)

ax_sec = ax.twinx()
ax_sec.plot(widths, acc_cpu, color='blue', label='cpu accuracy', marker='x',
            ls='dashed', markersize=5, lw=0.5)
ax_sec.plot(widths, acc_gpu, color='red', label='gpu accuracy', marker='x',
            ls='dashed', markersize=5, lw=0.5)
ax_sec.plot(widths, [f(x, *popt) for x in widths], color='green',
            label='fitted. accuracy', marker='x', ls='dashed', markersize=5)

ax.set_xlabel('layer width (#)')
ax.set_ylabel('elapsed time (s)')
ax.set_title('Neural net training times')
ax_sec.set_ylabel('Accuracy (%)')

ax.grid()
ax.legend()
ax_sec.legend()
plt.show()
