import re
import numpy as np
from matplotlib import pyplot as plt

runs = 5
iterations = 100
algos = ["sem"]

file_prefix = {}
avg = {}

for algo in algos:
    empty = [0 for _ in range(iterations)]
    avg[algo] = [empty.copy() for _ in range(runs)]
    file_prefix[algo] = "vgg19_" + algo + "_run_"

for algo in algos:
    for r in range(runs):
        with open(file_prefix[algo] + str(r+1) + ".out", "r") as f:
            c = 0
            for line in f:
                if "fitness" in line:
                    m = re.findall(r"\s(\d+\.\d+)\s", line)
                    if m:
                        avg[algo][r][c] += float(m[0])
                        c += 1
        
# for algo in algos:
#     for i in range(len(avg[algo])):
#         avg[algo][i] /= runs


fig, ax = plt.subplots(ncols=1, nrows=1, tight_layout=True)

for algo in algos:
    for r in range(runs):
        ax.plot(avg[algo][r], linewidth=2)

# ax.legend(prop={'size': 14})
ax.set_xlim(0, iterations)
# ax.set_ylim(0.18, 0.3)
ax.grid(b=True, linestyle="--")
plt.savefig('converge.png')

# for n in avg:
#     with open(n + ".txt", "w") as f:
#         for value in avg[n]:
#             f.write(str(value) + "\n")