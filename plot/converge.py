import re
import numpy as np
from matplotlib import pyplot as plt

runs = 5
evals = 3000
algos = ["sem"]

file_prefix = {}
avg = {}

for algo in algos:
    empty = [0 for _ in range(evals)]
    avg[algo] = [empty.copy() for _ in range(runs)]
    file_prefix[algo] = "vgg19_" + algo + "_run_"

for algo in algos:
    for r in range(runs):
        run_best = float("inf")
        with open(file_prefix[algo] + str(r+1) + ".out", "r") as f:
            c = 0
            for line in f:
                if c >= evals:
                    break
                if "search end" in line:
                    break
                if "Fitness" in line:
                    m = re.findall(r".+Pruned ratio: (\d+\.\d+), Test Acc: (\d+\.\d+), Fitness: (\d+\.\d+)", line)
                    if m:
                        run_best = min(float(m[0][2]), run_best)
                        avg[algo][r][c] += run_best
                        c += 1

# for algo in algos:
#     for i in range(len(avg[algo])):
#         avg[algo][i] /= runs


fig, ax = plt.subplots(ncols=1, nrows=1, tight_layout=True)

for algo in algos:
    for r in range(runs):
        ax.plot(avg[algo][r], linewidth=2)

# ax.legend(prop={'size': 14})
ax.set_xlim(0, evals)
# ax.set_ylim(0.18, 0.3)
ax.grid(b=True, linestyle="--")
plt.savefig('converge.png')

# for n in avg:
#     with open(n + ".txt", "w") as f:
#         for value in avg[n]:
#             f.write(str(value) + "\n")