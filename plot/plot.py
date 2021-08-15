import copy
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.core.numeric import full

runs = 5
model = "vgg19"
algo = "sem"
res = []
run_data = []
run_data_rev = []

for r in range(runs):
    with open(model + "_" + algo + "_run_" + str(r+1) + ".out") as f:
        for line in f:
            if "ration 100" in line:
                m = re.match(r".+fitness\s\d\.\d+\s(.+)", line)
                if m :
                    run_data.append(eval(m.groups(1)[0]))
                break

full_run_data = copy.deepcopy(run_data)
solution_len = len(run_data[0])
section_1 = [x[:solution_len // 2] for x in full_run_data]

for i in range(len(run_data)):
    for j in range(len(run_data[i])):
        if run_data[i][j] < 0.5:
            run_data[i][j] = 1
        else:
            run_data[i][j] = 0
    run_data[i] = np.array(run_data[i])
    run_data[i] = run_data[i][:solution_len // 2]
    run_data_rev.append(np.argwhere(run_data[i] == 0))
    run_data[i] = np.argwhere(run_data[i] == 1)
    run_data[i] = np.squeeze(run_data[i])
    run_data_rev[i] = np.squeeze(run_data_rev[i])

fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(6,8))
# ax[0].eventplot(run_data[1], 'horizontal', linewidths=5, lineoffsets=0)
# ax[0].eventplot(run_data_rev[1], 'horizontal', linewidths=5, lineoffsets=0, color="red")
# ax[0].eventplot(run_data[2], 'horizontal', linewidths=5, lineoffsets=2)
# ax[0].eventplot(run_data_rev[2], 'horizontal', linewidths=5, lineoffsets=2, color="red")
# ax[0].eventplot(run_data[3], 'horizontal', linewidths=5, lineoffsets=500)
# ax[0].eventplot(run_data_rev[3], 'horizontal', linewidths=5, lineoffsets=500, color="red")
# ax[0].eventplot(run_data[4], 'horizontal', linewidths=5, lineoffsets=6)
# ax[0].eventplot(run_data_rev[4], 'horizontal', linewidths=5, lineoffsets=6, color="red")

color1 = "#ffd166"
color2 = "#118ab2"
width = 0.7
height = 0.7

ax[0].plot([0,0])
for i in range(len(run_data)):
    for j in range(len(run_data[i])):
        alpha = abs(section_1[i][j] - 0.5) * 2
        ax[0].add_patch(Rectangle(
            xy=(run_data[i][j]-width/2, i-height/2) ,width=width, height=height,
            linewidth=1, color=color1, fill=True, alpha=alpha))
    for j in range(len(run_data_rev[i])):
        alpha = abs(section_1[i][j] - 0.5) * 2
        ax[0].add_patch(Rectangle(
            xy=(run_data_rev[i][j]-width/2, i-height/2) ,width=width, height=height,
            linewidth=1, color=color2, fill=True, alpha=alpha))
    # ax[0].scatter(run_data[i],     [i] * len(run_data[i]),     color=color1,    marker="s", s=solution_len / 2 / 2)
    # ax[0].scatter(run_data_rev[i], [i] * len(run_data_rev[i]), color=color2,   marker="s", s=solution_len / 2 / 2)
    ax[1].plot(full_run_data[i][solution_len // 2:], '-')
# ax[0].legend(prop={'size': 14})
ax[0].set_xlim(-1, solution_len // 2)
ax[0].set_ylim(-1, 5)
ax[0].grid(b=True, linestyle="--")
plt.savefig('solutions-' + model + '.png')

# with open("plot.out", "r") as f:
#     for line in f:
#         arr = eval(line)
#         for i in range(len(arr)):
#             arr[i] = 1 if arr[i] > 0.5 else 0
#         res.append(arr)
    
# with open("plot2.out", "w") as f:
#     for r in res:
#         f.write(str(r) + "\n")

