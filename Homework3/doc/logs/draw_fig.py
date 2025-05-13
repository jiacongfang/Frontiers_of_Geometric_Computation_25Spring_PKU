import matplotlib.pyplot as plt
import pandas as pd
import os


def tensorboard_smoothing(x, smooth=0.6):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (1 + weight)
        weight = (weight + 1) * smooth
    return x


# fig, ax1 = plt.subplots(1, 1)
# pointnet = pd.read_csv("ocnn_test_acc.csv")
# pointnet2 = pd.read_csv("ocnn_train_acc.csv")

# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)

# ax1.plot(
#     pointnet["Step"],
#     tensorboard_smoothing(pointnet["Value"], smooth=0.6),
#     color="#3399FF",
# )

# ax1.plot(
#     pointnet2["Step"],
#     tensorboard_smoothing(pointnet2["Value"], smooth=0.6),
#     color="#FF6666",
# )

# ax1.legend(["Test Acc", "Train Acc"], loc="best")

# ax1.set_xlabel("epoch")
# ax1.set_ylabel("classification acc", color="black")

# plt.show()

# os.makedirs("./figures", exist_ok=True)

# fig.savefig(fname="./figures/ocnn_acc" + ".pdf", format="pdf")


fig, ax1 = plt.subplots(1, 1)
pointnet = pd.read_csv("ocnn_train_loss.csv")
pointnet2 = pd.read_csv("ocnn_train_acc.csv")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.plot(
    pointnet["Step"],
    tensorboard_smoothing(pointnet["Value"], smooth=0.6),
    color="#3399FF",
)


ax1.legend(["ocnn"], loc="best")

ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss", color="black")

plt.show()

os.makedirs("./figures", exist_ok=True)

fig.savefig(fname="./figures/ocnn_loss" + ".pdf", format="pdf")
