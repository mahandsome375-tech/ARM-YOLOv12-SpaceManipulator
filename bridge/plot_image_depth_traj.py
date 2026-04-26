import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D




BASE_DIR = os.path.dirname(__file__)
traj_path = os.path.join(BASE_DIR, "station_traj_fourmods.npy")
traj = np.load(traj_path)

t   = traj[:, 0]
cx  = traj[:, 1]
cy  = traj[:, 2]
d   = traj[:, 3]





win = min(len(d) // 2 * 2 - 1, 21)
if win < 5:
    win = 5
if win >= len(d):
    win = len(d) - 1 if len(d) % 2 == 0 else len(d)
if win % 2 == 0:
    win -= 1

d_smooth = savgol_filter(d, win, 3)







z_plot = 1.0 / (d_smooth + 1e-6)
z_label = "Inverse Depth / Closeness"








fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(
    cx, cy, z_plot,
    c=z_plot,
    cmap="viridis",
    s=8
)
ax.plot(cx, cy, z_plot, linewidth=1.2, alpha=0.9)


ax.set_xlabel("cx (pixel)")
ax.set_ylabel("cy (pixel)")
ax.set_zlabel(z_label)
ax.set_title("Image–Depth Trajectory of Target Motion")


cbar = plt.colorbar(sc, pad=0.08)
cbar.set_label(z_label)


ax.view_init(elev=28, azim=135)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "image_depth_trajectory_spiral.png"), dpi=600)
plt.show()