



import numpy as np


in_path  = r"bridge\station_traj_fourmods.npy"
out_path = r"bridge\station_traj_fourmods_calib.npy"


data = np.load(in_path)

t    = data[:, 0]
cx   = data[:, 1]
cy   = data[:, 2]
d    = data[:, 3]
tau0 = data[:, 4]
phi  = data[:, 5]
sigma = data[:, 6]

dt = float(np.mean(np.diff(t)))





calib_idx = np.array([
    30,
    120,
    210,
    270
], dtype=int)


d_meas = d[calib_idx]



d_gt = np.array([0.82, 0.65, 0.51, 0.35], dtype=float)


a, b = np.polyfit(d_meas, d_gt, 1)
print("Calibration coefficients: a = %.6f, b = %.6f" % (a, b))


d_calib = a * d + b


d_calib = np.clip(d_calib, 0.0, 1.0)


d_dot = np.gradient(d_calib, dt)
eps = 1e-6
tau = -d_calib / (d_dot + eps)


data_out = np.stack([t, cx, cy, d_calib, tau, phi, sigma], axis=1)
np.save(out_path, data_out)

print("Calibrated trajectory saved to:", out_path)
