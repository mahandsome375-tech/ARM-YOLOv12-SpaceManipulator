import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D





data = np.load(r'bridge\station_traj_fourmods.npy')


print('shape =', data.shape)


t, cx, cy, d, tau, phi, sigma = data.T

print('t     :', t[:5])
print('cx    :', cx[:5])
print('cy    :', cy[:5])
print('d     :', d[:5])
print('tau   :', tau[:5])
print('phi   :', phi[:5])
print('sigma :', sigma[:5])


plt.figure()
plt.plot(t, d)
plt.title('d vs t')

plt.figure()
plt.plot(t, tau)
plt.title('tau vs t')

plt.figure()
plt.plot(t, phi)
plt.title('phi vs t')

plt.figure()
plt.plot(t, sigma)
plt.title('sigma vs t')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cx, cy, d)
ax.set_xlabel('cx (pixel)')
ax.set_ylabel('cy (pixel)')
ax.set_zlabel('d (normalized depth)')
ax.set_title('image+depth trajectory')
plt.show()
