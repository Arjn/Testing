from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import scipy.stats as stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.integrate import odeint
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cmap = plt.cm.get_cmap('Paired')

# analytical solution to the falling body with drag
# def model(y, t):
#     g = -9.81
#     b = 200
#     x, v = y
#     dvdt = [v, (-0.0034 * g * np.exp(-x / 22000) * v**2) / (2 * b) + g]
#     return dvdt
#
#
# y0 = [1e7, 0.0]
# t = np.linspace(0, 3000, 101)
# sol = odeint(model, y0, t)
# plt.plot(t, sol[:, 0])
# plt.show()

# State is array of 6 elements: [x, y, z, xdot, ydot, zdot]

dt = 0.1
np.random.seed(500)
true_b = 400
std_true_object = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
true_starting_conditions = np.array([10., 10., 1e5, 1000., 1000., 500.])#np.array([500., -1000., 1e5, 100., 100., 500.])
est_starting_conditions = np.array([10., 10., 1.001e5, 900., 1100., 450.])#np.array([450., -900., 1.2e5, 90., 110., 450.])
observer_position = np.array([-1.e6, -1.e5, 100.]) #np.array([-8e2, -7e2, 0.])
observer_std = np.array([0., 0.05, 0.05])
P = np.diag([10.**2, 10.**2, 1e2**2, 20**2, 20**2, 20**2])
Q = np.diag([1e0, 1e0, 1e0, 0.5, 0.5, 0.5])


g = 9.81



def norm_angles(ang):
    ang = np.mod(ang, 2*np.pi)
    if ang > np.pi:
        ang -= 2*np.pi
    return ang



class falling_object(object):
    def __init__(self, X, std_mdl, b):
        self.X = X
        self.b = b
        self.std_model = std_mdl
        self.meas = [0, 0, 0]

    def update(self, dt):
        self.X[0] += self.X[3] * dt + randn() * self.std_model[0]
        self.X[1] += self.X[4] * dt + randn() * self.std_model[1]
        self.X[2] += self.X[5] * dt + randn() * self.std_model[2]
        self.X[3] += randn()*self.std_model[3]
        self.X[4] += randn() * self.std_model[4]
        self.X[5] += ((0.0034 * g * np.exp(-self.X[2] / 22000) * self.X[5] ** 2) / (2 * self.b) - g)*dt + randn()\
                     * self.std_model[5]
        return self.X


def f_cv(x, dt):
    """ state transition function for a 3D
    accelerating object under gravity"""
    b = 400
    x[0] = x[0] + x[3] * dt
    x[1] = x[1] + x[4] * dt
    x[2] = x[2] + x[5] * dt
    x[3] = x[3]
    x[4] = x[4]
    x[5] = x[5] + ((0.0034 * g * np.exp(-x[2] / 22000) * x[5] ** 2) / (2 * b) - g)*dt
    #print(x)
    #print()
    return x


def h_observer(x):
    """Measurement function -
    measuring only position"""
    dX = [x[0] - h_observer.pos[0], x[1] - h_observer.pos[1], x[2] - h_observer.pos[2]]
    r = np.sqrt(dX[0]**2 + dX[1]**2 + dX[2]**2)
    theta = np.arccos(dX[2]/r)
    phi = np.arctan2(dX[1], dX[0])
    return [theta, phi]


class observer(object):
    def __init__(self, observer_pos, stds):
        self.pos = observer_pos
        self.r_std = stds[0]
        self.theta_std = stds[1]
        self.phi_std = stds[2]
        self.r = 0.
        self.storage = []

    def observe(self, x):
        self.dX = [x[0] - self.pos[0], x[1] - self.pos[1], x[2] - self.pos[2]]
        self.r_true = np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2)
        self.theta_true = np.arccos(self.dX[2] / self.r_true)
        self.phi_true = np.arctan2(self.dX[1], self.dX[0])

    def noisy_observe(self, x):
        self.observe(x)
        self.r = self.r_true + randn()*self.r_std
        self.theta = self.theta_true + randn()*self.theta_std
        self.phi = self.phi_true + randn()*self.phi_std
        self.measure = [self.r, self.theta, self.phi]
        self.storage.append(self.measure.copy())

h_observer.pos = observer_position

points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-1.)
ukf = UKF(dim_x=6, dim_z=2, fx=f_cv, hx=h_observer, dt=dt, points=points)
ukf.x = np.array([est_starting_conditions])
ukf.R = np.diag([observer_std[1]**2, observer_std[2]**2]) #observer_std[0]**2
#ukf.H = np.array([[1]])
ukf.P = P
ukf.Q = Q#Q_matrix(dt, Q_variance) #np.diag([0.0, 0.0, 0.0, 00., 00., 00.])#
uxs = []
trux = []
truv = []
measx = []
time = []
covarx = []
covary = []
covarv = []
zs = np.arange(0, 1435 + dt, dt)

f = falling_object(true_starting_conditions, std_true_object, true_b)
station = observer(observer_position, observer_std)
t = 0
while f.X[2] >= 0:
    f.update(dt)
    station.noisy_observe(f.X)
    ukf.predict()
    ukf.update([station.theta, station.phi])
    measx.append(station.r)
    uxs.append(ukf.x.copy())
    trux.append(f.X.copy())
    covarx.append(ukf.P[2][2])
    covary.append(ukf.P[1][1])
    covarv.append(ukf.P[5][5])
    t += dt
    time.append(t)
    #print(f.X[2])
    #print(ukf.x[2])
    #print(ukf.P)
    #print("TIME = ", t)
    #print([station.r, station.theta, station.phi])
    print(f.X[2])
    #print(ukf.x)

uxs = np.array(uxs)
trux = np.array(trux)
station.storage = np.array(station.storage)
print(len(covarx))
print(len(uxs))

# # VELOCITY ERROR
plt.figure(1)
plt.plot(time, (uxs[:, 3] - trux[:, 3]), 'xkcd:blue', label='x-error',linewidth=3.0)
plt.plot(time, (uxs[:, 4] - trux[:, 4]),'xkcd:green', label='y-error',linewidth=3.0)
plt.plot(time, (uxs[:, 5] - trux[:, 5]), 'xkcd:orange', label='z-error',linewidth=3.0)
plt.plot(time, np.array(covarv)*3, 'r')
plt.plot(time, np.array(covarv)*-3, 'r')
plt.title('non linear eqn velocity error')
plt.legend()
#
# POSITION ERROR
plt.figure(2)
plt.plot(time, (uxs[:, 0] - trux[:, 0]), 'xkcd:blue', label='x-error', linewidth=3.0)
plt.plot(time, (uxs[:, 1] - trux[:, 1]), 'xkcd:green', label='y-error', linewidth=3.0)
plt.plot(time, (uxs[:, 2] - trux[:, 2]), 'xkcd:orange', label='z-error', linewidth=3.0)
plt.plot(time, np.array(covarx)*3, 'r')
plt.plot(time, np.array(covarx)*-3, 'r')
plt.title('non linear eqn position error')
plt.legend()

mean_x_error = [np.mean(uxs[:, 0] - trux[:, 0]), np.mean(uxs[:, 1] - trux[:, 1]), np.mean(uxs[:, 2] - trux[:, 2])]
mean_v_error = [np.mean(uxs[:, 3] - trux[:, 3]), np.mean(uxs[:, 4] - trux[:, 4]), np.mean(uxs[:, 5] - trux[:, 5])]

std_x = [np.std(uxs[:, 0] - trux[:, 0]), np.std(uxs[:, 1] - trux[:, 1]), np.std(uxs[:, 2] - trux[:,2])]
std_v = [np.std(uxs[:, 3] - trux[:, 3]), np.std(uxs[:, 4] - trux[:, 4]), np.std(uxs[:, 5] - trux[:, 5])]
#
print("mean x error = ", mean_x_error)
print("std x error = ", std_x)
print("mean v error = ", mean_v_error)
print("std v error = ", std_v)

# VELOCITY
plt.figure(3)
plt.plot(time, uxs[:, 5], 'xkcd:blue', label='filter vz', linewidth=3.0)
plt.plot(time, uxs[:, 3], 'xkcd:green', label='filter vx', linewidth=3.0)
plt.plot(time, uxs[:, 4], 'xkcd:orange', label='filter vy', linewidth=3.0)
plt.plot(time, trux[:, 5], linestyle='--', color='xkcd:dark blue', label=('model z'))
plt.plot(time, trux[:, 3],linestyle='--', color='xkcd:dark green', label=('model x'))
plt.plot(time, trux[:, 4], linestyle='--', color='xkcd:dark orange', label=('model y'))
#plt.scatter(time, measx, label='measure')
plt.title('non linear eqn velocity')
plt.legend()


# POSITION IN TIME
plt.figure(5)
plt.plot(time, uxs[:, 2], 'xkcd:blue', label='filter z', linewidth=3.0)
plt.plot(time, uxs[:, 0], 'xkcd:green', label='filter x', linewidth=3.0)
plt.plot(time, uxs[:, 1], 'xkcd:orange', label='filter y', linewidth=3.0)
plt.plot(time, trux[:, 2],  linestyle='--', color='xkcd:dark blue', label=('model z'))
plt.plot(time, trux[:, 0], linestyle='--', color='xkcd:dark green', label=('model x'))
plt.plot(time, trux[:, 1], linestyle='--', color='xkcd:dark orange', label=('model y'))
# #plt.scatter(time, measx, s=2, label='measure')
plt.title('non linear eqn position')
plt.legend()

# POSITION IN SPACE
fig = plt.figure(6)
ax = fig.gca(projection='3d')
ax.plot(uxs[:, 0], uxs[:, 1], uxs[:, 2], 'xkcd:blue', label='filter POS 2D')
ax.plot(trux[:, 0], trux[:, 1], trux[:, 2], linestyle='--', color='xkcd:dark blue', label=('model'))
ax.scatter(observer_position[0], observer_position[1], observer_position[2], s=50, marker='x', label='observer')
plt.title('non linear eqn position')
plt.legend()

# Observations
plt.figure(7)
plt.plot(time, station.storage[:, 1], 'b', label='theta')
plt.plot(time, station.storage[:, 2], 'r', label='phi')
plt.title('observations')
plt.legend()

plt.show()
#print('UKF standard deviation {:.3f} meters'.format(np.std(uxs - xs)))
