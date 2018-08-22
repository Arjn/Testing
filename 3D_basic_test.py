from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import scipy.stats as stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.integrate import odeint
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt


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


std_x = [10.0, 10.]
std_model = 0.#.001
g = 9.81
np.random.seed(200)

dt = 0.01
# std_true_object = np.array([0.0, 0.0, 0.0, 0.0])
# true_starting_conditions = np.array([100., 100., 100., 100.])
# est_starting_conditions = np.array([100., 100., 100, 10.])
# observer_position = np.array([0., 0.])
# observer_std = np.array([0.0, 0.0, 0.0])


class falling_object(object):
    def __init__(self, pos, vel, b, std, std_mdl):
        self.x = pos
        self.v = vel
        self.b = b
        self.std_x = std
        self.std_model = std_mdl
        self.meas = [0, 0, 0, 0]

    def update(self, dt):
        self.x[0] += self.v[0]*dt + randn()*self.std_model*self.x[0]
        self.x[1] += self.v[1] * dt + randn() * self.std_model * self.x[1]
        self.v[0] += randn()*self.std_model*self.v[0]
        self.v[1] += ((0.0034 * g * np.exp(-self.x[1] / 22000) * self.v[1] ** 2) / (2 * self.b) - g)*dt + randn()*self.std_model*self.v[1]
        return self.x

    def measure(self):
        self.meas[0] = randn()*self.std_x[0] + self.x[0]
        self.meas[1] = randn() * self.std_x[0] + self.x[1]
        self.meas[2] = randn() * self.std_x[1] + self.v[0]
        self.meas[3] = randn() * self.std_x[1] + self.v[1]
        return self.meas



def f_cv(x, dt):
    """ state transition function for a 1D
    accelerating object"""
    g = 9.81
    b = 200.
    x[0] = x[0] + x[2]*dt
    x[1] = x[1] + x[3]*dt
    x[2] = x[2]
    x[3] = x[3] + ((0.0034 * g * np.exp(-x[1] / 22000) * x[3] ** 2) / (2 * b) - g)*dt
    return x


def h_cv(x):
    """Measurement function -
        measuring only position"""
    return np.array([x[0], x[1]])


starting_conditions = [100., 1.00e4, 0., 0., 200.]

points = MerweScaledSigmaPoints(n=4, alpha=.0001, beta=2., kappa=-1)
ukf = UKF(dim_x=4, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=points)
ukf.x = np.array([1000., 1.00e3, 0., 0.])
ukf.R = np.diag([[std_x[0]**2, std_x[0]**2]])
#ukf.H = np.array([[1]])
ukf.Q = np.diag([100., 100., 1000., 1000.])
uxs = []
trux = []
truv = []
measx = []
time = []
covarx = []
covarv = []
zs = np.arange(0, 1435 + dt, dt)

f = falling_object(starting_conditions[0:2], starting_conditions[2:4], starting_conditions[4], std_x, std_model)
t = 0
while f.x[1] >= 0:
    ukf.predict()
    f.measure()
    print(f.x[1])
    ukf.update([f.meas[0], f.meas[1]])
    measx.append(f.meas[1])
    uxs.append(ukf.x.copy())
    trux.append(f.x.copy())
    truv.append(f.v.copy())
    print("COVARIANCE = ")
    print(ukf.P)
    print()
    covarx.append(np.sqrt(ukf.P[0][0]))
    covarv.append(np.sqrt(ukf.P[1][1]))
    t += dt
    time.append(t)
    f.update(dt)

uxs = np.array(uxs)
trux = np.array(trux)
print(trux)
truv = np.array(truv)
measx = np.array(measx)

print(len(covarx))
print(len(uxs))
plt.figure(1)
plt.plot(time, (uxs[:, 3] - truv[:, 1]), label='velocity difference')
plt.plot(time, np.array(covarv)*3, 'r')
plt.plot(time, np.array(covarv)*-3, 'r')
plt.title('non linear eqn velocity error')
plt.legend()

# mean_V_error = np.mean(uxs[:, 1] - truv)
# std_v_error = np.std(uxs[:, 1] - truv)
# mean_x_error = np.mean(uxs[:, 0] - trux)
# std_x_error = np.std(uxs[:, 0] - trux)

# print("mean x error = ", mean_x_error)
# print("std x error = ", std_x_error)
# print("mean v error = ", mean_V_error)
# print("std v error = ", std_v_error)

plt.figure(2)
plt.plot(time, uxs[:, 3], label='filter')
plt.plot(time, truv[:,1], label=('model'))
#plt.scatter(time, measx, label='measure')
plt.title('non linear eqn velocity')
plt.legend()

plt.figure(3)
plt.plot(time, (uxs[:, 1]-trux[:,1]), 'y', label='position difference')
plt.plot(time, np.array(covarx)*3, 'r')
plt.plot(time, np.array(covarx)*-3, 'r')
plt.title('non linear eqn position')
plt.legend()


plt.figure(4)
plt.plot(time, uxs[:, 1], 'y', label='filter')
plt.plot(time, trux[:,1], 'b', label='model')
plt.scatter(time, measx, s=1, label='measure')
plt.title('non linear eqn position')
plt.legend()

plt.show()
#print('UKF standard deviation {:.3f} meters'.format(np.std(uxs - xs)))
