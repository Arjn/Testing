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
def model(y, t):
    g = -9.81
    b = 200
    x, v = y
    dvdt = [v, (-0.0034 * g * np.exp(-x / 22000) * v**2) / (2 * b) + g]
    return dvdt

#
# y0 = [1e6, 0.0]
# t = np.linspace(0, 480, 101)
# sol = odeint(model, y0, t)
# plt.plot(t, sol[:, 0])



dt = 0.1
std_x = 0.01
g = 9.81
np.random.seed(200)


class falling_object(object):
    def __init__(self, pos, vel, b, std):
        self.x = pos
        self.v = vel
        self.b = b
        self.std_x = std
        self.meas = [0, 0]

    def update(self, dt):
        self.x += self.v*dt
        self.v += ((0.0034 * g * np.exp(-self.x / 22000) * (self.v ** 2)) / (2 * self.b) - g)*dt
        return self.x

    def measure(self):
        self.meas[0] = randn()*self.std_x * self.x + self.x
        self.meas[1] = randn() * self.std_x * self.v + self.v
        return self.meas



def f_cv(x, dt):
    """ state transition function for a 1D
    accelerating object"""
    g = -9.81
    x[0] = x[0] + x[1]*dt
    x[1] = x[1] + g*dt
    return x


def h_cv(x):
    """Measurement function -
        measuring only position"""
    return np.array([x[0]])


starting_conditions = [1e5, 0., 2000]

points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=0)
ukf = UKF(dim_x=2, dim_z=1, fx=f_cv, hx=h_cv, dt=dt, points=points)
ukf.x = np.array([1.001e5, 0.])
ukf.R = np.array([[1e3]])
#ukf.H = np.array([[1]])
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.5)
uxs = []
trux = []
truv = []
measx = []
zs = np.arange(0, 465 + dt, dt)
time = []
t = 0

f = falling_object(starting_conditions[0], starting_conditions[1], starting_conditions[2], std_x)

while f.x >= 0:
    f.update(dt)

    f.measure()
    ukf.predict()
    ukf.update(f.meas[0])
    uxs.append(ukf.x.copy())
    measx.append(f.meas[0])
    trux.append(f.x)
    truv.append(f.v)
    t += dt
    time.append(t)
uxs = np.array(uxs)

plt.figure(1)
plt.plot(time, (uxs[:, 1] - truv), label='velocity difference')

plt.title('linear eqn velocity error')
plt.grid()
plt.legend()

plt.figure(2)
plt.plot(time, uxs[:, 1], label='filter')
plt.plot(time, truv, label=('model'))
#plt.scatter(zs, measx, label='measure')
plt.title('linear eqn velocity')
plt.grid()
plt.legend()

plt.figure(3)
plt.plot(time, (uxs[:, 0]-trux), 'y', label='position difference')
plt.title('linear eqn position error')
plt.grid()
plt.legend()
#
#
plt.figure(4)
plt.plot(time, uxs[:, 0], 'y', label='filter')
plt.plot(time, trux, 'b', label='model')
plt.scatter(time, measx, s=1, label='measure')
plt.title('linear eqn position')
plt.grid()
plt.legend()
#
plt.show()
#print('UKF standard deviation {:.3f} meters'.format(np.std(uxs - xs)))
