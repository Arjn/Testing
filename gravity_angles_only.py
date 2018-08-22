# from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import Filter
import scipy.stats as stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.integrate import odeint
from numpy.random import randn
from scipy.linalg import inv
from numpy.random import rand
from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt
import decimal
import math
import csv
import copy
from mpl_toolkits.mplot3d import Axes3D

cmap = plt.cm.get_cmap('Paired')
ANGLES_ONLY = False

def zero_inv(Q):
    """
    This function is used in the FIM calculation process. For the gravitational dynamics, the noise comes from unknown
    force perturbations which affect the velocity directly - not the position. However to calculate the FIM, the inverse
    of the Q matrix is required. To get around this, the matrix is resized and only the noise contributing factors (non-
    zero) components are inverted.
    :param Q: noise matrix
    :return: inverted noise matrix
    """

    temp = np.diag([Q[2,2], Q[3,3], Q[4,4]])
    temp = inv(temp)
    return np.diag([Q[0,0], Q[1,1], temp[0,0], temp[1,1],temp[2,2]])

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    if result.size % 2 != 0:
        t = result.size - 1
    else:
        t = result.size
    # print(t)
    return result[int(t/2):]/max(result[int(t/2):])



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
X_t = []
with open('true_pos', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        temp = []
        for i in row:
            temp.extend([float(i)])
        X_t.append(temp)
X_t = np.array(X_t).T

constants = [-0.59783, 13.406, 3.986e5, 6374.]

tracker_noise = 0.17e-3 #rad
tracker_rng_noise = 1e-3

dt = 0.1
np.random.seed(5000)
P = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1.])
Q = np.diag([1e-2, 1e-2, 2.4062e-5, 2.4062e-5, 1e-6])
# Q = np.array([[dt**6/9 + dt**4/4, 0, dt**5/6 + dt**3/2, 0, 0], [0, dt**6/9 + dt**4/4, 0, dt**5/6 + dt**3/2, 0], [dt**5/6 + dt**3/2, 0, dt**4/4 + dt**2, 0, 0], [0, dt**5/6 + dt**3/2, 0, dt**4/4 + dt**2, 0], [0, 0, 0, 0, 0]])*2.4062e-5
std_true_object = np.array([np.sqrt(2.4062e-5), np.sqrt(2.4062e-5), 0])
est_starting_conditions = np.array([6500.4, 349.14, -1.8093, -6.7967, 0])#np.array([500., -1000., 1e5, 100., 100., 500.])
true_starting_conditions = np.array([6500.4+normal(0,np.sqrt(P[0,0])), 349.14+normal(0,np.sqrt(P[1,1])), -1.8093+normal(0,np.sqrt(P[2,2])),
                                    -6.7967+normal(0,np.sqrt(P[3,3])), 0.6932 ])# make library of positions of observers
observer_position = np.array([[6374., 0]])
observer_std = np.array([tracker_noise, tracker_rng_noise])


kf_update_rate = 0.1
sensor_sampling_time = 0.1
# Q = np.array([[sensor_sampling_time ** 3 * phi/3, sensor_sampling_time ** 2 * phi/2, 0, 0],
#  [sensor_sampling_time ** 2 * phi / 2, sensor_sampling_time * phi, 0, 0],
#  [0, 0, sensor_sampling_time ** 3 * phi/3, sensor_sampling_time ** 2 * phi/2],
#  [0, 0, sensor_sampling_time ** 3 * phi / 3, sensor_sampling_time ** 2 * phi / 2]])
# Q_range = [0,100]



#parameters = np.concatenate([[tracker_noise, dt], std_true_object, est_starting_conditions, true_starting_conditions, P, Q])

def get_val_from_range(range, plus_minus=None):
    t = rand(1,1)
    t = -1 if t[0][0]<0.5 else 1
    x = rand(1,1)
    result = (x[0][0]*(range[1]-range[0]) + range[0])*t if plus_minus is True else x[0][0]*(range[1]-range[0]) +range[0]
    return result

def norm_angles(ang):
    ang_temp = np.mod(ang, 2*np.pi)
    # print("\n\n\n\n\n")
    # print(ang-ang_temp)
    # print("\n\n\n\n\n")
    if ang_temp > np.pi and abs(ang-(ang_temp-2*np.pi)) > 1e-4:
        ang = ang_temp - 2*np.pi
    return ang


class falling_object(object):
    def __init__(self, X, std_mdl, b0, H0, mu, R0):
        self.X = X
        self.std_model = std_mdl
        self.meas = [0, 0, 0]
        self.b0 = b0
        self.H0 = H0
        self.mu = mu
        self.R0 = R0

    def update(self, dt):
        #Re-entry dynamics equations
        self.b = self.b0*np.exp(self.X[4])
        self.R = np.sqrt(self.X[0] **2 + self.X[1] **2)
        self.V = np.sqrt(self.X[2] ** 2 + self.X[3] ** 2)
        self.D = self.b*np.exp((self.R0 - self.R)/self.H0)*self.V
        self.G = -self.mu/self.R**3



        self.X[0] += self.X[2] * dt
        self.X[1] += self.X[3] * dt
        self.X[2] += (self.D*self.X[2] + self.G*self.X[0])*dt + np.random.normal(0, self.std_model[0])
        self.X[3] += (self.D*self.X[3] + self.G*self.X[1])*dt + np.random.normal(0, self.std_model[1])
        self.X[4] += np.random.normal(0, self.std_model[2])

        self.F_Dot(dt)

        return self.X

    def F_Dot(self, dt):
        self.f_bar = []
        dxdX = np.array([1,0,dt,0,0])
        self.f_bar.append(dxdX)

        dydX = np.array([0,1,0,dt,0])
        self.f_bar.append(dydX)

        dvxdX= np.array([-((self.X[0]*self.D*self.X[2])/self.R*self.H0 + (3*self.mu*self.X[0]**2)/self.R**5 - self.mu/self.R**3)*dt,
                         -(self.X[1]*self.D*self.X[2]/self.R*self.H0 + 3*self.mu*self.X[0]*self.X[1]/self.R**5)*dt,
                         (self.D*self.X[2]**2/self.V**2 + self.D)*dt + 1,
                         (self.D*self.X[2]*self.X[3]/self.V**2)*dt,
                         (self.D*self.X[2])*dt])
        self.f_bar.append(dvxdX)

        dvydX= np.array([-(self.X[0]*self.D*self.X[3]/self.R*self.H0 + 3*self.mu*self.X[0]*self.X[1]/self.R**5)*dt,
                        -((self.X[1]*self.D*self.X[3])/self.R*self.H0 + (3*self.mu*self.X[1]**2)/self.R**5 - self.mu/self.R**3)*dt,
                         (self.D*self.X[3]**2/self.V**2 + self.D)*dt,
                         (self.D*self.X[2]*self.X[3]/self.V**2)*dt + 1,
                         (self.D*self.X[3])*dt])
        self.f_bar.append(dvydX)

        dBdX = np.array([0,0,0,0,1])
        self.f_bar.append(dBdX)


        self.f_bar = np.array(self.f_bar)


def f_cv(X, dt):
    """ state transition function for a 3D
    accelerating object under gravity"""

    b0, H0, mu, R0 = constants

    b = b0 * np.exp(X[4])
    R = np.sqrt(X[0] ** 2 + X[1] ** 2)
    V = np.sqrt(X[2] ** 2 + X[3] ** 2)
    D = b * np.exp((R0 - R) /H0) * V
    G = -mu / R ** 3

    X[0] += X[2] * dt
    X[1] += X[3] * dt
    X[2] += (D * X[2] + G * X[0]) * dt
    X[3] += (D * X[3] + G * X[1]) * dt
    X[4] = X[4]
    return X


def h_observer(x, marks):
    """Measurement function -
    measuring only position"""
    estimated_obs_angles = []
    for i in range(0, len(marks)):
        dX = [x[0] - marks[i][0],x[1] - marks[i][1]]
        r = np.sqrt(dX[0]**2 + dX[1]**2)
        # theta = np.arccos(dX[2]/r)
        phi = np.arctan2(dX[1], dX[0])
        estimated_obs_angles.extend([phi, r]) if ANGLES_ONLY is False else estimated_obs_angles.extend(
            [phi])
        # print(np.array(estimated_obs_angles))

    return np.array(estimated_obs_angles)

def residual_h(a, b):
    if any(isinstance(el, list) for el in a):
        a = [item for sublist in a for item in sublist]
    if any(isinstance(el, list) for el in b):
        b = [item for sublist in b for item in sublist]

    y = a - b

    if ANGLES_ONLY is False:
        # data in format [[theta1, r], [theta2, r], ...]
        j = 0
        for i in range(0, len(y)):
            if j is not 1:
                q = y[i]
                y[i] = norm_angles(y[i])
                j += 1
            else:
                j = 0
            # if q != y[i]:
            #     print("NORMALISED ANGLES!", b, "!=", y[i])
    else:
        for i in range(0, len(y)):
            q = y[i]
            y[i] = norm_angles(y[i])
            # if q != y[i]:
            #     print("NORMALISED ANGLES!", b, "!=", y[i])
    return y

# def residual_x(a, b):
#     y = a - b
#     for i in range(0, len(y)-1):
#         y[i] = norm_angles(y[i])
#     return y

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    if ANGLES_ONLY is False:
        # data in format [[theta1, phi1, r], [theta2, phi2, r], ...]

        j = 0
        for z in range(0, z_count):
            if j is not 1:
                sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]), Wm))
                sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]), Wm))

                x[z] = np.arctan2(sum_sin1, sum_cos1)
                j += 1
            else:
                x[z] = np.sum(np.dot(sigmas[:, z], Wm))
                j = 0

    else:
        for z in range(0, z_count):
            sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]), Wm))
            sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]), Wm))

            x[z] = np.arctan2(sum_sin1, sum_cos1)

    return x


class AngSensor(object):
    def __init__(self, observer_pos, stds, update):
        self.pos = observer_pos
        self.r_std = 0 if ANGLES_ONLY else stds[2]#stds[0]
        self.theta_std = stds[0]
        self.phi_std = stds[1]
        self.r = []
        self.r_true = np.zeros([len(self.pos), 1])
        self.theta_true = np.zeros([len(self.pos), 1])
        self.phi_true = np.zeros([len(self.pos), 1])
        self.storage_phi = [[0]]
        self.theta = []
        self.phi = []
        self.time = 0
        self.update_rate = update
        self.measure_ready = False

    def observe(self, x):
        for i in range(0, len(self.pos)):
            self.dX = [x[0] - self.pos[i][0], x[1]-self.pos[i][1]]
            self.r_true[i] = np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2)
            self.phi_true[i] = np.arctan2(self.dX[1], self.dX[0])
            self.H_bar(x)


    def H_bar(self,x):
        self.h_bar = []
        drdx = [x[0]/self.r_true[0][0], x[1]/self.r_true[0][0],0,0,0]
        self.h_bar.append(drdx)

        dthetadx = [-self.dX[1]/(self.dX[0]**2 + self.dX[1]**2), 1/(self.dX[0]**2 + self.dX[1]**2),
                             0,0,0]
        self.h_bar.append(dthetadx)
        self.h_bar = np.array(self.h_bar)
        # print(self.h_bar)

    def noisy_observe(self, x, time):
        self.measure_ready = False
        self.observe(x)
        self.measure = []
        self.theta = []
        self.phi = []
        #print(np.mod(time, self.update_rate))
        if self.time >= self.update_rate:
            # print("TRUE")
            self.time = 0
            for i in range(0, len(self.pos)):
                self.r = self.r_true[i] + np.random.normal(0, self.r_std)
                self.phi.append(self.phi_true[i] + np.random.normal(0, self.phi_std))
                #print(self.theta, self.phi)
                if ANGLES_ONLY:
                    mix = [float(self.phi[i])]
                else:
                    mix = [float(self.phi[i]), float(self.r)]
                self.measure.extend(mix)
            for i in range(0, len(self.pos),2):
                self.storage_phi[i].append(self.measure[i+1])
            self.measure = np.array(self.measure)
            self.measure_ready = True

x_error_storage = []
v_error_storage = []

for iterations in range(0,1):
    print(iterations)
    true_starting_conditions = np.array([6500.4 + normal(0, np.sqrt(P[0, 0])), 349.14 + normal(0, np.sqrt(P[1, 1])),
                                     -1.8093 + normal(0, np.sqrt(P[2, 2])),
                                     -6.7967 + normal(0, np.sqrt(P[3, 3])), 0.6932])
    Q_storage = []
    Q_best = 0
    Q_best_num = 10000000
    # for i in range(0,100):
    #     print(i)
    #     true_starting_conditions = np.array([0., 0., 3000 * np.cos(np.deg2rad(45)), 3000 * np.sin(
    #         np.deg2rad(45))])  # np.array([500., -1000., 1e5, 100., 100., 500.])
    #     est_starting_conditions = np.array([10, 0, 3010 * np.cos(np.deg2rad(45)), 3010 * np.sin(np.deg2rad(45))])
    h_observer.pos = observer_position

    dec = decimal.Decimal(str(dt))
    dec = abs(dec.as_tuple().exponent)
    points = Filter.MerweScaledSigmaPoints(n=5, alpha=1, beta=2., kappa=1)
    mult = 1 if ANGLES_ONLY else 2
    ukf = Filter.UKF(dim_x=5, dim_z=mult * len(observer_position), fx=f_cv, hx=h_observer, dt=dt, points=points,
              z_mean_fn=z_mean, residual_z=residual_h)
    ukf.x = np.array([est_starting_conditions])
    if ANGLES_ONLY:
        observer_std = np.array([tracker_noise, tracker_noise])
        ukf.R_store = np.diag([observer_std[0] ** 2])
    else:
        observer_std = np.array([tracker_noise, tracker_noise, tracker_rng_noise])
        ukf.R_store = np.diag([(observer_std[0])**2, (observer_std[2])**2])
    ukf.R = ukf.R_store
    # ukf.H = np.array([[1]])
    ukf.P = P

    # ukf.Q = np.diag([get_val_from_range(Q_range),get_val_from_range(Q_range), get_val_from_range(Q_range),
    #                  get_val_from_range(Q_range)])#Q_matrix(dt, Q_variance) #np.diag([0.0, 0.0, 0.0, 00., 00., 00.])#
    uxs = []
    trux = []
    truv = []
    measx = []
    time = []
    covarx = []
    covary = []
    covarv = []
    kalman_t = []
    meas_pos = []
    update_time = []
    zs = np.arange(0, 1435 + dt, dt)
    ukf.Q = Q
    f = falling_object(true_starting_conditions, std_true_object, constants[0], constants[1], constants[2], constants[3])
    station = AngSensor(observer_position, observer_std, sensor_sampling_time)
    t = 0
    i = 0
    J = inv(ukf.P)
    J_storage = []
    while t < 200:
        f.update(dt)
        # f.X = X_t[i, 0:4]
        station.time += dt
        station.observe(f.X)
        station.noisy_observe(f.X, t)
        ukf.predict()
        if station.measure_ready is True:
            # print("UPDATE")
            ukf.update(station.measure, hx_args=(observer_position,))
            kalman_t.append(t)
            measx.append([station.r, station.phi])
            x_m = station.r * np.cos(station.phi) + observer_position[0][0]
            y_m = station.r * np.sin(station.phi) + observer_position[0][1]
            meas_pos.append(np.array([(f.X[0]-x_m).flatten(), (f.X[1] - y_m).flatten()]).flatten())
            update_time.append(t)
            uxs.append(ukf.x.copy())
            trux.append(f.X.copy())
            time.append(t)
            covarx.append(np.mean([ukf.P[0][0], ukf.P[1][1]]))
            covary.append(ukf.P[2][2])
            covarv.append(np.mean([ukf.P[2][2], ukf.P[3][3]]))
            J = inv(ukf.Q) + np.dot(station.h_bar.T, inv(ukf.R)).dot(station.h_bar) - np.dot(np.dot(inv(ukf.Q), f.f_bar),
                inv(J + np.dot(f.f_bar.T, inv(ukf.Q)).dot(f.f_bar)), np.dot(inv(ukf.Q), f.f_bar))

            J_storage.append(np.diag(J))


        t = round(t+dt, dec)

    #print("Z = ", f.X[2], "TIME = ", t)
    #print(ukf.x[1])
    #print(ukf.P)
    #print([station.r, station.theta, station.phi])
    # print(t)
    i += 1
    J_storage = np.array(J_storage)
    CRLB = (1/(J_storage))
    uxs = np.array(uxs)
    trux = np.array(trux)



    mean_x_error = [np.mean(np.sqrt((uxs[:, 0] - trux[:, 0])**2)), np.mean(np.sqrt((uxs[:, 1] - trux[:, 1])**2))]
    mean_v_error = [np.mean(np.sqrt((uxs[:, 2] - trux[:, 2])**2)), np.mean(np.sqrt((uxs[:, 3] - trux[:, 3])**2))]

    x_error_storage.append(mean_x_error)
    v_error_storage.append(mean_v_error)
np.savetxt("trux.csv", trux, delimiter=",")
np.savetxt("x_error_storage.csv", np.array(x_error_storage), delimiter=",")
np.savetxt('v_error_storage.csv', np.array(v_error_storage), delimiter=",")

print('mean x error = %f' % np.mean(np.array(x_error_storage)))
print('mean v error = %f' % np.mean(np.array(v_error_storage)))

std_x = [np.std(abs(uxs[:, 0] - trux[:, 0])), np.std(abs(uxs[:, 1] - trux[:, 1]))]
std_v = [np.std(abs(uxs[:, 2] - trux[:, 2])), np.std(abs(uxs[:, 3] - trux[:, 3]))]
#
print("mean x error = ", mean_x_error)
print("std x error = ", std_x)
print("mean v error = ", mean_v_error)
print("std v error = ", std_v)
sum = np.sum(mean_x_error) + np.sum(mean_v_error) + np.sum(std_x) + np.sum(std_v)
print(sum)
    # if sum < Q_best_num:
    #     Q_best = ukf.Q
    #     Q_best_num = sum
    #     print(sum)
    #     print(Q_best)
    # temp = [Q]
    # Q_storage.append(np.concatenate([np.diag(ukf.Q), mean_x_error, std_x, mean_v_error, std_v]))
# station.storage_phi = np.array(station.storage_phi)
# station.storage_theta = np.array(station.storage_theta)
# print(Q_best)
# strings = ['QX','QY','QVX','QVY']
# Q_storage = np.array(Q_storage)
# for i in range(0,4):
#     plt.figure(i+1)
#     plt.plot(Q_storage[:,i], Q_storage[:,4], '*', label='mean x')
#     plt.plot(Q_storage[:,i], Q_storage[:,5],'*', label='mean y')
#     plt.plot(Q_storage[:,i], Q_storage[:,6],'*', label='std x')
#     plt.plot(Q_storage[:,i], Q_storage[:,7],'*', label='std y')
#     plt.plot(Q_storage[:,i], Q_storage[:,8],'*', label='mean vx')
#     plt.plot(Q_storage[:, i], Q_storage[:, 9],'*', label='mean vy')
#     plt.plot(Q_storage[:, i], Q_storage[:, 10],'*', label='std vx')
#     plt.plot(Q_storage[:, i], Q_storage[:, 11],'*', label='std vy')
#     plt.xlabel('Q_value')
#     plt.ylabel('means and stds')
#     plt.yscale('log')
#     plt.title('%s' % strings[i])
#     plt.legend()


# # VELOCITY ERROR
plt.figure(1)
plt.plot(time, abs(uxs[:, 2] -  trux[:, 2]), 'xkcd:blue', label='x-error',linewidth=1.0)
plt.plot(time, abs(uxs[:, 3] -  trux[:,3]),'xkcd:green', label='y-error',linewidth=1.0)
# plt.plot(time, (uxs[:, 5] - trux[:, 5]), 'xkcd:orange', label='z-error',linewidth=3.0)
# plt.plot(time, np.array(covarv)*-3, 'r')
plt.plot(update_time, np.sqrt(CRLB[:,2]**2 + CRLB[:,3]**2), label='CRLB vx')
plt.plot(time, np.array(covarv)*3, 'r')
plt.title('Velocity Error Re-entry Problem')
plt.xlabel('Time [s]')
plt.ylabel('Error [km/s]')
plt.yscale('log')
plt.legend()

np.savetxt("ukf_v_error.csv", abs(uxs[:, 2] -  trux[:, 2]), delimiter=",")
#
# POSITION ERROR
plt.figure(2)
plt.plot(update_time, abs(np.array(meas_pos)[:,0]),'xkcd:blue', label='measurements-only x', linewidth=0.1)
plt.plot(update_time, abs(np.array(meas_pos)[:,1]),'xkcd:green', label='measurements-only y', linewidth=0.1)
plt.plot(time, abs((uxs[:, 0] -  trux[:, 0])), 'xkcd:royal blue', label='x-error', linewidth=1.0)
plt.plot(time, np.array(covarx)*3, 'xkcd:red', label='position-covariance')
# plt.plot(time, np.multiply(-3,np.array(covarx)), 'xkcd:black', label='position-covariance')
plt.plot(time, abs(uxs[:, 1] -  trux[:, 1]), 'xkcd:bright green', label='y-error', linewidth=1.0)
plt.plot(update_time, np.sqrt(CRLB[:,0]**2+ CRLB[:,1]**2), label='CRLB x')
# plt.plot(time, (uxs[:, 2] - trux[:, 2]), 'xkcd:orange', label='z-error', linewidth=3.0)
plt.yscale('log')
#plt.plot(time, np.array(covarx)*3, 'r')
#plt.plot(time, np.array(covarx)*-3, 'r')
plt.title('Position Error Re-entry Problem')
plt.xlabel('Time [s]')
plt.ylabel('Error [km]')
plt.legend()

plt.figure(8)
plt.plot(time, autocorr((uxs[:, 0] -  trux[:, 0])), label='x-autocorrelation')
plt.plot(time, autocorr((uxs[:, 1] -  trux[:, 1])), label='y-autocorrelation')
plt.title('error autocorrelation')
plt.legend()

plt.figure(7)


plt.title('position error based only on measurements')
plt.legend()

# VELOCITY
plt.figure(3)
plt.plot(time, uxs[:, 2], 'xkcd:green', label='filter vx', linewidth=3.0)
plt.plot(time, uxs[:, 3], 'xkcd:orange', label='filter vy', linewidth=3.0)
plt.plot(time,  trux[:, 2],linestyle='--', color='xkcd:dark green', label=('model x'))
plt.plot(time,  trux[:, 3], linestyle='--', color='xkcd:dark orange', label=('model y'))
#plt.scatter(time, measx, label='measure')
plt.title('non linear eqn velocity')
plt.legend()


# POSITION IN TIME
plt.figure(5)
plt.plot(time, uxs[:, 0], 'xkcd:green', label='filter x', linewidth=1.0)
plt.plot(time, uxs[:, 1], 'xkcd:orange', label='filter y', linewidth=1.0)
plt.plot(time,  trux[:, 0], linestyle='--', color='xkcd:dark green', label=('model x'))
plt.plot(time,  trux[:, 1], linestyle='--', color='xkcd:dark orange', label=('model y'))
# #plt.scatter(time, measx, s=2, label='measure')
plt.title('non linear eqn position')
plt.yscale('log')
plt.legend()

# POSITION IN SPACE
plt.figure(6)
plt.plot(uxs[:, 0], uxs[:, 1], 'xkcd:blue', label='filter POS 2D')
plt.plot( X_t[:, 0],  X_t[:, 1], linestyle='--', color='xkcd:dark blue', label=('model'))
plt.scatter(observer_position[:][:, 0], observer_position[:][:, 1], marker='x', label='observer')
plt.title('non linear eqn position')
plt.legend()

plt.figure(7)
plt.plot(time, uxs[:, 4] , 'xkcd:royal blue', label='aero_error', linewidth=1.0)
plt.plot(time, trux[:, 4], label='true aero', linewidth=1.0)
plt.title('Ballistic coefficient')
plt.legend()

#Observations
# plt.figure(7)
# for i in range(0, len(observer_position)):
#     if(len(station.storage_theta[i][1:]) > 2):
#         plt.plot(time[:], np.array(station.storage_theta[i][1:]), 'b', label='theta')
#         plt.plot(time[:], np.array(station.storage_phi[i][1:]), 'r', label='phi')
# plt.title('observations')
# plt.legend()

plt.show()
# #print('UKF standard deviation {:.3f} meters'.format(np.std(uxs - xs)))
