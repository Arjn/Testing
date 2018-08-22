from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import scipy.stats as stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.integrate import odeint
from numpy.random import randn
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
import decimal
import math
from mpl_toolkits.mplot3d import Axes3D
import traceback

cmap = plt.cm.get_cmap('Paired')

ANGLES_ONLY = True

def mag(x):
    q = 0
    for i in len(x):
        q += i ** 2
    return np.sqrt(q)

def get_val_from_range(range, plus_minus=None):
    t = rand(1,1)
    t = -1 if t[0][0]<0.5 else 1
    x = rand(1,1)
    result = (x[0][0]*(range[1]-range[0]) + range[0])*t if plus_minus is True else x[0][0]*(range[1]-range[0]) +range[0]
    return result

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


# observer_position = np.array([[-1.e1, 1.e1, 1.e1],[-1.e2, -1.e2, 0.],[1.e3, -1.e3, 1.e3], [1.e4, 0., 1e4],
#                               [0, 1e6, 1e6], [0, -1e5, -1e5]])

observer_position = np.array([[-3.e11, 1.e11, -1.e11],[-1.e11, -1.e11, 100.],[1.e11, -1.e11, 1.e11], [1.e11, 0., 1e11], [1e11, 1e11, 1e11]])
true_starting_conditions = np.array([10., 10., 1e7, 100., 100., 500.])

tracker_noise = 0.09225998470098455 #ARCSECOND

dt = 0.100
np.random.seed(500)
true_b = 400.
std_true_object = np.array([104, 105, 103,  13.,  10., 22.])
true_starting_conditions = np.array([10., 10., 1e7, 1000., 1000., 500.])#np.array([500., -1000., 1e5, 100., 100., 500.])
est_starting_conditions = np.array([1.14047361e+02, 1.15387855e+02, 1.00001038e+07, 1.13055514e+02, 1.10708489e+02, 5.22758571e+02])# make library of positions of observers
observer_position = np.array([[-3.e1, 1.e1, -1.e1],[-1.e3, -1.e3, 100.],[1.e4, -1.e4, 1.e4], [1.e5, 0., 1e5], [1e6, 1e6, 1e7]])
observer_std = np.array([np.deg2rad(tracker_noise/3600), np.deg2rad(tracker_noise/3600)])
P = np.diag([10178, 10178, 10178, 406 , 406, 406])
Q = np.diag([9.9, 9.99, 9.99, 0.8, 0.8, 0.8])
tracker_rng_noise = 1e1
sensor_sampling_time = 0.5

g = 9.81

alpha_range = [-1,1]
beta_range = [0,10]
kappa_range = [-5,5]

def norm_angles(ang):
    ang_temp = np.mod(ang, 2*np.pi)
    # print("\n\n\n\n\n")
    # print(ang-ang_temp)
    # print("\n\n\n\n\n")
    if ang_temp > np.pi and abs(ang-(ang_temp-2*np.pi)) > 1e-4:
        ang = ang_temp - 2*np.pi
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
    return x


def h_observer(x, marks):
    """Measurement function -
    measuring only position"""
    estimated_obs_angles = []
    for i in range(0, len(marks)):
        dX = [marks[i][0] - x[0], marks[i][1] - x[1], marks[i][2] - x[2]]
        r = np.sqrt(dX[0]**2 + dX[1]**2 + dX[2]**2)
        theta = np.arccos(dX[2]/r)
        phi = np.arctan2(dX[1], dX[0])
        estimated_obs_angles.extend([theta, phi, r]) if ANGLES_ONLY is False else estimated_obs_angles.extend([theta, phi])
    #print(np.array(estimated_obs_angles))
    return np.array(estimated_obs_angles)

def residual_h(a, b):
    if any(isinstance(el, list) for el in a):
        a = [item for sublist in a for item in sublist]
    if any(isinstance(el, list) for el in b):
        b = [item for sublist in b for item in sublist]

    y = a - b

    if ANGLES_ONLY is False:
        # data in format [[theta1, phi1, r], [theta2, phi2, r], ...]
        j = 0
        for i in range(0, len(y)):
            if j is not 2:
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
            if j is not 2:
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
        self.r_std = 0#stds[0]
        self.theta_std = stds[0]
        self.phi_std = stds[1]
        self.r = []
        self.r_true = np.zeros([len(self.pos), 1])
        self.theta_true = np.zeros([len(self.pos), 1])
        self.phi_true = np.zeros([len(self.pos), 1])
        self.storage_theta = [[0], [0], [0], [0], [0], [0]]
        self.storage_phi = [[0], [0], [0], [0], [0], [0]]
        self.theta = []
        self.phi = []
        self.update_rate = update
        self.measure_ready = False
        self.internal_clock = 0

    def observe(self, x):
        for i in range(0, len(self.pos)):
            self.dX = [self.pos[i][0]-x[0], self.pos[i][1]-x[1], self.pos[i][2]-x[2]]
            self.r_true[i] = np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2)
            self.theta_true[i] = np.arccos(self.dX[2] / self.r_true[i])
            self.phi_true[i] = np.arctan2(self.dX[1], self.dX[0])

    def noisy_observe(self, x, time):
        self.measure_ready = False
        self.observe(x)
        self.measure = []
        self.theta = []
        self.phi = []
        #print(np.mod(time, self.update_rate))
        if self.internal_clock >= self.update_rate:
            #print("TRUE")
            for i in range(0, len(self.pos)):
                self.r = self.r_true[i] + randn()*self.r_std
                self.theta.append(self.theta_true[i] + randn()*self.theta_std)
                self.phi.append(self.phi_true[i] + randn()*self.phi_std)
                #print(self.theta, self.phi)
                if ANGLES_ONLY:
                    mix = [float(self.theta[i]), float(self.phi[i])]
                else:
                    mix = [float(self.theta[i]), float(self.phi[i]), float(self.r)]
                self.measure.extend(mix)
            for i in range(0, len(self.pos),2):
                self.storage_theta[i].append(self.measure[i])
                self.storage_phi[i].append(self.measure[i+1])
            self.measure = np.array(self.measure)
            self.measure_ready = True
            self.internal_clock = 0
        self.internal_clock += (time - self.internal_clock)


h_observer.pos = observer_position

start_error_x = []
start_error_v = []

end_error_x = []
end_error_v = []

good_x = 0
bad_x = 0

good_v = 0
bad_v = 0

failed = 0
plt.figure(1)
mean_pos = []
param_save_good = []
good_errors = []
param_save_bad = []
bad_errors = []
mean_vel_good = []
mean_vel_bad = []
mean_pos_good = []
mean_pos_bad = []
iteration_num = 0
for i in range(0,3000):
    print(i)
    print()

    parameters = np.concatenate([[tracker_noise, dt], std_true_object, est_starting_conditions, true_starting_conditions, P, Q])

    dec = decimal.Decimal(str(dt))
    dec = abs(dec.as_tuple().exponent)
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-1.)
    mult = 2 if ANGLES_ONLY else 3
    ukf = UKF(dim_x=6, dim_z=mult*len(observer_position), fx=f_cv, hx=h_observer, dt=dt, points=points, z_mean_fn=z_mean,
              residual_z=residual_h)
    ukf.x = np.array([est_starting_conditions])
    if ANGLES_ONLY:
        observer_std = np.array([np.deg2rad(tracker_noise / 3600), np.deg2rad(tracker_noise / 3600)])
        ukf.R_store = np.diag([observer_std[0] ** 2, observer_std[1] ** 2])
    else:
        observer_std = np.array([np.deg2rad(tracker_noise / 3600), np.deg2rad(tracker_noise / 3600), tracker_rng_noise])
        ukf.R_store = np.diag([observer_std[0]**2, observer_std[1]**2, observer_std[2]**2])

    ukf.R = ukf.R_store
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
    kalman_t = []
    zs = np.arange(0, 1435 + dt, dt)

    f = falling_object(true_starting_conditions, std_true_object, true_b)
    station = AngSensor(observer_position, observer_std, sensor_sampling_time)
    t = 0
    print("dt = %f" % dt)
    while f.X[2] >= 0:
        f.update(dt)
        station.noisy_observe(f.X, t)
        try:
            ukf.predict()
        except:
            print("Not worked")
            print()
            break

        if station.measure_ready is True:
            #print("UPDATE")
            try:
                ukf.update(station.measure, hx_args=(observer_position,))
            except:
                print("Not worked")
                print()
                break
            kalman_t.append(t)
        measx.append(station.r)
        uxs.append(ukf.x.copy())
        trux.append(f.X.copy())
        covarx.append(np.mean([ukf.P[0][0],ukf.P[1][1],ukf.P[2][2]]))
        covarv.append(np.mean([ukf.P[3][3], ukf.P[4][4], ukf.P[5][5]]))
        t = round(t+dt, dec)
        time.append(t)


    uxs = np.array(uxs)
    trux = np.array(trux)



    if len(uxs) > 1:

        uxs_pos = np.sqrt(uxs[:, 0] ** 2 + uxs[:, 1] ** 2 + uxs[:, 2] ** 2)
        uxs_speed = np.sqrt(uxs[:, 3] ** 2 + uxs[:, 4] ** 2 + uxs[:, 5] ** 2)
        trux_pos = np.sqrt(trux[:, 0] ** 2 + trux[:, 1] ** 2 + trux[:, 2] ** 2)
        trux_speed = np.sqrt(trux[:, 3] ** 2 + trux[:, 4] ** 2 + trux[:, 5] ** 2)

        start_error_x.extend([uxs_pos[0] - trux_pos[0]])
        end_error_x.extend([uxs_pos[-1] - trux_pos[-1]])

        start_error_v.extend([uxs_speed[0] - trux_speed[0]])
        end_error_v.extend([uxs_speed[-1] - trux_speed[-1]])

        mean_pos.extend([np.mean(uxs_pos)])

        if abs(covarx[0]) > abs(covarx[-1]):
            good_x += 1
            print("Pos good")
            plt.plot(uxs_pos, label='filter position')
            plt.plot(trux_pos, 'r--', label='true position')
            plt.title('position error')
            param_save_good.append(np.concatenate([[i], parameters]))
            iteration_num = i
            mean_pos_good.append(np.concatenate([[i], [np.mean(uxs_pos - trux_pos)]]))
        else:
            bad_x += 1
            print("Pos bad")
            param_save_bad.append(np.concatenate([[i], parameters]))
            # plt.semilogy(np.subtract(uxs_pos, trux_pos), color='r', label='%s' % str(i))
            # plt.title('position error')
            iteration_num = i
            mean_pos_bad.append(np.concatenate([[i], [np.mean(uxs_pos - trux_pos)]]))

        print("position error start = %f \t position error end = %f" %(start_error_x[-1], end_error_x[-1]))
        print("position covar start = %f \t position covar end = %f" % (covarx[0], covarx[-1]))
        print()

        if abs(covarv[0]) > abs(covarv[-1]):
            good_v += 1
            print("Vel good")
            plt.plot(np.subtract(uxs_speed, trux_speed), label='%s' % str(i))
            plt.title('speed error')
            param_save_good.append(np.concatenate([[i], parameters])) if iteration_num is not i else 0
            mean_vel_good.append(np.concatenate([[i], [np.mean(uxs_speed - trux_speed)]]))
            iteration_num = i
        else:
            bad_v +=1
            print("Vel bad")
            param_save_bad.append(np.concatenate([[i], parameters])) if iteration_num is not i else 0
            # plt.semilogy(np.subtract(uxs_speed, trux_speed),color='r', label='%s' % str(i))
            # plt.title('speed error')
            mean_vel_bad.append(np.concatenate([[i], [np.mean(uxs_speed - trux_speed)]]))
            iteration_num = i

        print("speed error start = %f \t speed error end = %f" % (start_error_v[-1], end_error_v[-1]))
        print("velocity covar start = %f \t velocity covar end = %f" % (covarv[0], covarv[-1]))
        print()





    else:
        failed += 1

plt.legend()
print("Successful position = %d" % good_x)
print("Successful velocity = %d" % good_v)

print("Unsuccessful position = %d" % bad_x)
print("Unsuccessful velocity = %d" % bad_v)

print("Failed = %d" % failed)

print("mean of mean filter positions = %f" % np.mean(mean_pos))
print("std of mean filter positions = %f" % np.std(mean_pos))

np.savetxt("working_iteration_params.csv", param_save_good, delimiter="\t")
np.savetxt("failed_iteration_params.csv", param_save_bad, delimiter="\t")

np.savetxt("working_means_pos.csv", np.array(mean_pos_good), delimiter="\t")
np.savetxt("failed_means_pos.csv", np.array(mean_pos_bad), delimiter="\t")

np.savetxt("working_means_speed.csv", np.array(mean_vel_good), delimiter="\t")
np.savetxt("failed_means_speed.csv", np.array(mean_vel_bad), delimiter="\t")

plt.figure(1)
plt.semilogy(mean_pos)
plt.title("Mean position error")

plt.show()
