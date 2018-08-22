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
import keyboard

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

observer_position = np.array([[0, 0, 1.e5],[1.e5, 0, 0.],[0, 1.e5, 0], [0., 0., 1e5],
                              [1e5, 1e5, 1e5]])



tracker_noise_range = [0.001, 1000] #ARCSECOND
tracker_range_noise = [1e2, 1e3]

dt_range = [0.1, 100]
np.random.seed(400)
true_b = 400.
std_true_object_pos_range = np.array([1e2, 1e3])
std_true_object_vel_range = np.array([1e1, 1e2])



est_starting_conditions_range_pos = np.array([1e2, 1e3])
est_starting_conditions_range_vel = np.array([1, 1e2])

P_range_pos = [100 ** 2,1e3 ** 2]
P_range_vel = [1e1 ** 2,1e2 ** 2]
Q_range_pos = [0.01e1,1e1]
Q_range_vel = [0.01e-1,1e0]
sensor_sampling_time_range = [0.01,100]

g = 9.81



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

def FIM(self, dx):
    """
    Computes the derivative of the observation equations for use in the Fisher information matrix
    :param dx:
    :return:
    """
    r = (np.sqrt((dx[0]) ** 2 + (dx[1]) ** 2 + (dx[2]) ** 2))
    dR = np.array([(dx[0] / r), (dx[1] / r), (dx[2] / r), 0, 0, 0])
    dRT = dR.T
    dtheta = np.array([(dx[2] * dx[0]) / (r ** 2 * np.sqt(dx[0] ** 2 + dx[1] ** 2)),
                       (dx[2] * dx[1]) / (r ** 2 * np.sqt(dx[0] ** 2 + dx[1] ** 2)),
                       np.sqrt(dx[0] ** 2 + dx[1] ** 2) / r ** 2, 0, 0, 0])
    dthetaT = dtheta.T
    dphi = np.array([(dx[1] / dx[0] ** 2 + dx[1] ** 2), (dx[1] / dx[0] ** 2 + dx[1] ** 2),
                     0, 0, 0, 0])
    dphiT = dphi.T
    return self.theta_std* np.multiply(dthetaT, dtheta) + self.phi_std * np.multiply(dphiT, dphi)

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
iteration_num_good = 0
iteration_num_bad = 0
param_zeros = np.zeros(34)
param_save_good.append(param_zeros)
param_save_bad.append(param_zeros)

for i in range(0, 100000):
    print(i)
    print()
    true_starting_conditions = np.array([500., -1000., 1e5, 100., 100., 500.])#np.array([10., 10., 1e7, 100., 100., 500.])
    holder = true_starting_conditions  # np.array([500., -1000., 1e5, 100., 100., 500.])

    tracker_noise = get_val_from_range(tracker_noise_range)
    tracker_rng_noise = get_val_from_range(tracker_range_noise)
    dt = abs(get_val_from_range(dt_range))

    temp = std_true_object_pos_range
    std_true_object_pos = [get_val_from_range(temp), get_val_from_range(temp), get_val_from_range(temp)]

    temp = std_true_object_vel_range
    std_true_object_vel = [get_val_from_range(temp), get_val_from_range(temp),
                           get_val_from_range(temp)]
    std_true_object = []
    std_true_object.extend(std_true_object_pos)
    std_true_object.extend(std_true_object_vel)
    std_true_object = np.array(std_true_object)


    temp = est_starting_conditions_range_pos
    est_starting_pos = [get_val_from_range(temp, True), get_val_from_range(temp, True),
                           get_val_from_range(temp, False)]

    temp = est_starting_conditions_range_vel
    est_starting_vel = [get_val_from_range(temp, True), get_val_from_range(temp, True),
                           get_val_from_range(temp, True)]

    true_starting_conditions = holder
    est_starting_conditions = []
    est_starting_conditions.extend(true_starting_conditions[0:3]+std_true_object_pos)
    est_starting_conditions.extend(true_starting_conditions[3:6] + std_true_object_vel)
    est_starting_conditions = np.array(est_starting_conditions)
    est_starting_conditions[2] = abs(est_starting_conditions[2])*2 if est_starting_conditions[2]<0 or est_starting_conditions[2]<est_starting_conditions[5] else est_starting_conditions[2]

    t1 = get_val_from_range(P_range_pos)
    t2 = get_val_from_range(P_range_vel)
    P = [t1, t1, t1, t2, t2, t2]

    t1 = get_val_from_range(Q_range_pos)
    t2 = get_val_from_range(Q_range_vel)
    Q = [t1, t1, t1, t2, t2, t2] # set the variables

    sensor_sampling_time = get_val_from_range(sensor_sampling_time_range)

    parameters = np.concatenate([[tracker_noise, dt, sensor_sampling_time], std_true_object, est_starting_conditions, true_starting_conditions, P, Q])

    P = np.diag(P)
    Q = np.diag(Q)
    # print("tracker noise")
    # print(tracker_noise)
    # print()
    # print("dt")
    # print(dt)
    # print()
    # print("std true object")
    # print(std_true_object)
    # print()
    # print("est_starting_conditions")
    # print(est_starting_conditions)
    # print()
    # print("true_starting_conditions")
    # print(true_starting_conditions)
    # print()
    # print("P")
    # print(P)
    # print()
    # print("Q")
    # print(Q)
    # print()

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
        #print(f.X[2])
        f.update(dt)
        #station.observe(f.X)
        station.noisy_observe(f.X, t)
        try:
            ukf.predict()
        except:
            print("\n\n\n\nNot worked\n\n\n\n")
            print()
            break

        if station.measure_ready is True:
            #print("UPDATE")
            try:
                ukf.update(station.measure, hx_args=(observer_position,))
            except:
                print("\n\n\n\nNOT WORKED\n\n\n\n")
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
        #print("Z = ", f.X[2], "TIME = ", t)
        #print(ukf.x[2])
        #print(ukf.P)
        #print([station.r, station.theta, station.phi])
        #print(ukf.x)

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
        # plt.plot(np.subtract(uxs_pos, trux_pos), label='filter position error')
        # #plt.plot(trux_pos, 'r--', label='true position')
        # plt.title('position error')
        # plt.plot(3*np.array(covarx), 'r')
        # plt.plot(np.multiply(-3,np.array(covarx)), 'r')
        # plt.show()

        # plt.figure(1)
        # plt.plot(time, (uxs[:, 3] - trux[:, 3]), 'xkcd:blue', label='x-error', linewidth=3.0)
        # plt.plot(time, (uxs[:, 4] - trux[:, 4]), 'xkcd:green', label='y-error', linewidth=3.0)
        # plt.plot(time, (uxs[:, 5] - trux[:, 5]), 'xkcd:orange', label='z-error', linewidth=3.0)
        # # plt.plot(time, np.array(covarv)*3, 'r')
        # # plt.plot(time, np.array(covarv)*-3, 'r')
        # plt.title('non linear eqn velocity error')
        # plt.legend()
        # #
        # # POSITION ERROR
        # plt.figure(2)
        # plt.plot(time, (uxs[:, 0] - trux[:, 0]), 'xkcd:blue', label='x-error', linewidth=3.0)
        # plt.plot(time, covarx, 'xkcd:black', label='position-covariance')
        # plt.plot(time, np.multiply(-1, covarx), 'xkcd:black', label='position-covariance')
        # plt.plot(time, (uxs[:, 1] - trux[:, 1]), 'xkcd:green', label='y-error', linewidth=3.0)
        # plt.plot(time, (uxs[:, 2] - trux[:, 2]), 'xkcd:orange', label='z-error', linewidth=3.0)
        # # plt.plot(time, np.array(covarx)*3, 'r')
        # # plt.plot(time, np.array(covarx)*-3, 'r')
        # plt.title('non linear eqn position error')
        # plt.legend()

        #plt.show()

        if abs(covarx[0]) > abs(covarx[-1]) and abs(end_error_x[-1]) < abs(start_error_x[-1]):
            good_x += 1
            print("Pos good")
            # plt.plot(uxs_pos, label='filter position')
            # plt.plot(trux_pos, 'r--', label='true position')
            # plt.title('position error')
            param_save_good.append(np.concatenate([[i], parameters]))
            iteration_num_good = i
            mean_pos_good.append(np.concatenate([[i], [np.mean(uxs_pos - trux_pos)]]))
        else:
            bad_x += 1
            print("Pos bad")
            param_save_bad.append(np.concatenate([[i], parameters]))
            # plt.semilogy(np.subtract(uxs_pos, trux_pos), color='r', label='%s' % str(i))
            # plt.title('position error')
            iteration_num_bad = i
            mean_pos_bad.append(np.concatenate([[i], [np.mean(uxs_pos - trux_pos)]]))

        print("position error start = %f \t position error end = %f" %(start_error_x[-1], end_error_x[-1]))
        print("position covar start = %f \t position covar end = %f" % (covarx[0], covarx[-1]))
        print()

        if abs(covarv[0]) > abs(covarv[-1]) and abs(end_error_v[-1]) < abs(start_error_x[-1]):
            good_v += 1
            print("Vel good")
            # plt.plot(np.subtract(uxs_speed, trux_speed), label='%s' % str(i))
            # plt.title('speed error')
            param_save_good.append(np.concatenate([[i], parameters])) if iteration_num_good is not i else 0
            mean_vel_good.append(np.concatenate([[i], [np.mean(uxs_speed - trux_speed)]]))
            iteration_num_good = i
        else:
            bad_v +=1
            print("Vel bad")
            param_save_bad.append(np.concatenate([[i], parameters])) if iteration_num_bad is not i else 0
            # plt.semilogy(np.subtract(uxs_speed, trux_speed),color='r', label='%s' % str(i))
            # plt.title('speed error')
            mean_vel_bad.append(np.concatenate([[i], [np.mean(uxs_speed - trux_speed)]]))
            iteration_num_bad = i

        print("speed error start = %f \t speed error end = %f" % (start_error_v[-1], end_error_v[-1]))
        print("velocity covar start = %f \t velocity covar end = %f" % (covarv[0], covarv[-1]))
        print()





    else:
        failed += 1

    # if keyboard.is_pressed('q'):  # if key 'q' is pressed
    #     print('You Pressed A Key!')
    #     break  # finishing the loop

plt.legend()
print("Successful position = %d" % good_x)
print("Successful velocity = %d" % good_v)

print("Unsuccessful position = %d" % bad_x)
print("Unsuccessful velocity = %d" % bad_v)

print("Failed = %d" % failed)

print("mean of mean filter positions = %f" % np.mean(mean_pos))
print("std of mean filter positions = %f" % np.std(mean_pos))

np.savetxt("working_iteration_params.csv", np.array(param_save_good), delimiter="\t")
np.savetxt("failed_iteration_params.csv", np.array(param_save_bad), delimiter="\t")

np.savetxt("working_means_pos.csv", np.array(mean_pos_good), delimiter="\t")
np.savetxt("failed_means_pos.csv", np.array(mean_pos_bad), delimiter="\t")

np.savetxt("working_means_speed.csv", np.array(mean_vel_good), delimiter="\t")
np.savetxt("failed_means_speed.csv", np.array(mean_vel_bad), delimiter="\t")

plt.figure(1)
plt.semilogy(mean_pos)
plt.title("Mean position error")


# station.storage_phi = np.array(station.storage_phi)
# station.storage_theta = np.array(station.storage_theta)
# print(len(covarx))
# print(len(uxs))
#
# # # VELOCITY ERROR
# plt.figure(1)
# plt.plot(time, (uxs[:, 3] - trux[:, 3]), 'xkcd:blue', label='x-error',linewidth=3.0)
# plt.plot(time, (uxs[:, 4] - trux[:, 4]),'xkcd:green', label='y-error',linewidth=3.0)
# plt.plot(time, (uxs[:, 5] - trux[:, 5]), 'xkcd:orange', label='z-error',linewidth=3.0)
# #plt.plot(time, np.array(covarv)*3, 'r')
# #plt.plot(time, np.array(covarv)*-3, 'r')
# plt.title('non linear eqn velocity error')
# plt.legend()
# #
# # POSITION ERROR
# plt.figure(2)
# plt.plot(time, (uxs[:, 0] - trux[:, 0]), 'xkcd:blue', label='x-error', linewidth=3.0)
# plt.plot(time, (uxs[:, 1] - trux[:, 1]), 'xkcd:green', label='y-error', linewidth=3.0)
# plt.plot(time, (uxs[:, 2] - trux[:, 2]), 'xkcd:orange', label='z-error', linewidth=3.0)
# #plt.plot(time, np.array(covarx)*3, 'r')
# #plt.plot(time, np.array(covarx)*-3, 'r')
# plt.title('non linear eqn position error')
# plt.legend()
#
# mean_x_error = [np.mean(uxs[:, 0] - trux[:, 0]), np.mean(uxs[:, 1] - trux[:, 1]), np.mean(uxs[:, 2] - trux[:, 2])]
# mean_v_error = [np.mean(uxs[:, 3] - trux[:, 3]), np.mean(uxs[:, 4] - trux[:, 4]), np.mean(uxs[:, 5] - trux[:, 5])]
#
# std_x = [np.std(uxs[:, 0] - trux[:, 0]), np.std(uxs[:, 1] - trux[:, 1]), np.std(uxs[:, 2] - trux[:,2])]
# std_v = [np.std(uxs[:, 3] - trux[:, 3]), np.std(uxs[:, 4] - trux[:, 4]), np.std(uxs[:, 5] - trux[:, 5])]
# #
# print("mean x error = ", mean_x_error)
# print("std x error = ", std_x)
# print("mean v error = ", mean_v_error)
# print("std v error = ", std_v)
#
# # VELOCITY
# plt.figure(3)
# plt.plot(time, uxs[:, 5], 'xkcd:blue', label='filter vz', linewidth=3.0)
# plt.plot(time, uxs[:, 3], 'xkcd:green', label='filter vx', linewidth=3.0)
# plt.plot(time, uxs[:, 4], 'xkcd:orange', label='filter vy', linewidth=3.0)
# plt.plot(time, trux[:, 5], linestyle='--', color='xkcd:dark blue', label=('model z'))
# plt.plot(time, trux[:, 3],linestyle='--', color='xkcd:dark green', label=('model x'))
# plt.plot(time, trux[:, 4], linestyle='--', color='xkcd:dark orange', label=('model y'))
# #plt.scatter(time, measx, label='measure')
# plt.title('non linear eqn velocity')
# plt.legend()
#
#
# # POSITION IN TIME
# plt.figure(5)
# plt.plot(time, uxs[:, 2], 'xkcd:blue', label='filter z', linewidth=3.0)
# plt.plot(time, uxs[:, 0], 'xkcd:green', label='filter x', linewidth=3.0)
# plt.plot(time, uxs[:, 1], 'xkcd:orange', label='filter y', linewidth=3.0)
# plt.plot(time, trux[:, 2],  linestyle='--', color='xkcd:dark blue', label=('model z'))
# plt.plot(time, trux[:, 0], linestyle='--', color='xkcd:dark green', label=('model x'))
# plt.plot(time, trux[:, 1], linestyle='--', color='xkcd:dark orange', label=('model y'))
# # #plt.scatter(time, measx, s=2, label='measure')
# plt.title('non linear eqn position')
# plt.legend()
#
# # POSITION IN SPACE
# fig = plt.figure(6)
# ax = fig.gca(projection='3d')
# ax.plot(uxs[:, 0], uxs[:, 1], uxs[:, 2], 'xkcd:blue', label='filter POS 2D')
# ax.plot(trux[:, 0], trux[:, 1], trux[:, 2], linestyle='--', color='xkcd:dark blue', label=('model'))
# ax.scatter(observer_position[:][:, 0], observer_position[:][:, 1], observer_position[:][:, 2], marker='x', label='observer')
# plt.title('non linear eqn position')
# plt.legend()
#
# #Observations
# # plt.figure(7)
# # for i in range(0, len(observer_position)):
# #     if(len(station.storage_theta[i][1:]) > 2):
# #         plt.plot(time[:], np.array(station.storage_theta[i][1:]), 'b', label='theta')
# #         plt.plot(time[:], np.array(station.storage_phi[i][1:]), 'r', label='phi')
# # plt.title('observations')
# # plt.legend()

plt.show()
#print('UKF standard deviation {:.3f} meters'.format(np.std(uxs - xs)))
