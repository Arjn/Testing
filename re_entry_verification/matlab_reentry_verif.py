import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

X_t = []
X_ukf = []

with open('true_pos', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        temp = []
        for i in row:
            temp.extend([float(i)])
        X_t.append(temp)
X_t = np.array(X_t).T

with open('UKF_est_pos', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        temp = []
        for i in row:
            temp.extend([float(i)])
        X_ukf.append(temp)
X_ukf = np.array(X_ukf).T

df = pd.read_csv('model_vel_error.csv')
labels = ['x', 'y', 'vx', 'vy', 'rho']

model_x = pd.read_csv('trux.csv')

# for i in range(0,5):
#     plt.figure(i)
#     plt.plot(X_t[:,i], label='verified %s' % labels[i])
#     plt.plot(df._values[:,i], label='%s' % labels[i])
#     plt.legend()


plt.figure(1)
for i in range(0,5):
    plt.semilogy(abs(model_x._values[:,i] - X_t[1:,i]), label='%s'%labels[i])
plt.xlabel('Time [ms]')
plt.ylabel('Error [-]')
plt.legend()
plt.title('Error between implemented dynamic model and model used in verified filter')


plt.figure(2)
plt.plot(model_x._values[:,0], label='model X')
plt.plot(X_t[:,0], label='verified X')
plt.plot(model_x._values[:,1], label='model Y')
plt.plot(X_t[:,1], label='verified Y')
plt.legend()


plt.figure(3)
plt.plot(model_x._values[:,4], label='model rho')
plt.plot(X_t[:,4], label='verified rho')
plt.legend()
plt.show()


# plt.semilogy(abs(X_t[:,2] - X_ukf[:,2]), label='verified model error: mean = %f' % np.mean(abs(X_t[:,2] - X_ukf[:,2])), linewidth=1.)
# # plt.plot(df._values, label='unverified model error: mean = %f' % np.mean(df._values), linewidth=1.3)
# # plt.plot(abs(abs(df._values[:,i] - X_ukf[:,1]) - mod_vel_error.flatten().T[1:]), label='difference')
# plt.xlabel('Time [ms]')
# plt.ylabel('Error [-]')
# plt.title('verified model and SATANS UKF model for re-entry problem [ballistic coeff] ')
# plt.legend()
# plt.show()

