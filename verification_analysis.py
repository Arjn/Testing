import csv
import numpy as np
import matplotlib.pyplot as plt

#parameters = np.concatenate([[tracker_noise, dt], std_true_object, est_starting_conditions, true_starting_conditions, P, Q])

failed_iter_params = []
failed_means_pos = []
failed_means_speed = []

work_iter_params = []
work_means_pos = []
work_means_speed = []

def make_float(string_list):
    return [float(i) for i in string_list]

with open('failed_iteration_params.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        failed_iter_params.append(make_float(row))
print('1')

with open('working_iteration_params.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        #print(row)
        work_iter_params.append(make_float(row))
print('2')
with open('failed_means_pos.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        #print(row)
        failed_means_pos.append(make_float(row))
print('3')
with open('failed_means_speed.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        #print(row)
        failed_means_speed.append(make_float(row))
print('4')
with open('working_means_pos.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        #print(row)
        work_means_pos.append(make_float(row))
print('5')
with open('working_means_speed.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter='\t')
    for row in int:
        #print(row)
        work_means_speed.append(make_float(row))
print('6')

failed_iter_params_pos = []
failed_iter_params_speed = []
failed_means_pos = np.array(failed_means_pos)
failed_means_speed = np.array(failed_means_speed)
failed_iter_params = np.array(failed_iter_params)
for i in range(0,len(failed_iter_params)):
    pos = False
    if failed_iter_params[i,0] in failed_means_pos[:, 0]:
        failed_iter_params_pos.append(failed_iter_params[i,:])
    if failed_iter_params[i,0] in failed_means_speed[:,0]:
        failed_iter_params_speed.append(failed_iter_params[i,:])

work_iter_params_pos = []
work_iter_params_speed = []
work_means_pos = np.array(work_means_pos)
work_means_speed = np.array(work_means_speed)
work_iter_params = np.array(work_iter_params)
b = 100

for i in range(0,len(work_iter_params)):
    pos = False
    if len(work_means_pos) > 0:
        if work_iter_params[i,0] in work_means_pos[:, 0] and b != i:
            work_iter_params_pos.append(work_iter_params[i,:])
    if work_iter_params[i,0] in work_means_speed[:,0] and b != i:
        work_iter_params_speed.append(work_iter_params[i,:])
    b = i

work_iter_params_pos = np.array(work_iter_params_pos)
# plt.figure(2)
# ax = plt.gca()
# x = np.array([np.mean(i) for i in work_iter_params_pos[:,1]])
# ax.plot(x, work_means_pos[:,1], '*')
# ax.set_yscale('log')
# plt.title('ang')
# plt.xlabel('ang [m]')
# plt.xlabel('mean position error[m]')
# plt.savefig('ang_pos_worked.png')


failed_iter_params_pos = np.array(failed_iter_params_pos)
# plt.figure(3)
# ax = plt.gca()
# x = np.array([np.mean(i) for i in failed_iter_params_pos[:,1]])
# ax.plot(x, failed_means_pos[:,1], '*')
# ax.set_yscale('log')
# plt.title('ang')
# plt.xlabel('ang [m]')
# plt.ylabel('mean position error [m]')
# plt.savefig('ang_pos_failed.png')
#
failed_iter_params_speed = np.array(failed_iter_params_speed)
# plt.figure(4)
# ax = plt.gca()
# x = np.array([np.mean(i) for i in failed_iter_params_speed[:,1]])
# ax.plot(x, failed_means_speed[:,1], '*')
# ax.set_yscale('log')
# plt.title('ang')
# plt.xlabel('ang [m/s]')
# plt.ylabel('mean speed error [m/s]')
# plt.savefig('ang_speed_failed.png')

work_iter_params_speed = np.array(work_iter_params_speed)
# plt.figure(5)
# ax = plt.gca()
# x = np.array([np.mean(i) for i in work_iter_params_speed[:,1]])
# ax.plot(x, work_means_speed[:,1], '*')
# ax.set_yscale('log')
# plt.title('ang')
# plt.xlabel('ang [m/s]')
# plt.ylabel('mean error [m/s]')
# plt.savefig('ang_speed_good.png')


a = 1
y1 = np.array(work_means_pos[:,1])
x1 = np.array(work_iter_params_pos[:, a])

y2 = np.array(failed_means_pos[:,1])
x2 = np.array(failed_iter_params_pos[:, a])
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y)[0]

# line_y = (m*x + c)

plt.figure(1)
ax = plt.gca()
#plt.plot(abs(failed_iter_params_speed[:,a]), abs(failed_means_speed[:,1]), 'o', color='r', markersize=1, label='Failed runs')
plt.plot(abs(x2[:]), abs(y2), 'o', color='r', label='Failed runs', markersize=1)
plt.plot(abs(x1[1:]), abs(y1), 'o', color='b', label='Successful runs', markersize=1)
plt.xlabel('sensor error [arcsec]')
plt.ylabel('mean error [m]')
plt.title('Failed and successful simulations position with sensor angular error')

#plt.plot(x, line_y, color='r', label='Fitted line')
plt.legend()
ax.set_yscale('log')


y1 = np.array(work_means_speed[:,1])
x1 = np.array(work_iter_params_speed[:, a])

y2 = np.array(failed_means_speed[:,1])
x2 = np.array(failed_iter_params_speed[:, a])
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y)[0]

# line_y = (m*x + c)

plt.figure(3)
ax = plt.gca()
#plt.plot(abs(failed_iter_params_speed[:,a]), abs(failed_means_speed[:,1]), 'o', color='r', markersize=1, label='Failed runs')
plt.plot(abs(x2[:]), abs(y2), 'o', color='r', label='Failed runs', markersize=1)
plt.plot(abs(x1[1:]), abs(y1), 'o', color='b', label='Successful runs', markersize=1)
plt.xlabel('sensor error [arcsec]')
plt.ylabel('mean error [m/s]')
plt.title('Failed and successful simulations speed with sensor angular error')

#plt.plot(x, line_y, color='r', label='Fitted line')
plt.legend()
ax.set_yscale('log')


plt.figure(2)
ax = plt.gca()
plt.hist(failed_iter_params_pos[:, a], 100, color='r', label='Failed runs')
plt.hist(work_iter_params_pos[:, a], 100, color='b', label='Successful runs')
plt.legend()
plt.xlabel('sensor error [arcsec]')
plt.ylabel('number of simulation runs')
plt.title('Failed and successful simulations speed with sensor angular error')


plt.show()

