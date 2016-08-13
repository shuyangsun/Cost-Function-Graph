import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def get_cost_function(h_theta_x, logistic=False):
    def cost_func(thetas, training_set):
        result = 0
        for training_sample in training_set:
            nonlocal logistic
            if logistic:
                hx = h_theta_x(thetas, training_sample[:-1])
                y = training_sample[-1]
                result += -y * math.log(hx) - (1 - y) * math.log(1 - hx)
            else:
                result += pow(h_theta_x(thetas, training_sample[:-1]) - training_sample[-1], 2)
        result *= (1/(2 * len(training_set)))
        return result
    return cost_func


def hypothesis_linear_regression(thetas, training_sample):
    return thetas[0] + sum(training_sample * thetas[1:])


def hypothesis_logistic_regression(thetas, training_sample):
    z = thetas[0] + sum(training_sample * thetas[1:])
    return 1 / (1 + np.e ** (-z))

my_training_set = np.array([0, 1,
                            1, 0,
                            2, 1,
                            3, 0]).reshape(4, 2)
my_cost_func_linear_regression = get_cost_function(hypothesis_linear_regression)
my_cost_func_logistic_regression_non_convex = get_cost_function(hypothesis_logistic_regression)
my_cost_func_logistic_regression_convex = get_cost_function(hypothesis_logistic_regression, logistic=True)

my_thetas_lst = []

cost_lst_linear_regression = []
cost_lst_logistic_regression_non_convex = []
cost_lst_logistic_regression_convex = []

start = -10
stop = 10
step = 1
for theta_0 in np.arange(start, stop, step):
    for theta_1 in np.arange(start, stop, step):
        my_thetas_lst.append(np.array([theta_0, theta_1]))

for my_thetas in my_thetas_lst:
    cost_linear = my_cost_func_linear_regression(my_thetas, my_training_set)
    cost_logistic_non_convex = my_cost_func_logistic_regression_non_convex(my_thetas, my_training_set)
    cost_logistic_convex = my_cost_func_logistic_regression_convex(my_thetas, my_training_set)
    cost_lst_linear_regression.append(cost_linear)
    cost_lst_logistic_regression_non_convex.append(cost_logistic_non_convex)
    cost_lst_logistic_regression_convex.append(cost_logistic_convex)

theta_0_lst = [x[0] for x in my_thetas_lst]
theta_1_lst = [x[-1] for x in my_thetas_lst]

plt.close('all')

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.add_subplot(221, projection='3d')
ax.set_title('Linear Regression Cost Function (convex)')
ax.plot_trisurf(theta_0_lst, theta_1_lst, cost_lst_linear_regression, cmap=cm.jet, linewidth=0.1)
plt.xlabel('theta 0')
plt.ylabel('theta 1')

ax = fig.add_subplot(222, projection='3d')
ax.set_title('Logistic Regression Cost Function (non-convex)')
ax.plot_trisurf(theta_0_lst, theta_1_lst, cost_lst_logistic_regression_non_convex, cmap=cm.jet, linewidth=0.1)
plt.xlabel('theta 0')
plt.ylabel('theta 1')

ax = fig.add_subplot(223, projection='3d')
ax.set_title('Logistic Regression Cost Function (non-convex)')
ax.plot_trisurf(theta_0_lst, theta_1_lst, cost_lst_logistic_regression_convex, cmap=cm.jet, linewidth=0.1)
plt.xlabel('theta 0')
plt.ylabel('theta 1')

plt.show()
