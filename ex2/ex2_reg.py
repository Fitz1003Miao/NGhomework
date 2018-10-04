import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop

def mapFeature(x1, x2):
    degree = 2
    m = len(x1)

    x = np.ones([m, 1])    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            x = np.c_[x, (x1 ** (i - j)) * (x2 ** j)]

    return x

def sigmoid(x):
    g = 1. / (1 + np.e ** (-1 * x))
    return g

def costFunction(theta, x, y, Lambda):
    h = sigmoid(x.dot(theta))
    m = len(y)
    theta_temp = theta.copy()
    theta_temp[0] = 0
    J = 1. / m * -1 * (np.transpose(y).dot(h) + np.transpose(1 - y).dot(1 - h)) + Lambda / (2 * m) * np.transpose(theta_temp).dot(theta_temp)
    return J

def gradient(theta, x, y, Lambda):
    h = sigmoid(x.dot(theta))
    m = len(y)
    theta_temp = theta.copy()
    theta_temp[0] = 0

    g = 1. / m * np.transpose(x).dot(h - y) + Lambda / m * theta_temp
    return g

if __name__ == "__main__":
    data = np.loadtxt(fname = "ex2data2.txt", delimiter = ",")
    x = data[:, 0:2]
    y = data[:, 2]

    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plot1 = plt.scatter(x[pos, 0], x[pos, 1], marker = "o", c = "b")
    plot2 = plt.scatter(x[neg, 0], x[neg, 1], marker = "x", c = "r")

    plt.legend([plot1, plot2], ["Admitted", "No Admitted"], loc = "upper right")

    x = mapFeature(x[:, 0], x[:, 1])

    m,n = x.shape
    initial_theta = np.zeros([n , 1])

    initial_lambda = 0.1

    result = scop.fmin_bfgs(f = costFunction, x0 = initial_theta, fprime = gradient, args = (x, y, initial_lambda))
    print(result)

    x = np.linspace(-1, 1.5, 50)
    y = np.linspace(-1, 1.5, 50)
    z = np.zeros([len(x), len(y)])

    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = mapFeature(np.reshape(x[i], [-1, 1]), np.reshape(y[j], [-1, 1])).dot(result)

    z = z.T

    plt.contour(x, y, z, [0, 0.1])
    plt.show()
    