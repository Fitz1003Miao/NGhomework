# import numpy as np
# import scipy.optimize
# import matplotlib.pyplot as plt

# def sigmoid(x):
#     g = 1. / (1 + np.e ** (-1 * x))
#     g = np.reshape(g, [len(g),])
#     return g

# def costFunction(theta, x, y):
#     h = sigmoid(x.dot(theta))
#     m = len(y)
#     J = 1. / m * -1 * (np.transpose(y).dot(np.log(h)) + np.transpose(1 - y).dot(np.log(1 - h)))
#     return J

# def gradient(theta, x, y):
#     h = sigmoid(x.dot(theta))
#     m,n = x.shape
#     g = np.reshape(1. / m * (np.transpose(x).dot(h - y)), [n,])

#     return g


# def predict(theta, x, y):
#     result = sigmoid(x.dot(theta))
#     result[np.where(result >= 0.5)] = 1
#     result[np.where(result < 0.5)] = 0

#     print(np.mean(np.float64(result ==y)))

# if __name__ == "__main__":
#     data = np.loadtxt(fname = "ex2data1.txt", delimiter = ",")
#     x = data[:, 0:2]
#     y = data[:, 2]

#     m,n = x.shape
#     x = np.c_[np.ones([m,1]), x]
#     initial_theta = np.zeros([n + 1, 1])
#     result = np.reshape(scipy.optimize.fmin_bfgs(f = costFunction, x0 = initial_theta, fprime = gradient, args = (x,y)),[n+1, 1])
#     # print(gradient(initial_theta, x, y))
#     print(result)
#     # predict(result, x, y)
#     print(sigmoid(np.array([1, 45, 85]).dot(result)))

#     pos = np.where(y == 1)
#     neg = np.where(y == 0)
    
#     plot1 = plt.scatter(x[pos, 1], x[pos, 2], marker = 'o', c = "r")
#     plot2 = plt.scatter(x[neg, 1], x[neg, 2], marker = 'x', c = "b")

#     plt.legend([plot1, plot2], ["Admitted", "Not admitted"], loc = "upper right")
    
#     x = np.linspace(30, 100, 100)
#     y = np.linspace(30, 100, 100)

#     z = np.zeros([len(x), len(y)])
#     for i in range(len(x)):
#         for j in range(len(y)):
#             z[i][j] = np.array([1, x[i], y[j]]).dot(result)

#     z = np.transpose(z)
#     plt.contour(x,y,z,[0,0.01],linewidth=2.0)
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop

def sigmoid(x):
    g = 1. / (1 + np.e ** (-1 * x))
    return g

def costFunction(theta, x, y):
    h = sigmoid(x.dot(theta))
    m = len(y)
    J = 1. / m * -1 * (np.transpose(y).dot(np.log(h)) + np.transpose(1 - y).dot(np.log(1 - h)))
    return J

def gradient(theta, x, y):
    h = np.reshape(sigmoid(x.dot(theta)), [-1, 1])
    m, n = x.shape
    g = np.reshape(1. / m * (np.transpose(x).dot(h - y)), [n, ])
    return g

if __name__ == "__main__":
    data = np.loadtxt(fname = "ex2data1.txt", delimiter = ",")
    
    x = data[:, 0:2]
    y = data[:, 2]

    m, n = x.shape
    x = np.c_[np.ones([m, 1]), x]
    y = np.reshape(y, [-1, 1])

    initial_theta = np.zeros([n + 1, 1])
    result = np.reshape(scop.fmin_bfgs(f = costFunction, x0 = initial_theta, fprime = gradient, args = (x, y)), [-1, 1])
    print(result)

    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plot1 = plt.scatter(x[pos, 1], x[pos, 2], marker = "o", c = "b")
    plot2 = plt.scatter(x[neg, 1], x[neg, 2], marker = "x", c = "r")

    plt.legend([plot1, plot2], ["Admitted", "No-Admitted"], loc = "upper right")

    x = np.linspace(30, 100, 100)
    y = np.linspace(30, 100, 100)
    z = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = np.array([1, x[i], y[j]]).dot(result)

    z = z.T
    plt.contour(x, y, z, [0, 0.01], linewidth=2.0)

    plt.show()