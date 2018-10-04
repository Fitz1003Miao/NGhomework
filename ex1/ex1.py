import numpy as np
import matplotlib.pyplot as plt

def costFunction(theta, x, y):
    h = x.dot(theta)
    m = len(y)
    J = 1. / (2 * m) * np.transpose(h - y).dot(h - y)

    return J

def gradientDescent(theta, x, y, alpha, iterations):

    m = len(y)
    for i in range(iterations):
        h = x.dot(theta)
        temp = theta - alpha / m * (np.transpose(x).dot(h - y))
        theta = temp.copy()

    return theta

if __name__ == "__main__":
    data = np.loadtxt(fname = "ex1data1.txt", delimiter = ",")
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x, y, marker = ".", c = "r")
    plt.show()

    m = len(y)
    x = np.c_[np.ones([m, 1]), x]
    y = np.reshape(y, [-1, 1])
    initial_theta = np.zeros([2, 1])

    print(costFunction(initial_theta, x, y)) 

    alpha = 0.01
    iterations = 1500

    theta = gradientDescent(initial_theta, x, y, alpha, iterations)

    result = x.dot(theta)

    plt.scatter(x[:, 1], y, marker = ".", c = "r")
    plt.plot(x[:, 1], result)
    plt.show()