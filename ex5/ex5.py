#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop
import scipy.io as scio

def LinearRegCostFunction(theta, x, y, Lambda):
    y = y.reshape(-1, 1)
    h = np.dot(x, theta).reshape(-1, 1)

    err = h - y
    m = len(y)

    temp = theta.copy().reshape(-1, 1)
    temp[0,0] = 0
    temp = np.dot(np.transpose(temp), temp)

    J = 1. / (2 * m) * np.transpose(err).dot(err) + Lambda * 1.0 / (2 * m) * temp
    return J

def LinearRegGradient(theta, x, y, Lambda):
    y = y.reshape(-1, 1)
    h = np.dot(x, theta).reshape(-1, 1)
    
    err = h - y
    m = len(y)

    temp = theta.copy().reshape(-1, 1)
    temp[0, 0] = 0
    
    grad = 1. / m * np.transpose(x).dot(err) + Lambda * 1.0 / m * temp
    return grad.reshape(-1, )

def trainLinearReg(x, y, Lambda):
    theta = np.zeros([x.shape[1], 1])
    result = scop.fmin_bfgs(f = LinearRegCostFunction, x0 = theta, fprime = LinearRegGradient, args = (x, y, Lambda))
    return result

def learningCurve(x, y, xval, yval, Lambda):
    m = x.shape[0]
    J_train = np.zeros([m])
    J_cv = np.zeros([m])
    for i in range(1, m + 1):
        theta = trainLinearReg(x[0:i, :], y[0:i, :], Lambda).reshape(-1, 1)
        J_train[i - 1] = LinearRegCostFunction(theta, x[0:i, :], y[0:i, :], 0)
        J_cv[i - 1] = LinearRegCostFunction(theta, xval, yval, 0)

    plot1, = plt.plot(np.arange(1, m + 1), J_train, "g-")
    plot2, = plt.plot(np.arange(1, m + 1), J_cv, "b-")

    plt.legend([plot1, plot2], ["Train", "Cross Validation"], loc = "upper right")
    plt.show()

def polyFeatures(x, degree):
    x_poly = np.zeros([x.shape[0], degree])
    for i in range(1, degree + 1):
        x_poly[:, i - 1] = (x ** i).reshape(1, -1)

    return x_poly

def featureNormalize(x):
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis = 0, ddof = 1)
    x = (x - mu) / sigma

    return x, mu, sigma

def plotFit(theta, xmin, xmax, mu, sigma, degree):
    x = np.arange(xmin - 15, xmax + 15)
    x_poly = polyFeatures(x, degree)
    x_poly_normal = (x_poly - mu) / sigma
    x_poly_normal = np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal]

    result = x_poly_normal.dot(theta)
    plt.plot(x, result, "b-")
    plt.show()

def validationCurve(Lambda_vec, x, y, xval, yval):
    J_train = np.zeros([len(Lambda_vec)])
    J_val = np.zeros([len(Lambda_vec)])
    for i in range(len(Lambda_vec)):
        theta = trainLinearReg(x, y, Lambda_vec[i]).reshape(-1, 1)
        J_train[i] = LinearRegCostFunction(theta, x, y, 0)
        J_val[i] = LinearRegCostFunction(theta, xval, yval, 0)
    
    plot1, = plt.plot(Lambda_vec, J_train, "g-")
    plot2, = plt.plot(Lambda_vec, J_val, "b-")

    plt.legend([plot1, plot2], ["Train", "Cross Validation"], loc = "upper right")
    plt.show()
    


if __name__ == "__main__":
    data = scio.loadmat("ex5data1.mat")
    x = data["X"]
    y = data["y"]

    m = len(y)
    plt.scatter(x, y, marker = "x", c = "r")
    theta = np.ones([2, 1])

    Lambda = 1.
    J = LinearRegCostFunction(theta, np.c_[np.ones([m, 1]), x], y, Lambda)
    print(J)

    grad = LinearRegGradient(theta, np.c_[np.ones([m, 1]), x], y, Lambda)
    print(grad)

    Lambda = 0
    result = trainLinearReg(np.c_[np.ones([m, 1]), x], y, Lambda)
    X = np.arange(-60, 40)
    Y = np.c_[np.ones([X.shape[0], 1]), X].dot(result)
    plt.plot(X, Y, "b-")

    plt.show()

    xval = data["Xval"]
    yval = data["yval"]
    learningCurve(np.c_[np.ones([x.shape[0], 1]), x], y, np.c_[np.ones([xval.shape[0], 1]), xval], yval, Lambda)

    xtest = data["Xtest"]
    ytest = data["ytest"]

    degree = 8
    x_poly = polyFeatures(x, degree)
    x_poly_normal, mu, sigma = featureNormalize(x_poly)

    xval_poly = polyFeatures(xval, degree)
    xval_poly_normal = (xval_poly - mu) / sigma

    xtest_poly = polyFeatures(xtest, degree)
    xtest_poly_normal = (xtest_poly - mu) / sigma

    theta = trainLinearReg(np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal], y, Lambda)

    plt.scatter(x, y, marker = "x", c = "r")
    plotFit(theta, np.min(x), np.max(x), mu, sigma, degree)

    learningCurve(np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal], y, np.c_[np.ones([xval_poly_normal.shape[0], 1]), xval_poly_normal], yval, Lambda)
    Lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.])
    validationCurve(Lambda_vec, np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal], y, np.c_[np.ones([xval_poly_normal.shape[0], 1]), xval_poly_normal], yval)

    theta = trainLinearReg(np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal], y, 3.)
    J_train = LinearRegCostFunction(theta, np.c_[np.ones([x_poly_normal.shape[0], 1]), x_poly_normal], y, 0)
    J_cv = LinearRegCostFunction(theta, np.c_[np.ones([xval_poly_normal.shape[0], 1]), xval_poly_normal], yval, 0)
    J_test = LinearRegCostFunction(theta, np.c_[np.ones([xtest_poly_normal.shape[0], 1]), xtest_poly_normal], ytest, 0)

    print(theta)