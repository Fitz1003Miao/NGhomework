import numpy as np

def featureNormalize(x):
    means = np.mean(x, axis = 0)   
    sigma = np.std(x, axis = 0, ddof = 1)
    x = (x - means) / sigma
    return x, means, sigma

def costFunction(theta, x, y):
    h = x.dot(theta)
    err = h - y
    m = len(y)
    J = 1. / (2 * m) * np.transpose(err).dot(err)
    return J

def gradient(theta, x, y):
    h = x.dot(theta)
    err = h - y
    m = len(y)
    g = 1. / m * np.transpose(x).dot(err)
    return g

def gradientDescent(theta, x, y, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        h = x.dot(theta)
        theta -= alpha * gradient(theta, x, y)

    return theta

if __name__ == "__main__":
    data = np.loadtxt(fname = "ex1data2.txt", delimiter = ",")

    x = data[:, 0:2]
    y = data[:, 2]

    x, means_x, sigma_x = featureNormalize(x)
    
    m,n = x.shape
    x = np.c_[np.ones([m, 1]), x]
    y = np.reshape(y, [-1, 1])
    initial_theta = np.zeros([n + 1, 1])
    
    alpha = 0.01
    iterations = 8500

    theta = gradientDescent(initial_theta, x, y, alpha, iterations)
    print(theta)
    
    




