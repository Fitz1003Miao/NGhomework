import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop
import scipy.io as scio

def displayData(x):
    padding = 1
    display_array = -1 * np.ones([padding + 10 * (20 + padding), padding + 10 * (20 + padding)])
    
    for i in range(10):
        for j in range(10):
            display_array[padding + i * (20 + padding) : padding + i * (20 + padding) + 20, padding + j * (20 + padding) : padding + j * (20 + padding) + 20] = np.reshape(x[i * 10 + j,:], [20, 20], order = 'F')
    
    plt.imshow(display_array, cmap = "gray")
    plt.axis("off")
    plt.show()

def sigmoid(x):
    g = 1. / (1 + np.e ** (-1 * x))
    return g

def costFunction(theta, x, y, Lambda):
    h = sigmoid(x.dot(theta))
    m = len(y)
    theta_temp = theta.copy()
    theta_temp[0] = 0
    J = 1. / m * -1 * (np.transpose(y).dot(np.log(h)) + np.transpose(1 - y).dot(np.log(1 - h))) + Lambda / (2 * m) * np.transpose(theta_temp).dot(theta_temp)

    return J

def gradient(theta, x, y, Lambda):
    h = sigmoid(x.dot(theta))
    m = len(y)
    y = np.reshape(y, [-1,])

    theta_temp = theta.copy()
    theta_temp[0] = 0

    g = 1. / m * np.transpose(x).dot(h - y) + Lambda / m * theta_temp
    return g

def OneVsAll(x, y, num_labels, Lambda):
    m,n = x.shape
    all_theta = np.zeros([n + 1, num_labels])
    x = np.c_[np.ones([m, 1]), x]

    for i in range(1, num_labels + 1):
        initial_theta = all_theta[:, i - 1]
        class_y = (y == i)
        result = scop.fmin_bfgs(f = costFunction, x0 = initial_theta, fprime = gradient, args = (x, class_y, Lambda))

        all_theta[:, i - 1] = result.reshape(1, -1)
    return all_theta

def predictOneVsAll(theta, x, y, num_labels, Lambda):
    m,n = x.shape
    
    x = np.c_[np.ones([m, 1]), x]
    result = x.dot(theta)

    p = np.max(result, axis = 1)
    for i in range(m):
        for j in range(1, num_labels + 1):
            if p[i] == result[i, j - 1]:
                p[i] = j
                break

    p = p.reshape(-1, 1)
    print(np.sum(p == y, axis = 0) * 1.0 / m * 100)

if __name__ == "__main__":
    data = scio.loadmat(file_name = "ex3data1.mat")
    x = data['X']
    y = data['y']

    # m,n = x.shape
    # selected = [s for s in [np.random.randint(i - i, m - 1) for i in range(100)]]

    # displayData(x[selected,:])

    # num_labels = 10
    # Lambda = 0.1

    # all_theta = OneVsAll(x, y, num_labels, Lambda)
    # predictOneVsAll(all_theta, x, y, num_labels, Lambda)

    x = data['X']
    y = data['y']

    m,n = x.shape
    data = scio.loadmat(file_name = "ex3weights.mat")
    
    theta1 = data['Theta1'].T
    theta2 = data['Theta2'].T

    x = np.c_[np.ones([m, 1]), x]
    result1 = sigmoid(x.dot(theta1))
    result1 = np.c_[np.ones([m, 1]), result1]

    result2 = sigmoid(result1.dot(theta2))
    
    num_labels = 10
    p = np.max(result2, axis = 1)

    for i in range(m):
        for j in range(1, num_labels + 1):
            if p[i] == result2[i, j - 1]:
                p[i] = j
                break

    p = np.reshape(p, [-1, 1])
    print(np.sum(p == y, axis = 0) * 1.0 / m)
            
