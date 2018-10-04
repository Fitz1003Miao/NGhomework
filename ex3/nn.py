#-*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as scop
import scipy.io as scio
import matplotlib.pyplot as plt

def debugInitWeight(L_in, L_out):
    W = np.zeros([L_in, L_out])
    p = np.arange(1, L_in * L_out + 1)
    W = np.sin(p).reshape(W.shape) / 10

    b = np.zeros([1, L_out])
    p = np.arange(1, L_out + 1)
    b = np.sin(p).reshape(b.shape) / 10
    return W, b 

def sigmoid(x):
    g = 1. / (1 + np.e ** (-1 * x))
    return g

def sigmoidGradient(x):
    g = sigmoid(x) * (1 - sigmoid(x))
    return g

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda):
    length = len(nn_params)
    w1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size).copy()

    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape(1, hidden_layer_size).copy()

    w2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + hidden_layer_size * output_layer_size].reshape(hidden_layer_size, output_layer_size).copy()

    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + hidden_layer_size * output_layer_size : length].reshape(1, output_layer_size).copy()

    n0 = x
    z1 = x.dot(w1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(w2) + b2
    n2 = sigmoid(z2)

    m = x.shape[0]
    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    term = np.r_[w1.reshape(-1, 1), w2.reshape(-1, 1)]
    term = np.transpose(term).dot(term)
    J = 1. / (2 * m) * np.trace((n2 - class_y).dot(np.transpose(n2 - class_y))) + Lambda / (2 * m) * term
    # J = 1. / m * -1 * np.trace(class_y.dot(np.transpose(np.log(n2))) + (1 - class_y).dot(np.transpose(np.log(1 - n2))))
    return np.ravel(J)

def nnGradient(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lmabda):
    length = len(nn_params)
    w1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape(1, hidden_layer_size).copy()
    w2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + hidden_layer_size * output_layer_size].reshape(hidden_layer_size, output_layer_size).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + hidden_layer_size * output_layer_size : length].reshape(1, output_layer_size).copy()

    n0 = x
    z1 = x.dot(w1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(w2) + b2
    n2 = sigmoid(z2)

    m = x.shape[0]
    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    delta2 = (n2 - class_y) * sigmoidGradient(z2)
    w2_grad = np.transpose(n1).dot(delta2)
    b2_grad = np.sum(delta2, axis = 0)

    delta1 = delta2.dot(np.transpose(w2)) * sigmoidGradient(z1)
    w1_grad = np.transpose(n0).dot(delta1)
    b1_grad = np.sum(delta1, axis = 0)

    grad = np.r_[w1_grad.reshape(-1, 1), b1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1), b2_grad.reshape(-1, 1)] / m + Lmabda / m * np.r_[w1.reshape(-1, 1), np.zeros(b1.shape).reshape(-1, 1), w2.reshape(-1, 1), np.zeros(b2.shape).reshape(-1, 1)]
    
    return grad

def checkGradient(Lambda = 0):
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 3

    m = 100
    w1, b1 = debugInitWeight(input_layer_size, hidden_layer_size)
    w2, b2 = debugInitWeight(hidden_layer_size, output_layer_size)

    x = np.random.randint(-10, 10, [m, input_layer_size])
    y = np.mod(np.random.randint(1, 10, [m, 1]), output_layer_size)
    nn_params = np.r_[w1.reshape(-1, 1), b1.reshape(-1, 1), w2.reshape(-1, 1), b2.reshape(-1, 1)]

    grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)

    step = np.zeros([nn_params.shape[0]])
    num_grad = np.zeros([nn_params.shape[0]])
    e = 1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss_1 = nnCostFunction(nn_params - step.reshape(-1, 1), input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)
        loss_2 = nnCostFunction(nn_params + step.reshape(-1, 1), input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)
        num_grad[i] = (loss_2 - loss_1) / (2 * e)
        step[i] = 0

    res = np.c_[grad, num_grad]
    print(res)

def displayData(x):
    padding = 1
    display_array = -1 * np.ones([padding + (20 + padding) * 10, padding + (20 + padding) * 10])

    for i in range(10):
        for j in range(10):
            display_array[padding + (20 + padding) * i : padding + (20 + padding) * i + 20, padding + (20 + padding) * j : padding + (20 + padding) * j + 20] = x[i * 10 + j, :].reshape(20, 20, order = 'F')

    plt.imshow(display_array)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    checkGradient(Lambda = 0.5)

    data = scio.loadmat("data_digits.mat")
    x = data['X']
    y = data['y']

    m = len(y)
    selected = [np.random.randint(i - i, m) for i in range(100)]

    displayData(x[selected, :])

