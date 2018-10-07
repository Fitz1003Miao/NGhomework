#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as scop
import time

def debugInitWeight(L_in, L_out):
    W = np.zeros([L_in, L_out])
    p = np.arange(1, L_in * L_out + 1)
    W = np.sin(p).reshape(W.shape) / 10
    
    b = np.zeros([1, L_out])
    p = np.arange(1, L_out + 1)
    b = np.sin(p).reshape(b.shape) / 10

    return W,b

def sigmoid(x):
    g = 1. / (1 + np.e ** (-1 * x))
    return g

def sigmoidGradient(x):
    g = sigmoid(x) * (1 - sigmoid(x))
    return g

def nnGradient_MSE(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda):
    length = len(nn_params)
    W1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape([input_layer_size, hidden_layer_size]).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape([1, hidden_layer_size]).copy()
    W2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size].reshape([hidden_layer_size, output_layer_size]).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size : length].reshape([1, output_layer_size]).copy()

    m,n = x.shape
    
    # 正向传播
    n0 = x
    z1 = n0 .dot(W1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(W2) + b2
    n2 = sigmoid(z2)

    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    # BP
    delta2 = 1. / m * (n2 - class_y) * sigmoidGradient(z2)
    w2_grad = np.transpose(n1).dot(delta2)
    b2_grad = np.sum(delta2, axis = 0)
    
    delta1 = delta2.dot(np.transpose(W2)) * sigmoidGradient(z1)

    w1_grad = np.transpose(n0).dot(delta1)
    b1_grad = np.sum(delta1, axis = 0)

    grad = np.r_[w1_grad.reshape(-1, 1), b1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1), b2_grad.reshape(-1, 1)] + Lambda / m * np.r_[W1.reshape(-1, 1), np.zeros(b1.shape).reshape(-1, 1), W2.reshape(-1, 1), np.zeros(b2.shape).reshape(-1, 1)]

    return np.ravel(grad)

def nnCostFunction_MSE(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda):
    length = len(nn_params)
    W1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape([input_layer_size, hidden_layer_size]).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape([1, hidden_layer_size]).copy()
    W2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size].reshape([hidden_layer_size, output_layer_size]).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size : length].reshape([1, output_layer_size]).copy()

    m, n = x.shape
    
    # 正向传播
    n0 = x

    z1 = n0.dot(W1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(W2) + b2
    n2 = sigmoid(z2)

    term = np.r_[W1.reshape(-1, 1), W2.reshape(-1, 1)]
    term = np.transpose(term).dot(term)

    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    J = 1. / (2 * m) * np.trace((n2 - class_y).dot(np.transpose(n2 - class_y))) + Lambda / (2 * m) * term
    return np.ravel(J)

def nnGradient_CRE(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda):
    length = len(nn_params)
    W1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape([input_layer_size, hidden_layer_size]).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape([1, hidden_layer_size]).copy()
    W2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size].reshape([hidden_layer_size, output_layer_size]).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size : length].reshape([1, output_layer_size]).copy()

    m,n = x.shape
    
    # 正向传播
    n0 = x
    z1 = n0 .dot(W1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(W2) + b2
    n2 = sigmoid(z2)

    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    # BP
    # delta2 = 1. / m * (n2 - class_y) * sigmoidGradient(z2)
    delta2 = 1. / m * -1 * (class_y - n2) / (n2 * (1 - n2)) * sigmoidGradient(z2)
    w2_grad = np.transpose(n1).dot(delta2)
    b2_grad = np.sum(delta2, axis = 0)
    
    delta1 = delta2.dot(np.transpose(W2)) * sigmoidGradient(z1)

    w1_grad = np.transpose(n0).dot(delta1)
    b1_grad = np.sum(delta1, axis = 0)

    grad = np.r_[w1_grad.reshape(-1, 1), b1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1), b2_grad.reshape(-1, 1)] + Lambda / m * np.r_[W1.reshape(-1, 1), np.zeros(b1.shape).reshape(-1, 1), W2.reshape(-1, 1), np.zeros(b2.shape).reshape(-1, 1)]

    return np.ravel(grad)

def nnCostFunction_CRE(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda):
    length = len(nn_params)
    W1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape([input_layer_size, hidden_layer_size]).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape([1, hidden_layer_size]).copy()
    W2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size].reshape([hidden_layer_size, output_layer_size]).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size : length].reshape([1, output_layer_size]).copy()

    m, n = x.shape
    
    # 正向传播
    n0 = x

    z1 = n0.dot(W1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(W2) + b2
    n2 = sigmoid(z2)

    term = np.r_[W1.reshape(-1, 1), W2.reshape(-1, 1)]
    term = np.transpose(term).dot(term)

    class_y = np.zeros([m, output_layer_size])
    for i in range(output_layer_size):
        class_y[:, i] = (y == i).reshape(1, -1)

    # J = 1. / (2 * m) * np.trace((n2 - class_y).dot(np.transpose(n2 - class_y))) + Lambda / (2 * m) * term
    J = 1. / m * -1 * np.trace(class_y.dot(np.transpose(np.log(n2))) + (1 - class_y).dot(np.transpose(np.log(1 - n2)))) + Lambda / (2 * m) * term
    return np.ravel(J)
    

def checkGradient(Lambda = 0):
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 3
    m = 100

    W1, b1 = debugInitWeight(input_layer_size, hidden_layer_size)
    W2, b2 = debugInitWeight(hidden_layer_size, output_layer_size)

    nn_params = np.r_[W1.reshape(-1, 1), b1.reshape(-1, 1), W2.reshape(-1, 1), b2.reshape(-1, 1)]
    x = np.random.randint(1, 5, [m, input_layer_size])
    y = np.mod(np.random.randint(1, 10, [m, 1]), output_layer_size)

    grad = nnGradient_CRE(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)

    step = np.zeros([nn_params.shape[0]])
    num_grad = np.zeros([nn_params.shape[0]])

    e = 1e-4
    for i in range(len(nn_params)):
        step[i] = e
        loss1 = nnCostFunction_CRE(nn_params - step.reshape(-1, 1), input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)
        loss2 = nnCostFunction_CRE(nn_params + step.reshape(-1, 1), input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda)
        num_grad[i] = (loss2 - loss1) / (2 * e)
        step[i] = 0

    result = np.c_[grad, num_grad]
    print(result)

def displayData(x):
    padding = 1
    display_array = -1 * np.ones([10 * (20 + padding) + padding, 10 * (20 + padding) + padding])
    for i in range(10):
        for j in range(10):
            display_array[padding + i * (20 + padding) : padding + i * (20 + padding) + 20, padding + j * (20 + padding) : padding + j * (20 + padding) + 20] = x[i * 10 + j, :].reshape(20, 20, order = "F")
    
    plt.imshow(display_array)
    plt.axis("off")
    plt.show()

def predict(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y):
    length = len(nn_params)
    W1 = nn_params[0 : input_layer_size * hidden_layer_size].reshape([input_layer_size, hidden_layer_size]).copy()
    b1 = nn_params[input_layer_size * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size].reshape([1, hidden_layer_size]).copy()
    W2 = nn_params[(input_layer_size + 1) * hidden_layer_size : (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size].reshape([hidden_layer_size, output_layer_size]).copy()
    b2 = nn_params[(input_layer_size + 1) * hidden_layer_size + (hidden_layer_size) * output_layer_size : length].reshape([1, output_layer_size]).copy()

    m,n = x.shape

    n0 = x
    z1 = n0.dot(W1) + b1
    n1 = sigmoid(z1)

    z2 = n1.dot(W2) + b2
    n2 = sigmoid(z2)

    p = np.zeros([m, 1])
    max_h = np.max(n2, axis = 1)
    for i in range(output_layer_size):
        p[np.where(max_h == n2[:, i])] = i

    print(p)
    accuray = np.sum(p==y) * 1.0 / m * 100
    print("准确率为 %f%%" % accuray)


if __name__ == "__main__":
    # checkGradient(Lambda = 0.5)
    start = time.time()
    data = scio.loadmat("data_digits.mat")
    
    x = data['X']
    y = data['y']

    m,n = x.shape
    selected = [np.random.randint(i - i, m) for i in range(100)]
    displayData(x[selected, :])

    input_layer_size = n
    hidden_layer_size = 25
    output_layer_size = np.max(y) - np.min(y) + 1
    print(np.max(y), np.min(y))

    Lambda = 1

    W1, b1 = debugInitWeight(input_layer_size, hidden_layer_size)
    W2, b2 = debugInitWeight(hidden_layer_size, output_layer_size)
    nn_params = np.r_[W1.reshape(-1, 1), b1.reshape(-1, 1), W2.reshape(-1, 1), b2.reshape(-1, 1)]
    predict(nn_params, input_layer_size, hidden_layer_size, output_layer_size, x, y)
    result = scop.fmin_bfgs(f = nnCostFunction_CRE, x0 = nn_params, fprime = nnGradient_CRE, args = (input_layer_size, hidden_layer_size, output_layer_size, x, y, Lambda))

    stop = time.time()
    print("执行时间:",stop - start)
    
    predict(result, input_layer_size, hidden_layer_size, output_layer_size, x, y)


    
    