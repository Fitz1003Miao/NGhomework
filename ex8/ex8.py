import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

def visData(x):
    plt.plot(x[:, 0], x[:, 1], "bx")

    plt.xlim(0, 30)
    plt.ylim(0, 30)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")

    plt.show()

def estimateGussian(x):
    mu = np.mean(x, axis = 0)
    sigma2 = np.std(x, axis = 0, ddof = 1) ** 2
    return mu, sigma2

def multivariateGussian(x, mu, sigma2):
    D = x.shape[1]
    if sigma2.shape[0] > 1:
        sigma2 = np.diag(sigma2)

    x = x - mu
    k = 1 / ((2 * np.pi) ** (D / 2) * np.linalg.det(sigma2) ** 0.5)

    p = k * np.exp(-0.5 * np.sum(x.dot(np.linalg.inv(sigma2)) * x, axis = 1))

    return p

def visualizeFit(x, mu, simga):
    X = np.arange(0, 36, 0.5)
    Y = np.arange(0, 36, 0.5)
    X1, X2 = np.meshgrid(X, Y)
    Z = multivariateGussian(np.c_[X1.reshape(-1, 1), X2.reshape(-1, 1)], mu, simga)
    Z = Z.reshape(X1.shape)

    plt.plot(x[:, 0], x[:, 1], "bx")
    if np.sum(np.isinf(Z).astype(float)) == 0:
        CS = plt.contour(X1, X2, Z, 10. ** np.arange(-20, 0, 3), color = 'black', linewidth = .5)

    plt.show()

def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    step = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        cvPrecision = pval < epsilon
        
        tp = np.sum((cvPrecision == 1) & (yval == 1).ravel()).astype(float)
        fp = np.sum((cvPrecision == 1) & (yval == 0).ravel()).astype(float)
        fn = np.sum((cvPrecision == 0) & (yval == 1).ravel()).astype(float)

        precision = tp / (tp + fp)
        recision = tp / (tp + fn)
        F1 = 2 * (precision * recision) / (precision + recision)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


if __name__ == "__main__":
    data = scio.loadmat("ex8data1.mat")
    x = data["X"]

    visData(x)
    mu, sigma2 = estimateGussian(x)
    p = multivariateGussian(x, mu, sigma2)

    visualizeFit(x, mu, sigma2)

    xval = data["Xval"]
    yval = data["yval"]
    pval = multivariateGussian(xval, mu, sigma2)
    
    epsilon, F1 = selectThreshold(yval, pval)
    outliers = np.where(p < epsilon)
    plt.plot(x[outliers, 0], x[outliers, 1], 'o', markeredgecolor = 'r', markerfacecolor = 'w', markersize = 10)
    visData(x)