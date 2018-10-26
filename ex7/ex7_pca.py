import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

def featureNormalize(x):
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0, ddof = 1)
    return (x - mean) / std, mean, std

def drawline(plt,p1,p2,line_type):
    plt.plot(np.array([p1[0],p2[0]]),np.array([p1[1],p2[1]]),line_type)

def projectData(x, u, k):
    Z = x.dot(u[:, 0:k])
    return Z

def recoverData(x, u, k):
    x_recover = x.dot(np.transpose(u[:, 0:k]))
    return x_recover

def displayData(x):
    m, n = x.shape
    rows = np.int32(np.sqrt(m))
    cols = np.int32(m / rows)
    
    padding = 1
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n/width)
    display_array = -1 * np.ones([padding + rows * (height + padding), padding + cols * (width + padding)])

    for row in range(rows):
        for col in range(cols):
            display_array[padding + row * (height + padding) : padding + row * (height + padding) + height, padding + col * (width + padding) : padding + col * (width + padding) + width] = np.reshape(x[row * cols + col, :], (height, width), order = "F")

    plt.imshow(display_array, cmap = "gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    data = scio.loadmat("ex7data1.mat")
    x = data["X"]
    plt.scatter(x[:, 0], x[:, 1], c = "b", marker = "o")
    # plt.show()

    x_normal, mu, std = featureNormalize(x)
    n = len(x_normal)
    u,s,v = np.linalg.svd(np.transpose(x_normal).dot(x_normal) / n)

    drawline(plt, mu, mu + s[0] * u[:, 0], "r-")
    plt.axis("square")
    plt.show()

    K = 1
    Z = projectData(x_normal, u, K)
    x_recover = recoverData(Z, u, K)

    plt.scatter(x_normal[:, 0], x_normal[:, 1], c = "b", marker = "o")
    plt.scatter(x_recover[:, 0], x_recover[:, 1], c = "r", marker = "o")
    for i in range(len(x_normal)):
        drawline(plt, x_normal[i,:], x_recover[i,:], "--k")

    plt.axis("square")
    plt.show()


    data = scio.loadmat("ex7faces.mat")
    x = data["X"]
    displayData(x[0:100, :])

    x_normal, mu, std = featureNormalize(x)
    m = len(x)
    u, s, v = np.linalg.svd(np.transpose(x_normal).dot(x_normal) / m)
    K = 100

    Z = projectData(x_normal, u, K)
    displayData(np.transpose(u[:, 0:36]))
    displayData(Z[0:100, :])
    x_recover = recoverData(Z, u, K)
    displayData(x_recover[0:100, :])
