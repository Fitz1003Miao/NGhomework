#-*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

def findClosestCentroids(x, centroids):
    k = len(centroids)
    idx = np.zeros([x.shape[0], 1])
    dist = np.zeros([x.shape[0], k])
    for i in range(k):
        dist[:, i] = np.sqrt(np.sum((x - centroids[i]) ** 2, axis = 1)).reshape(1, -1)

    # idx = np.where(dist == np.min(dist, axis = 1).reshape(-1, 1))[1]
    idx = np.argmin(dist, axis = 1).reshape(-1, 1) # 比 np.where 强的地方在于每一行的值都是唯一的,不会出现一行有多个相等的最大值
    return idx

def computeCentroids(x, idx, K):
    centroids = np.zeros([K, x.shape[1]])
    for i in range(K):
        centroids[i, :] = np.mean(x[np.where(idx == i)[0],:], axis = 0)
    return centroids

def runkMeans(x, initial_centroids, iterations, plot_progress = False):
    centroids = initial_centroids
    idx = np.zeros([x.shape[0], 1])
    K = len(initial_centroids)

    if plot_progress:
        plt.scatter(x[:, 0], x[:, 1], c = "r", marker = ".")
        plt.scatter(centroids[:, 0], centroids[:, 1], c = "g", marker = "*")
        plt.scatter(centroids[1, 0], centroids[1, 1], c = "b", marker = "*")
        plt.scatter(centroids[2, 0], centroids[2, 1], c = "y", marker = "*")
            
        plt.show()

    for i in range(iterations):
        idx = findClosestCentroids(x, centroids)
        centroids = computeCentroids(x, idx, K)
        if plot_progress:
            plt.scatter(x[:, 0], x[:, 1], c = "r", marker = ".")
            plt.scatter(centroids[0, 0], centroids[0, 1], c = "g", marker = "*")
            plt.scatter(centroids[1, 0], centroids[1, 1], c = "b", marker = "*")
            plt.scatter(centroids[2, 0], centroids[2, 1], c = "y", marker = "*")
            
            plt.show()
    return centroids, idx

def kMeansInitCentroids(x, K):
    randperm = np.random.permutation(x.shape[0])
    return x[randperm[0:K], :]

if __name__ == "__main__":
    data = scio.loadmat("ex7data2.mat")
    x = data["X"]
    K = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    idx = findClosestCentroids(x, initial_centroids)
    centroids = computeCentroids(x, idx, K)
    iterations = 10
    runkMeans(x, initial_centroids, iterations, plot_progress = True)

    img = plt.imread("bird_small.png")
    img /= 255
    img_shape = img.shape
    img = img.reshape(img_shape[0] * img_shape[1], 3)
    
    K = 16
    iterations = 10

    initial_centroids = kMeansInitCentroids(img, K)
    centroids, idx = runkMeans(img, initial_centroids, iterations)
    idx = findClosestCentroids(img, centroids)
    
    X_recovered = centroids[idx, :] * 255
    X_recovered = X_recovered.reshape(img_shape[0], img_shape[1], 3)
    plt.imsave("recovered.jpg", X_recovered)