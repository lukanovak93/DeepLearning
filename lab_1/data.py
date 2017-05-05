import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def sample_gmm_2d(K, C, N):
    X = np.zeros((K*N, 2), dtype=float)
    Y = np.zeros((K*N), dtype=float)
    for i in range(K):
        mean = np.random.uniform(-10, 10)
        sigma = np.random.uniform(0, 5)
        c_i = np.random.choice(range(C))
        for j in range(N):
            X[i*N+j] = np.random.uniform(mean, sigma, 2)
            Y[i*N+j] = c_i
    return (X, Y)

def eval_perf_binary(Y, Y_):
    accuracy = float(sum(Y == Y_)) / len(Y)
    precision = float(sum((Y == Y_) == (Y == 1))) / sum(Y)
    recall = float(sum((Y == Y_) == (Y == 1))) / sum(Y_)
    precisions = []
    yy = Y == Y_
    correct = 0
    for i in range(len(yy)):
        if yy[i]:
            correct += 1
            precisions.append(correct / (i + 1))
    avg_precision = np.mean(precisions)
    return accuracy, precision, recall, avg_precision

def graph_data(X, Y_, Y, special=[]):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(Y_))
    for i in range(X.shape[0]):
        if Y_[i] == Y[i]:
            marker = 'o'
        else:
            marker = 's'
        if i in special:
            size = 40
        else:
            size = 20
        plt.scatter(X[i, 0], X[i, 1], marker=marker, c=Y_[i], cmap='Greys', norm=norm, s=size)
    plt.show()

def graph_surface(fun, rect, offset=0.0, width=1000, height=1000):
    x1 = np.linspace(rect[0][0], rect[1][0], height)
    x2 = np.linspace(rect[0][1], rect[1][1], width)
    xx, yy = np.meshgrid(x1, x2)
    h = fun(np.stack([xx.flatten(), yy.flatten()], axis=1))
    if len(h.shape) > 1:
        h = h[:, 1]
    h = h.reshape(xx.shape)
    plt.contour(xx, yy, h, colors='black', levels=[offset])
    m = abs(h.flatten().max())
    norm = matplotlib.colors.Normalize(vmin=offset-m,vmax=m+offset)
    plt.pcolormesh(xx, yy, h, norm=norm)

def myDummyDecision(X):
    score = X[:,0] + X[:,1] - 5
    return score

if __name__ == "__main__":
    np.random.seed(1000)
    tf.random.seed(1000)
    (X, Y_) = sample_gmm_2d(4, 2, 30)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0.5)
    graph_data(X, Y_, myDummyDecision(X))
