import numpy as np
import data

def fcann2_train(X, Y_, iterations=1000, param_delta=0.05, param_lambda=1e-3, h=5):
    Y_ = Y_.astype(int)
    N, D = X.shape
    K = max(Y_) + 1

    W1 = np.random.randn(D, h)
    b1 = np.zeros((1, h))
    W2 = np.random.randn(h, K)
    b2 = np.zeros((1, K))

    for i in range(iterations):
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        # exponated classification results
        scores = np.dot(hidden_layer, W2) + b2
        expscores = np.exp(scores)

        # softmax denominator
        sumexp = np.sum(expscores, axis=1, keepdims=True)

        # log-probabilities of classes
        probs = expscores / sumexp
        logprobs = -np.log(probs[range(N), Y_])

        # loss function
        loss  = np.sum(logprobs) / N + param_lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        # print
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # loss derivation over components by results
        dL_ds = probs
        dL_ds[range(N), Y_] -= 1
        dL_ds /= N

        # parameters gradients
        grad_W2 = np.dot(hidden_layer.T, dL_ds)
        grad_b2 = np.sum(dL_ds, axis=0, keepdims=True)

        grad_hidden = np.dot(dL_ds, W2.T)
        grad_hidden[hidden_layer <= 0] = 0

        grad_W1 = np.dot(X.T, grad_hidden)
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

        # updated parameters
        W1 += -param_delta * grad_W1
        b1 += -param_delta * grad_b1
        W2 += -param_delta * grad_W2
        b2 += -param_delta * grad_b2
    return W1, b1, W2, b2

def fcann2_classify(W1, b1, W2, b2):
    def classify(X):
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        return scores
    return classify

if __name__ == '__main__':
    (X, Y_) = data.sample_gmm_2d(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    classify = fcann2_classify(W1, b1, W2, b2)
    Y = classify(X)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(classify, bbox)
    data.graph_data(X, Y_, np.argmax(Y, axis=1))
