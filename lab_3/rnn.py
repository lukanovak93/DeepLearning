import numpy as np

class RNN:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(scale=1e-2, size=(vocab_size, hidden_size)) # input projection
        self.W = np.random.normal(scale=1e-2, size=(hidden_size, hidden_size)) # hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1)) # input bias

        self.V = np.random.normal(scale=1e-2, size=(hidden_size, vocab_size)) # output projection
        self.c = np.zeros((vocab_size, 1)) # output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        minibatch_size = x.shape[0]
        h = []
        h_prev=np.array(h_prev)
        for i in range(minibatch_size):
            a_t = np.squeeze(x[i].dot(U)) + np.squeeze(h_prev[i].dot(W)) + np.squeeze(b)
            h.append(np.expand_dims(np.tanh(a_t), axis=0))
        cache = np.array(h), h_prev, x, W.copy()
        return h, cache


    def rnn_forward(self, x, h0, U, W, b):
        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        cache = []
        h_current = h0
        h = np.empty((x.shape[0], 0, self.hidden_size))
        for i in range(self.sequence_length):
            h_current, current_cache = self.rnn_step_forward(x[:, i, :], h_current, U, W, b)
            h = np.append(h, h_current, axis=1)
            cache.append(current_cache)

        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        h, h_prev, x, W = cache
        minibatch_size = x.shape[0]
        dh_prev = np.zeros((self.hidden_size, 1))
        dU = np.zeros((self.vocab_size, self.hidden_size))
        dW = np.zeros((self.hidden_size, self.hidden_size))
        db = np.zeros((self.hidden_size, 1))

        for i in range(minibatch_size):
            da = np.multiply(np.expand_dims(grad_next[i], axis=1), (1 - h[i] ** 2).T)
            dh_prev += W.dot(da)
            dU += np.expand_dims(x[i], axis=1).dot(da.T)
            dW += da.dot(h_prev[i])
            db += da

        return dh_prev, dU, dW, db


    def rnn_backward(self, dh, cache):
        dU = np.zeros((self.vocab_size, self.hidden_size))
        dW = np.zeros((self.hidden_size, self.hidden_size))
        db = np.zeros((self.hidden_size, 1))
        print('dh ' + str(dh.shape))
        print('cache ' + str(np.array(cache).shape))
        for i in range(self.sequence_length - 1, -1, -1):
            dh, dU_i, dW_i, db_i = self.rnn_step_backward(dh[:, i, :], cache[i])
            dU += dU_i
            dW += dW_i
            db += db_i

        return np.clip(dU, -5, 5), np.clip(dW, -5, 5), np.clip(db, -5, 5)

    def output(self, h, V, c):
        batch_size = h.shape[0]
        probs = []
        for i in range(batch_size):
            h_i = np.expand_dims(h[i], axis=1)
            o = V.T.dot(h_i) + c
            s = np.sum(o)
            probs.append(np.exp(o.T) / s)
        probs = np.array(probs)
        return probs

    def output_loss_and_grads(self, h, V, c, y):
        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a one-hot vector of dimension
        #     vocabulary size x 1 - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[timestep] = np.zeros((vocabulary_size, 1))
        #   y[timestep][batch_y[timestep]] = 1

        #     where y might be a dictionary.
        loss = 0.0
        probs = []
        for i in range(self.sequence_length):
            p = self.output(h[:, i, :], V, c)
            loss += -np.log(p * y)
            probs.append(p)
        probs = np.stack(probs, axis=1)

        batch_size = h.shape[0]
        dV, dc, dh = [], [], []
        for i in range(batch_size):
            dV.append(h[i].T.dot(np.squeeze(probs[i]) - y[i]))
            dc.append(probs[i] - y[i])
            dh.append((np.squeeze(probs[i]) - y[i]).dot(V.T))

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h
        return loss, np.array(dh), np.array(dV), np.array(dc)

    def update(self, dU, dW, db, dV, dc):
        delta = 1e-7
        # update memory matrices
        self.memory_U += dU**2
        self.memory_W += dW**2
        self.memory_b += db**2
        self.memory_V += dV**2
        self.memory_c += dc**2
        # perform the Adagrad update of parameters
        self.U -= self.learning_rate / (delta + np.sqrt(self.memory_U)) * dU
        self.W -= self.learning_rate / (delta + np.sqrt(self.memory_W)) * dW
        self.b -= self.learning_rate / (delta + np.sqrt(self.memory_b)) * db
        self.V -= self.learning_rate / (delta + np.sqrt(self.memory_V)) * dV
        self.c -= self.learning_rate / (delta + np.sqrt(self.memory_c)) * dc

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc)
        return loss, np.zeros((self.hidden_size, 1)) # h0 ???

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((dataset.batch_size, 1, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((dataset.batch_size, hidden_size, 1))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        batch_size, sequence_length = y.shape # sequence length vec postavljen
        x_oh, y_oh = np.zeros((batch_size, sequence_length, vocab_size)), np.zeros((batch_size, sequence_length, vocab_size))
        for i in range(batch_size):
            for j in range(sequence_length):
                y_oh[i, j, y[i, j]] = 1
                x_oh[i, j, x[i, j]] = 1

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)

        if batch % sample_every == 0:
            # run sampling (2.2)
            pass
        batch += 1
