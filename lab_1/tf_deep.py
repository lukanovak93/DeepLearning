import tensorflow as tf
import numpy as np
import data

class TFDeep:
    def __init__(self, layers, param_delta=0.5, param_lambda=0.1):
        D = layers[0]
        C = layers[-1]
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.h = layers
        self.W = []
        self.b = []
        for i in range(1, len(self.h)):
            self.W.append(tf.Variable(tf.random_normal([self.h[i-1], self.h[i]]), name='W'+str(i)))
            self.b.append(tf.Variable(tf.random_normal([self.h[i]]), name='b'+str(i)))

        a = self.X
        for i in range(len(self.W) - 1):
            a = tf.nn.relu(tf.matmul(a, self.W[i]) + self.b[i])
        self.probs = tf.nn.softmax(tf.matmul(a, self.W[-1]) + self.b[-1])

        regularization_loss = 0
        for w in self.W:
            regularization_loss += tf.nn.l2_loss(w)
        self.loss = tf.reduce_mean(-tf.reduce_sum(
            tf.mul(tf.log(tf.clip_by_value(self.probs,1e-10,1.0)), self.Yoh_), reduction_indices=1
            )) + param_lambda * regularization_loss

        # trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(param_delta, self.global_step,
                                                    1, 1-1e-4, staircase=True)
        trainer = tf.train.AdamOptimizer(1e-4)
        self.train_step = trainer.minimize(self.loss)
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter):
            self.global_step.assign(self.global_step + 1)
            loss = self.session.run([self.loss, self.train_step] + self.W + self.b,
                                            feed_dict={self.X: X, self.Yoh_: Yoh_})[0]
            if i % 10 == 0:
                print(i, loss)

    def train_es(self, X, Yoh_, X_val, Yoh_val, param_niter, patience):
        best_loss = -1
        best_W = None
        best_b = None
        j = 0

        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter):
            j += 1
            loss = self.session.run([self.loss, self.train_step] + self.W + self.b,
                                            feed_dict={self.X: X, self.Yoh_: Yoh_})[0]
            if i % 10 == 0:
                print(i, loss)
                val_loss = self.session.run([self.loss], feed_dict={self.X: X_val, self.Yoh_: Yoh_val})[0]
                if best_W == None or val_loss < best_loss:
                    j = 0
                    best_loss = val_loss
                    best_W = self.W
                    best_b = self.b

            if j > patience:
                self.W = best_W
                self.b = best_b
                break

    def train_mb(self, X, Yoh_, param_niter, batch_size):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter):
            n_batches = int(X.shape[0] / batch_size)
            indices = np.random.permutation(X.shape[0])
            for j in range(n_batches):
                from_i = j * batch_size
                to_i = min((j+1) * batch_size, X.shape[0])
                loss = self.session.run([self.loss, self.train_step] + self.W + self.b,
                                        feed_dict={self.X: X[indices[from_i : to_i], :],
                                                   self.Yoh_: Yoh_[indices[from_i : to_i], :]})[0]
            if i % 10 == 0:
                print(i, loss)

    def eval(self, X):
        return self.session.run([self.probs], feed_dict={self.X: X})[0]

    def count_params(self):
        for v in tf.trainable_variables():
            print(v.name)
        total_count = 0
        for i in range(1, len(self.h)):
            total_count += self.h[i] * self.h[i-1]
        total_count += sum(self.h[1:])
        print("Total parameter count: " + str(total_count))

if __name__ == '__main__':
    (X, Y_) = data.sample_gmm_2d(6, 2, 10)
    N, D = X.shape
    C = 2
    Yoh_ = np.zeros((N, C))
    Yoh_[range(N), Y_.astype(int)] = 1
    model = TFDeep([2, 3, 2])
    model.train(X, Yoh_, 1000)
    probs = model.eval(X)
    model.count_params()
    Y = np.argmax(probs, axis=1)
    print(data.eval_perf_binary(Y, np.argmax(Yoh_, axis=1)))
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(model.eval, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
