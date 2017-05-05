import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf_deep import TFDeep
import numpy as np
from sklearn import svm
import data

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

N = mnist.train.images.shape[0]
D = mnist.train.images.shape[1]
C = mnist.train.labels.shape[1]

Y_train_ = np.argmax(mnist.train.labels, axis=1)
Y_test_ = np.argmax(mnist.test.labels, axis=1)

tf.set_random_seed(1000)
np.random.seed(1000)

# Model 748x10
print("Model 748x10")
model = TFDeep([784, 10])
model.train(mnist.train.images, mnist.train.labels, 100)
Y_train = np.argmax(model.eval(mnist.train.images), axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
print("Train accuracy: {}".format(sum(Y_train == Y_train_) / Y_train_.shape[0]))
print("Test accuracy: {}".format(sum(Y_test == Y_test_) / Y_test_.shape[0]))

# Model 784x100x10
print("Model 748x100x10")
model = TFDeep([784, 100, 10])
model.train(mnist.train.images, mnist.train.labels, 100)
Y_train = np.argmax(model.eval(mnist.train.images), axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
print("Train accuracy: {}".format(sum(Y_train == Y_train_) / Y_train_.shape[0]))
print("Test accuracy: {}".format(sum(Y_test == Y_test_) / Y_test_.shape[0]))

# Early stopping
tf.set_random_seed(1000)
np.random.seed(1000)
print("Model 748x10 with early stopping")
indices = np.random.permutation(mnist.train.images.shape[0])
n = int(0.8 * N)
train_i, val_i = indices[:n], indices[n:]
X_train = mnist.train.images[train_i, :]
Y_train = mnist.train.labels[train_i, :]
X_val = mnist.train.images[val_i, :]
Y_val = mnist.train.labels[val_i, :]
model = TFDeep([784, 10])
model.train_es(X_train, Y_train, X_val, Y_val, 100, 20)
Y_train = np.argmax(model.eval(mnist.train.images), axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
print("Train accuracy: {}".format(sum(Y_train == Y_train_) / Y_train_.shape[0]))
print("Test accuracy: {}".format(sum(Y_test == Y_test_) / Y_test_.shape[0]))

# Stochastic gradient descent
tf.set_random_seed(1000)
np.random.seed(1000)
print("Model 748x10 with stochastic gradient descent")
model = TFDeep([784, 10])
model.train_mb(mnist.train.images, mnist.train.labels, 100, 100)
Y_train = np.argmax(model.eval(mnist.train.images), axis=1)
Y_test = np.argmax(model.eval(mnist.test.images), axis=1)
print("Train accuracy: {}".format(sum(Y_train == Y_train_) / Y_train_.shape[0]))
print("Test accuracy: {}".format(sum(Y_test == Y_test_) / Y_test_.shape[0]))

# # SVM with gaussian kernel
# clf = svm.SVC( decision_function_shape='ovo', max_iter=10)
# clf.fit(mnist.train.images, np.argmax(mnist.train.labels, axis=1))
# Y = clf.predict(mnist.train.images)
#
# # SVM with linear kernel
# clf = svm.LinearSVC(multi_class='ovo')
# clf.fit(mnist.train.images, np.argmax(mnist.train.labels, axis=1))
# Y = clf.predict(mnist.train.images)
