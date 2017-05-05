import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
import pickle

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def sample_prob(probs):
"""Sampling of vector x by probability vector p(x=1) = probs"""
    return tf.nn.relu(
        tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def draw_weights(W, shape, N, interpolation="bilinear"):
    """Weights visualization

    W -- weight vector
    shape -- tuple of dimensions for 2D weights representation - usually, input picture 		dimensions, ex (28,28)
    N -- number of weights vectors
    """
    image = Image.fromarray( tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/20)), 20),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)

def draw_reconstructions(ins, outs, states, shape_in, shape_state):

    """Visualization of input and matching reconstructions and state of the hidden layer:
	ins -- input vectors
	outs -- reconstructed vectors
	states -- hidden layer state vector
	shape_in -- input dimensions, ex (28, 28)
	shape_state -- dimensions for 2D state visualization
    """

    plt.figure(figsize=(8, 12*4))
    for i in range(20):
        plt.subplot(20, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.subplot(20, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.subplot(20, 4, 4*i + 3)
        plt.imshow(states[i][0:(shape_state[0] * shape_state[1])].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
    plt.tight_layout()

Nv = 784
v_shape = (28,28)
Nh = 100
h1_shape = (10,10)

gibbs_sampling_steps = 1
# learning rate
alpha = 0.1 

g1 = tf.Graph()
with g1.as_default():
    X1 = tf.placeholder("float", [None, 784])
    w1 = weights([Nv, Nh])
    vb1 = bias([Nv])
    hb1 = bias([Nh])

    h0_prob = tf.nn.sigmoid(tf.matmul(X1, w1) + hb1)
    h0 = sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = tf.nn.sigmoid(tf.matmul(h1, w1, transpose_b=True) + vb1)
        v1 = sample_prob(v1_prob)
        h1_prob = tf.nn.sigmoid(tf.matmul(v1, w1) + hb1)
        h1 = sample_prob(h1_prob)

    # positive phase
    w1_positive_grad = tf.matmul(X1, h0, transpose_a=True)

    # negative phase
    w1_negative_grad = tf.matmul(v1, h1, transpose_a=True)

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    # network parameter refreshing - RBM learning start
    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1_prob, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0))

    out1 = (update_w1, update_vb1, update_hb1)

    # reconstruction of the input vector - probability p(v=1)
    v1_prob = tf.nn.sigmoid(tf.matmul(h1, w1, transpose_b=True) + vb1) # [? x Nv]

    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.initialize_all_variables()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

with tf.Session(graph=g1) as sess:
    sess.run(initialize1)
    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([err_sum1, out1], feed_dict={X1: batch})

        if i%(int(total_batch/10)) == 0:
            print("Batch count: " + str(i) + "  Avg. reconstruction error: " + str(err))

    w1s = w1.eval()
    vb1s = vb1.eval()
    hb1s = hb1.eval()
    pickle.dump((w1s, vb1s, hb1s), open('weights.pickle', 'wb'))
    vr, h1_probs, h1s = sess.run([v1_prob, h1_prob, h1], feed_dict={X1: teX[0:20, :]})

# weights visualization
draw_weights(w1s, v_shape, Nh)

# reconstruction and states visualization
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape)
plt.show()
