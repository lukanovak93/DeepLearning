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
Nh2 = 100
h2_shape = (10,10)

gibbs_sampling_steps = 2
alpha = 0.1

g2 = tf.Graph()
with g2.as_default():
    X2 = tf.placeholder("float", [None, Nv])
    w1s, vb1s, hb1s = pickle.load(open("weights.pickle", "rb"))
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = weights([Nh, Nh2])
    vb2 = bias([Nh])
    hb2 = bias([Nh2])

    # visible layer of the second RBM
    v2_prob = tf.nn.sigmoid(tf.matmul(X2, w1a) + hb1a)
    v2 = sample_prob(v2_prob)
    # hidden layer of the second RBM
    h2_prob = tf.nn.sigmoid(tf.matmul(v2, w2) + hb2)
    h2 = sample_prob(h2_prob)
    h3 = h2

    for step in range(gibbs_sampling_steps):
        v3_prob = tf.nn.sigmoid(tf.matmul(h3, w2, transpose_b=True) + vb2)
        v3 = sample_prob(v3_prob)
        h3_prob = tf.nn.sigmoid(tf.matmul(v3, w2) + hb2)
        h3 = sample_prob(h3_prob)

    w2_positive_grad = tf.matmul(v2, h2, transpose_a=True)
    w2_negative_grad = tf.matmul(v3, h3, transpose_a=True)

    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(v2)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_vb2 = tf.assign_add(vb2, alpha * tf.reduce_mean(v2 - v3, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2 - h3, 0))

    out2 = (update_w2, update_vb2, update_hb2)

    # input reconstruction by top layer state vector h3
    v4_prob = tf.nn.sigmoid(tf.matmul(h3, w2, transpose_b=True) + hb1a)
    v4 = sample_prob(v4_prob)
    v5_prob = tf.nn.sigmoid(tf.matmul(v4, w1a, transpose_b=True) + vb1a)

    err2 = X2 - v5_prob
    err_sum2 = tf.reduce_mean(err2 * err2)

    initialize2 = tf.initialize_all_variables()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

with tf.Session(graph=g2) as sess:
    sess.run(initialize2)

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([err_sum2, out2], feed_dict={X2: batch})

        if i%(int(total_batch/10)) == 0:
            print("Batch count: ", i, "  Avg. reconstruction error: ", err)

    w2s, vb2s, hb2s = sess.run([w2, vb2, hb2], feed_dict={X2: batch})
    vr2, h3_probs, h3s = sess.run([v5_prob, h3_prob, h3], feed_dict={X2: teX[0:50,:]})

    pickle.dump((w1s, w2s, hb1s, vb1s, hb2s, vb2s), open('weights_2.pickle', 'wb'))

# weights visualization
draw_weights(w2s, h1_shape, Nh2, interpolation="nearest")

# reconstruction and states visualization
draw_reconstructions(teX, vr2, h3s, v_shape, h2_shape)
plt.show()
