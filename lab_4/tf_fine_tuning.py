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


beta = 0.01
Nv = 784
v_shape = (28,28)
Nh = 100
h1_shape = (10,10)
Nh2 = 100
h2_shape = (10,10)
gibbs_sampling_steps = 2

g3 = tf.Graph()
with g3.as_default():
    X3 = tf.placeholder("float", [None, Nv])
    w1s, w2s, hb1s, vb1s, hb2s, vb2s = pickle.load(open("weights_2.pickle", "rb"))
    w1_up = tf.Variable(w1s)
    w1_down = tf.Variable(tf.transpose(w1s))
    w2a = tf.Variable(w2s)
    hb1_up = tf.Variable(hb1s)
    hb1_down = tf.Variable(vb2s)
    vb1_down = tf.Variable(vb1s)
    hb2a = tf.Variable(hb2s)

    # wake pass
    h1_up_prob = tf.nn.sigmoid(tf.matmul(X3, w1_up) + hb1_up)
    h1_up = sample_prob(h1_up_prob) # s^{(n)} u pripremi
    v1_up_down_prob = tf.nn.sigmoid(tf.matmul(h1_up, w1_down) + vb1_down)
    v1_up_down = sample_prob(v1_up_down_prob) # s^{(n-1)\mathit{novo}} u pripremi

    # top RBM Gibs passes
    h2_up_prob = tf.nn.sigmoid(tf.matmul(h1_up, w2a) + hb2a)
    h2_up = sample_prob(h2_up_prob)
    h4 = h2_up

    for step in range(gibbs_sampling_steps):
        h1_down_prob = tf.nn.sigmoid(tf.matmul(h4, w2a) + hb1_down)
        h1_down = sample_prob(h1_down_prob)
        h4_prob = tf.nn.sigmoid(tf.matmul(h1_down, w2a) + hb2a)
        h4 = sample_prob(h4_prob)

    # sleep pass
    v1_down_prob = tf.nn.sigmoid(tf.matmul(h1_down, w1_down) + vb1_down)
    v1_down = sample_prob(v1_down_prob) # s^{(n-1)} u pripremi
    h1_down_up_prob = tf.nn.sigmoid(tf.matmul(v1_down, w1_up) + hb1_up)
    h1_down_up = sample_prob(h1_down_up_prob) # s^{(n)\mathit{novo}} u pripremi


    # generative weights update during wake pass
    update_w1_down = tf.assign_add(w1_down, beta * tf.matmul(tf.transpose(h1_up), X3 - v1_up_down_prob) / tf.to_float(tf.shape(X3)[0]))
    update_vb1_down = tf.assign_add(vb1_down, beta * tf.reduce_mean(X3 - v1_up_down_prob, 0))

    # top RBM update
    w2_positive_grad = tf.matmul(h1_up, h2_up, transpose_a=True)
    w2_negative_grad = tf.matmul(h1_down, h4, transpose_a=True)
    dw3 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1_up)[0])
    update_w2 = tf.assign_add(w2a, beta * dw3)
    update_hb1_down = tf.assign_add(hb1_down, beta * tf.reduce_mean(h1_up - h1_down, 0))
    update_hb2 = tf.assign_add(hb2a, beta * tf.reduce_mean(h2_up - h4, 0))

    # recognition weights update during sleep pass
    update_w1_up = tf.assign_add(w1_up, beta * tf.matmul(tf.transpose(v1_down_prob), h1_down - h1_down_up) / tf.to_float(tf.shape(X3)[0]))
    update_hb1_up = tf.assign_add(hb1_up, beta * tf.reduce_mean(h1_down - h1_down_up, 0))###########^ #####

    out3 = (update_w1_down, update_vb1_down, update_w2, update_hb1_down, update_hb2, update_w1_up, update_hb1_up)

    err3 = X3 - v1_down_prob
    err_sum3 = tf.reduce_mean(err3 * err3)

    initialize3 = tf.initialize_all_variables()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

with tf.Session(graph=g3) as sess:
    sess.run(initialize3)
    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([err_sum3, out3], feed_dict={X3: batch})

        if i%(int(total_batch/10)) == 0:
            print("Batch count: ", i, "  Avg. reconstruction error: ", err)

    w2ss, w1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs = sess.run(
        [w2a, w1_up, w1_down, hb2a, hb1_up, hb1_down, vb1_down], feed_dict={X3: batch})
    vr3, h4s, h4_probs = sess.run([v1_down_prob, h4, h4_prob], feed_dict={X3: teX[0:20,:]})

# weights visualization
draw_weights(w1_ups, v_shape, Nh)
draw_weights(w1_downs.T, v_shape, Nh)
draw_weights(w2ss, h1_shape, Nh2, interpolation="nearest")

# reconstruction and states visualization
Npics = 5
plt.figure(figsize=(8, 12*4))
for i in range(20):
    plt.subplot(20, Npics, Npics*i + 1)
    plt.imshow(teX[i].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Test input")
    plt.axis('off')
    plt.subplot(20, Npics, Npics*i + 2)
    plt.subplot(20, Npics, Npics*i + 4)
    plt.imshow(vr3[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 3")
    plt.axis('off')
    plt.subplot(20, Npics, Npics*i + 5)
    plt.imshow(h4s[i][0:Nh2].reshape(h2_shape), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Top states 3")
    plt.axis('off')
plt.tight_layout()
plt.show()
