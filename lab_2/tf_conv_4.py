from cifar_data import *
import tensorflow as tf
import math
import skimage as ski
import skimage.io
from matplotlib import pyplot as plt

SAVE_DIR = '/home/luka/Documents/fer/deep_learning/lab_2/save/zad_4'

# Helper functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_nxn(x, n, k):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, k, k, 1], padding='SAME')

def evaluate(y_, y, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(y_.shape[0]):
        actual = y_[i]
        predicted = y[i]
        confusion_matrix[actual][predicted] += 1
    correct_predictions = 0
    for i in range(num_classes):
        correct_predictions += confusion_matrix[i][i]
    accuracy = correct_predictions / (y_.shape[0])
    return (accuracy, confusion_matrix)

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[3]
  num_channels = w.shape[2]
  k = w.shape[0]
  assert w.shape[0] == w.shape[1]
  w = w.reshape(k, k, num_channels, num_filters)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


# Data info
num_classes = 10
img_height = 32
img_width = 32
num_channels = 3
weight_decay = 1e-4
N = train_x.shape[0]
epochs = 8
batch_size = 50
num_batches = N // batch_size
global_step = tf.Variable(1e-1, trainable=False)
learning_rate = 5e-2

# Initialize plot data
plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

# Network
y_ = tf.placeholder(tf.float32, [None, num_classes])
x = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])

W_conv1 = weight_variable([5, 5, num_channels, 16])
b_conv1 = bias_variable([16])

h_1 = max_pool_nxn(tf.nn.relu(conv2d(x, W_conv1) + b_conv1), 3, 2)

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv_2 = bias_variable([32])

h_2 = max_pool_nxn(tf.nn.relu(conv2d(h_1, W_conv2) + b_conv_2), 3, 2)
h_2_flat = tf.reshape(h_2, [-1, 8 * 8 * 32])

W_fc1 = weight_variable([8 * 8 * 32, 256])
b_fc1 = bias_variable([256])

h_3 = tf.nn.relu(tf.matmul(h_2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([256, 128])
b_fc2 = bias_variable([128])

h_4 = tf.nn.relu(tf.matmul(h_3, W_fc2) + b_fc2)

W_fc3 = weight_variable([128, num_classes])
b_fc3 = bias_variable([num_classes])

y = tf.matmul(h_4, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_) +
    weight_decay * (tf.nn.l2_loss(W_conv1)**2 + tf.nn.l2_loss(W_conv2)**2 + tf.nn.l2_loss(W_fc1)**2 +
    tf.nn.l2_loss(W_fc2)**2 + tf.nn.l2_loss(W_fc3)**2))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Data preprocessing
train_y_oh = np.zeros((N, num_classes))
for i in range(N):
    train_y_oh[i][train_y[i]] = 1

test_y_oh = np.zeros((test_y.shape[0], num_classes))
for i in range(test_y.shape[0]):
    test_y_oh[i][test_y[i]] = 1

# Training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        train_x, train_y_oh = shuffle_data(train_x, train_y_oh)
        for i in range(num_batches):
            lower_bound = i * batch_size
            upper_bound = min((i + 1) * batch_size, train_x.shape[0])
            batch = (train_x[lower_bound:upper_bound, ...], train_y_oh[lower_bound:upper_bound, ...])

            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
                print("epoch %d, step %d/%d, training accuracy %g"%(epoch, i, N//batch_size, train_accuracy))
                draw_conv_filters(epoch, i, W_conv1.eval(), SAVE_DIR)

    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_x, y_: test_y_oh}))
