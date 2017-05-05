import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

Y = a * X + b

loss = (Y - Y_)**2

loss_grad_a = tf.reduce_sum(2 * (a * X + b - Y_) * X)
loss_grad_b = tf.reduce_sum(2 * (a * X + b - Y_))

trainer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads_and_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    val_loss, _, val_a, val_b, val_grads_and_vars, val_loss_grad_a, val_loss_grad_b = sess.run(
        [loss, train_op, a, b, grads_and_vars, loss_grad_a, loss_grad_b],
        feed_dict={X: [1, 2], Y_: [3, 5]}
    )
    val_grads = list(map(lambda x: x[0], val_grads_and_vars))
    print(i,val_loss, val_a,val_b, val_grads, [val_loss_grad_a, val_loss_grad_b])
