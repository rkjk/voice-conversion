import tensorflow as tf
import numpy as np
from sys import exit

source_data = np.load('source_input.npy')
target_data = np.load('target_input.npy')

inp = tf.convert_to_tensor(source_data)
outp = tf.convert_to_tensor(target_data)

learning_rate = 0.001
num_steps = 100
batch_size = 10
dispay_step = 50

n_hidden_1 = 10
n_hidden_2 = 10
num_input = 13
num_output = 13

X = tf.placeholder('float32', [1, num_input])
Y = tf.placeholder('float32', [1,num_output])

# Store layers weight & bias
weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
                    }
biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                    'out': tf.Variable(tf.random_normal([num_output]))
                    }


def neural_network(x):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out'] + biases['out'])

    return out_layer

regressor = neural_network(X)
loss_op = tf.reduce_sum(tf.pow(regressor - Y, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for x,y in zip(source_data, target_data):
        x = x.reshape(-1,13)
        y = y.reshape(-1,13)
        #print(sess.run(X, feed_dict={X:x}))
        #print(sess.run(Y, feed_dict={Y:y}))

        sess.run(train_op, feed_dict={X:x, Y:y})

        loss = sess.run(loss_op, feed_dict={X:x, Y:y})

    saver.save(sess, './my_test_model', global_step=1000)
