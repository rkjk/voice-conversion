import tensorflow as tf
import numpy as np
from sys import exit

#source_data = np.load('source_input.npy')
#target_data = np.load('target_input.npy')
source_data = np.load('source_input_logfbank.npy')
target_data = np.load('target_input_logfbank.npy')

learning_rate = 0.1
num_steps = 100
batch_size = 10

n_hidden_1 = 10
n_hidden_2 = 10
num_input = source_data.shape[1]
num_output = target_data.shape[1]


X = tf.placeholder('float32', [1, num_input], name="X")
Y = tf.placeholder('float32', [1,num_output], name="Y")

# Store layers weight & bias
weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name='h1'),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
                    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]), name='out_h')
                    }
biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
                'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
                    'out': tf.Variable(tf.random_normal([num_output]), name='out_b')
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
        x = x.reshape(-1,num_input)
        y = y.reshape(-1,num_output)
        #print(sess.run(X, feed_dict={X:x}))
        #print(sess.run(Y, feed_dict={Y:y}))

        sess.run(train_op, feed_dict={X:x, Y:y})

        loss = sess.run(loss_op, feed_dict={X:x, Y:y})

    saver.save(sess, './models/my_test_model-logfbank')
