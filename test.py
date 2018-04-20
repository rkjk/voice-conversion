import tensorflow as tf
import numpy as np
import python_speech_features as psf
from sys import exit

sess = tf.Session()

saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

b1 = sess.run('b1:0')
b2 = sess.run('b2:0')
out_b = sess.run('out_b:0')
h2 = sess.run('h2:0')
h1 = sess.run('h1:0')
out_h = sess.run('out_h:0')


def neural_network(x):


    layer_1 = tf.add(tf.matmul(x, h1), b1)
    print("layer_1 :", layer_1.get_shape())
    layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
    print("layer_2 :", layer_2.get_shape())
    out_layer = tf.matmul(layer_2, out_h + out_b)


    return out_layer

#rt = tf.constant(1, shape=(1,13), dtype=tf.float32)
rt = np.ones(13).reshape(-1,13)

X = tf.placeholder('float32', [1, 13], name="X")
#Y = tf.placeholder('float32', [1, 13], name="Y")
Y = neural_network(X)

source = [np.random.rand(13).reshape(-1,13), np.ones(13).reshape(-1,13)]

target = []

with sess.as_default():

    for x in source:

        sess.run(Y, feed_dict={X:x})
        target.append(sess.run(Y, feed_dict={X:x}))

print(target)
