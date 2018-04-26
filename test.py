import tensorflow as tf
import numpy as np
import python_speech_features as psf
from sphfile import SPHFile
import scipy.io.wavfile
from sys import exit

sess = tf.Session()

#saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver = tf.train.import_meta_graph('models/my_test_model-logfbank.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))

b1 = sess.run('b1:0')
b2 = sess.run('b2:0')
out_b = sess.run('out_b:0')
h2 = sess.run('h2:0')
h1 = sess.run('h1:0')
out_h = sess.run('out_h:0')


def neural_network(x):
    """
    Feedforward NN to evaluate the function
    """


    layer_1 = tf.add(tf.matmul(x, h1), b1)

    layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
    
    out_layer = tf.matmul(layer_2, out_h + out_b)

    return out_layer



filename = '/home/raghav/sem2/speech/proj/TIMIT/TIMIT/TRAIN/DR6/MABC0/SA1.WAV'
source = SPHFile(filename)
source.write_wav('source.wav')
fs, source = scipy.io.wavfile.read('source.wav')
source = psf.logfbank(source, fs)
print(source.shape)
num_input = source.shape[1]

X = tf.placeholder('float32', [1, num_input], name="X")
Y = neural_network(X)

target = []

with sess.as_default():

    for x in source:
        x = x.reshape(-1,num_input)

        sess.run(Y, feed_dict={X:x})
        target.append(sess.run(Y, feed_dict={X:x}).reshape(-1,1))
        #target.append(sess.run(Y, feed_dict={X:x}))

target = np.squeeze(target)
print(target.shape)
#np.save('converted-speech.npy', target)
