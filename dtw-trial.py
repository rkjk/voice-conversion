import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import scipy.io.wavfile
import python_speech_features as psf

x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance, path)

fs, w1 = scipy.io.wavfile.read('./sa1.wav')
fs, w2 = scipy.io.wavfile.read('./sa2.wav')

data1 = psf.mfcc(w1, fs)
data2 = psf.mfcc(w2, fs)
print(data1.shape)
print(data2.shape)

distance, path = fastdtw(data1, data2, dist=euclidean)
print(path[-20:])

