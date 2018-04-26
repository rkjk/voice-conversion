import numpy as np
import os
import python_speech_features as psf
import scipy.io.wavfile
from sys import exit
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def dtw_list_store(source, target, source_list, target_list):

    """
    Store the Time aligned frames for the Source and Target in separate lists
    List is passed as reference.
    List is 3 dimensional: (#frames, #features, #files)
    So each time aligned series is stored separately for a user

    Arguments:

    source - Source segment for the particular phoneme
    target - Target segment for the particular phoneme
    dtw_source - List containing source mfcc that we will append to
    dtw_target - Corresponding target mfcc list
    """

    dtw_source = []
    dtw_target = []

    fs, source = scipy.io.wavfile.read(source)
    fs, target = scipy.io.wavfile.read(target)


    #source = psf.mfcc(source, 16000)
    #target = psf.mfcc(target, 16000)

    source, energy = psf.fbank(source, 16000)
    target, energy = psf.fbank(target, 16000)

    distance, path = fastdtw(source, target, dist=euclidean)

    for vertex in path:
        dtw_source.append(source[vertex[0],:])
        dtw_target.append(target[vertex[1],:])

    dtw_source = np.array(dtw_source)
    dtw_target = np.array(dtw_target)


    source_list.append(dtw_source)
    target_list.append(dtw_target)

def filesearch():
    dir_root = '/home/raghav/sem2/speech/proj/vcc2016_training/'
    dir1 = dir_root + 'SF1/'
    dir2 = dir_root + 'TF1/'

    source_list = []
    target_list = []

    for sfile in os.listdir(dir1):
        sf1 = dir1+sfile
        tf1 = dir2+sfile

        dtw_list_store(sf1, tf1, source_list,target_list)
        print(sfile, " is done")
    
    for (x,y) in zip(source_list, target_list):
        assert(x.shape==y.shape)

    np.save('source_input_vcc_melfbank.npy', source_list)
    np.save('target_input_vcc_melfbank.npy', target_list)


filesearch()
