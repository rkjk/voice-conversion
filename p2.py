import numpy as np
import os
import scipy.io.wavfile
import python_speech_features as psf
from sphfile import SPHFile
from sys import exit
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw




def create_dict(filename):

    """ 
    Function to take as input PHN file
    Read the PHN file line by line and store in a dictionary with phonemes as keys
    """


    # Read Phone transcription
    with open(filename, "r") as phn_file:
        phn_content = phn_file.readlines()

    phn_content = [x.strip() for x in phn_content]

    source_dict = {}

    for i, data in enumerate(phn_content):
        li = list(data.split(" "))
        li[0] = int(li[0])
        li[1] = int(li[1])

        if(li[2] in source_dict.keys()): 
            source_dict[li[2]][i] = [li[0],li[1]]
        else:
            temp_dict = {i:[li[0], li[1]]}
            source_dict[li[2]] = temp_dict

    return source_dict




def dtw_list_store(source, target, dtw_source, dtw_target):

    """
    Store the Time aligned frames for the Source and Target in separate lists
    List is passed as reference, a common list for every user.

    Arguments:

    source - Source segment for the particular phoneme
    target - Target segment for the particular phoneme
    dtw_source - List containing source mfcc that we will append to
    dtw_target - Corresponding target mfcc list
    """



    source = psf.mfcc(source, 16000)
    target = psf.mfcc(target, 16000)

    source_list = []
    target_list = []

    distance, path = fastdtw(source, target, dist=euclidean)

    for vertex in path:
        dtw_source.append(source[vertex[0],:])
        dtw_target.append(target[vertex[1],:])

    return dtw_source, dtw_target


def feature_dtw(source_list, target_list, source_dict, target_dict, source_wav, target_wav):

    ## Just collect pairs of source and target MFCCs disregarding time time/phoneme information

    """Iterate over each phoneme present in the source dictionary.
    If the phoneme is also found in the target, do DTW between all pairs of frames
    and get the distance and path. 
    Store the corresponding pairs in a list

    Note: The list is common for the user, the dict is file/utterance specific

    Arguments:

    source_list - list contain the source mfcc vectors (common for the given user)
    target_list - correspnding target user list
    source_dict - The dictionary containing the PNH info for a particular file for a particular user
    target_dict - Corresponding dict for the target user
    source_wav - One Wav file of source user
    target_wav = corresponding wav file for target
    """

    for source_keys in source_dict.keys():

        if(source_keys in target_dict.keys()):

            source_phn_dict = source_dict[source_keys]
            target_phn_dict = target_dict[source_keys]


            for count1, i in enumerate(source_phn_dict.keys()):
                for count2, j in enumerate(target_phn_dict.keys()):
                    l1 = source_phn_dict[i]
                    l2 = target_phn_dict[j]
                    source_temp = source_wav[l1[0] : l1[1]]
                    target_temp = target_wav[l2[0] : l2[1]]

                    source_list, target_list = dtw_list_store(source_temp, target_temp, source_list, target_list)


    assert(len(source_list) == len(target_list))

    return source_list, target_list
