from p2 import *
from sys import exit
import os



def create_io_vectors(source_phn_file, source_wav_file, target_phn_file, target_wav_file, source_list, target_list):

    source_dict = create_dict(source_phn_file)
    target_dict = create_dict(target_phn_file)

    source = SPHFile(source_wav_file)
    target = SPHFile(target_wav_file)

    source.write_wav('source.wav')
    target.write_wav('target.wav')

    fs, source = scipy.io.wavfile.read('source.wav')
    fs, target = scipy.io.wavfile.read('target.wav')

    ## Lists to hold the time aligned mfcc coeffs

    #source_list, target_list = feature_dtw(source_list, target_list, source_dict, target_dict, source, target)
    feature_dtw(source_list, target_list, source_dict, target_dict, source, target)

    #return source_list, target_list


def user_io():

    dir_root = '/home/raghav/sem2/speech/proj/TIMIT/TIMIT/TRAIN/DR6/'
    dir1 = dir_root + 'MABC0/'
    dir2 = dir_root + 'MAJP0/'

    source_list = []
    target_list = []

    f1, p1 = 'SA1.WAV','SA1.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p1, dir2+f1
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SA2.WAV','SA2.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p1, dir2+f1
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SI781.WAV','SI781.PHN'
    f2, p2 = 'SI1074.WAV','SI1074.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SI1620.WAV','SI1620.PHN'
    f2, p2 = 'SI1704.WAV','SI1704.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SI2041.WAV','SI2041.PHN'
    f2, p2 = 'SI2334.WAV','SI2334.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)
    
    f1, p1 = 'SX151.WAV','SX151.PHN'
    f2, p2 = 'SX84.WAV','SX84.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SX241.WAV','SX241.PHN'
    f2, p2 = 'SX174.WAV','SX174.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SX331.WAV','SX331.PHN'
    f2, p2 = 'SX354.WAV','SX354.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    f1, p1 = 'SX421.WAV','SX421.PHN'
    f2, p2 = 'SX444.WAV','SX444.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p2, dir2+f2
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)

    source_list = np.array(source_list)
    target_list = np.array(target_list)

    print(source_list.shape, target_list.shape)

    np.save('source_input.npy', source_list)
    np.save('target_input.npy', target_list)

def user_io_2():

    """
    Given Two Speakers, iterate through all their utterances, and for
    every pair of utterances map phones and do DTW. So this we we exhaustively
    get all possible phone combinations
    """

    dir_root = '/home/raghav/sem2/speech/proj/TIMIT/TIMIT/TRAIN/DR6/'
    dir1 = dir_root + 'MABC0/'
    dir2 = dir_root + 'MAJP0/'

    source_list = []
    target_list = []

    for sfile in os.listdir(dir1):
        for tfile in os.listdir(dir2):
            if(sfile.endswith(".WAV") and tfile.endswith(".WAV")):
                    sphn = sfile[:-3]+"PHN"
                    tphn = tfile[:-3]+"PHN"
                    print(sfile, sphn, tfile, tphn)
                    s_phn, s_wav, t_phn, t_wav = dir1+sphn, dir1+sfile, dir2+tphn, dir2+tfile
                    create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)
                    
    source_list = np.array(source_list)
    target_list = np.array(target_list)

    print(source_list.shape, target_list.shape)

    np.save('source_input_logfbank.npy', source_list)
    np.save('target_input_logfbank.npy', target_list)


user_io_2()
