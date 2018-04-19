from p2 import *



def create_io_vectors(source_phn_file, source_wav_file, target_phn_file, target_wav_file, source_list, target_list):

    source_dict = create_dict(phn_file)
    target_dict = create_dict(dir2+'SA1.PHN')

    source = SPHFile(dir1+'SA1.WAV')
    target = SPHFile(dir1+'SA1.WAV')

    source.write_wav('source.wav')
    target.write_wav('target.wav')

    fs, source = scipy.io.wavfile.read('source.wav')
    fs, target = scipy.io.wavfile.read('target.wav')

    ## Lists to hold the time aligned mfcc coeffs

    source_dtw, target_dtw = feature_dtw(source_list, target_list, source_dict, target_dict, source, target)


def user_io():

    dir_root = '/home/raghav/sem2/speech/proj/TIMIT/TIMIT/TRAIN/DR6/'
    dir1 = dir_root + 'MABC0/'
    dir2 = dir_root + 'MAJP0/'

    source_list = []
    target_list = []
    f1, p1 = 'SA1.WAV','SA1.PHN'
    s_phn, s_wav, t_phn, t_wav = dir1+p1, dir1+f1, dir2+p1, dir2+f1
    source_list, target_list = create_io_vectors(s_phn, s_wav, t_phn, t_wav, source_list, target_list)
