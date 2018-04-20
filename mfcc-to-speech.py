import numpy as np
import librosa,librosa.display
import matplotlib.pyplot as plt

#mfccs = np.load('converted-speech.npy')

#mfccs = mfccs.T

def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)



# load
filename = 'source.wav'
y, sr = librosa.load(filename)

# calculate mfcc
sr = 16000
Y = librosa.stft(y)
mfccs = librosa.feature.mfcc(y,sr=sr, n_mfcc=13, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
mfccs = mfccs[:,:-2]
print(mfccs.shape)



n_mfcc = mfccs.shape[0]
n_mel = 26
dctm = librosa.filters.dct(n_mfcc, n_mel)
n_fft = 512
mel_basis = librosa.filters.mel(sr, n_fft)


# Empirical scaling of channels to get ~flat amplitude mapping.
bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))


# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, mfccs)))

#recon_stft[recon_stft==0] = 1e-10

# Impose reconstructed magnitude on white noise STFT.

excitation = np.random.randn(y.shape[0])
E = librosa.stft(excitation)

print("E :", E.shape)
print("recon_stft :", recon_stft.shape)

recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

# Output
librosa.output.write_wav('output.wav', recon, sr)


plt.style.use('seaborn-darkgrid')
plt.figure(1)
plt.subplot(211)
librosa.display.waveplot(y, sr)
plt.subplot(212)
librosa.display.waveplot(recon,sr)
plt.show()

