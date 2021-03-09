import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = './pre_processing_music2.wav'

#waveform
signal , sr = librosa.load(file, sr=22050) # sr * T ->22050 * 180
librosa.display.waveplot(signal, sr=sr)

plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

#-

# fft -> spectrum
fft = np.fft.fft(signal)

magnitute = np.abs(fft)
frequency = np.linspace(0,sr, len(magnitute))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitute = magnitute[:int(len(magnitute)/2)]

# plt.plot(frequency, magnitute)
# plt.xlabel('frequancy')
# plt.ylabel('magnitute')
# plt.show()

plt.plot(left_frequency, left_magnitute)
plt.xlabel('Left Frequancy')
plt.ylabel('Left Magnitute')
plt.show()



# stft -> spectrogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()

# logged ieasy to display
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()



# MFCCs
MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length = hop_length, n_mfcc=13)

librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('MFCC')
plt.colorbar()
plt.show()
