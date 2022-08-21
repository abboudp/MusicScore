import librosa
import matplotlib.pyplot as plt
import librosa.display

# 1. Load an audio file
audio_path = 'tokyo.m4a'
x, sr = librosa.load(audio_path)
print('Data X Type and Sample Rate SR Type', type(x), type(sr))
print('Data X Size and Sample Rate', x.shape, sr)

# 2. Visual audio
plt.figure(figsize=(14,5))
librosa.display.waveplot(x, sr=sr)
plt.savefig('Waveform .jpg')

# 3. Sound spectrum
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')
plt.colorbar()
plt.savefig('Sound spectrum .jpg')

# 4. Frequency axis converted to a diameter axis
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.savefig('Sound Spectrum - Tricious axis .jpg')

# 5. Audio Save
# librosa.output.write_wav('/source/d3/ContentGeneration/result/example.wav', x, sr)
