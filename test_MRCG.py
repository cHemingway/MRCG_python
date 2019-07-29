import MRCG
import os
import librosa
import time

script_path = os.path.dirname(os.path.abspath( __file__ ))
wav_dir = os.path.join(script_path,'example/SNR103F3MIC021002_ch01.wav')
audio, sr = librosa.load(wav_dir, sr=16000)
# sr,audio = scipy.io.wavfile.read(wav_dir)
print('success to load sample wav-file')
s = time.clock()
samp_mrcg = MRCG.mrcg_extract(audio,sr)
e = time.clock()
print('success to extract features')
print(e-s)
