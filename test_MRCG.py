import MRCG
import os
import scipy.io.wavfile
import time

script_path = os.path.dirname(os.path.abspath( __file__ ))
wav_dir = os.path.join(script_path,'example/SNR103F3MIC021002_ch01.wav')
sr,audio = scipy.io.wavfile.read(wav_dir)
audio = audio.astype(float) / 32767 # Convert to range -1 to 1
print('success to load sample wav-file')
s = time.clock()
samp_mrcg = MRCG.mrcg_extract(audio,sr)
e = time.clock()
print('success to extract features')
print(e-s)
