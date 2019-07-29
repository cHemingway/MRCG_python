import os
import unittest

import scipy.io.wavfile
import numpy as np
import MRCG

class Test_mrcg_extract(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        wav = os.path.join(script_path, 'test_data/SNR103F3MIC021002_ch01.wav')
        sr, audio = scipy.io.wavfile.read(wav)
        self.sr = sr
        self.audio = audio.astype(float) / 32767  # Convert to range -1 to 1

    def test_extract(self):
        samp_mrcg = MRCG.mrcg_extract(self.audio, self.sr)
        # Check the type
        self.assertIsNotNone(samp_mrcg)
        self.assertIsInstance(samp_mrcg, np.ndarray)
        # TODO Check values against original MATLAB code
