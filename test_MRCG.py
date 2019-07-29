import os
import unittest

import scipy.io.wavfile, scipy.io.matlab
import numpy as np
from matplotlib import transforms, pyplot as plt
import MRCG

TEST_FILE = 'test_data/SNR103F3MIC021002_ch01'


class Test_mrcg_extract(unittest.TestCase):
    def setUp(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        # Load audio
        wav = os.path.join(script_path, TEST_FILE + '.wav')
        sr, audio = scipy.io.wavfile.read(wav)
        self.sr = sr
        self.audio = audio.astype(float) / 32767  # Convert to range -1 to 1
        # Load matlab .mat file
        mat = os.path.join(script_path, TEST_FILE + '.mat')
        mat_dict = scipy.io.matlab.loadmat(mat)
        self.mrcg = mat_dict['features']

    def test_extract(self):
        samp_mrcg = MRCG.mrcg_extract(self.audio, self.sr)
        # Plot for reference
        self.plot_mrcg(samp_mrcg)
        # Check the type
        self.assertIsNotNone(samp_mrcg)
        self.assertIsInstance(samp_mrcg, np.ndarray)
        # Check size and values against original MATLAB code result
        self.assertEquals(self.mrcg.shape, samp_mrcg.shape)
        np.testing.assert_almost_equal(samp_mrcg, self.mrcg, decimal=4)


    def plot_mrcg(self, mrcg, filename='mrcg_comparison.png'):
        ''' Utility function to save plot of our MRCG to a file '''
        fig, (ref_ax, our_ax, diff_ax) = plt.subplots(1, 3, 
                                                     sharex=True,
                                                     sharey=True)
        fig.set_size_inches(10, 7)
        format_kwargs = {
            'cmap':'jet', # Use full range color map for clarity    
        }
        
        ref_ax.imshow(self.mrcg, **format_kwargs)
        ref_ax.set_title("MATLAB")
        our_ax.imshow(mrcg, **format_kwargs)
        our_ax.set_title("Python")
        
        # Plot difference
        diff = np.abs(self.mrcg - mrcg)
        diff_ax.imshow(diff, **format_kwargs)
        diff_ax.set_title("Differences")

        # Save figure, minimal padding/border
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
