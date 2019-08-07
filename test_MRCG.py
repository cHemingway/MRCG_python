import os
import unittest

import scipy.io.wavfile, scipy.io.matlab
import numpy as np
from matplotlib import transforms, pyplot as plt
import MRCG

TEST_FILE = 'test_data/SNR103F3MIC021002_ch01'


class Test_mrcg(object):
    ''' Base class for testing MRCG '''

    # Args to set tolerance for np.testing.assert_allclose
    tolerance_kwargs = {'rtol': 1e-5, 'atol': 1e-6}

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
        self.mat_dict = mat_dict
        self.mrcg = self.mat_dict['mrcg']


class Test_gammatone(Test_mrcg, unittest.TestCase):
    def test_value(self):
        ''' Compare gammatone value against MATLAB implementation '''
        known_g = self.mat_dict['g']

        sig = self.audio

        # TODO, don't recalculate beta, but use code
        beta = 1000 / np.sqrt(sum(map(lambda x: x*x, sig)) / len(sig))
        sig = sig*beta
        sig = sig.reshape(len(sig), 1)

        our_g = MRCG.gammatone(sig, 64, self.sr)

        # Check shape
        self.assertEqual(our_g.shape, known_g.shape)

        # Check values are close
        np.testing.assert_allclose(
            our_g, known_g, **Test_mrcg.tolerance_kwargs)

    def test_numChan(self):
        ''' Check channel count is correct '''
        sig = np.random.randn(10000)
        for num_chan in (32, 64, 128, 256, 255):
            g = MRCG.gammatone(sig, num_chan)
            self.assertEqual(num_chan, g.shape[0])


class Test_beta(Test_mrcg, unittest.TestCase):
    def test_value(self):
        ''' Compare beta value against MATLAB implementation '''
        good_beta = self.mat_dict['beta']
        our_beta = MRCG.get_beta(self.audio)
        # FIXME high tolerance of 0.1%, why?
        tolerance_kwargs = Test_mrcg.tolerance_kwargs
        tolerance_kwargs['rtol'] = 1e-04
        tolerance_kwargs['atol'] = 0 # Check only relative tolerance, not abs
        np.testing.assert_allclose(good_beta, our_beta, **tolerance_kwargs)



class Test_mrcg_extract(Test_mrcg, unittest.TestCase):

    def test_extract(self):
        ''' Test final MRCG matches MATLAB implementation '''
        samp_mrcg = MRCG.mrcg_extract(self.audio, self.sr)
        # Plot for reference
        self.plot_mrcg(samp_mrcg)
        # Check the type
        self.assertIsNotNone(samp_mrcg)
        self.assertIsInstance(samp_mrcg, np.ndarray)
        # Check size and values against original MATLAB code result
        self.assertEqual(self.mrcg.shape, samp_mrcg.shape)
        np.testing.assert_allclose(samp_mrcg, self.mrcg, **Test_mrcg.tolerance_kwargs)


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
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    unittest.main()