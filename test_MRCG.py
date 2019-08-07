import os
import unittest
import cProfile

import scipy.io.wavfile, scipy.io.matlab
import numpy as np
from matplotlib import transforms, pyplot as plt
import MRCG

TEST_FILE = 'test_data/SNR103F3MIC021002_ch01'


class Test_mrcg(object):
    ''' Base class for testing MRCG '''

    # Args to set tolerance for np.testing.assert_allclose
    tolerance_kwargs = {
        'rtol': 1e-5, 
        'atol': 0       # Don't check absolute tolerance, only relative
    }

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

        # Define some constants
        # Each cochleogram is 64 long, and we have 4 of them, so 4 * 64 = 256
        # Note they are still 393 wide, which we do not explicitly state
        self.all_coch_len = 256


class Test_gammatone(Test_mrcg, unittest.TestCase):
    def test_value(self):
        ''' Compare gammatone value against MATLAB implementation '''
        known_g = self.mat_dict['g']

        sig = self.audio

        # Scale using beta as recommended
        beta = MRCG.get_beta(sig)
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
        np.testing.assert_allclose(good_beta, our_beta, **tolerance_kwargs)


class Test_all_cochleagrams(Test_mrcg, unittest.TestCase):

    def setUp(self):
        super().setUp()
        sig = self.audio
        self.sampFreq = self.sr
        beta = MRCG.get_beta(sig)
        sig = sig*beta
        sig = sig.reshape(len(sig), 1)
        self.g = MRCG.gammatone(sig, 64, self.sampFreq)

    def test_value(self):
        ''' Test all cochleagrams match MATLAB implementation '''
        # Get all cochleagrams and flatten
        c1, c2, c3, c4 = MRCG.all_cochleagrams(self.g, self.sampFreq)
        all_cochleas = np.concatenate([c1, c2, c3, c4], 0)
        # Get what MATLAB generated
        good_all_cochleas = self.mrcg[0:self.all_coch_len]
        # Compare
        np.testing.assert_allclose(all_cochleas, good_all_cochleas)

    def test_concat(self):
        ''' Test all_cochs are correctly concatanated into MRCG '''
        # Could also have put this in Test_mrcg_extract instead
        c1, c2, c3, c4 = MRCG.all_cochleagrams(self.g, self.sampFreq)
        all_cochleas = np.concatenate([c1, c2, c3, c4], 0)
        # Get MRCG, should be [all_cochleas; delta; delta2]
        samp_mrcg = MRCG.mrcg_extract(self.audio, self.sr)
        # Check they are _exactly_ equal, as concatanation should not modify
        np.testing.assert_equal(all_cochleas, samp_mrcg[0:self.all_coch_len])


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


    def test_all_cochleas(self):
        ''' Test cochleagrams in output are correct '''
        samp_mrcg = MRCG.mrcg_extract(self.audio, self.sr)
        good_all_cochleas = self.mrcg[0:self.all_coch_len]
        our_all_cochleas = samp_mrcg[0:self.all_coch_len]

        # Compare
        np.testing.assert_allclose(our_all_cochleas, good_all_cochleas,
                                   **Test_mrcg.tolerance_kwargs)



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
    # If we call python -m cProfile test_MRCG.py, we get no tests!
    # See https://stackoverflow.com/q/11645285
    # So instead we include profiling in the script directly. Not ideal, but OK
    pr = cProfile.Profile()
    pr.enable()
    try: # Wrap in try so we still save stats on exception
        unittest.main()
    finally: # We don't want to _catch_ the exception as that would hide it
        pr.disable()
        pr.dump_stats(__file__ + ".prof")
