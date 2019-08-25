import time
import os

import numpy as np
from numpy import matlib
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft
from scipy import signal, io as sio

epsc = 0.000001

# Load loundness scaling matrix from same folder as this file
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
FMAT_PATH = os.path.join(SCRIPT_FOLDER, 'f_af_bf_cf.mat')
FMAT = sio.loadmat(FMAT_PATH)


def get_beta(sig):
    beta = 1000.0 / np.sqrt(np.mean((np.square(sig))))
    return beta


def mrcg_extract_components(sig, sampFreq=16000, window_size=0.020):
    '''Extract individual components of the MRCG, not concatanated together'''
    beta = get_beta(sig)
    sig = sig*beta
    sig = sig.reshape(len(sig), 1)
    g = gammatone(sig, 64, sampFreq)

    cochlea1, cochlea2, cochlea3, cochlea4 = all_cochleagrams(g, sampFreq, window_size)
    all_cochleas = np.concatenate([cochlea1, cochlea2, cochlea3, cochlea4], 0)

    del0 = deltas(all_cochleas)
    ddel = deltas(deltas(all_cochleas, 5), 5)

    return ([cochlea1, cochlea2, cochlea3, cochlea4], del0, ddel)


def mrcg_extract(sig, sampFreq=16000, window_size=0.020):
    ''' Extract the MRCG, fully concatanated into one vector '''
    cochs, del0, ddel = mrcg_extract_components(sig, sampFreq, window_size)
    all_cochleas = np.concatenate(cochs, 0)
    output = np.concatenate((all_cochleas, del0, ddel), 0)
    return output


def all_cochleagrams(g, sampFreq, window_size=0.020):
    ''' Get all cochleagrams '''

    coch_shift = window_size / 2.0
    coch2_size = window_size * 10.0
    cochlea1 = np.log10(cochleagram(
        g, int(sampFreq * window_size), int(sampFreq * coch_shift)))
    cochlea2 = np.log10(cochleagram(
        g, int(sampFreq * coch2_size), int(sampFreq * coch_shift)))
    cochlea1 = cochlea1[:, :]
    cochlea2 = cochlea2[:, :]
    cochlea3 = get_avg(cochlea1, 5, 5)
    cochlea4 = get_avg(cochlea1, 11, 11)

    return cochlea1, cochlea2, cochlea3, cochlea4


def gammatone(insig, numChan=128, fs=16000):
    fRange = [50, 8000]
    filterOrder = 4
    gL = 2048
    sigLength = len(insig)
    phase = np.zeros([numChan, 1])
    erb_b = hz2erb(fRange)

    ###################
    erb_b_diff = (erb_b[1]-erb_b[0])/(numChan-1)
    erb = np.arange(erb_b[0], erb_b[1]+epsc, erb_b_diff)
    cf = erb2hz(erb)
    b = [1.019 * 24.7 * (4.37 * x / 1000 + 1) for x in cf]
    gt = np.zeros([numChan, gL])
    tmp_t = np.arange(1, gL+1)/fs
    for i in range(numChan):
        gain = 10**((loudness(cf[i])-60)/20)/3*(2 * np.pi * b[i] / fs)**4
        tmp_temp = gain*(fs**3)*tmp_t**(filterOrder - 1)*np.exp(-2 * np.pi * b[i] * tmp_t) * np.cos(2 * np.pi * cf[i] * tmp_t + phase[i])
        tmp_temp2 = np.reshape(tmp_temp, [1, gL])
        gt[i, :] = tmp_temp2

    sig = np.reshape(insig, [sigLength, 1])
    gt2 = np.transpose(gt)
    resig = np.matlib.repmat(sig, 1, numChan)
    r = np.transpose(fftfilt(gt2, resig, numChan))
    return r


def hz2erb(hz):
    erb1 = 0.00437
    # erb2 = [x * erb1 for x in hz]
    # erb3 = [x + 1 for x in erb2]
    erb2 = np.multiply(erb1, hz)
    erb3 = np.subtract(erb2, -1)
    erb4 = np.log10(erb3)
    erb = 21.4 * erb4
    return erb


def erb2hz(erb):
    hz = [(10**(x/21.4)-1)/(0.00437) for x in erb]
    return hz


def loudness(freq):
    dB = 60
    # af = [2.3470,2.1900,2.0500,1.8790,1.7240,1.5790,1.5120,1.4660,1.4260,1.3940,1.3720,1.3440,1.3040,1.2560,1.2030,1.1350,1.0620,1.0000,0.9670,0.9430,0.9320,0.9330,0.9370,0.9520,0.9740,1.0270,1.1350,1.2660,1.5010]
    # bf = [0.0056,0.0053,0.0048,0.0040,0.0038,0.0029,0.0026,0.0026,0.0026,0.0026,0.0025,0.0025,0.0023,0.0020,0.0016,0.0011,0.0005,0,-0.0004,-0.0007,-0.0009,-0.0010,-0.0010,-0.0009,-0.0006,0,0.0009,0.0021,0.0049]
    # cf = [74.3000,65.0000,56.3000,48.4000,41.7000,35.5000,29.8000,25.1000,20.7000,16.8000,13.8000,11.2000,8.9000,7.2000,6.0000,5.0000,4.4000,4.2000,3.7000, 2.6000, 1.0000,-1.2000,-3.6000,-3.9000,-1.1000,6.6000,15.3000,16.4000,11.6000]
    # ff = np.multiply([0.0020,0.0025,0.0032,0.0040,0.0050,0.0063,0.0080,0.0100,0.0125,0.0160,0.0200,0.0250,0.0315,0.0400,0.0500,0.0630,0.0800,0.1000,0.1250,0.1600,0.2000,0.2500,0.3150,0.4000,0.5000,0.6300,0.8000,1.0000,1.2500],10000)
    af = FMAT['af'][0]
    bf = FMAT['bf'][0]
    cf = FMAT['cf'][0]
    ff = FMAT['ff'][0]
    i = 0
    while ff[i] < freq:
        i = i + 1

    afy = af[i - 1] + (freq - ff[i - 1]) * \
        (af[i] - af[i - 1]) / (ff[i] - ff[i - 1])
    bfy = bf[i - 1] + (freq - ff[i - 1]) * \
        (bf[i] - bf[i - 1]) / (ff[i] - ff[i - 1])
    cfy = cf[i - 1] + (freq - ff[i - 1]) * \
        (cf[i] - cf[i - 1]) / (ff[i] - ff[i - 1])
    loud = 4.2 + afy * (dB - cfy) / (1 + bfy * (dB - cfy))
    return loud


def fftfilt(b, x, nfft):
    fftflops = [18, 59, 138, 303, 660, 1441, 3150, 6875, 14952, 32373, 69762,
                149647, 319644, 680105, 1441974, 3047619, 6422736, 13500637, 28311786,
                59244791, 59244791*2.09]
    nb, _ = np.shape(b)
    nx, mx = np.shape(x)
    n_min = 0
    while 2**n_min < nb-1:
        n_min = n_min+1
    n_temp = np.arange(n_min, 21 + epsc, 1)
    # n = [2 ** x for x in n_temp]
    n = np.power(2, n_temp)
    fftflops = fftflops[n_min-1:21]
    # L = [x -(nb-1) for x in n]
    L = np.subtract(n, nb-1)
    lenL = np.size(L)
    # temp_ind = [np.ceil(nx/ L[x])*fftflops[x] for x in range(lenL)]
    # ind = temp_ind.index(int(np.min(temp_ind)))
    temp_ind0 = np.ceil(np.divide(nx, L))
    temp_ind = np.multiply(temp_ind0, fftflops)
    temp_ind = np.array(temp_ind)
    # ind = temp_ind.index(int(np.min(temp_ind)))
    ind = np.argmin(temp_ind)
    nfft = int(n[ind])
    L = int(L[ind])
    b_tr = np.transpose(b)
    B_tr = fft(b_tr, nfft)
    B = np.transpose(B_tr)
    y = np.zeros([nx, mx])
    istart = 0
    while istart < nx:
        iend = min(istart+L, nx)
        if (iend - istart) == 1:
            X = x[0][0]*np.ones([nx, mx])
        else:
            xtr = np.transpose(x[istart:iend][:])
            Xtr = fft(xtr, nfft)
            X = np.transpose(Xtr)
        # temp_Y =np.transpose([a * b for a, b in zip(B, X)])
        temp_Y = np.transpose(np.multiply(B, X))
        Ytr = ifft(temp_Y, nfft)
        Y = np.transpose(Ytr)
        yend = np.min([nx, istart + nfft])
        y[istart:yend][:] = y[istart:yend][:] + np.real(Y[0:yend-istart][:])

        istart = istart + L
    # y = np.real(y)
    return y


def cochleagram(r, winLength=320, winShift=160):
    numChan, sigLength = np.shape(r)
    increment = winLength / winShift
    M = np.floor(sigLength / winShift)
    a = np.zeros([numChan, int(M)])
    rs = np.square(r)
    rsl = np.concatenate((np.zeros([numChan, winLength-winShift]), rs), 1)
    for m in range(int(M)):
        temp = rsl[:, m*winShift: m*winShift+winLength]
        a[:, m] = np.sum(temp, 1)

    return a


def cochleagram_keep(r, winLength=320, winShift=160):
    numChan, sigLength = np.shape(r)
    increment = winLength / winShift
    M = np.floor(sigLength / winShift)
    a = np.zeros([numChan, int(M)])
    for m in range(int(M)):
        for i in range(numChan):
            if m < increment:
                a[i, m] = (sum(map(lambda x: x*x, r[i, 0:(m+1)*winShift])))
            else:
                startpoint = (m - increment) * winShift
                a[i, m] = (
                    sum(map(lambda x: x*x, r[i, int(startpoint):int(startpoint) + winLength])))
    return a


def get_avg(m, v_span, h_span):
    nr, nc = np.shape(m)
    # out = np.zeros([nr+2*h_span,nc+2*h_span])
    fil_size = (2 * v_span + 1) * (2 * h_span + 1)
    meanfil = np.ones([1+2*h_span, 1+2*h_span])
    meanfil = np.divide(meanfil, fil_size)

    out = signal.convolve2d(m, meanfil, boundary='fill',
                            fillvalue=0, mode='same')
    return out


def deltas(x, w=9):
    nr, nc = np.shape(x)
    if nc == 0:
        d = x
    else:
        hlen = int(np.floor(w / 2))
        w = 2 * hlen + 1
        win = np.arange(hlen, int(-(hlen+1)), -1)
        temp = x[:, 0]
        fx = np.matlib.repmat(temp.reshape([-1, 1]), 1, int(hlen))
        temp = x[:, nc-1]
        ex = np.matlib.repmat(temp.reshape([-1, 1]), 1, int(hlen))
        xx = np.concatenate((fx, x, ex), 1)
        d = signal.lfilter(win, 1, xx, 1)
        d = d[:, 2*hlen:nc+2*hlen]

    return d
