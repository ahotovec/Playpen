# Trying out some functions
# Currently just cross-correlation, but it's a start...

import numpy as np
from scipy.fftpack import fft, ifft

def xcorr(data):

    # Find amplitudes, rotate so events are in columns
    amps = np.amax(data, axis=1)
    data = (data.T * 1/amps)
    
    # Prep some other necessary terms
    M, N = np.shape(data)
    l = np.roll(np.linspace((-M/2+1), (M/2), M, endpoint=True), M/2+1)*0.01
    wcoeff = 1/np.sqrt(sum(data*data));
    C = np.eye(N)
    L = np.zeros((N, N))
    
    # Get FFT of traces
    X = fft(data, axis=0)
    Xc = np.conjugate(X)
    
    # Loop through rows of similarity matrix
    for n in range(N):
        cols = range(n, N)
        CC = np.multiply(np.tile(X[:, n], (len(cols),1)).T, Xc[:, cols])
        corr = np.real(ifft(CC, axis=0))
        maxval = np.amax(corr, axis=0)
        indx = np.argmax(corr, axis=0)
        C[n, cols] = np.multiply(maxval, wcoeff[cols])*wcoeff[n]
        L[n, cols] = l[indx]
    
    # Fill in the lower triangular part of matrix
    C = C + C.T - np.eye(N)
    L = L - L.T

    return C, L

