'''
Trying out PCA on the FFT of my unaligned triggers
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


ffttrigs = np.load('ffttrigs.npy')
pca = PCA()
pca.fit(ffttrigs)

# Throws error for complex numbers:
# /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sklearn/utils/validation.py:83: ComplexWarning: Casting complex values to real discards the imaginary part
#  return X.astype(np.float32 if X.dtype == np.int32 else np.float64)

plt.semilogx(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Still, ~80 dimensions explains 95%...
