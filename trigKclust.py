'''
Trying out KMeans on the unaligned waveform data itself
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load up the waveform data as a matrix of (numtrigs x numsamples) in size
trigdata = np.load('trigdata.npy')

# Do a quick PCA
pca = PCA().fit(trigdata)
plt.semilogx(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Find some cluster centers using KMeans
est = KMeans(n_clusters=18)
clusters = est.fit_predict(trigdata)

fig = plt.figure(figsize=(8, 3))
for i in range(18):
     ax = fig.add_subplot(3, 6, 1 + i, xticks=[], yticks=[])
     ax.plot(est.cluster_centers_[i])

plt.show()

# Shiny.
