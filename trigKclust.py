'''
Trying out KMeans on the unaligned waveform data itself
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load up the waveform data as a matrix of (numtrigs x numsamples) in size
trigdata = np.load('trigdata.npy')

# Normalize to maximum amplitude
amps = np.amax(trigdata, axis=1)
trigdata = (trigdata.T * 1/amps).T

# Plot the data!
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1, xlabel='Sample', ylabel='Trigger')
ax.imshow(trigdata, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

# Do a quick PCA
pca = PCA().fit(trigdata)
fig = plt.figure() 
ax = plt.semilogx(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# Find some cluster centers using KMeans
nclust = 24 # Change this as you please
nrows = 4 # for plotting
est = KMeans(n_clusters=nclust)
clusters = est.fit_predict(trigdata)

fig = plt.figure(figsize=(15,9))
for i in range(nclust):
     ax = fig.add_subplot(nrows, np.ceil(nclust/nrows), 1 + i, xticks=[], yticks=[])
     ax.plot(est.cluster_centers_[i])

fig = plt.figure(figsize=(15,9))
for i in range(nclust):
    ax = fig.add_subplot(nrows, np.ceil(nclust/nrows), 1 + i, xticks=[])
    ax.imshow(trigdata[clusters==i,:], aspect='auto', cmap='RdBu', vmin=-1, vmax=1,
              interpolation='nearest')

plt.show()

# Shiny.
