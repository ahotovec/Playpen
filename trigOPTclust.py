'''
Trying out OPTICS to cluster the correlated waveforms
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from redpy import xcorr
import redoptics as redop

# Load waveform data as a matrix of (n_trigs, n_samples) in size
# Should be in same directory, made by 'trigday.py'
trigdata = np.load('trigdata.npy')
trigdata = trigdata[0:500,:] # Only take first part, otherwise very time consuming!
N = np.shape(trigdata)[0]

# Normalize to maximum amplitude
amps = np.amax(trigdata, axis=1)
trigdata = (trigdata.T * 1/amps).T

# First pass cross-correlate
t = time.time()
C, L = xcorr(trigdata)
print('Time to correlate first run: ' + repr(time.time()-t) + ' seconds')

# Realign
LMax = np.int(np.max(np.abs(L))*100)
W_adj = np.hstack((np.zeros((N,LMax)),trigdata,np.zeros((N,LMax))))
# Find the index of the event with the smallest mean abs(lag)
centerevent = np.argmin(np.median(np.abs(L),axis=1))
tshift = L[centerevent,:]*100
for i in range(N):
    W_adj[i,:] = np.roll(W_adj[i,:], np.int(tshift[i]))
# Cut out the ends
W_adj = W_adj[:,LMax+250:LMax+1024+250]

# Correlate again on (hopefully) better aligned waveforms
t = time.time()
C, L = xcorr(W_adj)
print('Time to correlate second run: ' + repr(time.time()-t) + ' seconds')

# Show the unordered C and L matrices
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(C, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(L, aspect='auto', vmin=-5, vmax=5, interpolation='nearest', cmap='RdBu')

# Cluster with OPTICS
t = time.time()
ttree = redop.setOfObjects(1-C)
redop.prep_optics(ttree,1)
redop.build_optics(ttree,1)
print('Time to cluster: ' + repr(time.time()-t) + ' seconds')

# Plot the reachability diagram
fig = plt.figure(figsize=(12,9))
plt.plot(ttree._reachability[ttree._ordered_list])

# Put the C, L, and waveform matrices in same order as ordered list
order = ttree._ordered_list
Co = C[order,:]
Co = Co[:,order]
Lo = L[order,:]
Lo = Lo[:,order]
Wo = W_adj[order,:]

# Plot the ordered C and L matrices (shiny!)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(Co, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(Lo, aspect='auto', vmin=-5, vmax=5, interpolation='nearest', cmap='RdBu')

# Plot the ordered waveforms
fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(Wo, aspect='auto', vmin=-1, vmax=1, interpolation='nearest', cmap='RdBu')

#Estimate the number of clusters with CC>0.7
redop.ExtractDBSCAN(ttree,0.3)
core_samples = ttree._index[ttree._is_core[:] > 0]
labels = ttree._cluster_id[:]
n_clusters_ = max(ttree._cluster_id)
print('Estimated number of clusters ' + repr(n_clusters_))

plt.show()
