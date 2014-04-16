"""
Playing around with ObsPy
20140411: Trying to load data and trigger it, then cut out the triggers
20140415: Working on adding PCA of triggers
"""

from obspy.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.trigger import classicSTALTA, triggerOnset

# Grab a day from IRIS (HSR is at Mount St. Helens)
client = Client("IRIS")
# This is a time period where I know lots of repeaters are happening
t = UTCDateTime("2004-11-24T00:00:00.000")

print('Grabbing waveforms...')
st = client.get_waveforms("UW", "HSR", "--", "EHZ", t, t + 86400)
print('Done!')

# Detrend and merge, fill gaps with zeros, bandpass
st = st.detrend()
st = st.merge(method=1, fill_value=0)
st = st.filter('bandpass', freqmin=1.0, freqmax=10.0, corners=2, zerophase=True)

# print("Close the window to continue...")
# Helicorder plot
# st.plot(type='dayplot')

# STA/LTA ttrigger
print('Triggering')
tr = st[0]
cft = classicSTALTA(tr.data, 80, 700) # 0.8 s short, 7 s long
# NOTE HERE: first trigger will be no earlier than 699 samples in!
# Be sure to add some padding when looking at consecutive days

# Plot triggers
# plotTrigger(tr, cft, 3, 2)

# Grab triggers [on off], in samples
on_off = triggerOnset(cft, 3, 2)

# Go through the trigger list, don't allow triggers within 10 seconds of each other
# Slice out the traces, we want to correlate them next!

# First trigger
print('Cutting out triggers')
trigtimes = [on_off[0,0]]
trigs = st.slice(t - 10 + 0.01*trigtimes[0], t + 20 + 0.01*trigtimes[0])

for n in range(1,len(on_off)):
    if on_off[n,0] > trigtimes[-1] + 1000:
        trigtimes.append(on_off[n,0])
        trigs = trigs.append(st[0].slice(t - 10 + 0.01*trigtimes[-1], t + 20 + 0.01*trigtimes[-1]))

print('Number of triggers: ' + repr(len(trigs)))

# Check on some of the triggers to see if they're okay
# trigs.plot(type='dayplot', vertical_scaling_range=500)
# print("Check to see if the triggers look okay. Close each window to continue...")
# for n in range(0,25):
#     trigs[n].plot()




# PCA of triggers
#
# fft of each trace
# create matrix n(events)xm(samples)
# from sklearn.decomposition import PCA
# import numpy as np
# X - np.random.random((270,250))
# pca = PCA()
# pca.fit(X)
#

