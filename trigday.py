"""
Playing around with ObsPy
20140411: Trying to load data and trigger it, then cut out the triggers
20140415: Working on adding PCA of triggers
20140417: Moved PCA and clustering tests to different file, save triggers
"""

from obspy.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.trigger import classicSTALTA, triggerOnset
import numpy as np

# Grab a day from IRIS (HSR is at Mount St. Helens)
client = Client("IRIS")
# This is a time period where I know lots of repeaters are happening
t = UTCDateTime("2004-11-24T00:00:00.000")
savename = 'trigdata.npy'

print('Grabbing waveforms...')
st = client.get_waveforms("UW", "HSR", "--", "EHZ", t - 10, t + 86420)
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

# Grab triggers [on off], in samples
on_off = triggerOnset(cft, 3, 2)

# Go through the trigger list, don't allow triggers within 10 seconds of each other
# Slice out the traces, save them to an array we can manipulate later

# Find first trigger that's after the start of the day
# Last trigger has to be before end of day
trigtime = 0
print('Cutting out triggers to save: time consuming!!')
for n in range(0,len(on_off)):
    if on_off[n,0] > trigtime + 1000:
        if trigtime is 0:
            trigtime = on_off[n,0]
            trigs = st.slice(t - 15 + 0.01*trigtime, t + 5.47 + 0.01*trigtime)
            trigdata = trigs[-1].data
        else:
            trigtime = on_off[n,0]
            if trigtime < 8641000:
                trigs = trigs.append(st[0].slice(t - 15 + 0.01*trigtime, t + 5.47 + 0.01*trigtime))
                trigdata = np.vstack((trigdata, trigs[-1].data))

print('Number of triggers: ' + repr(len(trigs)))
print('Shape of trigdata: ' + repr(trigdata.shape))

# Save for future playing with
np.save(savename, trigdata)
