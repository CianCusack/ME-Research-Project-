import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 23.0       # sample rate, Hz
cutoff = 1  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
#data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
data = np.array([339, 317, 306, 292, 290, 279, 260, 266, 256, 244, 240, 234, 242, 238, 224, 228, 236, 232, 225, 238, 254, 267, 284, 296, 295, 304, 328, 344, 350, 351, 352, 356, 356, 364, 355, 346, 329, 306, 305, 310, 308, 339, 364, 370, 325, 312, 308, 298, 298, 285, 285, 236, 224, 188, 164, 156, 152, 148, 144, 150, 154, 149, 146, 140, 124, 114, 117, 110, 108, 102, 77, 46, 48, 36, 46, 31, 32, 31, 1280, 1280, 1280, 35, 35, 31, 31, 37, 32, 53, 48, 48, 40, 38, 37, 31, 37, 31, 32, 31, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 31, 33, 31, 33, 32, 33, 32, 36, 48, 58, 70, 72, 61, 69, 100, 138, 208, 295, 356, 406, 454, 498, 519, 502, 506, 506, 494, 506, 518, 504, 514, 508, 468, 423, 396, 396, 416, 426, 439, 453, 492, 504, 556, 610, 656, 718, 736, 780, 788, 764, 748, 724, 709, 690, 664, 638, 600, 535, 468, 408, 403, 346, 308, 326, 336, 364, 412, 396, 389, 404, 402, 424, 434, 430, 470, 475, 504, 522, 525, 556, 581, 592, 616, 627, 652, 670, 687, 702, 700, 695, 696, 698, 698, 680, 664, 654, 646, 642, 636, 608, 588, 559, 534, 536, 528, 531, 526, 497, 506, 511, 490, 490, 490, 484, 457, 448, 418, 412, 408, 422, 439, 456, 488, 529, 571, 594, 632, 646, 673, 694, 698, 706, 712, 700, 709, 704, 716, 716, 712, 720, 31, 31, 34, 35, 33, 38, 45, 48, 53, 44, 31, 32, 701, 688, 668, 649, 630, 614, 605, 584, 578, 571, 567, 570, 573, 592, 610, 636, 658, 673, 658, 654, 650, 653, 656, 660, 666])
# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data[:115], 'b-', label='data')
plt.plot(t, y[:115], 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()