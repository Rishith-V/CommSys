import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.fft import fft

# Parameters
start_time = 0
stop_time = 1
fm = 10  # Maximum frequency component in Hertz for the given spectrum
fs = 10 * fm
ts = 1 / fs

mx = np.array([])
tvect = np.array([])

for T in range(30):
    time = np.arange(start_time + T, stop_time + T, ts)
    U = random.randint(1, 5)
    m_t = U * np.cos(2 * fm * time)
    
    mx = np.append(mx, m_t)
    tvect = np.append(tvect, time)

    print(f"Loop {T+1}/30: Now plotting cumulative signal up to {stop_time + T} seconds.")  #for debugging
    
    mf = fft(mx) / fs
    N = len(mf)
    mf_abs_sorted = np.fft.fftshift(abs(mf))
    freq_axis = np.linspace(-fs / 2, fs / 2, N)
    
    # Time Domain
    plt.figure(1)
    plt.clf()
    plt.plot(tvect, mx)
    plt.title(f'Time Domain, Time=({start_time + T}, {stop_time + T})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Save the final figure after the last iteration
    if T == 29:
        plt.savefig('sample_final_time.svg')
    
    # Frequency Domain
    plt.figure(2)
    plt.clf()
    plt.plot(freq_axis, mf_abs_sorted)
    plt.title(f'Frequency Domain, Time=({start_time + T}, {stop_time + T})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.ylim(0, 0.5)
    plt.grid(True)

    plt.pause(1.2)  #pause to view the plots

# Save frequency-domain figure
plt.savefig('sample_final_freq.svg')
plt.show()
