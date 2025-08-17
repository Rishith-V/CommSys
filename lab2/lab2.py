import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.io.wavfile import write
from playsound import playsound

fs = 48000
T = 30
t = np.arange(0, T, 1/fs)
A = 0.1

choice = input("Enter 'busy' or 'dial' or 'ringing' to generate the desired tone: ").lower()

if choice == 'busy':
    print("Generating busy tone...")
    f1 = 480
    f2 = 620
    on_time = 0.5
    off_time = 0.5
    filename = 'busy_tone.wav'

elif choice == 'dial':
    print("Generating dial tone...")
    f1 = 350
    f2 = 440
    on_time = 1
    off_time = 0
    filename = 'dial_tone.wav'

elif choice == 'ringing':
    print("Generating dial tone...")
    f1 = 440
    f2 = 480
    on_time = 2
    off_time = 4
    filename = 'dial_tone.wav'

else:
    print("Invalid choice. Please run the script again and enter 'busy' or 'dial'.")
    exit()
    
cadence_period = on_time + off_time

continue_tone = A * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))

time_in_cadence = t % cadence_period
envelope = time_in_cadence < on_time

generated_signal = continue_tone * envelope

print("plotting signals...")

plt.figure(figsize=(12, 5))
plt.plot(t, generated_signal)
plt.title('Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

plt.xlim(0, 6)
plt.ylim(-0.25, 0.25)

plt.savefig('time.svg')

mf = fft(generated_signal) / fs
N = len(mf)
mf_abs_sorted = np.fft.fftshift(abs(mf))
freq_axis = np.linspace(-fs/2, fs/2, N)

plt.figure(figsize=(12, 5))
plt.plot(freq_axis, mf_abs_sorted)

plt.title('Frequency-Domain Signal (Spectrum)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.xlim(0, 1000)

plt.savefig('freq.svg')
plt.show

print("Plotting Complete")

# Saving file using scipy
print(f"Saving audio to {filename}...")
scaled_signal = np.int16(generated_signal / np.max(np.abs(generated_signal)) * 32767)
write(filename, int(fs), scaled_signal)

# Playing the sound using playsound 
try:
    print(f"Playing '{filename}'...")
    playsound(filename)
    print("Playback finished.")
except Exception as e:
    print(f"Error playing sound: {e}")