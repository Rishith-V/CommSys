import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import os

Fs = 20000
T  = 0.2
t  = np.arange(-T/2, T/2, 1/Fs)

def sinc_lab(x):
    return np.sinc(x/np.pi)

def mag_spectrum(x, Fs):
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1/Fs))
    return f, np.abs(X)/len(x)

def lpf_impulse(t, B):
    return 2*B * sinc_lab(2*np.pi*B*t)

B1 = 500
B2 = 400
f0 = 110

os.makedirs("plots", exist_ok=True)
plot_num = 1

def save_plot(x, y, title):
    global plot_num
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(f"plots/plot_{plot_num:02d}.png", dpi=300)
    plt.close()
    plot_num += 1

m = lpf_impulse(t, B1)
g = lpf_impulse(t, B2)

save_plot(t, m, f"m(t) for B1={B1} Hz")
save_plot(t, g, f"g(t) for B2={B2} Hz")

f, Mmag = mag_spectrum(m, Fs)
save_plot(f, Mmag, f"|M(f)| for B1={B1} Hz")

f, Gmag = mag_spectrum(g, Fs)
save_plot(f, Gmag, f"|G(f)| for B2={B2} Hz")

y_full = fftconvolve(m, g, mode="full")
start = (len(y_full) - len(m)) // 2
y = y_full[start:start+len(m)]

save_plot(t, y, f"y(t)=m*g (B1={B1} Hz, B2={B2} Hz)")

f, Ymag = mag_spectrum(y, Fs)
save_plot(f, Ymag, f"|Y(f)| (B1={B1} Hz, B2={B2} Hz)")

m_tones = sum((1.0/k) * np.cos(2*np.pi*k*f0*t) for k in range(1, 6))
save_plot(t, m_tones, "m(t) five-tone")

f, Mtones = mag_spectrum(m_tones, Fs)
save_plot(f, Mtones, "|M(f)| five-tone")

y_full = fftconvolve(m_tones, g, mode="full")
start = (len(y_full) - len(m_tones)) // 2
y_tones = y_full[start:start+len(m_tones)]

save_plot(t, y_tones, f"y(t) five-tone via B2={B2} Hz")

f, Ytones = mag_spectrum(y_tones, Fs)
save_plot(f, Ytones, f"|Y(f)| five-tone via B2={B2} Hz")
