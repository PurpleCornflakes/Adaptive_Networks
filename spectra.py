import numpy as np
import matplotlib.pyplot as plt

r = np.load("r_seq.npy")

def truncate(r):
    N = len(r)
    k = N-1
    while r[k] == 1:
        k -= 1
    return k

t_r = r[:truncate(r)+1]
N = len(t_r)

ft_r = np.fft.fft(t_r)
ft_r = ft_r[1:N//2+1]
f = np.fft.fftfreq(len(t_r))[1:N//2+1]

plt.plot(np.log(f), np.log(np.abs(ft_r)**2))
plt.show()
