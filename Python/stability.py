import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

plt.rcParams['figure.figsize'] = [8, 8]

# Load the S-parameter file
bjt = rf.Network('1to10G.s2p')  # Ensure filename matches

# Stability calculations (precompute for all frequencies)


def sqabs(x): return np.square(np.absolute(x))


delta = bjt.s11.s * bjt.s22.s - bjt.s12.s * bjt.s21.s
rs = np.abs((bjt.s12.s * bjt.s21.s) / (sqabs(bjt.s11.s) - sqabs(delta)))
cs = np.conj(bjt.s11.s - delta * np.conj(bjt.s22.s)) / \
    (sqabs(bjt.s11.s) - sqabs(delta))
rl = np.abs((bjt.s12.s * bjt.s21.s) / (sqabs(bjt.s22.s) - sqabs(delta)))
cl = np.conj(bjt.s22.s - delta * np.conj(bjt.s11.s)) / \
    (sqabs(bjt.s22.s) - sqabs(delta))

# k calculation
k = (1 - sqabs(bjt.s11.s) - sqabs(bjt.s22.s) + sqabs(delta)) / \
    (2 * np.abs(bjt.s12.s * bjt.s21.s))

mu = (1 - sqabs(bjt.s11.s)) / (np.abs(bjt.s22.s -
                                      (delta*np.conj(bjt.s11.s)))+np.abs(bjt.s21.s*bjt.s12.s))

# MAximum available gain (MAG) calculation
MAG = np.abs(bjt.s12.s/bjt.s21.s) * (k-np.sqrt(np.square(k)-1))

# Function to generate stability circle points


def calc_circle(c, r):
    theta = np.linspace(0, 2*np.pi, 1000)
    return c + r * np.exp(1.0j*theta)


# Loop over each frequency and plot individually
# for idx in range(len(bjt.f)):
#     plt.figure()

#     # Plot S11 and S22 for the current frequency
#     freq_point = bjt[idx]
#     freq_point.plot_s_smith(m=0, n=0, label='S11',
#                             marker='o', color='green')  # S11
#     freq_point.plot_s_smith(m=1, n=1, label='S22',
#                             marker='o', color='purple')  # S22

#     # Generate stability circles for this frequency
#     source_circle = calc_circle(cs[idx][0, 0], rs[idx][0, 0])
#     load_circle = calc_circle(cl[idx][0, 0], rl[idx][0, 0])

#     # Plot source stability circle
#     n_source = rf.Network(frequency=rf.Frequency(
#         bjt.f[idx], unit='Hz'), s=source_circle)
#     n_source.plot_s_smith(
#         color='red', label='Source Stability Circle', marker='')

#     # Plot load stability circle
#     n_load = rf.Network(frequency=rf.Frequency(
#         bjt.f[idx], unit='Hz'), s=load_circle)
#     n_load.plot_s_smith(color='blue', label='Load Stability Circle', marker='')

#     plt.title(f'Frequency: {bjt.f[idx]/1e9} GHz')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


k = np.squeeze(k)  # Remove any singleton dimensions
delta = np.squeeze(delta)  # Remove any singleton dimensions
mu = np.squeeze(mu)  # Remove any singleton dimensions
MAG = np.squeeze(MAG)  # Remove any singleton dimensions

# print mag at 4 Ghz
print(bjt.f)
print((np.isclose(bjt.f, 4e9, atol=1.e-3)))
val = MAG[np.where(np.isclose(bjt.f, 4e9, atol=1e6))]
print('MAG at 4 GHz:', sum(val)/len(val))

# Plotting k values
plt.figure()
plt.plot(bjt.f/1e9, k)
plt.plot(bjt.f/1e9, np.abs(delta))
plt.plot(bjt.f/1e9, mu)
plt.plot(bjt.f/1e9, MAG)
# add line in 4 Ghz
plt.axvline(x=4, color='gray', linestyle='--')
plt.title('K Factor')
plt.xlabel('Frequency (GHz)')
plt.ylabel('K')
plt.grid()
plt.legend(['K', '|$\Delta$|', '$\mu$', 'MAG'])
plt.show()
