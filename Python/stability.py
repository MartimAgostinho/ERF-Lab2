import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

plt.rcParams['figure.figsize'] = [8, 8]

# Load the S-parameter file
bjt = rf.Network('1to10G.s2p', )

# Define a function to calculate the square of the absolute value


def sqabs(x): return np.square(np.absolute(x))


# delta is the determinant of the S-parameter matrix
delta = bjt.s11.s * bjt.s22.s - bjt.s12.s * bjt.s21.s

# Source and load reflection coefficients
# rs and rl are the source and load reflection coefficients
# cs and cl are the source and load circle centers
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


# Filter for 4 GHz
freq_idx = np.where(np.isclose(bjt.f, 4e9, atol=1e6))[
    0][0]
freq_point = bjt[freq_idx]

# Generate stability circles for 4 GHz
source_circle = calc_circle(cs[freq_idx][0, 0], rs[freq_idx][0, 0])
load_circle = calc_circle(cl[freq_idx][0, 0], rl[freq_idx][0, 0])

# Plot S11 and S22 for 4 GHz
plt.figure()
s11 = freq_point.s11.s[0, 0]  # Extrair o valor numérico de S11
s22 = freq_point.s22.s[0, 0]  # Extrair o valor numérico de S22
print('S11:', s11)
print('S22:', s22)
print('|s11|: ', np.abs(s11))
print('|s22|: ', np.abs(s22))

# Adicionar os valores numéricos ao gráfico
freq_point.plot_s_smith(
    m=0, n=0, label=f'S11 = {s11}', marker='o', color='green')  # S11
freq_point.plot_s_smith(
    m=1, n=1, label=f'S22 = {s22}', marker='o', color='purple')  # S22

# Calcular B1, B2, C1 e C2
delta4GHz = delta[freq_idx][0, 0]
print('delta4GHz:', delta4GHz)
B1 = 1 + sqabs(s11) - sqabs(s22) - sqabs(delta4GHz)
print('B1:', B1)
B2 = 1 + sqabs(s22) - sqabs(s11) - sqabs(delta4GHz)
print('B2:', B2)
C1 = s11 - (delta4GHz * np.conj(s22))
print('C1:', C1)
C2 = s22 - (delta4GHz * np.conj(s11))
print('C2:', C2)

# Calcular ros (ρs) e rol (ρL)
ros = (B1 - np.sqrt(np.square(B1) - 4 * sqabs(C1))) / (2 * C1)
rol = (B2 - np.sqrt(np.square(B2) - 4 * sqabs(C2))) / (2 * C2)

# Exibir os resultados para 4 GHz
print("Frequency (GHz):", bjt.f[freq_idx] / 1e9)
print("ros (ρs):", ros)
print("rol (ρL):", rol)

zs = (1 + ros) / (1 - ros)
print("zs:", zs)
zl = (1 + rol) / (1 - rol)
print("zl:", zl)

# Plot source stability circle
n_source = rf.Network(frequency=rf.Frequency(
    bjt.f[freq_idx], unit='Hz'), s=source_circle)
n_source.plot_s_smith(color='red', label='Source Stability Circle', marker='')

# Plot load stability circle
n_load = rf.Network(frequency=rf.Frequency(
    bjt.f[freq_idx], unit='Hz'), s=load_circle)
n_load.plot_s_smith(color='blue', label='Load Stability Circle', marker='')

rf.plotting.plot_smith(zl, marker='o', color='blue',
                       label=f'Zl = {zl}')  # Plot Zl
rf.plotting.plot_smith(zs, marker='o', color='red',
                       label=f'Zs = {zs}')  # Plot Z

plt.title(f'Frequency: {bjt.f[freq_idx]/1e9} GHz')
plt.legend()
plt.tight_layout()
plt.show()

k = np.squeeze(k)  # Remove any singleton dimensions
delta = np.squeeze(delta)  # Remove any singleton dimensions
mu = np.squeeze(mu)  # Remove any singleton dimensions
MAG = np.squeeze(MAG)  # Remove any singleton dimensions

# Print MAG at 4 Ghz
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
