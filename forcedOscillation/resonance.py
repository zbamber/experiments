import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('resonance_data.csv', delimiter=',')
frequency = data[:, 0]
v0 = data[:, 1]
v0_error = data[:, 2]

plt.figure(figsize=(8,5))
plt.errorbar(frequency, v0, yerr=v0_error, fmt='o-', capsize=3, color='blue', label='Velocity')
plt.title('Velocity Resonance Curve')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Velocity Amplitude (mV)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/velocity_resonance_curve.png')

x0 = v0 / frequency
x0_err = v0_error / frequency

plt.figure(figsize=(8,5))
plt.errorbar(frequency, x0, yerr=x0_err, fmt='s-', capsize=3, color='blue', label='Displacement')
plt.title('Displacement Resonance Curve')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Displacement Amplitude (Arbitrary Units)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/displacement_resonance_curve.png')

P = v0**2
P_err = 2 * v0 * v0_error

max_idx = np.argmax(P)
P_max = P[max_idx]
f0 = frequency[max_idx]
P_half = P_max / 2

# Robust calculation without needing scipy
left_freqs = frequency[:max_idx+1]
left_P = P[:max_idx+1]
idx_below = np.where(left_P <= P_half)[0][-1]
idx_above = np.where(left_P > P_half)[0][0]
m = (left_P[idx_above] - left_P[idx_below]) / (left_freqs[idx_above] - left_freqs[idx_below])
f1 = left_freqs[idx_below] + (P_half - left_P[idx_below]) / m

f2_est = f0 + (f0 - f1)
fwhm_est = f2_est - f1
Q_est = f0 / fwhm_est

plt.figure(figsize=(8,5))
plt.errorbar(frequency, P, yerr=P_err, fmt='^-', capsize=3, color='blue', label='Power')
plt.axvline(x=f0, color='red', linestyle='-', alpha=0.5, label=f'Natural Freq (f0) = {f0:.3f} Hz')
plt.axhline(y=P_half, color='green', linestyle='--', label=f'Half Power (P_max/2) = {P_half:,.0f}')
plt.axvline(x=f1, ymin=0, ymax=0.5, color='green', linestyle=':')
plt.text(f1, P_half*1.05, f'{f1:.3f} Hz', ha='right', color='green')
plt.axvline(x=f2_est, ymin=0, ymax=0.5, color='green', linestyle=':', alpha=0.5)
plt.text(f2_est, P_half*1.05, f'~{f2_est:.3f} Hz\n(est)', ha='left', color='green')
plt.title(f'Power Resonance Curve')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Power (Arbitrary Units)')
plt.xlim(f0 - 0.4, f0 + 0.4) # Dynamic zoom added here to match appearance
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/power_resonance_curve.png')

phase_data = np.loadtxt('frequency_phase_data.csv', delimiter=',')
frequency_phase = phase_data[:, 0]
phase = phase_data[:, 1]

plt.figure(figsize=(8,5))
plt.plot(frequency_phase, phase, 'd-', color='blue', label='Phase')
plt.title('Phase Resonance Curve')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Phase difference (Degrees)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/phase_resonance_curve.png')