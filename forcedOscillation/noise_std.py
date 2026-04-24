import numpy as np
from scipy.optimize import curve_fit

data = np.loadtxt('noise2.csv')
time = data[:,0]
time_shifted = time - time.min()
voltage = data[:,1]

def sine_wave(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

p0 = [(voltage.max() - voltage.min())/2, 14.33, 0.0, np.mean(voltage)]
popt, _ = curve_fit(sine_wave, time_shifted, voltage, p0=p0)

fitted_curve = sine_wave(time_shifted, *popt)
residuals = voltage - fitted_curve

residual_std = np.std(residuals)

print(f"Original Raw Standard Deviation: {np.std(voltage):.6f} V")
print(f"Standard Deviation of Residuals: {residual_std:.6f} V")