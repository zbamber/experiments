import glob
import numpy as np
import os

# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================
TUBE_RADIUS   = 0.1e-3    # m
TUBE_LENGTH   = 30.0e-3   # m
V_IN          = 666.3e-6  # m³  calculated from expansion (150cc @ 1982mbar -> system @ 364.2mbar)
T_ROOM        = 293.15    # K
R_GAS         = 8.314     # J/(mol·K)

# Residual pressure (mbar)
# Helium: optimized from fit as 1.167 mbar to remove tail droop
# Argon: suspected lower due to faster pumping. Trying 0.2 mbar.
PR_VALUES = {
    'argon': 0.2,
    'helium': 1.167
}

# PT2 Accuracy: 15% absolute, 2% repeatability. 
# We use 2% for relative fitting errors.
PT2_REL_ERR = 0.02

# Gas-specific calibration factors for TTR91 (PT2)
# P_actual = C_gas * P_reading
GAS_CALIBRATION = {
    'argon': 1.7,
    'helium': 0.8
}

# Molecular weights (kg/mol)
GAS_MOLAR_MASS = {
    'argon': 39.948e-3,
    'helium': 4.0026e-3
}

# Typical collision diameters (m) for Knudsen number calculation
GAS_DIAMETER = {
    'argon': 3.4e-10,
    'helium': 2.6e-10
}

def get_knudsen_number(P_Pa, gas_name):
    """Calculate Knudsen number Kn = lambda / (2*a)"""
    d = GAS_DIAMETER.get(gas_name, 3e-10)
    # Mean free path lambda = (k*T) / (sqrt(2) * pi * d^2 * P)
    k_B = 1.38e-23
    mfp = (k_B * T_ROOM) / (np.sqrt(2) * np.pi * d**2 * P_Pa)
    return mfp / (2 * TUBE_RADIUS)

def extract_data(file_path):
    """Extract Time and PT2 (3rd column)"""
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(file_path, skiprows=1)
    
    # Structure: Time, PT1, PT2
    return data[:, 0], data[:, 2]

def process_file(file_path):
    parts = file_path.replace('\\', '/').split('/')
    gas_name = next((p.lower() for p in parts if p.lower() in GAS_CALIBRATION), 'argon')
    
    time, pt2_raw = extract_data(file_path)
    
    c_gas = GAS_CALIBRATION[gas_name]
    P_actual = pt2_raw * c_gas
    
    # p_r is in mbar, same as P_actual here
    p_r = PR_VALUES.get(gas_name, 1.0)
    
    # Transformation: y = ln(P - p_r)
    # We must ensure P > p_r
    # We add a small epsilon to avoid log(0)
    valid_mask = (P_actual > (p_r + 0.05)) 
    if not np.any(valid_mask):
        print(f"Skipping {file_path}: No data points where P > p_r + 0.05 ({p_r} mbar)")
        return

    time = time[valid_mask]
    P_actual = P_actual[valid_mask]
    
    y = np.log(P_actual - p_r)
    
    # Error propagation: delta_y = delta_P / (P - p_r)
    # delta_P = 0.02 * P (repeatability)
    delta_y = (PT2_REL_ERR * P_actual) / (P_actual - p_r)
    
    # Identify molecular regime: Kn > 0.1 (Capturing transition)
    P_Pa = P_actual * 100.0
    Kn = get_knudsen_number(P_Pa, gas_name)
    is_molecular = (Kn > 0.1).astype(int)
    
    print(f'Processed {file_path} ({gas_name})')
    print(f'  Points: {len(y)}, Molecular points: {np.sum(is_molecular)}')
    print(f'  P_start: {P_actual[0]:.2f} mbar, P_end: {P_actual[-1]:.2f} mbar')
    print(f'  Kn_start: {Kn[0]:.3f}, Kn_end: {Kn[-1]:.3f}')
    print(f'  p_r used: {p_r}')

    output_path = file_path.replace('.dat', '_processed.csv')
    header = 'time_s,y,y_error,is_molecular'
    np.savetxt(output_path, np.column_stack((time, y, delta_y, is_molecular)), 
               delimiter=',', header=header, comments='')

def main():
    patterns = [
        'data/molecular flow/argon/*.dat',
        'data/molecular flow/helium/*.dat'
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            if '_processed' in f: continue
            process_file(f)

if __name__ == '__main__':
    main()
