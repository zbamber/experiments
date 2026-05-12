import glob
import numpy as np
import os
import pandas as pd

# ============================================================
# EXPERIMENTAL PARAMETERS — update these for your setup
# ============================================================
TUBE_RADIUS   = 0.1e-3    # m   inner radius (a) = 0.1 mm from PDF
TUBE_LENGTH   = 30.0e-3   # m   length (l) = 30 mm from PDF
T_ROOM        = 293.15    # K   room temperature (~20 °C)
V_IN          = 666.3e-6  # m³  calculated from expansion (150cc @ 1982mbar -> system @ 364.2mbar)
P_OUT_MBAR    = 1023      # mbar atmospheric pressure
DELTA_P_MBAR  = 5         # mbar DU2000 sensor uncertainty
RE_CRIT       = 100       # threshold for non-linear behavior at start
P_MIN_MARGIN  = 3         # exclude tail where P_in < P_out + P_MIN_MARGIN * delta_p
                           #     (region where uncertainty diverges)

# ============================================================
# GAS PROPERTIES  (literature values at ~20 °C, 1 atm)
# ============================================================
GAS_PROPERTIES = {
    'argon': {
        'molar_mass': 39.948e-3,   # kg/mol
        'viscosity':   2.27e-5,    # Pa·s
    },
    'helium': {
        'molar_mass':  4.0026e-3,  # kg/mol
        'viscosity':   1.96e-5,    # Pa·s
    },
}

R_GAS = 8.314  # J/(mol·K)


# ============================================================
# DATA LOADING
# ============================================================
def extract_data(file_path):
    # Try loading with comma or tab
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except:
        data = np.loadtxt(file_path, skiprows=0) # Fallback for no header/tab
    
    # User said: columns are time, PT1 and PT2. 
    # But some files might have 4 columns (time, time_err, pt1, pt1_err)
    if data.shape[1] >= 4:
        return data[:, 0], data[:, 2] # time, pt1
    elif data.shape[1] >= 2:
        return data[:, 0], data[:, 1] # time, pt1
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


# ============================================================
# REYNOLDS NUMBER
# ============================================================
def compute_reynolds(pt1_mbar, time_s, gas_name):
    """
    Reynolds number for gas flowing through the restriction tube.
    """
    props = GAS_PROPERTIES.get(gas_name, GAS_PROPERTIES['argon'])
    M     = props['molar_mass']   # kg/mol
    eta   = props['viscosity']    # Pa·s

    pt1_Pa = pt1_mbar * 100.0                      # mbar → Pa
    dPdt   = np.gradient(pt1_Pa, time_s)           # Pa/s  (negative: falling)

    Re = (2.0 * M * V_IN * np.abs(dPdt)) / (np.pi * TUBE_RADIUS * eta * R_GAS * T_ROOM)
    return Re


# ============================================================
# TRIM DETECTION
# ============================================================
def find_laminar_start(Re):
    """
    Return the first index at which Re < RE_CRIT and remains so for all
    subsequent points.
    """
    for i in range(len(Re)):
        if np.all(Re[i:] < RE_CRIT):
            return i
    return len(Re) - 1


def find_laminar_end(pt1_mbar):
    """
    Return one-past-the-last index to keep, excluding the tail.
    """
    margin  = P_MIN_MARGIN * DELTA_P_MBAR
    valid   = pt1_mbar > (P_OUT_MBAR + margin)
    if not np.any(valid):
        return 0
    return int(np.where(valid)[0][-1]) + 1  # exclusive


# ============================================================
# MAIN PROCESSING
# ============================================================
def process_all():
    # Search for files in common locations
    search_patterns = [
        'data/laminar/argon/*.dat',
        'data/laminar/helium/*.dat',
        'data/*.csv'
    ]
    
    P_out   = P_OUT_MBAR
    delta_p = DELTA_P_MBAR

    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern))

    if not files:
        print('No data files found. Check folder paths.')
        return

    for file_path in files:
        if '_processed' in file_path:
            continue
            
        # --- identify gas type -------------------------------------------
        parts    = file_path.replace('\\', '/').split('/')
        gas_name = next((p.lower() for p in parts if p.lower() in GAS_PROPERTIES), 'argon')
        
        try:
            time, pt1 = extract_data(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        # --- y value and propagated uncertainty (full dataset) -----------
        # Avoid log of negative or zero
        valid_mask = (pt1 > P_out)
        if not np.any(valid_mask):
            print(f"Skipping {file_path}: No data points where Pin > Pout")
            continue
            
        time = time[valid_mask]
        pt1 = pt1[valid_mask]
        
        y = np.log((pt1 - P_out) / (pt1 + P_out))

        y_error = ((2.0 / np.abs(pt1**2 - P_out**2))
                   * np.sqrt((P_out * delta_p)**2 + (pt1 * delta_p)**2))

        # --- Reynolds number → trim indices ------------------------------
        Re        = compute_reynolds(pt1, time, gas_name)
        i_start   = find_laminar_start(Re)
        i_end     = len(time)  # Include everything until the end (tail included)

        n_total   = len(time)
        in_laminar = np.zeros(n_total, dtype=int)
        
        # Ensure we have a valid window
        if i_start < i_end:
            in_laminar[i_start:i_end] = 1
        
        print(f'\nProcessed: {os.path.basename(file_path)} [{gas_name}]')
        print(f'  Total points: {n_total}')
        print(f'  Turbulent (start) cut: {i_start} points (Re_max={np.max(Re):.1f})')
        print(f'  Tail (end) cut: {n_total - i_end} points')
        print(f'  Laminar fit window: indices {i_start} to {i_end-1}')

        # --- save processed dataset for LSFR -------------------------------
        output_path = file_path.replace('.csv', '_processed.csv').replace('.dat', '_processed.csv')
        
        # Ensure we don't overwrite the original if it was already .csv (handled by _processed check)
        
        np.savetxt(
            output_path,
            np.column_stack((time, y, y_error, in_laminar)),
            delimiter=',',
            header='time_s,y,y_error,is_laminar',
            comments='',
        )

    print('\nProcessing complete.')


if __name__ == '__main__':
    process_all()
