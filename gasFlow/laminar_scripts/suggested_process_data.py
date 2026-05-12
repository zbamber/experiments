import glob
import numpy as np
import os

# ============================================================
# EXPERIMENTAL PARAMETERS — update these for your setup
# ============================================================
TUBE_RADIUS   = 0.5e-3    # m   inner radius of restriction tube at RT1
                           #     e.g. 0.5e-3 for a 1 mm diameter tube
T_ROOM        = 293.15    # K   room temperature (~20 °C)
V_IN          = 666.3e-6  # m³  calculated from expansion (150cc @ 1982mbar -> system @ 364.2mbar)
P_OUT_MBAR    = 1023      # mbar atmospheric pressure measured via bypass
DELTA_P_MBAR  = 5         # mbar DU2000 sensor uncertainty
RE_CRIT       = 2300      # critical Reynolds number for laminar–turbulent transition
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
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]  # time, pt1, pt2


# ============================================================
# REYNOLDS NUMBER
# ============================================================
def compute_reynolds(pt1_mbar, time_s, gas_name):
    """
    Reynolds number for gas flowing through the restriction tube.

    Derivation (ideal gas, isothermal, Hagen-Poiseuille geometry):
      - Moles in SV: n = P_in V_in / RT
      - Mass flow out: dm/dt = M V_in |dP_in/dt| / RT
      - Volume flow at inlet: Q = V_in |dP_in/dt| / P_in
      - Mean velocity: v = Q / (pi a^2)
      - Density at inlet: rho = M P_in / RT

      Re = rho v (2a) / eta
         = 2 M V_in |dP_in/dt| / (pi a eta R T)

    Note: P_in cancels, so Re depends only on |dP/dt| and known constants.

    Parameters
    ----------
    pt1_mbar : ndarray  P_in readings in mbar
    time_s   : ndarray  corresponding times in seconds
    gas_name : str      key into GAS_PROPERTIES dict

    Returns
    -------
    Re : ndarray  Reynolds number at each time point
    """
    props = GAS_PROPERTIES[gas_name]
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
    subsequent points — avoids picking a momentary dip in a turbulent signal.
    """
    for i in range(len(Re)):
        if np.all(Re[i:] < RE_CRIT):
            return i
    # Fallback: no clean laminar region found
    return len(Re) - 1


def find_laminar_end(pt1_mbar):
    """
    Return one-past-the-last index to keep, excluding the tail where
    P_in is dangerously close to P_out and y-uncertainty diverges.
    """
    margin  = P_MIN_MARGIN * DELTA_P_MBAR
    valid   = pt1_mbar > (P_OUT_MBAR + margin)
    if not np.any(valid):
        return 0
    return int(np.where(valid)[0][-1]) + 1  # exclusive


# ============================================================
# MAIN PROCESSING
# ============================================================
def laminar():
    folder_locations = ['./data/laminar/argon', './data/laminar/helium']
    P_out   = P_OUT_MBAR
    delta_p = DELTA_P_MBAR

    dat_files = []
    for folder in folder_locations:
        dat_files.extend(glob.glob(folder + '/*.dat'))

    if not dat_files:
        print('No .dat files found. Check folder paths.')
        return

    for file_path in dat_files:
        # --- identify gas type -------------------------------------------
        parts    = file_path.replace('\\', '/').split('/')
        gas_name = next((p.lower() for p in parts if p.lower() in GAS_PROPERTIES), None)
        if gas_name is None:
            print(f'WARNING: cannot determine gas type for {file_path} — skipping.')
            continue

        time, pt1, pt2 = extract_data(file_path)

        # --- y value and propagated uncertainty (full dataset) -----------
        y = np.log((pt1 - P_out) / (pt1 + P_out))

        y_error = ((2.0 / np.abs(pt1**2 - P_out**2))
                   * np.sqrt((P_out * delta_p)**2 + (pt1 * delta_p)**2))

        # --- Reynolds number → trim indices ------------------------------
        Re        = compute_reynolds(pt1, time, gas_name)
        i_start   = find_laminar_start(Re)
        i_end     = len(time)

        if i_start >= i_end:
            print(f'WARNING: no valid laminar window for {file_path}. '
                  f'Using all data.')
            i_start, i_end = 0, len(time)

        n_total   = len(time)
        n_kept    = i_end - i_start
        print(f'\n{os.path.basename(file_path)}  [{gas_name}]')
        print(f'  Re range (full):    {Re.min():.0f} – {Re.max():.0f}')
        print(f'  Turbulent region:   indices 0 – {i_start - 1}  '
              f'(Re > {RE_CRIT})')
        print(f'  Tail trimmed:       indices {i_end} – {n_total - 1}  '
              f'(P_in too close to P_out)')
        print(f'  Laminar fit window: indices {i_start} – {i_end - 1}  '
              f'({n_kept}/{n_total} points kept)')
        print(f'  Re at fit start:    {Re[i_start]:.0f}')
        print(f'  Time window:        {time[i_start]:.1f} s – '
              f'{time[i_end - 1]:.1f} s')

        # --- save diagnostics CSV (all points, for inspection) -----------
        output_dir = os.path.dirname(file_path).replace('data', 'processed_data')
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(file_path))[0]

        diag_path = os.path.join(output_dir, base + '_diagnostics.csv')
        in_laminar = np.zeros(n_total, dtype=int)
        in_laminar[i_start:i_end] = 1
        np.savetxt(
            diag_path,
            np.column_stack((time, pt1, y, y_error, Re, in_laminar)),
            delimiter=',',
            header='time_s,pt1_mbar,y,y_error,reynolds,in_laminar_fit',
            comments='',
        )

        # --- save trimmed dataset for LSFR -------------------------------
        output_path = os.path.join(output_dir, base + '_processed.csv')
        np.savetxt(
            output_path,
            np.column_stack((time[i_start:i_end],
                             y[i_start:i_end],
                             y_error[i_start:i_end])),
            delimiter=',',
            header='x,y,y_error',
            comments='',
        )

    print('\nProcessing complete.')


if __name__ == '__main__':
    laminar()