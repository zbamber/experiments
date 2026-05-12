import numpy as np
import pandas as pd
import glob
import os

# ============================================================
# CONSTANTS
# ============================================================
A         = 0.1e-3    # m (radius)
SIGMA_A   = 0.05 * A  # 5% manufacturer error
L         = 30.0e-3   # m
SIGMA_L   = 0.5e-3    # 0.5 mm error from PDF
V_IN      = 666.3e-6  # m³
SIGMA_VIN = 0.5e-6    # 0.5 cc error from calculation
P_OUT     = 1023e2    # Pa (1023 mbar)
SIGMA_P   = 5e2       # Pa (5 mbar)
T         = 293.15    # K
R_GAS     = 8.314     # J/(mol·K)
K_B       = 1.3806e-23 # J/K

GAS_DATA = {
    'argon': {
        'M': 39.948e-3,  # kg/mol
    },
    'helium': {
        'M': 4.0026e-3,  # kg/mol
    }
}

def fit_slope(file_path):
    data = pd.read_csv(file_path)
    # Use only laminar points
    laminar = data[data['is_laminar'] == 1]
    if len(laminar) < 2:
        return None, None
    
    x = laminar['time_s'].values
    y = laminar['y'].values
    w = 1.0 / (laminar['y_error'].values**2)
    
    # Weighted linear fit y = mx + c
    sum_w = np.sum(w)
    sum_wx = np.sum(w * x)
    sum_wy = np.sum(w * y)
    sum_wxx = np.sum(w * x**2)
    sum_wxy = np.sum(w * x * y)
    
    delta = sum_w * sum_wxx - sum_wx**2
    m = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
    sigma_m = np.sqrt(sum_w / delta)
    
    return m, sigma_m

def calculate_properties():
    results = []
    
    # Process all laminar data
    files = glob.glob('data/laminar/**/*_processed.csv', recursive=True)
    
    for f in files:
        gas = 'argon' if 'argon' in f.lower() else 'helium'
        m, sigma_m = fit_slope(f)
        if m is None: continue
        
        # 1. Viscosity (eta)
        # m = - (pi * a^4 * P_out) / (8 * eta * l * V_in)
        # eta = - (pi * a^4 * P_out) / (8 * m * l * V_in)
        eta = - (np.pi * A**4 * P_OUT) / (8.0 * m * L * V_IN)
        
        # Uncertainty propagation for eta
        rel_err_eta = np.sqrt(
            (4.0 * SIGMA_A / A)**2 +
            (SIGMA_P / P_OUT)**2 +
            (sigma_m / m)**2 +
            (SIGMA_L / L)**2 +
            (SIGMA_VIN / V_IN)**2
        )
        sigma_eta = eta * rel_err_eta
        
        # 2. Mean speed (c_bar)
        # c_bar = sqrt(8 * R * T / (pi * M))
        M = GAS_DATA[gas]['M']
        c_bar = np.sqrt(8.0 * R_GAS * T / (np.pi * M))
        
        # 3. Mean free path (lambda)
        # eta = 1/3 * rho * lambda * c_bar
        # rho = P_out * M / (R * T)
        rho = (P_OUT * M) / (R_GAS * T)
        lam = (3.0 * eta) / (rho * c_bar)
        
        # Uncertainty in lambda (simplified, mostly eta-dominated)
        sigma_lam = lam * rel_err_eta
        
        # 4. Collision cross-section (sigma)
        # lam = 1 / (sqrt(2) * n_v * sigma)
        # n_v = P_out / (k_B * T)
        n_v = P_OUT / (K_B * T)
        cs_sigma = 1.0 / (np.sqrt(2.0) * n_v * lam)
        
        # Uncertainty in sigma
        sigma_cs = cs_sigma * rel_err_eta
        
        # 5. Atomic diameter (d)
        # cs_sigma = pi * d^2
        d = np.sqrt(cs_sigma / np.pi)
        sigma_d = d * 0.5 * rel_err_eta
        
        results.append({
            'file': os.path.basename(f),
            'gas': gas,
            'm': m,
            'sigma_m': sigma_m,
            'eta': eta,
            'sigma_eta': sigma_eta,
            'c_bar': c_bar,
            'lambda': lam,
            'sigma_lambda': sigma_lam,
            'sigma': cs_sigma,
            'sigma_cs': sigma_cs,
            'd': d,
            'sigma_d': sigma_d
        })
        
    # Average results per gas
    df = pd.DataFrame(results)
    for gas in ['argon', 'helium']:
        gas_df = df[df['gas'] == gas]
        if gas_df.empty: continue
        
        print(f"\n{'='*40}")
        print(f" RESULTS FOR {gas.upper()}")
        print(f"{'='*40}")
        
        mean_eta = gas_df['eta'].mean()
        err_eta = gas_df['sigma_eta'].mean() # Average error
        
        mean_c = gas_df['c_bar'].mean()
        mean_lam = gas_df['lambda'].mean()
        err_lam = gas_df['sigma_lambda'].mean()
        
        mean_sigma = gas_df['sigma'].mean()
        err_sigma = gas_df['sigma_cs'].mean()
        
        mean_d = gas_df['d'].mean()
        err_d = gas_df['sigma_d'].mean()
        
        print(f"Viscosity (eta):        ({mean_eta*1e5:.3f} +/- {err_eta*1e5:.3f}) x 10^-5 Pa.s")
        print(f"Mean Speed (c_bar):     {mean_c:.1f} m/s")
        print(f"Mean Free Path (lambda): ({mean_lam*1e9:.1f} +/- {err_lam*1e9:.1f}) nm")
        print(f"Cross Section (sigma):  ({mean_sigma*1e20:.3f} +/- {err_sigma*1e20:.3f}) x 10^-20 m^2")
        print(f"Atomic Diameter (d):    ({mean_d*1e10:.3f} +/- {err_d*1e10:.3f}) Angstrom")

if __name__ == "__main__":
    calculate_properties()
