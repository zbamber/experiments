import os
import glob
import numpy as np
import importlib.util
import matplotlib.pyplot as plt

# Import LSFR-25.py dynamically
spec = importlib.util.spec_from_file_location("lsfr", "molecular_scripts/LSFR-25.py")
lsfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lsfr)

# Experimental Constants
V_IN = 150e-6  # m³
T_ROOM = 293.15 # K
TUBE_RADIUS = 0.1e-3 # m

# Gas-specific tube lengths
TUBE_LENGTHS = {
    'argon': 5.0e-3,
    'helium': 30.0e-3
}

GAS_CALIBRATION = {'argon': 1.7, 'helium': 0.8}
GAS_MOLAR_MASS = {'argon': 39.948e-3, 'helium': 4.0026e-3}

def get_theoretical_s(gas_name):
    """Equation 7: S = (2*pi*a^3)/(3*l) * sqrt(8*k*T / (pi*m))"""
    a = TUBE_RADIUS
    l = TUBE_LENGTHS.get(gas_name, 30.0e-3)
    T = T_ROOM
    m = GAS_MOLAR_MASS[gas_name] / 6.022e23
    k_B = 1.38e-23
    
    term1 = (2 * np.pi * a**3) / (3 * l)
    term2 = np.sqrt((8 * k_B * T) / (np.pi * m))
    return term1 * term2

def run_analysis():
    processed_files = glob.glob('data/molecular flow/**/*_processed.csv', recursive=True)
    
    results = []
    
    for f in processed_files:
        gas_name = 'argon' if 'argon' in f.lower() else 'helium'
        
        # Load processed data
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        # Columns: time_s, y, y_error, is_molecular
        mask = data[:, 3] == 1
        
        if not np.any(mask):
            print(f"Skipping {f}: No molecular points found.")
            continue
            
        x = data[mask, 0]
        y = data[mask, 1]
        y_err = data[mask, 2]
        
        # Perform fit
        params, uncertainties = lsfr.fitting_procedure(x, y, y_err)
        slope = params[0]
        s_exp = -slope * V_IN
        s_theo = get_theoretical_s(gas_name)
        
        results.append({
            'file': f,
            'gas': gas_name,
            's_exp': s_exp,
            's_theo': s_theo
        })
        
        # Update LSFR globals for clean plot
        lsfr.FILE_NAME = f
        lsfr.PLOT_TITLE = f"Molecular Flow: {gas_name.capitalize()}"
        lsfr.X_LABEL = "Time (s)"
        lsfr.Y_LABEL = "ln(P - pr)"
        lsfr.FIGURE_NAME = f.replace('_processed.csv', '_fit.png')
        
        # Create plot
        lsfr.create_plot(x, y, y_err, params, uncertainties)
        plt.close()

    # Print summary
    print("\n" + "="*80)
    print(f"{'File':<40} | {'S_exp (m³/s)':<15} | {'S_theo (m³/s)':<15} | {'Ratio'}")
    print("-" * 80)
    
    summary_by_gas = {'argon': [], 'helium': []}
    for r in results:
        ratio = r['s_exp'] / r['s_theo']
        print(f"{os.path.basename(r['file']):<40} | {r['s_exp']:>12.2e} | {r['s_theo']:>12.2e} | {ratio:.2f}")
        summary_by_gas[r['gas']].append(r['s_exp'])

    if summary_by_gas['argon'] and summary_by_gas['helium']:
        actual_ratio = np.mean(summary_by_gas['helium']) / np.mean(summary_by_gas['argon'])
        theo_ratio = np.sqrt(GAS_MOLAR_MASS['argon'] / GAS_MOLAR_MASS['helium'])
        print("\n" + "="*80)
        print(f"Experimental S_He / S_Ar: {actual_ratio:.2f}")
        print(f"Theoretical  S_He / S_Ar: {theo_ratio:.2f}")
        print("="*80)

if __name__ == '__main__':
    run_analysis()
