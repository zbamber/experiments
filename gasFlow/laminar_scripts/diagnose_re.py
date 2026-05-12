import numpy as np
import pandas as pd
import os
from process_data import compute_reynolds, extract_data, GAS_PROPERTIES, P_OUT_MBAR

def diagnose_file(file_path):
    gas_name = 'argon' if 'argon' in file_path.lower() else 'helium'
    time, pt1 = extract_data(file_path)
    
    # Filter for Pin > Pout
    mask = pt1 > P_OUT_MBAR
    time = time[mask]
    pt1 = pt1[mask]
    
    Re = compute_reynolds(pt1, time, gas_name)
    
    print(f"\nFile: {file_path}")
    print(f"Max Re: {np.max(Re):.2f} at start")
    print(f"Min Re: {np.min(Re):.2f} at end")
    print(f"First 5 Re values: {Re[:5]}")

if __name__ == "__main__":
    diagnose_file('data/laminar/argon/argon_1.dat')
    diagnose_file('data/laminar/helium/helium_1.dat')
