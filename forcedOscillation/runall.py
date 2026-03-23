import os
import subprocess
import glob

def run_lsfr_on_all_csvs():
    target_script = "lsfr-25.py"

    ignore = []
    
    csv_files = glob.glob("Data/*processed*.csv")
    
    if not csv_files:
        print("No CSV files found in this directory.")
        return

    print(f"Found {len(csv_files)} files. Starting processing...")

    for file in csv_files:
        if 'processed' not in file or file in ignore:
            continue
        print(f"--- Running {target_script} on {file} ---")
        
        try:
            subprocess.run(["python", target_script, file, 'imageOnly'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {file}: {e}")
        except FileNotFoundError:
            print(f"Error: Could not find '{target_script}' in this directory.")
            break

    print("\nAll files have been processed.")

if __name__ == "__main__":
    run_lsfr_on_all_csvs()