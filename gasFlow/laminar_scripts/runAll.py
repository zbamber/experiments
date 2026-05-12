import os
import subprocess
import glob

def run_lsfr_on_all_csvs():
    target_script = "laminar_scripts/LSFR-laminar.py"

    # Search recursively for processed files
    csv_files = glob.glob("data/**/*_processed.csv", recursive=True) + glob.glob("data/*_processed.csv")
    
    # Filter out duplicates and non-files
    csv_files = sorted(list(set([f for f in csv_files if os.path.isfile(f)])))

    if not csv_files:
        print("No processed CSV files found in the data/ directory.")
        return

    print(f"Found {len(csv_files)} files. Starting processing...")

    for file in csv_files:
        print(f"--- Running {target_script} on {file} ---")
        
        try:
            # We pass 'imageOnly' to avoid blocking on plt.show()
            subprocess.run(["python", target_script, file, 'imageOnly'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {file}: {e}")
        except FileNotFoundError:
            print(f"Error: Could not find '{target_script}' in this directory.")
            break

    print("\nAll files have been processed.")

if __name__ == "__main__":
    run_lsfr_on_all_csvs()