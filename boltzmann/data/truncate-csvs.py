import os
import glob

def truncate_csvs(lines_to_remove=8):
    """Reads all CSVs in the current directory, removes the last N lines, 
    and saves them to a 'truncated' subdirectory."""
    
    ignore = []

    # Create the target directory if it doesn't exist
    output_dir = 'truncated'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in this directory.")
        return

    print(f"Found {len(csv_files)} files. Starting truncation...")

    for file in csv_files:

        if 'processed' not in file or file in ignore:
            continue

        # Read the file (using utf-8-sig to handle Excel BOMs automatically)
        with open(file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
        # Make sure the file actually has enough lines to chop
        if len(lines) > lines_to_remove:
            truncated_lines = lines[:-lines_to_remove]
        else:
            print(f"Skipping {file}: Only has {len(lines)} lines.")
            continue
            
        # Save the new file into the subdirectory
        output_path = os.path.join(output_dir, file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(truncated_lines)
            
        print(f"Chipped {lines_to_remove} lines off {file} -> Saved to /{output_dir}")

    print("\nAll files successfully truncated!")

if __name__ == "__main__":
    truncate_csvs()