import os
import numpy as np
import nibabel as nib
import glob

# Paths
source_dir = "3T_data"
output_dir = "3T_data_npy"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_folder(class_name):
    # Handle case sensitivity (AD/ad, CN/cn)
    # Look for both lowercase and uppercase folder names
    folder_path = os.path.join(source_dir, class_name)
    if not os.path.exists(folder_path):
        folder_path = os.path.join(source_dir, class_name.lower())
    
    if not os.path.exists(folder_path):
        print(f"Warning: Could not find folder for {class_name}")
        return

    # Find all .nii files
    files = glob.glob(os.path.join(folder_path, "*.nii"))
    print(f"Found {len(files)} files in {class_name}...")

    for file_path in files:
        try:
            # 1. Load the NIfTI file
            img = nib.load(file_path)
            data = img.get_fdata()

            # 2. Normalize or resize if necessary (Optional but recommended)
            # For now, we just save the raw data array
            
            # 3. Create a new filename
            # We prepend the class name so we know what it is later (e.g., AD_patient1.npy)
            original_name = os.path.basename(file_path).replace('.nii', '')
            new_filename = f"{class_name}_{original_name}.npy"
            save_path = os.path.join(output_dir, new_filename)

            # 4. Save as .npy
            np.save(save_path, data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Process both classes
process_folder("AD")
process_folder("CN")

print(f"\nConversion complete! Files saved to '{output_dir}'.")