import os
import argparse

def generate_test_list(data_dir, output_file):
    """
    Scans a directory for .npy.h5 files and writes their basenames (without the .npy.h5 extension)
    to an output text file. This is used to create the test_vol.txt list.
    """
    try:
        print(f"Scanning directory: {data_dir}")
        # Get all files in the directory
        files = os.listdir(data_dir)
        
        # Filter for .npy.h5 files and process their names
        test_cases = []
        for f in files:
            # We are looking for files ending in .npy.h5
            if f.endswith('.npy.h5'):
                # Correctly remove the '.npy.h5' suffix to get the base case name.
                # For example: 'case0001-0061.npy.h5' -> 'case0001-0061'
                base_name = f[:-7] 
                test_cases.append(base_name)
        
        if not test_cases:
            print(f"Warning: No .npy.h5 files were found in '{data_dir}'.")
            print("The output file will be empty. Please check the data directory path.")
            # Create an empty file so the process doesn't fail later
            open(output_file, 'w').close()
            return

        # Sort the list alphabetically for consistent results
        test_cases.sort()
        
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the processed names to the output file
        with open(output_file, 'w') as f:
            for case in test_cases:
                f.write(case + '\n')
                
        print(f"✅ Successfully generated test list with {len(test_cases)} entries.")
        print(f"   Output file saved to: {output_file}")

    except FileNotFoundError:
        print(f"❌ Error: The data directory '{data_dir}' was not found.")
        exit(1) # Exit with an error code
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        exit(1) # Exit with an error code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically generate the test_vol.txt file from a directory of test data."
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help="Path to the directory containing the test data files (e.g., test_vol_h5)."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True,
        help="Path for the output list file (e.g., lists/lists_Synapse/test_vol.txt)."
    )
    
    args = parser.parse_args()
    
    generate_test_list(args.data_dir, args.output_file)
