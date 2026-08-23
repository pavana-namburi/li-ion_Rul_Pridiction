"""
NASA PCoE Li-ion Battery RUL Dataset Preprocessing
Extended for B0005, B0006, B0007, B0018, B0025, B0026

This script:
1. Loads .mat files from battery_data/ directory
2. Extracts discharge cycles
3. Computes RUL labels (capacity-threshold based, 70% per NASA standard)
4. Saves sample-level CSVs to data/ directory

Usage:
    python nasa_rul_preprocessing.py

Output:
    data/B0005_discharge.csv
    data/B0006_discharge.csv
    data/B0007_discharge.csv
    data/B0018_discharge.csv
    data/B0025_discharge.csv
    data/B0026_discharge.csv

Author: NASA PCoE RUL Preprocessing
Date: 2026-01-08
"""

from scipy.io import loadmat
import pandas as pd
from pathlib import Path
import traceback


# ==============================================================================
# CORE PREPROCESSING FUNCTIONS
# ==============================================================================

def load_data(mat_path, battery):
    """
    Load discharge cycles from NASA .mat file.
    
    Args:
        mat_path (str): Path to .mat file (e.g., 'battery_data/B0005.mat')
        battery (str): Battery name key in .mat file (e.g., 'B0005')
    
    Returns:
        pd.DataFrame: Sample-level discharge data with columns:
            cycle, ambient_temperature, capacity,
            voltage_measured, current_measured, temperature_measured,
            current_load, voltage_load, time
    
    Raises:
        FileNotFoundError: If .mat file not found
        KeyError: If battery key not in .mat structure
        ValueError: If no discharge cycles found
    """
    mat_path = Path(mat_path)
    
    # Check file exists
    if not mat_path.exists():
        raise FileNotFoundError(
            f"File not found: {mat_path}\n"
            f"Expected at: {mat_path.absolute()}"
        )
    
    # Load .mat file
    try:
        mat = loadmat(str(mat_path))
    except Exception as e:
        raise ValueError(
            f"Failed to load .mat file {mat_path}.\n"
            f"Error: {str(e)}\n"
            f"Make sure the file is a valid MATLAB .mat file."
        )
    
    # Check battery key exists
    if battery not in mat:
        available = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(
            f"Battery '{battery}' not found in {mat_path}.\n"
            f"Available keys: {available}"
        )
    
    # Extract cycle data
    counter = 0
    dataset = []

    try:
        cycle_count = len(mat[battery][0, 0]['cycle'][0])
    except Exception as e:
        raise ValueError(
            f"Cannot access cycle data in {battery}.\n"
            f"File structure might be different: {str(e)}"
        )
    
    # Iterate over all cycles
    for i in range(cycle_count):
        row = mat[battery][0, 0]['cycle'][0, i]

        # Only process discharge cycles
        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]

            # Iterate over all time steps in the discharge cycle
            num_samples = len(data[0][0]['Voltage_measured'][0])
            
            for j in range(num_samples):
                dataset.append([
                    counter + 1,  # cycle number (1-indexed)
                    ambient_temperature,
                    capacity,
                    data[0][0]['Voltage_measured'][0][j],
                    data[0][0]['Current_measured'][0][j],
                    data[0][0]['Temperature_measured'][0][j],
                    data[0][0]['Current_load'][0][j],
                    data[0][0]['Voltage_load'][0][j],
                    data[0][0]['Time'][0][j]
                ])

            counter += 1

    if not dataset:
        raise ValueError(
            f"No discharge cycles found for {battery}.\n"
            f"Checked {cycle_count} cycles, found 0 discharge cycles."
        )

    df = pd.DataFrame(dataset, columns=[
        'cycle',
        'ambient_temperature',
        'capacity',
        'voltage_measured',
        'current_measured',
        'temperature_measured',
        'current_load',
        'voltage_load',
        'time'
    ])
    
    return df


def calculate_RUL_capacity_based(df, battery_name, threshold=0.7):
    """
    Compute RUL (Remaining Useful Life) based on capacity threshold.
    
    EOL (End-of-Life) Definition:
        First cycle where capacity <= threshold * initial_capacity
        Per NASA PCoE standard: threshold = 0.7 (70% capacity fade)
    
    RUL Computation:
        RUL = EOL_cycle - current_cycle
    
    Args:
        df (pd.DataFrame): Sample-level discharge data from load_data()
        battery_name (str): Battery identifier (e.g., 'B0005') for logging
        threshold (float): Capacity threshold ratio (default 0.7)
                          Per NASA PCoE standard for 18650 Li-ion cells
    
    Returns:
        tuple: (df_with_rul, metadata_dict)
            - df_with_rul: Same DataFrame with added 'RUL' column
            - metadata_dict: (initial_capacity, eol_threshold, eol_cycle,
                              total_cycles, final_capacity)
    """
    # Aggregate capacity per cycle (take first, constant within discharge)
    cap_per_cycle = df.groupby('cycle')['capacity'].first()
    
    initial_capacity = cap_per_cycle.iloc[0]
    eol_capacity_threshold = threshold * initial_capacity
    
    # Find first cycle where capacity <= threshold
    eol_mask = cap_per_cycle <= eol_capacity_threshold
    
    if eol_mask.any():
        eol_cycle = cap_per_cycle[eol_mask].index[0]
    else:
        # Fallback: use last recorded cycle if threshold not reached
        eol_cycle = cap_per_cycle.index.max()
        final_cap_pct = (cap_per_cycle.iloc[-1] / initial_capacity) * 100
        print(
            f"\n⚠️  WARNING ({battery_name}): "
            f"Capacity never reached {threshold*100}% threshold.\n"
            f"    Minimum capacity: {cap_per_cycle.iloc[-1]:.3f} Ah "
            f"({final_cap_pct:.1f}% of initial).\n"
            f"    Using max cycle ({eol_cycle}) as fallback EOL."
        )
    
    # Add RUL column (RUL = cycles remaining)
    df = df.copy()
    df['RUL'] = eol_cycle - df['cycle']
    df['RUL'] = df['RUL'].clip(lower=0)  # Ensure non-negative
    
    eol_capacity_actual = cap_per_cycle.iloc[-1]
    total_cycles = cap_per_cycle.index.max()
    
    return df, (initial_capacity, eol_capacity_threshold, eol_cycle, 
                total_cycles, eol_capacity_actual)


def preprocess_battery_dataset(mat_path, battery_name, output_dir='./data', 
                               capacity_threshold=0.7):
    """
    Complete preprocessing pipeline for one battery.
    
    Args:
        mat_path (str): Path to .mat file
        battery_name (str): Battery identifier (e.g., 'B0005')
        output_dir (str): Directory to save CSVs (default './data')
        capacity_threshold (float): EOL capacity threshold (default 0.7)
    
    Returns:
        dict: Summary statistics for logging
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {battery_name}")
    print(f"{'='*70}")
    
    # Load raw data
    print(f"✓ Loading {battery_name}.mat...")
    df = load_data(mat_path, battery_name)
    print(f"  → Loaded {len(df):,} samples across {df['cycle'].max()} cycles")
    
    # Compute RUL
    print(f"✓ Computing RUL (capacity threshold: {capacity_threshold*100}%)...")
    df, (q0, q_eol_thresh, eol_cycle, total_cycles, q_final) = \
        calculate_RUL_capacity_based(df, battery_name, threshold=capacity_threshold)
    
    # Print statistics
    fade_pct = (1 - (q_final / q0)) * 100
    print(f"  → Initial capacity (Q0): {q0:.3f} Ah")
    print(f"  → EOL threshold: {q_eol_thresh:.3f} Ah ({capacity_threshold*100}% of Q0)")
    print(f"  → EOL cycle: {eol_cycle}")
    print(f"  → Final capacity: {q_final:.3f} Ah ({100 - fade_pct:.1f}% of initial)")
    print(f"  → Total cycles to EOL: {total_cycles}")
    print(f"  → RUL range: {df['RUL'].max():.0f} → {df['RUL'].min():.0f} cycles")
    
    # Save to CSV
    output_path = Path(output_dir) / f"{battery_name}_discharge.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"  → Rows: {len(df):,} | Columns: {len(df.columns)}")
    
    return {
        'battery': battery_name,
        'total_samples': len(df),
        'total_cycles': total_cycles,
        'initial_capacity': q0,
        'final_capacity': q_final,
        'fade_percentage': fade_pct,
        'eol_cycle': eol_cycle,
        'output_file': str(output_path)
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Preprocess all NASA PCoE batteries (B0005, B0006, B0007, B0018, B0025, B0026).
    """
    # Battery list: 6 batteries (core + accelerated)
    batteries = ["B0005", "B0006", "B0007", "B0018", "B0025", "B0026"]
    
    # Input directory (where .mat files are stored)
    data_input_dir = './battery_data'
    
    results = []
    
    print("\n" + "="*70)
    print("NASA PCoE Li-ion Battery RUL Dataset Preprocessing")
    print("="*70)
    print(f"Processing {len(batteries)} batteries: {', '.join(batteries)}")
    print(f"Input directory: {data_input_dir}")
    print(f"Output directory: ./data")
    print("="*70)
    
    # Process each battery
    for battery in batteries:
        mat_file = Path(data_input_dir) / f"{battery}.mat"
        
        try:
            result = preprocess_battery_dataset(
                mat_path=str(mat_file),
                battery_name=battery,
                output_dir='./data',
                capacity_threshold=0.7  # NASA standard: 70%
            )
            results.append(result)
        
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Skipping {battery}")
        
        except KeyError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Skipping {battery}")
        
        except ValueError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Skipping {battery}")
        
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR processing {battery}:")
            print(f"   {str(e)}")
            traceback.print_exc()
            print(f"   Skipping {battery}")
    
    # Summary report
    if results:
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        print(f"\n✓ Successfully processed {len(results)}/{len(batteries)} batteries")
        print("✓ All RUL CSV files are ready in ./data/ directory")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Verify CSV files:")
        print("   ls -lh data/B*.csv")
        print("\n2. Load and explore data:")
        print("   import pandas as pd")
        print("   df = pd.read_csv('data/B0005_discharge.csv')")
        print("   print(df.head())")
        print("\n3. Choose algorithm (LSTM, CNN-LSTM, or tabular)")
        print("\n4. Apply algorithm-specific preprocessing")
        print("   (normalization, sequencing, feature engineering)")
        print("\n5. Train & validate on B0005/B0006/B0007 (train)")
        print("   Validate on B0018, test on B0025/B0026")
        print("\n" + "="*70)
    else:
        print("\n" + "="*70)
        print("❌ NO BATTERIES WERE SUCCESSFULLY PROCESSED")
        print("="*70)
        print("Troubleshooting:")
        print("1. Check that battery_data/ directory exists")
        print("2. Check that all .mat files are in battery_data/")
        print("3. Verify file names: B0005.mat, B0006.mat, etc.")
        print("4. Run: ls battery_data/B*.mat")
        print("="*70)


if __name__ == "__main__":
    main()
