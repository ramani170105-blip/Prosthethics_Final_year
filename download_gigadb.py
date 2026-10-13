"""
download_gigadb.py
==================
Downloads the GigaDB "Intuitive Upper-Extremity Movement" Dataset
(Jeong et al. 2020) using the MOABB library.

This dataset contains:
- 60-channel EEG
- 7-channel EMG
- 25 Subjects
- 11 complex movement classes (e.g. grasp, pinch, open, flex)

The data is saved to `datasets/gigadb/`.
"""

import os
import mne
import numpy as np

# Ensure moabb is installed: pip install moabb
try:
    from moabb.datasets import Jeong2020
except ImportError:
    print("MOABB library not found. Please install it first:")
    print("pip install moabb")
    exit(1)

def download_and_extract_subject(subject_id=1, save_dir="datasets/gigadb"):
    """
    Downloads the dataset for a given subject and saves the raw 
    EEG+EMG files to the workspace for our pipeline to use.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Downloading GigaDB Dataset for Subject {subject_id}")
    print("This may take a few minutes (files are large)...")
    print("=" * 60)
    
    # Initialize the dataset
    dataset = Jeong2020()
    
    # This automatically downloads from Zenodo/GigaDB and caches it
    # We only fetch one subject to start so we can build the pipeline quickly
    data = dataset.get_data(subjects=[subject_id])
    
    subj_data = data[subject_id]
    
    # MOABB structures data as: subject -> session -> run -> mne.io.Raw
    for session_name, runs in subj_data.items():
        for run_name, raw in runs.items():
            
            # Create a filename based on subject, session, and run
            # e.g., datasets/gigadb/S01_session_1_run_1.fif
            filename = f"S{subject_id:02d}_{session_name}_{run_name}.fif"
            save_path = os.path.join(save_dir, filename)
            
            print(f"Saving: {save_path}")
            print(f"  Channels: {len(raw.ch_names)}")
            print(f"  Samples:  {raw.n_times}")
            print(f"  Duration: {raw.n_times / raw.info['sfreq']:.1f} seconds")
            
            # Save the raw data directly to our folder
            raw.save(save_path, overwrite=True)

    print("=" * 60)
    print(f"Download complete! Files saved to: {save_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    # Download Subject 1 and 2 to start testing the fusion pipeline
    for sub in range(1, 26):
        download_and_extract_subject(subject_id=sub)
