import os
import shutil
import glob
import numpy as np
import zipfile
import urllib.request
import ssl

# Fix for macOS Python SSL Certificate Error
ssl._create_default_https_context = ssl._create_unverified_context

from datasets.gigadb_loader import load_gigadb_subject
from preprocessing.signal_processor import preprocess_eeg, preprocess_emg

# =============================================================================
# STEP 1: PASTE YOUR ZENODO DOWNLOAD LINKS HERE!
# We found the exact Zenodo Record IDs for all 5 parts of the Jeong2020 dataset!
# The script will now automatically build the URLs for all 25 subjects.
# =============================================================================
SUBJECT_ZENODO_URLS = {}

# Part 1 (Subjects 1-5): Record 19021436
# Part 2 (Subjects 6-10): Record 19021439
# Part 3 (Subjects 11-15): Record 19021438
# Part 4 (Subjects 16-20): Record 19021435
# Part 5 (Subjects 21-25): Record 19021437

record_map = {
    19021436: range(4, 6),   # We skip 1,2,3 since you already processed them!
    19021439: range(6, 11),
    19021438: range(11, 16),
    19021435: range(16, 21),
    19021437: range(21, 26)
}

for record_id, subjects in record_map.items():
    for sub in subjects:
        SUBJECT_ZENODO_URLS[sub] = f"https://zenodo.org/records/{record_id}/files/sub{sub}.zip?download=1"

DATASET_ROOT = "datasets"
CACHE_DIR = os.path.join(DATASET_ROOT, "preprocessed_cache")

def download_with_progress(url, filename):
    print(f"-> 📥 Downloading {filename} from Zenodo...")
    urllib.request.urlretrieve(url, filename)

def run_rolling_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    all_X_eeg, all_X_emg, all_y, all_sub_ids = [], [], [], []
    
    if len(SUBJECT_ZENODO_URLS) == 0:
        print("❌ Error: Please open rolling_cache_builder.py and add your Zenodo links!")
        return

    for sub_id, url in SUBJECT_ZENODO_URLS.items():
        print(f"\n==================================================")
        print(f"   ROLLING CACHE: SUBJECT {sub_id} ")
        print(f"==================================================")
        
        zip_path = os.path.join(DATASET_ROOT, f"sub{sub_id}.zip")
        temp_dir = f"datasets/temp_extract_sub{sub_id}"
        override_dir = os.path.join(temp_dir, f"sub{sub_id}")
        
        # 1. Download directly from Zenodo
        try:
            download_with_progress(url, zip_path)
        except Exception as e:
            print(f"❌ Failed to download Subject {sub_id} from Zenodo: {e}")
            continue
            
        # 2. Extract
        print(f"-> 📦 Extracting {zip_path} to Mac SSD...")
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # 3. Preprocess into RAM
        print("-> 🧠 Preprocessing Brainwaves into RAM...")
        try:
            X_eeg_raw, X_emg_raw, y, sfreq = load_gigadb_subject(
                subject_id=sub_id, 
                task="multigrasp", 
                condition="realMove",
                override_data_dir=override_dir
            )
            X_eeg = preprocess_eeg(X_eeg_raw, sfreq)
            X_emg = preprocess_emg(X_emg_raw, sfreq)
            
            all_X_eeg.append(X_eeg)
            all_X_emg.append(X_emg)
            all_y.append(y)
            all_sub_ids.append(np.full(len(y), sub_id))
            print("✅ Subject processed successfully!")
            
        except Exception as e:
            print(f"❌ Failed to process subject {sub_id}: {e}")
            
        # 4. DELETE RAW FILES INSTANTLY TO SAVE SPACE
        print(f"-> 🗑️ Deleting massive raw files for Subject {sub_id}...")
        shutil.rmtree(temp_dir)
        os.remove(zip_path)
        print(f"🧹 Mac Hard Drive space recovered.")

    # 5. Save the Master Cache
    if len(all_y) > 0:
        print("\n==================================================")
        print(f"💾 Saving compressed Master Cache...")
        X_eeg_final = np.concatenate(all_X_eeg, axis=0)
        X_emg_final = np.concatenate(all_X_emg, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        sub_ids_final = np.concatenate(all_sub_ids, axis=0)
        
        num_subs = len(SUBJECT_ZENODO_URLS)
        np.save(os.path.join(CACHE_DIR, f"X_eeg_rolling_{num_subs}_subs.npy"), X_eeg_final)
        np.save(os.path.join(CACHE_DIR, f"X_emg_rolling_{num_subs}_subs.npy"), X_emg_final)
        np.save(os.path.join(CACHE_DIR, f"y_rolling_{num_subs}_subs.npy"), y_final)
        np.save(os.path.join(CACHE_DIR, f"sub_ids_rolling_{num_subs}_subs.npy"), sub_ids_final)
        print("✅ Rolling Cache Complete! You can now load this tiny file instantly in your training scripts.")
        
if __name__ == "__main__":
    run_rolling_cache()
