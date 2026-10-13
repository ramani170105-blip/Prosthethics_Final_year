import os
import gc
import torch
import numpy as np
import glob
import sys
sys.path.append("/content/Code")
from preprocessing.signal_processor import preprocess_eeg, preprocess_emg
from run_transformer import train_transformer
import datasets.gigadb_loader as gigadb_loader

# Override paths for Google Colab
gigadb_loader.DATASET_ROOT = "/content/drive/MyDrive/Prosthetics_Data"
DATASET_ROOT = "/content/drive/MyDrive/Prosthetics_Data"

def get_all_subject_ids():
    search_path_dirs = os.path.join(DATASET_ROOT, "sub*")
    downloaded_dirs = [p for p in glob.glob(search_path_dirs) if os.path.isdir(p)]
    return sorted(list(set([int(p.split('sub')[-1]) for p in downloaded_dirs])))

def main():
    print("==================================================")
    print("   MEMORY-SAFE COLAB TRAINING (ALL SUBJECTS)    ")
    print("==================================================")
    
    all_subject_ids = get_all_subject_ids()
    print(f"=> Found {len(all_subject_ids)} subjects in Google Drive: {all_subject_ids}")
    
    for sub_id in all_subject_ids:
        print(f"\n==================================================")
        print(f"   ADAPTING AI TO SUBJECT {sub_id} (RAM-SAFE)    ")
        print(f"==================================================")
        
        try:
            cache_dir = os.path.join(DATASET_ROOT, "preprocessed_cache")
            os.makedirs(cache_dir, exist_ok=True)
            eeg_cache = os.path.join(cache_dir, f"sub{sub_id}_eeg.npy")
            emg_cache = os.path.join(cache_dir, f"sub{sub_id}_emg.npy")
            y_cache = os.path.join(cache_dir, f"sub{sub_id}_y.npy")
            
            if os.path.exists(eeg_cache) and os.path.exists(emg_cache) and os.path.exists(y_cache):
                print(f">> CACHE HIT: Loading Subject {sub_id} from disk instantly...")
                X_eeg = np.load(eeg_cache)
                X_emg = np.load(emg_cache)
                y = np.load(y_cache)
            else:
                # Step A: Load Raw Data
                print(f"Loading raw data for Subject {sub_id}...")
                X_eeg_raw, X_emg_raw, y, sfreq = gigadb_loader.load_gigadb_subject(sub_id, task="multigrasp", condition="realMove")
                
                # Step B: Filter Data
                print("Applying biological filters...")
                X_eeg = preprocess_eeg(X_eeg_raw, sfreq)
                X_emg = preprocess_emg(X_emg_raw, sfreq)
                
                del X_eeg_raw, X_emg_raw
                gc.collect()
                
                print(f"Caching Subject {sub_id} to disk...")
                np.save(eeg_cache, X_eeg)
                np.save(emg_cache, X_emg)
                np.save(y_cache, y)
            
            # Step C: Train/Val/Test Split
            from sklearn.model_selection import train_test_split
            
            unique_classes, counts = np.unique(y, return_counts=True)
            valid_classes = unique_classes[counts >= 5]
            if len(valid_classes) < len(unique_classes):
                valid_mask = np.isin(y, valid_classes)
                X_eeg, X_emg, y = X_eeg[valid_mask], X_emg[valid_mask], y[valid_mask]
            
            unique_labels = np.unique(y)
            label_map = {val: idx for idx, val in enumerate(unique_labels)}
            y_mapped = np.array([label_map[val] for val in y])
            
            X_e_temp, X_e_te, X_m_temp, X_m_te, y_temp, y_te = train_test_split(
                X_eeg, X_emg, y_mapped, test_size=0.15, random_state=42, stratify=y_mapped
            )
            
            X_e_tr, X_e_val, X_m_tr, X_m_val, y_tr, y_val = train_test_split(
                X_e_temp, X_m_temp, y_temp, test_size=0.1764, random_state=42, stratify=y_temp
            )
            
            print(f"Training Transformer on Subject {sub_id}...")
            
            # CRITICAL: Change directory to Google Drive so the .pth file and logs sync to the Mac!
            os.chdir("/content/drive/MyDrive/Prosthetics_Data")
            
            train_transformer(
                sub_id, X_e_tr, X_e_val, X_e_te, X_m_tr, X_m_val, X_m_te, y_tr, y_val, y_te
            )
            
            # Change back to the code directory for any further processing
            os.chdir("/content/Code")
            
            del X_eeg, X_emg, y, y_mapped, X_e_temp, X_m_temp, y_temp
            del X_e_tr, X_e_val, X_e_te, X_m_tr, X_m_val, X_m_te, y_tr, y_val, y_te
            gc.collect()
            print(f"Subject {sub_id} complete! Cleared from memory.")
            
        except Exception as e:
            print(f"❌ Failed on Subject {sub_id}: {e}")

if __name__ == "__main__":
    main()
