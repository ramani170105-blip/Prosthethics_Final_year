import os
import gc
import torch
import numpy as np
from datasets.gigadb_loader import load_gigadb_subject, DATASET_ROOT
from preprocessing.signal_processor import preprocess_eeg, preprocess_emg
from run_transformer import train_transformer
import glob

def get_all_subject_ids():
    search_path_dirs = os.path.join(DATASET_ROOT, "sub*")
    downloaded_dirs = [p for p in glob.glob(search_path_dirs) if os.path.isdir(p)]
    sub_ids_from_dirs = [int(p.split('sub')[-1]) for p in downloaded_dirs if p.split('sub')[-1].isdigit()]
    
    # Also detect subjects in preprocessed cache
    cache_path = os.path.join(DATASET_ROOT, "preprocessed_cache", "sub*_eeg.npy")
    cached_files = glob.glob(cache_path)
    sub_ids_from_cache = [int(os.path.basename(p).split('sub')[-1].split('_')[0]) for p in cached_files if os.path.basename(p).split('sub')[-1].split('_')[0].isdigit()]
    
    return sorted(list(set(sub_ids_from_dirs + sub_ids_from_cache)))

def main():
    print("==================================================")
    print("   MEMORY-SAFE ADAPTIVE BCI BENCHMARK           ")
    print("==================================================")
    
    # 1. Detect all downloaded subjects
    all_subject_ids = get_all_subject_ids()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        target = int(sys.argv[1])
        if target in all_subject_ids:
            all_subject_ids = [target]
            print(f"\n=> Single Subject Mode: Targeting Subject {target} only!")
        else:
            print(f"\n⚠️ Subject {target} not found in {all_subject_ids}. Exiting.")
            return
    else:
        print(f"\n=> Found {len(all_subject_ids)} subjects on local disk: {all_subject_ids}\n")
    
    # 2. Loop through one subject at a time to save RAM!
    for sub_id in all_subject_ids:
        print(f"\n==================================================")
        print(f"   ADAPTING AI TO SUBJECT {sub_id} (RAM-SAFE)    ")
        print(f"==================================================")
        
        try:
            cache_dir = "datasets/preprocessed_cache"
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
                # Step A: Load Raw Data (1.9 GB)
                print(f"Loading raw data for Subject {sub_id}...")
                X_eeg_raw, X_emg_raw, y, sfreq = load_gigadb_subject(sub_id, task="multigrasp", condition="realMove")
                
                # Step B: Filter Data
                print("Applying biological filters...")
                X_eeg = preprocess_eeg(X_eeg_raw, sfreq)
                X_emg = preprocess_emg(X_emg_raw, sfreq)
                
                # Delete raw data to free RAM before training
                del X_eeg_raw, X_emg_raw
                gc.collect()
                
                # Save cache for next time
                print(f"Caching Subject {sub_id} to disk...")
                np.save(eeg_cache, X_eeg)
                np.save(emg_cache, X_emg)
                np.save(y_cache, y)
            
            # Step C: Few-Shot Augmentation & Train/Val/Test Split
            from sklearn.model_selection import train_test_split
            
            # Ensure NO classes are dropped! Augment rare classes (e.g. Class 13 & 14) so they have enough trials for stratified split
            unique_classes, counts = np.unique(y, return_counts=True)
            min_samples = 15
            augmented_eeg, augmented_emg, augmented_y = [X_eeg], [X_emg], [y]
            
            for cls, count in zip(unique_classes, counts):
                if count < min_samples:
                    deficit = min_samples - count
                    cls_mask = (y == cls)
                    cls_eeg = X_eeg[cls_mask]
                    cls_emg = X_emg[cls_mask]
                    
                    idx = np.random.choice(len(cls_eeg), size=deficit, replace=True)
                    synth_eeg = cls_eeg[idx].copy()
                    synth_emg = cls_emg[idx].copy()
                    
                    # 1. Subtle Gaussian noise (2% std)
                    synth_eeg += np.random.normal(0, 0.02 * (np.std(cls_eeg) + 1e-8), synth_eeg.shape)
                    synth_emg += np.random.normal(0, 0.02 * (np.std(cls_emg) + 1e-8), synth_emg.shape)
                    
                    # 2. Random temporal shift (jitter by -2 to +2 steps along time axis)
                    shift = np.random.randint(-2, 3)
                    if shift != 0:
                        synth_eeg = np.roll(synth_eeg, shift, axis=-1)
                        synth_emg = np.roll(synth_emg, shift, axis=-1)
                        
                    # 3. Amplitude scaling (0.95 to 1.05)
                    scale = np.random.uniform(0.95, 1.05)
                    synth_eeg *= scale
                    synth_emg *= scale
                    
                    augmented_eeg.append(synth_eeg)
                    augmented_emg.append(synth_emg)
                    augmented_y.append(np.full(deficit, cls))
                    print(f"   [Few-Shot Augmentation] Class {cls} inflated: {count} -> {min_samples} trials (zero classes dropped!)")
            
            X_eeg = np.concatenate(augmented_eeg, axis=0)
            X_emg = np.concatenate(augmented_emg, axis=0)
            y = np.concatenate(augmented_y, axis=0)
            
            unique_labels = np.unique(y)
            print(f"   [Model Architecture] Configured to learn ALL {len(unique_labels)} classes: {list(unique_labels)}")
            np.save(f"best_bci_model_sub{sub_id}_classes.npy", unique_labels)
            
            label_map = {val: idx for idx, val in enumerate(unique_labels)}
            y_mapped = np.array([label_map[val] for val in y])
            
            # Split 15% Test
            X_e_temp, X_e_te, X_m_temp, X_m_te, y_temp, y_te = train_test_split(
                X_eeg, X_emg, y_mapped, test_size=0.15, random_state=42, stratify=y_mapped
            )
            
            # Split remaining 85% into Train (70%) and Val (15%)
            X_e_tr, X_e_val, X_m_tr, X_m_val, y_tr, y_val = train_test_split(
                X_e_temp, X_m_temp, y_temp, test_size=0.1764, random_state=42, stratify=y_temp
            )
            
            print(f"Training Transformer on Subject {sub_id} with ALL {len(unique_labels)} classes...")
            train_transformer(
                sub_id, X_e_tr, X_e_val, X_e_te, X_m_tr, X_m_val, X_m_te, y_tr, y_val, y_te
            )
            
            # Step D: Delete everything from RAM so next subject can load!
            del X_eeg, X_emg, y, y_mapped, X_e_temp, X_m_temp, y_temp
            del X_e_tr, X_e_val, X_e_te, X_m_tr, X_m_val, X_m_te, y_tr, y_val, y_te
            gc.collect()
            print(f"Subject {sub_id} complete! Cleared from memory.")
            
        except Exception as e:
            print(f"❌ Failed on Subject {sub_id}: {e}")

if __name__ == "__main__":
    main()
