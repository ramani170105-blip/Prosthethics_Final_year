import os
import glob
import zipfile
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# Import modules
from datasets.gigadb_loader import load_gigadb_subject, DATASET_ROOT
from preprocessing.signal_processor import preprocess_eeg, preprocess_emg
from run_fusion_pipeline import extract_eeg_features, extract_emg_features

# Change this path to point to your local datasets folder!
DATASET_ROOT = "datasets"
CACHE_DIR = os.path.join(DATASET_ROOT, "preprocessed_cache")

def load_or_preprocess_data(task="multigrasp", condition="realMove"):
    """
    Checks if preprocessed data for ALL downloaded subjects exists on disk. 
    If not, loads all subjects locally, filters them, and saves them to disk.
    """
    cache_dir = "datasets/preprocessed_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # First, check if individual preprocessed subject caches exist in cache_dir
    cached_subjects = []
    for s in range(1, 26):
        if (os.path.exists(os.path.join(cache_dir, f"sub{s}_eeg.npy")) and
            os.path.exists(os.path.join(cache_dir, f"sub{s}_emg.npy")) and
            os.path.exists(os.path.join(cache_dir, f"sub{s}_y.npy"))):
            cached_subjects.append(s)
            
    if len(cached_subjects) > 0:
        print(f">> CACHE HIT: Found preprocessed data for {len(cached_subjects)} subjects: {cached_subjects}")
        all_eeg, all_emg, all_y, all_ids = [], [], [], []
        for s in cached_subjects:
            eeg_s = np.load(os.path.join(cache_dir, f"sub{s}_eeg.npy"))
            emg_s = np.load(os.path.join(cache_dir, f"sub{s}_emg.npy"))
            y_s = np.load(os.path.join(cache_dir, f"sub{s}_y.npy"))
            all_eeg.append(eeg_s)
            all_emg.append(emg_s)
            all_y.append(y_s)
            all_ids.append(np.full(len(y_s), s))
        X_eeg = np.concatenate(all_eeg, axis=0)
        X_emg = np.concatenate(all_emg, axis=0)
        y = np.concatenate(all_y, axis=0)
        sub_ids = np.concatenate(all_ids, axis=0)
        print(f">> Successfully loaded {len(y)} trials across subjects {cached_subjects} instantly from preprocessed cache!")
        return X_eeg, X_emg, y, sub_ids, 1000.0

    # We look for both unzipped folders and zip files
    search_path_dirs = os.path.join(DATASET_ROOT, "sub*")
    search_path_zips = os.path.join(DATASET_ROOT, "sub*.zip")
    
    downloaded_dirs = [p for p in glob.glob(search_path_dirs) if os.path.isdir(p)]
    downloaded_zips = [p for p in glob.glob(search_path_zips) if p.endswith('.zip')]
    
    def extract_id_from_dir(p): return int(p.split('sub')[-1])
    def extract_id_from_zip(p): return int(p.split('sub')[-1].replace('.zip', ''))
    
    sub_ids_from_dirs = [extract_id_from_dir(p) for p in downloaded_dirs]
    sub_ids_from_zips = [extract_id_from_zip(p) for p in downloaded_zips]
    
    all_subject_ids = list(set(sub_ids_from_dirs + sub_ids_from_zips))
    num_subs = len(all_subject_ids)
    
    if num_subs == 0:
        raise FileNotFoundError(
            f"Could not find any 'sub*' folders or 'sub*.zip' files inside {DATASET_ROOT}."
        )
    
    eeg_cache = os.path.join(cache_dir, f"X_eeg_{num_subs}_subs.npy")
    emg_cache = os.path.join(cache_dir, f"X_emg_{num_subs}_subs.npy")
    y_cache = os.path.join(cache_dir, f"y_{num_subs}_subs.npy")
    ids_cache = os.path.join(cache_dir, f"sub_ids_{num_subs}_subs.npy")
    
    if os.path.exists(eeg_cache) and os.path.exists(emg_cache) and os.path.exists(ids_cache):
        print(f">> CACHE HIT: Loading highly compressed data for {num_subs} subjects from disk...")
        X_eeg = np.load(eeg_cache)
        X_emg = np.load(emg_cache)
        y = np.load(y_cache)
        sub_ids = np.load(ids_cache)
        print(f"Loaded {len(y)} trials instantly.")
        return X_eeg, X_emg, y, sub_ids, 1000.0
        
    print(f">> CACHE MISS: No cache found for {num_subs} subjects. Building it now locally (This may take 90+ minutes)...")
    
    all_X_eeg, all_X_emg, all_y, all_sub_ids = [], [], [], []
    
    for sub_id in sorted(all_subject_ids):
        needs_extraction = sub_id not in sub_ids_from_dirs
        temp_dir = None
        override_dir = None
        
        if needs_extraction:
            zip_path = os.path.join(DATASET_ROOT, f"sub{sub_id}.zip")
            temp_dir = f"datasets/temp_extract_sub{sub_id}"
            override_dir = os.path.join(temp_dir, f"sub{sub_id}")
            
            print(f"--- LOCAL EXTRACTION: Extracting {zip_path} to Mac SSD... ---")
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print("Extraction complete! Streaming into memory...")
            
        try:
            X_eeg_raw, X_emg_raw, y, sfreq = load_gigadb_subject(
                subject_id=sub_id, 
                task=task, 
                condition=condition,
                override_data_dir=override_dir
            )
            
            # Apply biological filtering
            X_eeg = preprocess_eeg(X_eeg_raw, sfreq)
            X_emg = preprocess_emg(X_emg_raw, sfreq)
            
            all_X_eeg.append(X_eeg)
            all_X_emg.append(X_emg)
            all_y.append(y)
            all_sub_ids.append(np.full(len(y), sub_id))
        except Exception as e:
            print(f"Failed to load subject {sub_id}: {e}")
        finally:
            if needs_extraction and temp_dir and os.path.exists(temp_dir):
                print(f"Cleaning up massive temp files for sub{sub_id} to save disk space...")
                shutil.rmtree(temp_dir)
                
    X_eeg_final = np.concatenate(all_X_eeg, axis=0)
    X_emg_final = np.concatenate(all_X_emg, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    sub_ids_final = np.concatenate(all_sub_ids, axis=0)
    
    print(f"Caching aggregated {num_subs}-subject data to local disk...")
    np.save(eeg_cache, X_eeg_final)
    np.save(emg_cache, X_emg_final)
    np.save(y_cache, y_final)
    np.save(ids_cache, sub_ids_final)
    
    return X_eeg_final, X_emg_final, y_final, sub_ids_final, 1000.0

def main():
    print("========================================")
    print(" FINAL PIPELINE SETUP (WITH CACHING)    ")
    print("========================================")
    
    # 1. Load Data (with caching)
    X_eeg, X_emg, y, sfreq = load_or_preprocess_data()
    
    # 2. Extract Features
    print("\nExtracting EEG Bandpower Features...")
    eeg_feats = extract_eeg_features(X_eeg, sfreq)
    print("Extracting EMG Time-Domain Features...")
    emg_feats = extract_emg_features(X_emg)
    
    # 3. Train / Test Split
    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        eeg_feats, emg_feats, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Standard Scaling & PCA
    print("\nStandard Scaling & PCA...")
    eeg_scaler, emg_scaler = StandardScaler(), StandardScaler()
    X_eeg_tr = eeg_scaler.fit_transform(X_eeg_tr)
    X_eeg_te = eeg_scaler.transform(X_eeg_te)
    X_emg_tr = emg_scaler.fit_transform(X_emg_tr)
    X_emg_te = emg_scaler.transform(X_emg_te)
    
    pca = PCA(n_components=30)
    X_eeg_tr = pca.fit_transform(X_eeg_tr)
    X_eeg_te = pca.transform(X_eeg_te)
    
    # 5. Baseline: Train Unimodal Models
    lda_eeg = LDA().fit(X_eeg_tr, y_tr)
    lda_emg = LDA().fit(X_emg_tr, y_tr)
    
    eeg_acc = accuracy_score(y_te, lda_eeg.predict(X_eeg_te))
    emg_acc = accuracy_score(y_te, lda_emg.predict(X_emg_te))
    
    # 6. Adaptive Fusion Engine
    print("\nRunning Adaptive Fusion Engine...")
    eeg_q = np.std(X_eeg_tr) 
    emg_q = np.std(X_emg_tr)
    
    fusion_tr = AdaptiveFusionEngine(X_eeg_tr, X_emg_tr, eeg_quality=eeg_q, emg_quality=emg_q)
    fusion_te = AdaptiveFusionEngine(X_eeg_te, X_emg_te, eeg_quality=eeg_q, emg_quality=emg_q)
    
    X_fused_tr = fusion_tr.fuse().to_numpy()
    X_fused_te = fusion_te.fuse().to_numpy()
    
    # 7. Final Classification
    lda_fused = LDA().fit(X_fused_tr, y_tr)
    fused_acc = accuracy_score(y_te, lda_fused.predict(X_fused_te))
    
    # 8. Results
    print("\n========================================")
    print("             FINAL RESULTS              ")
    print("========================================")
    print(f"Random Guessing Baseline : {1/len(np.unique(y)):.2%}")
    print(f"EEG-Only Accuracy        : {eeg_acc:.2%}")
    print(f"EMG-Only Accuracy        : {emg_acc:.2%}")
    print(f"FUSED ACCURACY (NOVELTY) : {fused_acc:.2%}")
    print("========================================")

if __name__ == "__main__":
    main()
