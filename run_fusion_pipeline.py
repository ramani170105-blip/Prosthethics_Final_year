import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.signal import welch

# Import the new loader and your existing fusion engine
from datasets.gigadb_loader import load_gigadb_subject
from fusion.adaptive_fusion_engine import AdaptiveFusionEngine

# ---------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------
def extract_eeg_features(X_eeg, sfreq=1000):
    """ Extracts band power (Delta, Theta, Alpha, Beta, Gamma) from EEG epochs. """
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    features = []
    
    for epoch in X_eeg:
        epoch_feats = []
        for ch_data in epoch:
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=int(sfreq))
            for fmin, fmax in bands.values():
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                power = np.mean(psd[idx])
                epoch_feats.append(np.log(power + 1e-10))
        features.append(epoch_feats)
        
    return np.array(features)

def extract_emg_features(X_emg):
    """ Extracts standard time-domain features (RMS, MAV, VAR) from EMG epochs. """
    features = []
    for epoch in X_emg:
        epoch_feats = []
        for ch_data in epoch:
            rms = np.sqrt(np.mean(ch_data**2))
            mav = np.mean(np.abs(ch_data))
            var = np.var(ch_data)
            epoch_feats.extend([rms, mav, var])
        features.append(epoch_feats)
    return np.array(features)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print("========================================")
    print(" STARTING NOVEL EEG+EMG FUSION PIPELINE ")
    print("========================================")
    
    # 1. Load Data
    X_eeg, X_emg, y, sfreq = load_gigadb_subject(task="multigrasp", condition="realMove")
    
    # 2. Extract Features
    print("\nExtracting EEG Bandpower Features...")
    eeg_feats = extract_eeg_features(X_eeg, sfreq)
    print("Extracting EMG Time-Domain Features...")
    emg_feats = extract_emg_features(X_emg)
    
    # 3. Train / Test Split
    # We do a standard 70/30 split to prove the concept works
    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        eeg_feats, emg_feats, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Preprocessing (Scaling & PCA)
    print("\nPreprocessing (Scaling & PCA)...")
    eeg_scaler = StandardScaler()
    emg_scaler = StandardScaler()
    
    X_eeg_tr = eeg_scaler.fit_transform(X_eeg_tr)
    X_eeg_te = eeg_scaler.transform(X_eeg_te)
    X_emg_tr = emg_scaler.fit_transform(X_emg_tr)
    X_emg_te = emg_scaler.transform(X_emg_te)
    
    pca = PCA(n_components=30) # Reduce 300 EEG features to 30 to prevent overfitting
    X_eeg_tr = pca.fit_transform(X_eeg_tr)
    X_eeg_te = pca.transform(X_eeg_te)
    
    # 5. Baseline: Train Unimodal Models
    lda_eeg = LDA().fit(X_eeg_tr, y_tr)
    lda_emg = LDA().fit(X_emg_tr, y_tr)
    
    eeg_acc = accuracy_score(y_te, lda_eeg.predict(X_eeg_te))
    emg_acc = accuracy_score(y_te, lda_emg.predict(X_emg_te))
    
    # 6. Adaptive Fusion
    print("\nInitializing Adaptive Fusion Engine...")
    # Using simplistic proxy for quality: standard deviation across features for that modality
    # In a real hardware system, this would be electrode impedance.
    eeg_q = np.std(X_eeg_tr) 
    emg_q = np.std(X_emg_tr)
    
    # Fuse Train Data
    fusion_tr = AdaptiveFusionEngine(X_eeg_tr, X_emg_tr, eeg_quality=eeg_q, emg_quality=emg_q)
    X_fused_tr = fusion_tr.fuse().to_numpy()
    
    # Fuse Test Data
    fusion_te = AdaptiveFusionEngine(X_eeg_te, X_emg_te, eeg_quality=eeg_q, emg_quality=emg_q)
    X_fused_te = fusion_te.fuse().to_numpy()
    
    # 7. Train Fused Model
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
