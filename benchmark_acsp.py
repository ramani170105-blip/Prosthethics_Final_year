import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Import our custom logic
from main_setup import load_or_preprocess_data
from run_fusion_pipeline import extract_emg_features
from fusion.adaptive_fusion_engine import AdaptiveFusionEngine
from features.adaptive_csp import AdaptiveCSP

def benchmark_acsp():
    print("==================================================")
    print("       TESTING FIXED ADAPTIVE CSP (ACSP)          ")
    print("==================================================")
    
    # 1. Load Clean Data
    X_eeg, X_emg, y, sfreq = load_or_preprocess_data()
    
    # 2. Train / Test Split
    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        X_eeg, X_emg, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Extract Features
    print("\nExtracting EMG Time-Domain Features...")
    emg_feats_tr = extract_emg_features(X_emg_tr)
    emg_feats_te = extract_emg_features(X_emg_te)
    
    print("Extracting EEG Features using FIXED ACSP...")
    # Initialize our fixed ACSP with 6 spatial filters per class (3 top, 3 bottom)
    acsp = AdaptiveCSP(n_components=6, fs=sfreq)
    
    # Fit on training data and extract features
    acsp.fit(X_eeg_tr, y_tr)
    eeg_feats_tr = acsp.transform(X_eeg_tr)
    
    print("-> Simulating Smart Real-Time ACSP Adaptation on Test Set...")
    eeg_feats_te = []
    
    # Train a baseline EMG model just to give pseudo-labels to ACSP
    emg_scaler = StandardScaler()
    emg_tr_scaled = emg_scaler.fit_transform(emg_feats_tr)
    lda_emg = LinearDiscriminantAnalysis().fit(emg_tr_scaled, y_tr)
    
    adapt_count = 0
    skip_count = 0
    
    for i in range(len(X_eeg_te)):
        epoch_eeg = np.expand_dims(X_eeg_te[i], axis=0)
        epoch_emg = np.expand_dims(X_emg_te[i], axis=0)
        
        # 1. Extract feature for this single epoch using current spatial filters
        feat = acsp.transform(epoch_eeg)[0]
        eeg_feats_te.append(feat)
        
        # 2. THE FIX (SMART TEACHER): Get probabilities instead of blindly trusting
        emg_feat_scaled = emg_scaler.transform(extract_emg_features(epoch_emg))
        probs = lda_emg.predict_proba(emg_feat_scaled)[0]
        max_prob = np.max(probs)
        pseudo_label = lda_emg.classes_[np.argmax(probs)]
        
        # 3. Only adapt if the EMG model is highly confident (>85%)
        if max_prob > 0.85:
            acsp.adapt(epoch_eeg, pseudo_label)
            adapt_count += 1
        else:
            skip_count += 1
            
    eeg_feats_te = np.array(eeg_feats_te)
    print(f"   Adapted on {adapt_count} high-confidence trials. Skipped {skip_count} weak trials.")
    
    # 4. Standard Scaling (No PCA needed for ACSP because features are already optimized)
    print("\nStandard Scaling...")
    eeg_scaler = StandardScaler()
    X_eeg_tr_s = eeg_scaler.fit_transform(eeg_feats_tr)
    X_eeg_te_s = eeg_scaler.transform(eeg_feats_te)
    
    X_emg_tr_s = emg_tr_scaled
    X_emg_te_s = emg_scaler.transform(emg_feats_te)
    
    # 5. Build Adaptive Fusion Datasets
    print("Building Adaptive Fusion feature sets...")
    eeg_q = np.std(X_eeg_tr_s) 
    emg_q = np.std(X_emg_tr_s)
    
    fusion_tr = AdaptiveFusionEngine(X_eeg_tr_s, X_emg_tr_s, eeg_quality=eeg_q, emg_quality=emg_q)
    fusion_te = AdaptiveFusionEngine(X_eeg_te_s, X_emg_te_s, eeg_quality=eeg_q, emg_quality=emg_q)
    
    X_fused_tr = fusion_tr.fuse().to_numpy()
    X_fused_te = fusion_te.fuse().to_numpy()
    
    # 6. Train Final LDA Classifiers
    lda_eeg = LinearDiscriminantAnalysis().fit(X_eeg_tr_s, y_tr)
    lda_fused = LinearDiscriminantAnalysis().fit(X_fused_tr, y_tr)
    
    acc_eeg = accuracy_score(y_te, lda_eeg.predict(X_eeg_te_s))
    acc_emg = accuracy_score(y_te, lda_emg.predict(X_emg_te_s))
    acc_fused = accuracy_score(y_te, lda_fused.predict(X_fused_te))
    
    print("\n==================================================")
    print(f"Random Guessing Baseline : {1/len(np.unique(y)):.2%}")
    print(f"EMG-Only Accuracy        : {acc_emg:.2%}")
    print(f"EEG-Only (ACSP) Accuracy : {acc_eeg:.2%}")
    print(f"FUSED ACCURACY (NOVELTY) : {acc_fused:.2%}")
    print("==================================================")

if __name__ == "__main__":
    benchmark_acsp()
