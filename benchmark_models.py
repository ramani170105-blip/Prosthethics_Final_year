import warnings
warnings.filterwarnings('ignore') # Ignore sklearn convergence warnings for benchmarking

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Import our custom logic
from main_setup import load_or_preprocess_data
from run_fusion_pipeline import extract_eeg_features, extract_emg_features
from fusion.adaptive_fusion_engine import AdaptiveFusionEngine

def benchmark_models():
    print("==================================================")
    print("       ML BENCHMARKING SUITE (EEG+EMG FUSION)     ")
    print("==================================================")
    
    # 1. Load Clean Data (Uses Cache if available)
    X_eeg, X_emg, y, sfreq = load_or_preprocess_data()
    
    # 2. Extract Features
    print("\nExtracting features (This may take a moment)...")
    eeg_feats = extract_eeg_features(X_eeg, sfreq)
    emg_feats = extract_emg_features(X_emg)
    
    # 3. Train / Test Split
    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        eeg_feats, emg_feats, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Standard Scaling & PCA
    eeg_scaler, emg_scaler = StandardScaler(), StandardScaler()
    X_eeg_tr = eeg_scaler.fit_transform(X_eeg_tr)
    X_eeg_te = eeg_scaler.transform(X_eeg_te)
    X_emg_tr = emg_scaler.fit_transform(X_emg_tr)
    X_emg_te = emg_scaler.transform(X_emg_te)
    
    pca = PCA(n_components=30)
    X_eeg_tr = pca.fit_transform(X_eeg_tr)
    X_eeg_te = pca.transform(X_eeg_te)
    
    # 5. Build Adaptive Fusion Datasets
    print("Building Adaptive Fusion feature sets...")
    eeg_q = np.std(X_eeg_tr) 
    emg_q = np.std(X_emg_tr)
    
    fusion_tr = AdaptiveFusionEngine(X_eeg_tr, X_emg_tr, eeg_quality=eeg_q, emg_quality=emg_q)
    fusion_te = AdaptiveFusionEngine(X_eeg_te, X_emg_te, eeg_quality=eeg_q, emg_quality=emg_q)
    
    X_fused_tr = fusion_tr.fuse().to_numpy()
    X_fused_te = fusion_te.fuse().to_numpy()
    
    # 6. Define Classifiers
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM (Linear)": SVC(kernel="linear"),
        "SVM (RBF)": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    results = []
    
    print("\nStarting Benchmarks...\n")
    print(f"{'Model Name':<25} | {'EEG-Only':<10} | {'EMG-Only':<10} | {'Fused':<10}")
    print("-" * 65)
    
    # 7. Run Benchmarks
    for name, model in models.items():
        # EEG Only
        model.fit(X_eeg_tr, y_tr)
        acc_eeg = accuracy_score(y_te, model.predict(X_eeg_te))
        
        # EMG Only
        model.fit(X_emg_tr, y_tr)
        acc_emg = accuracy_score(y_te, model.predict(X_emg_te))
        
        # Fused
        model.fit(X_fused_tr, y_tr)
        acc_fused = accuracy_score(y_te, model.predict(X_fused_te))
        
        # Print Row
        print(f"{name:<25} | {acc_eeg*100:>8.2f}% | {acc_emg*100:>8.2f}% | {acc_fused*100:>8.2f}%")
        
        results.append({
            "name": name,
            "eeg": acc_eeg,
            "emg": acc_emg,
            "fused": acc_fused
        })
        
    print("-" * 65)
    
    # Find the absolute best Fused model
    best_model = max(results, key=lambda x: x['fused'])
    print(f"\n🏆 BEST PERFORMING FUSION MODEL: {best_model['name']} ({best_model['fused']*100:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    benchmark_models()
