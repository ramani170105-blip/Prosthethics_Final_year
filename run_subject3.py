import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from main_setup import load_or_preprocess_data
from run_fusion_pipeline import extract_eeg_features, extract_emg_features
from fusion.adaptive_fusion_engine import AdaptiveFusionEngine
from models.transformer_fusion import MultiModalCrossAttention


def run_advanced_ml(X_eeg, X_emg, y, sfreq):
    """Classical ML using the original proven pipeline:
    Band Power → PCA → AdaptiveFusionEngine → LDA/SVM/RF
    """
    unique_labels = np.unique(y)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y_mapped = np.array([label_map[l] for l in y])

    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        X_eeg, X_emg, y_mapped, test_size=0.3, random_state=42
    )

    eeg_feats_tr = extract_eeg_features(X_eeg_tr, sfreq)
    eeg_feats_te = extract_eeg_features(X_eeg_te, sfreq)
    emg_feats_tr = extract_emg_features(X_emg_tr)
    emg_feats_te = extract_emg_features(X_emg_te)

    # Scale + PCA (the approach that gave 70%+ LDA)
    eeg_scaler = StandardScaler()
    emg_scaler = StandardScaler()
    X_eeg_tr_s = eeg_scaler.fit_transform(eeg_feats_tr)
    X_eeg_te_s = eeg_scaler.transform(eeg_feats_te)
    X_emg_tr_s = emg_scaler.fit_transform(emg_feats_tr)
    X_emg_te_s = emg_scaler.transform(emg_feats_te)

    pca = PCA(n_components=min(30, X_eeg_tr_s.shape[1], X_eeg_tr_s.shape[0] - 1))
    X_eeg_tr_pca = pca.fit_transform(X_eeg_tr_s)
    X_eeg_te_pca = pca.transform(X_eeg_te_s)

    # Quality from raw scaled features (gives natural ~3:1 EEG:EMG ratio)
    eeg_q = np.std(X_eeg_tr_pca)
    emg_q = np.std(X_emg_tr_s)

    fusion_tr = AdaptiveFusionEngine(X_eeg_tr_pca, X_emg_tr_s, eeg_quality=eeg_q, emg_quality=emg_q)
    fusion_te = AdaptiveFusionEngine(X_eeg_te_pca, X_emg_te_s, eeg_quality=eeg_q, emg_quality=emg_q)

    X_train = fusion_tr.fuse().to_numpy()
    X_test  = fusion_te.fuse().to_numpy()

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_tr)
    lda_acc = lda.score(X_test, y_te)

    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_tr)
    svm_acc = svm.score(X_test, y_te)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_tr)
    rf_acc = rf.score(X_test, y_te)

    return lda_acc, svm_acc, rf_acc


def train_transformer(X_eeg, X_emg, y, num_epochs=100, batch_size=256, save_path='best_bci_model_sub3.pth'):
    """Trains the Multi-Modal Cross-Attention Transformer.
    Fixed Epoch Training with CPU-PyTorch Dynamic Data Augmentation.
    """
    print("-> Training Multi-Modal Transformer (CPU-PyTorch Accelerated Augmentation)...")

    # --- HOTFIX: Remove anomaly classes with <2 samples before Stratified Split ---
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 2]
    if len(valid_classes) < len(unique_classes):
        valid_mask = np.isin(y, valid_classes)
        X_eeg, X_emg, y = X_eeg[valid_mask], X_emg[valid_mask], y[valid_mask]
        print(f"   [Warning] Dropped {len(unique_classes) - len(valid_classes)} anomaly classes with <2 samples to allow stratified splitting.")

    X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te = train_test_split(
        X_eeg, X_emg, y, test_size=0.2, random_state=42, stratify=y
    )

    unique_labels = np.unique(y)
    label_map = {val: idx for idx, val in enumerate(unique_labels)}
    y_tr_mapped  = np.array([label_map[val] for val in y_tr])
    y_te_mapped  = np.array([label_map[val] for val in y_te])

    # --- ADAPTIVE OVERSAMPLING & NOISE AUGMENTATION ---
    unique_classes, class_counts_arr = np.unique(y_tr_mapped, return_counts=True)
    max_count = np.max(class_counts_arr)
    
    balanced_X_eeg, balanced_X_emg, balanced_y = [], [], []
    noise_level = 0.05
    
    for cls in unique_classes:
        cls_mask = (y_tr_mapped == cls)
        cls_X_eeg = X_eeg_tr[cls_mask]
        cls_X_emg = X_emg_tr[cls_mask]
        cls_count = len(cls_X_eeg)
        
        balanced_X_eeg.append(cls_X_eeg)
        balanced_X_emg.append(cls_X_emg)
        balanced_y.append(np.full(cls_count, cls))
        
        deficit = max_count - cls_count
        if deficit > 0:
            idx = np.random.choice(cls_count, size=deficit, replace=True)
            synth_X_eeg = cls_X_eeg[idx].copy()
            synth_X_emg = cls_X_emg[idx].copy()
            
            eeg_noise = np.random.normal(0, noise_level * np.std(cls_X_eeg), synth_X_eeg.shape)
            emg_noise = np.random.normal(0, noise_level * np.std(cls_X_emg), synth_X_emg.shape)
            synth_X_eeg += eeg_noise
            synth_X_emg += emg_noise
            
            balanced_X_eeg.append(synth_X_eeg)
            balanced_X_emg.append(synth_X_emg)
            balanced_y.append(np.full(deficit, cls))
            
    X_eeg_tr_balanced = np.concatenate(balanced_X_eeg, axis=0)
    X_emg_tr_balanced = np.concatenate(balanced_X_emg, axis=0)
    y_tr_mapped_balanced = np.concatenate(balanced_y, axis=0)
    
    shuffle_idx = np.random.permutation(len(y_tr_mapped_balanced))
    X_eeg_tr_balanced = X_eeg_tr_balanced[shuffle_idx]
    X_emg_tr_balanced = X_emg_tr_balanced[shuffle_idx]
    y_tr_mapped_balanced = y_tr_mapped_balanced[shuffle_idx]
    
    print(f"   [Data Balancing] Inflated training set from {len(y_tr_mapped)} to {len(y_tr_mapped_balanced)} perfectly balanced trials!")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   [Hardware: {device}]")

    # --- ULTRA FAST VRAM CACHING ---
    # Push the entire dataset to the GPU instantly to bypass the CPU-to-GPU bottleneck!
    X_eeg_tr_t  = torch.tensor(X_eeg_tr_balanced,  dtype=torch.float32).to(device)
    X_emg_tr_t  = torch.tensor(X_emg_tr_balanced,  dtype=torch.float32).to(device)
    y_tr_t      = torch.tensor(y_tr_mapped_balanced, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_eeg_tr_t, X_emg_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    X_eeg_te_t  = torch.tensor(X_eeg_te,  dtype=torch.float32).to(device)
    X_emg_te_t  = torch.tensor(X_emg_te,  dtype=torch.float32).to(device)
    y_te_t      = torch.tensor(y_te_mapped, dtype=torch.long).to(device)
    
    model = MultiModalCrossAttention(num_classes=len(unique_labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    max_epochs = 1000
    patience = 50
    best_train_loss = float('inf')
    epochs_no_impr = 0
    loss_history = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for b_eeg, b_emg, b_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(b_eeg, b_emg), b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_train_loss - 1e-4:
            best_train_loss = avg_loss
            epochs_no_impr = 0
            torch.save(model.state_dict(), save_path)
            print(f"   [Adaptive Learning] Epoch {epoch+1:03d} -> New Best Loss: {avg_loss:.4f} 🚀 (Model Saved)")
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"   ✅ Adaptive Stop triggered at epoch {epoch+1} (Loss Converged at {avg_loss:.4f})")
                break

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(X_eeg_te_t, X_emg_te_t, y_te_t), batch_size=batch_size, shuffle=False)
        for b_eeg, b_emg, b_y in test_loader:
            outputs = model(b_eeg, b_emg)
            _, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
            
            all_y_true.extend(b_y.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())

    acc = correct / total
    return acc, loss_history, all_y_true, all_y_pred


def main():
    print("==================================================")
    print("   PERSONALIZED ADAPTIVE BCI BENCHMARK           ")
    print("==================================================")

    X_eeg, X_emg, y, sub_ids, sfreq = load_or_preprocess_data()

    unique_subs = [3]
    print(f"\n=> Testing Subject 3 Only...\n")

    lda_scores, svm_scores, rf_scores, trans_scores = [], [], [], []

    for sub in unique_subs:
        print(f"\n==================================================")
        print(f"   ADAPTING AI TO SUBJECT {sub}")
        print(f"==================================================")

        mask     = (sub_ids == sub)
        X_e_sub  = X_eeg[mask]
        X_m_sub  = X_emg[mask]
        y_sub    = y[mask]

        lda_acc, svm_acc, rf_acc = run_advanced_ml(X_e_sub, X_m_sub, y_sub, sfreq)
        trans_acc, loss_history, y_true, y_pred = train_transformer(X_e_sub, X_m_sub, y_sub, save_path='best_bci_model_sub3.pth')
        lda_scores.append(lda_acc)
        svm_scores.append(svm_acc)
        rf_scores.append(rf_acc)
        trans_scores.append(trans_acc)

        print(f"\n[Subject {sub}] Transformer: {trans_acc:.2%} | LDA: {lda_acc:.2%} | RF: {rf_acc:.2%}")

    print("\n==================================================")
    print("         FINAL PERSONALIZED RESULTS              ")
    print("==================================================")
    print(f"Average LDA Accuracy           : {np.mean(lda_scores):.2%}")
    print(f"Average Random Forest Accuracy : {np.mean(rf_scores):.2%}")
    print(f"Average Non-Linear SVM Accuracy: {np.mean(svm_scores):.2%}")
    print(f"Average Transformer Accuracy   : {np.mean(trans_scores):.2%}")

    best = max(np.mean(lda_scores), np.mean(rf_scores), np.mean(svm_scores), np.mean(trans_scores))
    if best >= 0.85:
        print("\n✅ GOAL ACHIEVED: 85% barrier shattered!")
    else:
        print(f"\n⚠️  Best: {best:.2%}. We need more data or stronger augmentation to reach 85%.")
    print("==================================================")

if __name__ == "__main__":
    main()
