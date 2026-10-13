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
import time
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from main_setup import load_or_preprocess_data
from run_fusion_pipeline import extract_eeg_features, extract_emg_features
from fusion.adaptive_fusion_engine import AdaptiveFusionEngine
from models.transformer_fusion import MultiModalCrossAttention

import sys
import datetime
import random

def set_seed(seed=42):
    """Enforce determinism across all libraries for consistent accuracy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # Lock random seeds globally

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Start logging automatically
log_filename = f"bci_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = DualLogger(log_filename)
print(f"📄 Terminal output is being automatically saved to: {log_filename}\n")


def run_advanced_ml(X_eeg_tr, X_eeg_te, X_emg_tr, X_emg_te, y_tr, y_te, sfreq):
    """
    Runs the advanced Classical BCI pipeline on a specific train/test split.
    """

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


def train_transformer(sub_id, X_eeg_tr, X_eeg_val, X_eeg_te, X_emg_tr, X_emg_val, X_emg_te, y_tr, y_val, y_te, num_epochs=100, batch_size=256, max_epochs=1000, patience=15):
    """Trains the Multi-Modal Cross-Attention Transformer on a specific Train/Test split."""

    # --- Z-SCORE STANDARDIZATION ---
    eeg_mean = X_eeg_tr.mean(axis=(0, 2), keepdims=True)
    eeg_std = X_eeg_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    emg_mean = X_emg_tr.mean(axis=(0, 2), keepdims=True)
    emg_std = X_emg_tr.std(axis=(0, 2), keepdims=True) + 1e-8

    X_eeg_tr = (X_eeg_tr - eeg_mean) / eeg_std
    X_eeg_val = (X_eeg_val - eeg_mean) / eeg_std
    X_eeg_te = (X_eeg_te - eeg_mean) / eeg_std
    X_emg_tr = (X_emg_tr - emg_mean) / emg_std
    X_emg_val = (X_emg_val - emg_mean) / emg_std
    X_emg_te = (X_emg_te - emg_mean) / emg_std

    unique_labels = np.unique(y_tr)
    
    # --- ADAPTIVE OVERSAMPLING & NOISE AUGMENTATION ---
    unique_classes, class_counts_arr = np.unique(y_tr, return_counts=True)
    max_count = np.max(class_counts_arr)
    
    balanced_X_eeg, balanced_X_emg, balanced_y = [], [], []
    noise_level = 0.05
    
    for cls in unique_classes:
        cls_mask = (y_tr == cls)
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
    
    print(f"   [Data Balancing] Inflated training set from {len(y_tr)} to {len(y_tr_mapped_balanced)} perfectly balanced trials!")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   [Hardware: {device}]")

    X_eeg_tr_t  = torch.tensor(X_eeg_tr_balanced,  dtype=torch.float32).to(device)
    X_emg_tr_t  = torch.tensor(X_emg_tr_balanced,  dtype=torch.float32).to(device)
    y_tr_t      = torch.tensor(y_tr_mapped_balanced, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_eeg_tr_t, X_emg_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    X_eeg_val_t  = torch.tensor(X_eeg_val,  dtype=torch.float32).to(device)
    X_emg_val_t  = torch.tensor(X_emg_val,  dtype=torch.float32).to(device)
    y_val_t      = torch.tensor(y_val, dtype=torch.long).to(device)
    val_loader = DataLoader(TensorDataset(X_eeg_val_t, X_emg_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    X_eeg_te_t  = torch.tensor(X_eeg_te,  dtype=torch.float32).to(device)
    X_emg_te_t  = torch.tensor(X_emg_te,  dtype=torch.float32).to(device)
    y_te_t      = torch.tensor(y_te, dtype=torch.long).to(device)
    
    test_loader = DataLoader(TensorDataset(X_eeg_te_t, X_emg_te_t, y_te_t), batch_size=batch_size, shuffle=False)
    
    model = MultiModalCrossAttention(num_classes=len(unique_labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_test_acc = 0.0
    epochs_no_impr = 0
    loss_history = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for b_eeg, b_emg, b_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(b_eeg, b_emg)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += b_y.size(0)
            train_correct += (predicted == b_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        loss_history.append(avg_loss)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b_eeg, b_emg, b_y in val_loader:
                val_outputs = model(b_eeg, b_emg)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += b_y.size(0)
                val_correct += (val_predicted == b_y).sum().item()
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_acc)

        if epoch_val_acc > best_test_acc + 0.001:
            best_test_acc = epoch_val_acc
            epochs_no_impr = 0
            torch.save(model.state_dict(), f'best_bci_model_sub{sub_id}.pth')
            print(f"      [Epoch {epoch+1:03d}] Loss: {avg_loss:.4f} | Train: {epoch_train_acc:.1%} | Val: {epoch_val_acc:.1%} 🚀 (Saved)")
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"      ✅ Stop triggered at epoch {epoch+1} (Validation Acc plateaued at {best_test_acc:.1%})")
                break

    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for b_eeg, b_emg, b_y in train_loader:
            outputs = model(b_eeg, b_emg)
            _, predicted = torch.max(outputs.data, 1)
            train_total += b_y.size(0)
            train_correct += (predicted == b_y).sum().item()
    train_acc = train_correct / train_total

    # Final Test Set Evaluation & Latency Measurement
    model.eval()
    correct = 0
    total = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        start_time = time.time()
        for b_eeg, b_emg, b_y in test_loader:
            outputs = model(b_eeg, b_emg)
            _, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
            
            all_y_true.extend(b_y.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
            
        end_time = time.time()
        
    latency_ms = ((end_time - start_time) / total) * 1000  # ms per sample

    acc = correct / total
    f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    
    return acc, f1, train_acc, latency_ms, loss_history, all_y_true, all_y_pred


def main():
    print("==================================================")
    print("   PERSONALIZED ADAPTIVE BCI BENCHMARK           ")
    print("==================================================")

    X_eeg, X_emg, y, sub_ids, sfreq = load_or_preprocess_data()

    unique_subs = np.unique(sub_ids)
    print(f"\n=> Testing ALL {len(unique_subs)} Subjects for the Final Run...\n")

    lda_scores, svm_scores, rf_scores, trans_scores = [], [], [], []

    for sub in unique_subs:
        print(f"\n==================================================")
        print(f"   ADAPTING AI TO SUBJECT {sub}")
        print(f"==================================================")

        mask     = (sub_ids == sub)
        X_e_sub  = X_eeg[mask]
        X_m_sub  = X_emg[mask]
        y_sub    = y[mask]

        # Holdout Setup (Train 70% / Val 15% / Test 15%)
        unique_classes, counts = np.unique(y_sub, return_counts=True)
        valid_classes = unique_classes[counts >= 5]
        if len(valid_classes) < len(unique_classes):
            valid_mask = np.isin(y_sub, valid_classes)
            X_e_sub, X_m_sub, y_sub = X_e_sub[valid_mask], X_m_sub[valid_mask], y_sub[valid_mask]
            print(f"   [Warning] Dropped {len(unique_classes) - len(valid_classes)} anomaly classes with <5 samples for Holdout CV.")

        # Map labels robustly for the Subject
        unique_labels = np.unique(y_sub)
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        y_sub_mapped = np.array([label_map[val] for val in y_sub])

        # Split 15% for absolute final testing
        X_e_temp, X_e_te, X_m_temp, X_m_te, y_temp, y_te = train_test_split(
            X_e_sub, X_m_sub, y_sub_mapped, test_size=0.15, random_state=42, stratify=y_sub_mapped
        )
        
        # Split remaining 85% into Train (70% total) and Validation (15% total)
        X_e_tr, X_e_val, X_m_tr, X_m_val, y_tr, y_val = train_test_split(
            X_e_temp, X_m_temp, y_temp, test_size=0.1764, random_state=42, stratify=y_temp
        )
        
        print(f"\n   --- Training Phase (Holdout Method) ---")
        
        lda_acc, svm_acc, rf_acc = run_advanced_ml(X_e_tr, X_e_te, X_m_tr, X_m_te, y_tr, y_te, sfreq)
        trans_acc, f1, train_acc, latency, loss_history, y_true, y_pred = train_transformer(
            sub, X_e_tr, X_e_val, X_e_te, X_m_tr, X_m_val, X_m_te, y_tr, y_val, y_te
        )

        lda_scores.append(lda_acc)
        svm_scores.append(svm_acc)
        rf_scores.append(rf_acc)
        trans_scores.append(trans_acc)

        print(f"\n[Subject {sub} Final Holdout Result]")
        print(f"   Transformer -> Test Acc: {trans_acc:.2%} | F1: {f1:.4f} | Latency: {latency:.2f} ms")
        print(f"   Classical Base -> LDA: {lda_acc:.2%} | RF: {rf_acc:.2%}")

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
