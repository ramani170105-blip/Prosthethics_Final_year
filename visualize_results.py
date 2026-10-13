import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

from run_transformer import load_or_preprocess_data, train_transformer

def plot_learning_curve(loss_history, save_path="learning_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, color='teal', linewidth=2.5, label="Training Loss")
    plt.title("Transformer Learning Curve (Adaptive Data Augmentation)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Cross Entropy Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true label) to show percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt=".1f", cmap="mako", 
                cbar_kws={'label': 'Accuracy (%)'}, 
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title("Transformer Confusion Matrix (Subject 2 - ~88%)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Motor Intent", fontsize=14)
    plt.ylabel("True Motor Intent", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    print("Loading data...")
    X_eeg, X_emg, y, sub_ids, sfreq = load_or_preprocess_data()
    
    target_sub = 2
    print(f"Training on Subject {target_sub} to generate final plots...")
    
    sub_mask = (sub_ids == target_sub)
    X_e_sub = X_eeg[sub_mask]
    X_m_sub = X_emg[sub_mask]
    y_sub   = y[sub_mask]
    
    # We only care about the Transformer for the plots
    acc, loss_history, y_true, y_pred = train_transformer(X_e_sub, X_m_sub, y_sub, max_epochs=40, patience=5)
    
    print(f"\nTraining Complete. Accuracy: {acc*100:.2f}%")
    
    plot_learning_curve(loss_history)
    plot_confusion_matrix(y_true, y_pred)
