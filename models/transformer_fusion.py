import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeAdaptiveAvgPool1d(nn.Module):
    """ Fixes the Apple Silicon MPS crash for Adaptive Pooling """
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
        
    def forward(self, x):
        if x.device.type == 'mps':
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)

class EEGSpatialTemporalEncoder(nn.Module):
    """ Optimized CNN Frontend for BCI (inspired by EEGNet). Preserves frequencies and spatial patterns. """
    def __init__(self, in_channels, d_model):
        super().__init__()
        # 1. Temporal Convolution (learns frequencies like Alpha/Beta/Gamma)
        self.temporal = nn.Conv1d(in_channels, 32, kernel_size=32, padding=16)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 2. Spatial/Feature Convolution
        self.spatial = nn.Conv1d(32, d_model, kernel_size=8, stride=4, padding=4)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        self.pool = SafeAdaptiveAvgPool1d(16) # Compress sequence length to 16 for Transformer speed
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.elu(self.bn1(self.temporal(x)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.pool(x)
        return x.permute(2, 0, 1) # (seq_len, batch, d_model)

class MultiModalCrossAttention(nn.Module):
    def __init__(self, num_classes=7, d_model=32, n_heads=4):
        super().__init__()
        
        # Lightweight Encoders to prevent overfitting
        self.eeg_encoder = EEGSpatialTemporalEncoder(in_channels=60, d_model=d_model)
        self.emg_encoder = EEGSpatialTemporalEncoder(in_channels=6, d_model=d_model)
        
        # Transformer is heavily regularized
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=0.5)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
            nn.Dropout(0.5), # CRITICAL to prevent memorization
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_eeg, x_emg):
        # 1. Extract Spatial-Temporal Features natively (no destructive avg_pool here)
        eeg_seq = self.eeg_encoder(x_eeg) # (16, batch, d_model)
        emg_seq = self.emg_encoder(x_emg) # (16, batch, d_model)
        
        # 2. Cross-Attention
        fused_seq, _ = self.cross_attn(query=eeg_seq, key=emg_seq, value=emg_seq)
        fused_seq = self.layer_norm(fused_seq + eeg_seq)
        
        # 3. Transformer Temporal Processing
        out_seq = self.transformer_encoder(fused_seq)
        
        # 4. Global Average Pooling over the 16 time steps
        out_pooled = out_seq.mean(dim=0)
        
        return self.classifier(out_pooled)
