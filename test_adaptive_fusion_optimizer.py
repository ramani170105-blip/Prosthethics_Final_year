from fusion.adaptive_fusion_engine import AdaptiveFusionEngine
from optimization.adaptive_fusion_optimizer import AdaptiveFusionOptimizer


fusion = AdaptiveFusionEngine(
    eeg_features=X_eeg,
    emg_features=X_emg,
    eeg_quality=eeg_quality,
    emg_quality=emg_quality
)

fusion.calculate_weights()
fusion.calculate_feature_confidence()

fused = fusion.fuse()

feature_confidence = pd.concat([
    fusion.eeg_feature_weights,
    fusion.emg_feature_weights
])

feature_confidence.index = fused.columns

optimizer = AdaptiveFusionOptimizer(
    X=fused,
    y=y,
    feature_confidence=feature_confidence
)

importance = optimizer.calculate_feature_importance()

print("\nTop 20 Important Features")
print(importance.sort_values(ascending=False).head(20))

