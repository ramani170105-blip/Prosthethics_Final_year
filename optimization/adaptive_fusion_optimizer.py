import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class AdaptiveFusionOptimizer:

    def __init__(
        self,
        X,
        y,
        feature_confidence
    ):

        self.X = X.copy()

        self.y = y.copy()

        self.feature_confidence = feature_confidence

        self.feature_importance = None

        self.combined_score = None

        self.selected_features = None

        self.optimized_dataset = None


    # =====================================================
    # Feature Importance
    # =====================================================

    def calculate_feature_importance(self):

        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

        model.fit(self.X, self.y)

        self.feature_importance = pd.Series(
            model.feature_importances_,
            index=self.X.columns
        )

        print(f"Features : {len(self.feature_importance)}")

        return self.feature_importance

    