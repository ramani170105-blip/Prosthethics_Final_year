import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    f_classif
)

from sklearn.ensemble import RandomForestClassifier


class AdaptiveFeatureSelector:

    def __init__(self, X, y):

        self.X = X.copy()
        self.y = y.copy()

        self.selected_features = None
        self.selected_names = None

    # =====================================================
    # Variance Threshold
    # =====================================================

    def variance_selection(self, threshold=0.01):

        selector = VarianceThreshold(threshold)

        X_new = selector.fit_transform(self.X)

        names = self.X.columns[selector.get_support()]

        return pd.DataFrame(X_new, columns=names), names

    # =====================================================
    # Mutual Information
    # =====================================================

    def mutual_information_selection(self, k=200):

        selector = SelectKBest(
            mutual_info_classif,
            k=min(k, self.X.shape[1])
        )

        X_new = selector.fit_transform(
            self.X,
            self.y
        )

        names = self.X.columns[selector.get_support()]

        return pd.DataFrame(X_new, columns=names), names

    # =====================================================
    # ANOVA F-test
    # =====================================================

    def anova_selection(self, k=200):

        selector = SelectKBest(
            f_classif,
            k=min(k, self.X.shape[1])
        )

        X_new = selector.fit_transform(
            self.X,
            self.y
        )

        names = self.X.columns[selector.get_support()]

        return pd.DataFrame(X_new, columns=names), names

    # =====================================================
    # Random Forest Importance
    # =====================================================

    def random_forest_selection(self, top_n=200):

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(self.X, self.y)

        importance = pd.Series(
            model.feature_importances_,
            index=self.X.columns
        )

        importance = importance.sort_values(
            ascending=False
        )

        names = importance.head(top_n).index

        X_new = self.X[names]

        return X_new, names

    # =====================================================
    # Adaptive Selection
    # =====================================================

    def adaptive_select(self):

        print("\n" + "=" * 60)
        print("ADAPTIVE FEATURE SELECTION")
        print("=" * 60)

        samples = self.X.shape[0]
        features = self.X.shape[1]

        print(f"Samples  : {samples}")
        print(f"Features : {features}")

        # ---------- Strategy ----------

        if features > 1000:

            print("\nStrategy : Variance + Mutual Information")

            X1, _ = self.variance_selection()

            selector = SelectKBest(
                mutual_info_classif,
                k=min(200, X1.shape[1])
            )

            X2 = selector.fit_transform(X1, self.y)

            names = X1.columns[
                selector.get_support()
            ]

            self.selected_features = pd.DataFrame(
                X2,
                columns=names
            )

            self.selected_names = names

        elif samples > 500:

            print("\nStrategy : Random Forest")

            self.selected_features, self.selected_names = \
                self.random_forest_selection()

        else:

            print("\nStrategy : ANOVA")

            self.selected_features, self.selected_names = \
                self.anova_selection()

        print()

        print(
            f"Selected Features : "
            f"{self.selected_features.shape[1]}"
        )

        return self.selected_features

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print("\n" + "=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)

        print(
            f"Original Features : {self.X.shape[1]}"
        )

        print(
            f"Selected Features : {self.selected_features.shape[1]}"
        )

        print()

        print("First 20 Selected Features\n")

        print(
            list(self.selected_names[:20])
        )

    # =====================================================
    # Getter
    # =====================================================

    def get_dataset(self):

        return self.selected_features