import os
import json
import copy
import numpy as np
import pandas as pd


class EMGClassificationPipeline:
    """
    Orchestrates LOSO cross-validation across all models.

    Flow
    ----
    EMGMasterLoader → EMGSubjectSplitter → [LDA, RF, SVM]
        per fold: fit → predict → evaluate
        aggregate across folds
        save results to output/

    Parameters
    ----------
    loader   : EMGMasterLoader  (already .load()-ed)
    splitter : EMGSubjectSplitter
    models   : dict { name: sklearn_estimator }
    output_dir : str  directory to save results (CSVs + JSON)
    """

    def __init__(self, loader, splitter, models, output_dir="output/emg_classification"):
        self.loader = loader
        self.splitter = splitter
        self.models = models
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Results store: { model_name: [fold_result, ...] }
        self.fold_results = {name: [] for name in models}

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self):
        """
        Execute LOSO cross-validation for all models.
        Progress is printed fold-by-fold.
        """
        from classification.emg_classifiers import evaluate_fold

        n_folds = self.splitter.n_splits()

        print()
        print("=" * 70)
        print("EMG CLASSIFICATION — LOSO CROSS-VALIDATION")
        print("=" * 70)
        print(f"Models  : {list(self.models.keys())}")
        print(f"Subjects: {n_folds}  (one held-out per fold)")
        print(f"Classes : {len(self.splitter.classes_)}")
        print(f"Labels  : {list(self.splitter.classes_[:10])} ...")

        for fold, test_subj, X_tr, X_te, y_tr, y_te in self.splitter.splits():

            print()
            print(f"── Fold {fold:02d}/{n_folds}  test={test_subj}  "
                  f"train={X_tr.shape[0]:,}  test={X_te.shape[0]:,}")

            for name, model in self.models.items():

                # Clone model so each fold starts fresh
                clf = copy.deepcopy(model)

                result = evaluate_fold(clf, X_tr, X_te, y_tr, y_te)
                result["fold"] = fold
                result["test_subject"] = test_subj

                self.fold_results[name].append(result)

                print(
                    f"   {name:<14}  "
                    f"acc={result['accuracy']*100:.1f}%  "
                    f"f1={result['f1']*100:.1f}%"
                )

        return self.fold_results

    # ----------------------------------------------------------
    # Aggregate and Save
    # ----------------------------------------------------------

    def summarize(self):
        """
        Aggregate fold results and save:
          - output_dir/emg_loso_summary.csv   (per-model mean ± std)
          - output_dir/emg_loso_per_fold.csv  (per fold × model scores)
          - output_dir/emg_loso_metrics.json  (full summary with fold lists)

        Returns
        -------
        pd.DataFrame   summary table (one row per model)
        """
        from classification.emg_classifiers import aggregate_results, print_results

        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        summary_rows = []
        per_fold_rows = []

        for name, results in self.fold_results.items():

            summary = aggregate_results(results)
            print_results(name, summary)

            summary_rows.append({
                "Model":          name,
                "Accuracy_Mean":  round(summary["accuracy_mean"] * 100, 2),
                "Accuracy_Std":   round(summary["accuracy_std"] * 100, 2),
                "Precision_Mean": round(summary["precision_mean"] * 100, 2),
                "Precision_Std":  round(summary["precision_std"] * 100, 2),
                "Recall_Mean":    round(summary["recall_mean"] * 100, 2),
                "Recall_Std":     round(summary["recall_std"] * 100, 2),
                "F1_Mean":        round(summary["f1_mean"] * 100, 2),
                "F1_Std":         round(summary["f1_std"] * 100, 2),
            })

            for r in results:
                per_fold_rows.append({
                    "Model":       name,
                    "Fold":        r["fold"],
                    "TestSubject": r["test_subject"],
                    "Accuracy":    round(r["accuracy"] * 100, 2),
                    "Precision":   round(r["precision"] * 100, 2),
                    "Recall":      round(r["recall"] * 100, 2),
                    "F1":          round(r["f1"] * 100, 2),
                })

        summary_df = pd.DataFrame(summary_rows)
        per_fold_df = pd.DataFrame(per_fold_rows)

        # ── Save CSVs ──────────────────────────────────────────
        summary_path  = os.path.join(self.output_dir, "emg_loso_summary.csv")
        per_fold_path = os.path.join(self.output_dir, "emg_loso_per_fold.csv")
        json_path     = os.path.join(self.output_dir, "emg_loso_metrics.json")

        summary_df.to_csv(summary_path, index=False)
        per_fold_df.to_csv(per_fold_path, index=False)

        # JSON: save aggregated metrics without large numpy arrays
        json_data = {}
        for name, results in self.fold_results.items():
            agg = aggregate_results(results)
            # Remove per-fold raw lists from JSON to keep it readable
            json_data[name] = {
                k: v for k, v in agg.items()
                if not k.endswith("_folds")
            }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print()
        print("─" * 60)
        print(f"Summary CSV  → {summary_path}")
        print(f"Per-fold CSV → {per_fold_path}")
        print(f"Metrics JSON → {json_path}")

        return summary_df
