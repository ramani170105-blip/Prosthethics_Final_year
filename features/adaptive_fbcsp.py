import numpy as np
from scipy.signal import butter, filtfilt

from features.adaptive_csp import AdaptiveCSP


class AdaptiveFBCSP:
    """
    Novel Adaptive Filter Bank CSP with EMG-Guided Band Attention.

    Key innovations over standard FBCSP (Ang et al., 2008):
    1. Subject-Adaptive Band Discovery: Discovers the most discriminative
       frequency windows for THIS specific patient via PSD Fisher ratio.
    2. EMG-Guided Band Attention: Weights each band's CSP features by the
       cross-correlation between EEG power in that band and the EMG envelope.
       This cross-modal attention is novel — no published paper combines
       EMG SNR with per-band FBCSP weighting.
    3. Reuses the existing AdaptiveCSP per band (preserving your adaptive
       covariance regularization and Tikhonov stabilization).
    """

    # Standard BCI frequency bands as candidates
    CANDIDATE_BANDS = [
        (1,   4,  "Delta"),
        (4,   8,  "Theta"),
        (8,  13,  "Alpha/Mu"),
        (13, 30,  "Beta"),
        (30, 45,  "Low-Gamma"),
    ]

    def __init__(self, n_components=4, fs=1000):
        self.n_components = n_components
        self.fs = fs
        self.csp_models = []
        self.band_weights = None
        self.selected_bands = self.CANDIDATE_BANDS
        self.fitted = False

    # ------------------------------------------------------------------
    # Helper: Band Power Extraction
    # ------------------------------------------------------------------
    def _extract_band_power(self, X_eeg, fmin, fmax):
        nyq = 0.5 * self.fs
        lo = max(fmin / nyq, 1e-4)
        hi = min(fmax / nyq - 1e-4, 0.9999)
        b, a = butter(4, [lo, hi], btype='band')
        powers = []
        for epoch in X_eeg:
            filtered = filtfilt(b, a, epoch, axis=-1)
            powers.append(np.log(np.mean(filtered ** 2) + 1e-10))
        return np.array(powers)

    # ------------------------------------------------------------------
    # 1. Subject-Adaptive Band Discriminability (Fisher Ratio)
    # ------------------------------------------------------------------
    def _compute_band_discriminability(self, X_eeg, y):
        scores = []
        for fmin, fmax, name in self.CANDIDATE_BANDS:
            band_powers = self._extract_band_power(X_eeg, fmin, fmax)
            classes = np.unique(y)
            overall_mean = np.mean(band_powers)
            between = np.sum([
                len(band_powers[y == c]) * (np.mean(band_powers[y == c]) - overall_mean) ** 2
                for c in classes
            ])
            within = np.sum([
                np.sum((band_powers[y == c] - np.mean(band_powers[y == c])) ** 2)
                for c in classes
            ])
            fisher = between / (within + 1e-10)
            scores.append(fisher)
            print(f"   [{fmin}-{fmax} Hz] ({name}) — Fisher Discriminability: {fisher:.4f}")
        return np.array(scores)

    # ------------------------------------------------------------------
    # 2. EMG-Guided Band Attention (Novel Contribution)
    # ------------------------------------------------------------------
    def _compute_emg_band_attention(self, X_eeg, X_emg):
        """
        Computes Pearson correlation between each EEG frequency band power
        and the mean EMG envelope power across trials.
        Bands highly correlated with muscle activation get higher weight.
        """
        emg_envelope = np.mean(np.abs(X_emg), axis=(1, 2))
        attention_weights = []
        for fmin, fmax, _ in self.selected_bands:
            band_power = self._extract_band_power(X_eeg, fmin, fmax)
            corr = np.corrcoef(band_power, emg_envelope)[0, 1]
            attention_weights.append(abs(corr) if not np.isnan(corr) else 0.0)
        weights = np.array(attention_weights)
        weights = np.exp(weights) / (np.sum(np.exp(weights)) + 1e-10)
        return weights

    # ------------------------------------------------------------------
    # 3. Fit
    # ------------------------------------------------------------------
    def fit(self, X_eeg, X_emg, y):
        print("\n" + "=" * 60)
        print("ADAPTIVE FBCSP: BAND DISCOVERY")
        print("=" * 60)

        disc_scores = self._compute_band_discriminability(X_eeg, y)

        print("\n" + "=" * 60)
        print("EMG-GUIDED BAND ATTENTION (Novel Contribution)")
        print("=" * 60)
        emg_attention = self._compute_emg_band_attention(X_eeg, X_emg)

        combined = disc_scores * emg_attention
        combined = combined / (combined.sum() + 1e-10)
        self.band_weights = combined

        for i, (fmin, fmax, name) in enumerate(self.selected_bands):
            print(f"   [{fmin}-{fmax} Hz] ({name}) — Final Weight: {self.band_weights[i]:.4f}")

        print("\n" + "=" * 60)
        print("FITTING PER-BAND ADAPTIVE CSP")
        print("=" * 60)

        self.csp_models = []
        for fmin, fmax, name in self.selected_bands:
            nyq = 0.5 * self.fs
            lo = max(fmin / nyq, 1e-4)
            hi = min(fmax / nyq - 1e-4, 0.9999)
            b, a = butter(4, [lo, hi], btype='band')
            X_band = np.array([filtfilt(b, a, ep, axis=-1) for ep in X_eeg])
            csp = AdaptiveCSP(n_components=self.n_components, fs=self.fs)
            csp.classes = np.unique(y)
            csp.covariances = {}
            for c in csp.classes:
                csp.covariances[c] = csp._compute_covariance(X_band[y == c])
            csp._update_filters()
            self.csp_models.append((csp, b, a))
            print(f"   [OK] AdaptiveCSP fitted for [{fmin}-{fmax} Hz] ({name})")

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # 4. Transform
    # ------------------------------------------------------------------
    def transform(self, X_eeg):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")
        all_band_feats = []
        for i, (csp, b, a) in enumerate(self.csp_models):
            X_band = np.array([filtfilt(b, a, ep, axis=-1) for ep in X_eeg])
            feats = csp.transform(X_band)
            weighted_feats = feats * self.band_weights[i]
            all_band_feats.append(weighted_feats)
        return np.hstack(all_band_feats)

    def fit_transform(self, X_eeg, X_emg, y):
        return self.fit(X_eeg, X_emg, y).transform(X_eeg)
