import numpy as np
import scipy.linalg as la
from scipy.signal import butter, filtfilt

class AdaptiveCSP:
    """
    Fixed Adaptive Common Spatial Patterns (ACSP).
    Includes narrow 8-30Hz bandpass filtering and Covariance Regularization.
    """
    def __init__(self, n_components=4, learning_rate=0.05, fs=1000):
        self.n_components = n_components 
        self.learning_rate = learning_rate
        self.fs = fs
        self.covariances = {}
        self.filters = None
        self.classes = None
        
        # Create the 8-30Hz filter specifically for ACSP
        nyq = 0.5 * self.fs
        self.b, self.a = butter(4, [8.0/nyq, 30.0/nyq], btype='band')

    def _filter_eeg(self, X):
        """ Applies the 8-30Hz Mu/Beta rhythm filter. """
        return filtfilt(self.b, self.a, X, axis=-1)

    def _compute_covariance(self, X):
        """ Computes the regularized spatial covariance matrix. """
        _, channels, _ = X.shape
        cov = np.zeros((channels, channels))
        for epoch in X:
            cov += np.dot(epoch, epoch.T) / np.trace(np.dot(epoch, epoch.T))
        cov = cov / len(X)
        
        # THE FIX: Tikhonov Regularization (stabilizes the math when channels=60)
        cov += np.eye(channels) * 1e-4 
        return cov

    def fit(self, X, y):
        """ Learns the initial spatial filters. """
        self.classes = np.unique(y)
        self.covariances = {}
        
        # Filter first to remove noise!
        X_filt = self._filter_eeg(X)
        
        for c in self.classes:
            X_c = X_filt[y == c]
            self.covariances[c] = self._compute_covariance(X_c)
            
        self._update_filters()
        return self

    def _update_filters(self):
        """ Solves the generalized eigenvalue problem. """
        filters = []
        for c in self.classes:
            cov_c = self.covariances[c]
            cov_rest = np.mean([self.covariances[k] for k in self.classes if k != c], axis=0)
            
            # Generalized eigenvalue problem
            evals, evecs = la.eigh(cov_c, cov_c + cov_rest)
            
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            
            top_filters = evecs[:, :self.n_components // 2]
            bottom_filters = evecs[:, -(self.n_components // 2):]
            
            filters.append(top_filters)
            filters.append(bottom_filters)
            
        self.filters = np.hstack(filters)

    def transform(self, X):
        """ Extracts log-variance features. """
        # Filter first!
        X_filt = self._filter_eeg(X)
        
        features = []
        for epoch in X_filt:
            Z = np.dot(self.filters.T, epoch)
            var = np.var(Z, axis=1)
            features.append(np.log(var + 1e-10))
            
        return np.array(features)

    def adapt(self, X_new, y_pseudo):
        """ Adapts the spatial filters using EMA. """
        X_new_filt = self._filter_eeg(X_new)
        new_cov = self._compute_covariance(X_new_filt)
        
        old_cov = self.covariances[y_pseudo]
        self.covariances[y_pseudo] = (1 - self.learning_rate) * old_cov + (self.learning_rate) * new_cov
        
        self._update_filters()
