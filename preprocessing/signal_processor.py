import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, freq, fs, q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=-1)

def preprocess_eeg(X_eeg, fs=1000):
    """
    Applies biological preprocessing to EEG epochs.
    - Bandpass 1-40Hz (Isolates motor imagery rhythms: Mu, Beta, Gamma)
    - Notch 50Hz (Powerline noise removal)
    X_eeg shape: (epochs, channels, time)
    """
    print("  -> Applying EEG Bandpass (1-40 Hz)...")
    b, a = butter_bandpass(1.0, 40.0, fs, order=4)
    # filtfilt applies zero-phase filtering
    clean_eeg = filtfilt(b, a, X_eeg, axis=-1)
    
    print("  -> Applying EEG Notch (50 Hz)...")
    clean_eeg = notch_filter(clean_eeg, 50.0, fs)
    
    return clean_eeg

def preprocess_emg(X_emg, fs=1000):
    """
    Applies biological preprocessing to EMG epochs.
    - Bandpass 20-450Hz (Muscle activation envelope)
    - Notch 50Hz (Powerline noise)
    - Rectification (Absolute value)
    X_emg shape: (epochs, channels, time)
    """
    print("  -> Applying EMG Bandpass (20-450 Hz)...")
    b, a = butter_bandpass(20.0, 450.0, fs, order=4)
    clean_emg = filtfilt(b, a, X_emg, axis=-1)
    
    print("  -> Applying EMG Notch (50 Hz)...")
    clean_emg = notch_filter(clean_emg, 50.0, fs)
    
    print("  -> Applying EMG Rectification...")
    clean_emg = np.abs(clean_emg)
    
    return clean_emg
