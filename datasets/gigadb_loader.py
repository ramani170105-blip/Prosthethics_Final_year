import os
import mne
import numpy as np

# Change this path to point to your local datasets folder!
DATASET_ROOT = "datasets"


def load_gigadb_subject(subject_id=1, task="multigrasp", condition="realMove", override_data_dir=None):
    """
    Loads a specific subject, task, and condition from the GigaDB dataset.
    Returns synchronized (X_eeg, X_emg, y).
    
    Parameters:
    - subject_id: integer ID of the subject (1 to 25).
    - task: "multigrasp", "reaching", or "twist".
    - condition: "realMove" or "MI".
    - override_data_dir: optional path to load data from (e.g., local temp folder).
    """
    data_dir = override_data_dir if override_data_dir else os.path.join(DATASET_ROOT, f"sub{subject_id}")
    
    X_eeg_list, X_emg_list, y_list = [], [], []
    
    # We will iterate through all 3 sessions for the subject
    for session in [1, 2, 3]:
        file_prefix = f"session{session}_sub{subject_id}_{task}_{condition}"
        vhdr_path = os.path.join(data_dir, f"{file_prefix}.vhdr")
        
        if not os.path.exists(vhdr_path):
            continue
            
        print(f"Loading {vhdr_path}...")
        # preload=False forces MNE to read from disk, saving gigabytes of Mac RAM
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
        
        # Extract events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # In this dataset, stimulus events are typically labeled "Stimulus/S  X"
        # We only want actual movement cues, which usually have values between 1 and 20.
        keep_events = {k: v for k, v in event_id.items() if 'Stimulus' in k and v < 20}
        
        if not keep_events:
            continue
            
        # Extract epochs (0 to 4 seconds post-cue is standard for motor tasks)
        # event_repeated='drop' fixes the duplicate hardware timestamp glitch
        epochs = mne.Epochs(raw, events, event_id=keep_events, tmin=0, tmax=4.0, 
                            baseline=None, preload=True, event_repeated='drop', verbose=False)
        
        data = epochs.get_data() # shape: (n_epochs, n_channels, n_times)
        labels = epochs.events[:, 2]
        
        # Separate channels based on the header info
        ch_names = raw.ch_names
        eeg_indices = [i for i, ch in enumerate(ch_names) if 'EMG' not in ch and 'EOG' not in ch]
        emg_indices = [i for i, ch in enumerate(ch_names) if 'EMG' in ch and 'ref' not in ch.lower()]
        
        X_eeg_list.append(data[:, eeg_indices, :])
        X_emg_list.append(data[:, emg_indices, :])
        y_list.append(labels)
        
    if not X_eeg_list:
        raise ValueError("No valid data found in the directory.")
        
    X_eeg = np.concatenate(X_eeg_list, axis=0)
    X_emg = np.concatenate(X_emg_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Successfully loaded {len(y)} trials.")
    print(f"EEG shape: {X_eeg.shape} | EMG shape: {X_emg.shape}")
    
    return X_eeg, X_emg, y, raw.info['sfreq']

if __name__ == "__main__":
    X_eeg, X_emg, y, sfreq = load_gigadb_subject()
