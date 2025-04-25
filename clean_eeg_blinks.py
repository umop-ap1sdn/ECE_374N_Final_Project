import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyxdf import load_xdf
import mne
from mne.io import RawArray
from mne import create_info

# --- Configurable Parameters ---
threshold_std_factor = 4       # Multiplier for adaptive blink threshold
window_sec = 0.1               # Interpolation window (±100 ms)
merge_sec = 0.3                # Merge nearby blink events within 300 ms

# --- Input Path ---
root_path = input("Enter the path to the folder containing raw XDF files: ").strip()

# --- Find all XDF files recursively ---
xdf_files = []
for dirpath, _, filenames in os.walk(root_path):
    for f in filenames:
        if f.endswith(".xdf"):
            xdf_files.append(os.path.join(dirpath, f))

print(f"\nFound {len(xdf_files)} XDF files.")

# --- Process each file ---
for file_path in xdf_files:
    try:
        print(f"\nProcessing: {file_path}")
        streams, _ = load_xdf(file_path)

        # Identify EEG stream
        eeg_stream = next(s for s in streams if int(s['info']['channel_count'][0]) >= 32)
        data = eeg_stream['time_series']
        timestamps = eeg_stream['time_stamps']
        eeg_data = data[:, :32]
        eog_v = data[:, 37]  # AUX8 is channel index 37

        fs = int(round(1 / np.mean(np.diff(timestamps))))

        # --- Adaptive Threshold ---
        eog_baseline = np.median(eog_v)
        eog_std = np.std(eog_v)
        blink_threshold = threshold_std_factor * eog_std

        print(f"Using baseline: {eog_baseline:.2f}, std: {eog_std:.2f}, threshold: ±{blink_threshold:.2f}")

        # Detect blink events: deviation from baseline
        blink_indices = np.where(np.abs(eog_v - eog_baseline) > blink_threshold)[0]
        min_interval = int(merge_sec * fs)
        blink_events = [blink_indices[0]] if len(blink_indices) else []
        for idx in blink_indices[1:]:
            if idx - blink_events[-1] > min_interval:
                blink_events.append(idx)

        # --- Interpolate EEG Around Blinks ---
        window = int(window_sec * fs)
        clean_eeg = eeg_data.copy()
        for idx in blink_events:
            start = max(1, idx - window)
            end = min(len(eeg_data) - 2, idx + window)
            for ch in range(32):
                x = np.array([start - 1, end + 1])
                y = eeg_data[x, ch]
                f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                clean_eeg[start:end + 1, ch] = f(np.arange(start, end + 1))

        # --- Save Cleaned EEG as .fif in Same Folder ---
        folder = os.path.dirname(file_path)
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(folder, base + "_clean_raw.fif")

        # Create MNE RawArray and save
        ch_names = [f"EEG{i+1}" for i in range(32)]
        ch_types = ["eeg"] * 32
        info = create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        raw = RawArray(clean_eeg.T, info)
        raw.save(output_path, overwrite=True)

        print(f"Saved cleaned EEG as: {output_path}")

        # --- Optional: Save Blink Diagnostic Plot ---
        plt.figure(figsize=(14, 4))
        plt.plot(eog_v, label='Vertical EOG (AUX8)', linewidth=0.8)
        plt.axhline(eog_baseline, color='gray', linestyle='--', label='Median Baseline')
        plt.axhline(eog_baseline + blink_threshold, color='red', linestyle='--', label='+Threshold')
        plt.axhline(eog_baseline - blink_threshold, color='red', linestyle='--', label='-Threshold')
        plt.scatter(blink_events, eog_v[blink_events], color='black', s=10, label='Detected Blinks')
        plt.title(f"EOG Signal and Detected Blinks: {base}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, base + "_blink_diagnostics.png"))
        plt.close()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
