import numpy as np
import matplotlib.pyplot as plt
import mne
from pyxdf import load_xdf
import os

# --- CONFIGURATION ---
xdf_path = "ses-S001ecpostACS/eeg/sub-P1224_ses-S001ecpostACS_task-Default_run-001_eeg.xdf"
clean_fif_path = "ses-S001ecpostACS/eeg/sub-P1224_ses-S001ecpostACS_task-Default_run-001_eeg_clean_raw.fif"
fs_manual = None  # If sampling rate not detected properly, set it manually (e.g. fs_manual = 500)
blink_threshold = 150  # Amplitude threshold for "blink-like" events
channels_to_plot = list(range(32))  # EEG channels 0-31

# --- LOAD ORIGINAL XDF ---
streams, _ = load_xdf(xdf_path)
eeg_stream = next(s for s in streams if int(s['info']['channel_count'][0]) >= 32)
orig_eeg = eeg_stream['time_series'][:, :32]
timestamps = eeg_stream['time_stamps']
fs = fs_manual or int(round(1 / np.mean(np.diff(timestamps))))
time = np.linspace(0, len(orig_eeg)/fs, len(orig_eeg))

# --- LOAD CLEANED FIF ---
raw = mne.io.read_raw_fif(clean_fif_path, preload=True, verbose=False)
clean_eeg = raw.get_data().T  # shape: (samples, channels)

# --- SANITY CHECK ---
assert orig_eeg.shape == clean_eeg.shape, "Shape mismatch between original and cleaned EEG!"

# --- ANALYSIS + PLOTTING ---
print("\n--- Blink-like Event Count (|amp| > {}) ---".format(blink_threshold))
for ch in channels_to_plot:
    orig_blinks = np.sum(np.abs(orig_eeg[:, ch]) > blink_threshold)
    clean_blinks = np.sum(np.abs(clean_eeg[:, ch]) > blink_threshold)
    print(f"Channel {ch + 1:2d}: Original = {orig_blinks:5d}, Cleaned = {clean_blinks:5d}")

# --- PLOTTING IN BATCHES OF 4 CHANNELS ---
batch_size = 4
for i in range(0, len(channels_to_plot), batch_size):
    batch_chs = channels_to_plot[i:i + batch_size]
    fig, axes = plt.subplots(len(batch_chs), 1, figsize=(15, 8), sharex=True)

    if len(batch_chs) == 1:
        axes = [axes]

    for j, ch in enumerate(batch_chs):
        axes[j].plot(time, orig_eeg[:, ch], label="Original", alpha=0.5)
        axes[j].plot(time, clean_eeg[:, ch], label="Cleaned", alpha=0.7)
        axes[j].set_title(f"Channel {ch + 1}")
        axes[j].legend(loc='upper right')

    plt.xlabel("Time (s)")
    plt.suptitle(f"EEG Blink Artifact Cleaning – Channels {batch_chs[0] + 1} to {batch_chs[-1] + 1}", fontsize=14)
    plt.tight_layout()
    plt.show()
