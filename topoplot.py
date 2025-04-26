import scipy
import numpy as np
import mne
import matplotlib.pyplot as plt


# ch32Locations is the path to the ch32Locations.mat file (can be found on canvas)
# run_fisher_scores should be shaped as (n, 32) where n is the number of runs (number of topoplots to make), 32 is the EEG channels
# Ordering for EEG channels should be [FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']

# Returns the fig, ax from subplot creations, figures do not have titles or other labels by default, should be added after return
def make_topoplot(ch32Locations, run_fisher_scores):
    chLocs = scipy.io.loadmat(ch32Locations, struct_as_record=False, squeeze_me=True)
    chLocs = chLocs['ch32Locations']

    chLabels = []
    chX = []
    chY = []
    chZ = []

    for channel in chLocs:
        chLabels.append(channel.labels)
        chX.append(channel.X)
        chY.append(channel.Y)
        chZ.append(channel.Z)

    chX = np.array(chX) / 1000
    chY = np.array(chY) / 1000
    chZ = np.array(chZ) / 1000

    pos_dict = {name: np.array([x, y, z]) for name, x, y, z in zip(chLabels, chX, chY, chZ)}
    custom_montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')

    info = mne.create_info(ch_names=chLabels.tolist(), sfreq=512, ch_types='eeg')
    info.set_montage(custom_montage)

    fig, ax = plt.subplots(1, len(run_fisher_scores), figsize=(24, 6))
    
    count = 0
    for fisher_scores in run_fisher_scores:
        im, _ = mne.viz.plot_topomap(fisher_scores, info, cmap='viridis', show=False, axes=ax[count])
        cbar = plt.colorbar(im, ax=ax[count])
        cbar.set_label('Fisher Score')
        count += 1
    
    return fig, ax



