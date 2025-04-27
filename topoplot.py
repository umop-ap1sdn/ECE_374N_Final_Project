import scipy
import numpy as np
import mne
import matplotlib.pyplot as plt


# ch32Locations is the path to the ch32Locations.mat or ErrP_cap_chan_file.mat file (can be found on canvas)
# run_fisher_scores should be shaped as (n, 32) where n is the number of runs (number of topoplots to make), 32 is the EEG channels
# Ordering for EEG channels should be ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
# OR
# ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P3', 'P1', 'Pz', 'P2', 'P4', 'PO3', 'POz', 'PO4', 'O1', 'O2']

# Returns the fig, ax from subplot creations, figures do not have titles or other labels by default, should be added after return
meditation_cap_ordering = ['AF3', 'AF4', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'PO3', 'POZ', 'PO4', 'O1', 'O2']
tacs_cap_ordering = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
def make_topoplot(ch32Locations, run_fisher_scores, channel_ordering):
    chLocs = scipy.io.loadmat(ch32Locations, struct_as_record=False, squeeze_me=True)
    is_meditation = 'ch32Locations' not in chLocs.keys()
    
    if 'ch32Locations' in chLocs.keys():
        # tacs cap
        print(list(channel_ordering))
        print(tacs_cap_ordering)
        assert list(channel_ordering) == tacs_cap_ordering
        chLocs = chLocs['ch32Locations']
    else:
        # meditation cap
        print(list(channel_ordering))
        print(meditation_cap_ordering)
        assert list(channel_ordering) == meditation_cap_ordering
        chLocs = chLocs['chan']

    chLabels = []
    chX = []
    chY = []
    chZ = []

    for channel in chLocs:
        chLabels.append(channel.labels)
        # Rotate coordinates 90 degrees counterclockwise (left)
        # For a 90 degree rotation: x' = y, y' = -x
        chX.append(channel.Y)
        chY.append(channel.X)
        chZ.append(channel.Z)

    scale_factor = 10 if is_meditation else 1000
    chX = np.array(chX) / scale_factor
    chY = np.array(chY) / scale_factor
    chZ = np.array(chZ) / scale_factor

    pos_dict = {name: np.array([x, y, z]) for name, x, y, z in zip(chLabels, chX, chY, chZ)}
    print(pos_dict)
    custom_montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')

    info = mne.create_info(ch_names=chLabels, sfreq=512, ch_types='eeg')
    info.set_montage(custom_montage)


    fig, ax = plt.subplots(1, len(run_fisher_scores), figsize=(24, 6))
    
    count = 0
    for fisher_scores in run_fisher_scores:
        print(fisher_scores)
        im, _ = mne.viz.plot_topomap(fisher_scores, info, cmap='viridis', show=False, axes=ax, names=chLabels)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fisher Score')
        count += 1
    
    return fig, ax



