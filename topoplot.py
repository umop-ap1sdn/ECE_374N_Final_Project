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
def make_topoplot(ch32Locations, run_fisher_scores, channel_ordering, fisher_scores_names, figure_title = None):
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
    horizontal_scale_factor = 1.1 if is_meditation else 1
    vertical_scale_factor = 1.1 if is_meditation else 1
    chX = np.array(chX) / (scale_factor * horizontal_scale_factor)
    chY = np.array(chY) / (scale_factor * vertical_scale_factor)
    chZ = np.array(chZ) / scale_factor

    pos_dict = {name: np.array([x, y, z]) for name, x, y, z in zip(chLabels, chX, chY, chZ)}
    print(pos_dict)
    custom_montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')

    info = mne.create_info(ch_names=chLabels, sfreq=512, ch_types='eeg')
    info.set_montage(custom_montage)

    # Calculate optimal grid dimensions
    n_plots = len(run_fisher_scores)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Create figure with square-like dimensions
    fig_size = min(6 * n_cols, 6 * n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))
    ax = ax.flatten() if n_plots > 1 else [ax]
    
    # Find global min and max for consistent color scaling, ignoring nan values
    vmin = min(np.nanmin(scores) for scores in run_fisher_scores)
    vmax = max(np.nanmax(scores) for scores in run_fisher_scores)
    print(vmin, vmax)
    count = 0
    for fisher_scores, fisher_scores_name in zip(run_fisher_scores, fisher_scores_names):
        # replace nan with 0
        fisher_scores = np.nan_to_num(fisher_scores)
        print(fisher_scores)
        #put in the best cmap not viridis
        im, _ = mne.viz.plot_topomap(fisher_scores, info, cmap='plasma', show=False, axes=ax[count], 
                                    names=chLabels, vlim = (vmin, vmax))
        cbar = plt.colorbar(im, ax=ax[count])
        cbar.set_label('Fisher Score')
        ax[count].set_title(fisher_scores_name, fontsize=20)
        count += 1
    
    # Hide unused subplots
    for i in range(count, len(ax)):
        ax[i].set_visible(False)
    
    # plt.tight_layout()
    if figure_title:
        fig.suptitle(figure_title, fontsize=30)
    return fig, ax
