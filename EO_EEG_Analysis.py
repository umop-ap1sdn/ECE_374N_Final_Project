import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from topoplot import make_topoplot


def psd_window(psds, freqs, bands):
    bucket_psd = []
    
    for channel in psds:
        windows = {}
        for band in bands.keys():
            indeces = np.where((freqs >= bands[band][0]) & (freqs < bands[band][1]))[0]
            win = np.sum(channel[indeces])
            windows[band] = win
        bucket_psd.append(windows)
    
    return bucket_psd

def psd_window_samples(psds, freqs, bands):
    sample_windowed_psds = []

    for sample in psds:
        sample_windowed_psds.append(psd_window(sample, freqs, bands))
    
    return sample_windowed_psds

def psd_mean_std(psds):
    averaged = []
    for channel in range(np.shape(psds)[1]):
        channel_bands = {}
        for key in psds[0][channel].keys():
            vals = []
            for sample in range(len(psds)):
                vals.append(psds[sample][channel][key])
            
            mean = np.mean(vals)
            std = np.std(vals)
            channel_bands[key] = (mean, std)
        averaged.append(channel_bands)
    
    return averaged

def compute_fisher(mean_psd1, mean_psd2):
    fishers = []
    for channel in range(len(mean_psd1)):
        fisher_bands = {}
        for band in mean_psd1[channel].keys():
            fisher_bands[band] = (
                    np.abs(mean_psd1[channel][band][0] - mean_psd2[channel][band][0]) / 
                    np.clip(np.sqrt(np.square(mean_psd1[channel][band][1]) + np.square(mean_psd2[channel][band][1])), 0.005, None)
                )
            
        fishers.append(fisher_bands)
    
    return fishers

def fisher_heatmap(fisherscore, channels, bands, ax):
    remove_dict = []
    for channel in fisherscore:
        remove_dict.append(list(channel.values()))
    
    sns.heatmap(remove_dict, ax=ax, cmap='viridis', fmt='0.1f', annot=True, cbar_kws={"label": "Fisher Score"},
                yticklabels=channels, xticklabels=bands)

def shape_fisher_for_topoplots(fisherscore):
    remove_dict = []
    for channel in fisherscore:
        remove_dict.append(list(channel.values()))

    return np.transpose(remove_dict)

if __name__ == '__main__':
    meditation_channels = ['AF3', 'AF4', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'PO3', 'POZ', 'PO4', 'O1', 'O2']
    tacs_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']

    meditation_locs = 'data/ErrP_cap_chan_file.mat'
    tacs_locs = 'data/ch32Locations.mat'

    times = ['pre', 'post']
    subj_stim = {
        1: ['Meditation', 'tACS'],
        2: ['tACS', 'Meditation'],
        3: ['Meditation', 'Meditation'],
        4: ['tACS', 'Meditation']
    }

    frequency_bands = {
        'delta_low' : (0, 2),
        'delta_high' : (2, 4),
        'theta' : (4, 8),
        'alpha_low' : (8, 10),
        'alpha_high' : (10, 12),
        'beta_low' : (12, 20),
        'beta_high' : (20, 30),
        'gamma_low' : (30, 45),
        'gamma_high' : (45, 100)
    }

    data = None
    with open('data/EO_PSD_data.pkl', 'rb') as file:
        data = pickle.load(file)

    freqs = data['freqs']
    
    # print(len(psd_window_samples(data['P1222'][0]['eo'], freqs, frequency_bands)[0][0]))
    # print(np.shape(psd_window_samples(data['P1222'][0]['eo'], freqs, frequency_bands)))
    # 
    # eo_psd = psd_window_samples(data['P1222'][0]['eo'], freqs, frequency_bands)
    # nback1 = psd_window_samples(data['P1222'][0]['N2pre'], freqs, frequency_bands)
    # eo_avg = psd_mean_std(eo_psd)
    # nback1_avg = psd_mean_std(nback1)
    # 
    # # print((eo_avg[0]['delta_low']))
    # 
    # fisher = compute_fisher(eo_avg, nback1_avg)
    # print(fisher)

    for subj in subj_stim.keys():
        
        # Collect PSD
        eo1_psd = psd_window_samples(data[f'P122{subj}'][0]['eo'], freqs, frequency_bands)
        preN1_psd = psd_window_samples(data[f'P122{subj}'][0]['N2pre'], freqs, frequency_bands)
        postN1_psd = psd_window_samples(data[f'P122{subj}'][0]['N2post'], freqs, frequency_bands)

        eo2_psd = psd_window_samples(data[f'P122{subj}'][1]['eo'], freqs, frequency_bands)
        preN2_psd = psd_window_samples(data[f'P122{subj}'][1]['N2pre'], freqs, frequency_bands)
        postN2_psd = psd_window_samples(data[f'P122{subj}'][1]['N2post'], freqs, frequency_bands)

        # Average PSD
        eo1_psd = psd_mean_std(eo1_psd)
        preN1_psd = psd_mean_std(preN1_psd)
        postN1_psd = psd_mean_std(postN1_psd)
        
        eo2_psd = psd_mean_std(eo2_psd)
        preN2_psd = psd_mean_std(preN2_psd)
        postN2_psd = psd_mean_std(postN2_psd)

        # Fisher Scores
        eo1_vs_preN1 = compute_fisher(eo1_psd, preN1_psd)
        eo1_vs_postN1 = compute_fisher(eo1_psd, postN1_psd)
        preN1_vs_postN1 = compute_fisher(preN1_psd, postN1_psd)

        eo2_vs_preN2 = compute_fisher(eo2_psd, preN2_psd)
        eo2_vs_postN2 = compute_fisher(eo2_psd, postN2_psd)
        preN2_vs_postN2 = compute_fisher(preN2_psd, postN2_psd)

        day1_chan = meditation_channels
        day2_chan = meditation_channels

        day1_locs = meditation_locs
        day2_locs = meditation_locs

        if subj_stim[subj][0] == 'tACS':
            day1_chan = tacs_channels
            day1_locs = tacs_locs
        if subj_stim[subj][1] == 'tACS':
            day2_chan = tacs_channels
            day2_locs = tacs_locs

        # fig, ax = plt.subplots(2, 3, figsize=(18, 14))
        # plt.subplots_adjust(hspace=0.5)
        # fig.suptitle(f'Subject 122{subj} Fisher Score Heatmaps for Eyes Open and N-Back Sessions\nDay1: {subj_stim[subj][0]}, Day2: {subj_stim[subj][1]}')
        # 
        # ax[0, 0].set_title('Day 1 Eyes Open vs Pre N-Back Session')
        # fisher_heatmap(eo1_vs_preN1, day1_chan, frequency_bands.keys(), ax[0, 0])

        # ax[0, 1].set_title('Day 1 Eyes Open vs Post N-Back Session')
        # fisher_heatmap(eo1_vs_postN1, day1_chan, frequency_bands.keys(), ax[0, 1])

        # ax[0, 2].set_title('Day 1 Pre N-Back vs Post N-Back Session')
        # fisher_heatmap(preN1_vs_postN1, day1_chan, frequency_bands.keys(), ax[0, 2])

        # ax[1, 0].set_title('Day 2 Eyes Open vs Pre N-Back Session')
        # fisher_heatmap(eo2_vs_preN2, day2_chan, frequency_bands.keys(), ax[1, 0])

        # ax[1, 1].set_title('Day 2 Eyes Open vs Post N-Back Session')
        # fisher_heatmap(eo2_vs_postN2, day2_chan, frequency_bands.keys(), ax[1, 1])

        # ax[1, 2].set_title('Day 2 Pre N-Back vs Post N-Back Session')
        # fisher_heatmap(preN2_vs_postN2, day2_chan, frequency_bands.keys(), ax[1, 2])

        # fig.savefig(f'figs/subject122{subj}_nback_fisher.png', dpi=300)
        # 
        # plt.show()

        fig, ax = make_topoplot(day1_locs, shape_fisher_for_topoplots(eo1_vs_preN1), day1_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 1 ({subj_stim[subj][0]}) Eyes Open vs Pre N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day1_eo1_preN1_topoplot.png', dpi=300)
        
        fig, ax = make_topoplot(day1_locs, shape_fisher_for_topoplots(eo1_vs_postN1), day1_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 1 ({subj_stim[subj][0]}) Eyes Open vs Post N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day1_eo1_postN1_topoplot.png', dpi=300)

        fig, ax = make_topoplot(day1_locs, shape_fisher_for_topoplots(preN1_vs_postN1), day1_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 1 ({subj_stim[subj][0]}) Pre N-Back Session vs Post N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day1_preN1_postN1_topoplot.png', dpi=300)
        
        fig, ax = make_topoplot(day2_locs, shape_fisher_for_topoplots(eo2_vs_preN2), day2_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 2 ({subj_stim[subj][1]}) Eyes Open vs Pre N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day2_eo2_preN2_topoplot.png', dpi=300)
        
        fig, ax = make_topoplot(day2_locs, shape_fisher_for_topoplots(eo2_vs_postN2), day2_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 2 ({subj_stim[subj][1]}) Eyes Open vs Post N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day2_eo2_postN2_topoplot.png', dpi=300)
        
        fig, ax = make_topoplot(day2_locs, shape_fisher_for_topoplots(preN2_vs_postN2), day2_chan, frequency_bands.keys(), 
                                f'Subject 122{subj} Day 2 ({subj_stim[subj][1]}) Pre N-Back Session vs Post N-Back Session Topoplots')
        fig.savefig(f'figs/subject122{subj}_day2_preN2_postN2_topoplot.png', dpi=300)
        
        
        plt.show()

