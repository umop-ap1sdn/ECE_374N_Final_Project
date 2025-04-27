import numpy as np
import os
import mne
import pickle
from pyxdf import load_xdf

def split_eo_eeg(fif_path, n_outputs=8):
    data = mne.io.read_raw_fif(fif_path, preload=True).get_data()

    segments = []
    start = 0
    window = np.shape(data)[1] // n_outputs

    for i in range(n_outputs):
        segments.append(data[:, start:start+window])
        start += window
    
    return np.array(segments)

def split_eo_eeg_dir(fif_dir, n_outputs=8):
    for file in os.listdir(fif_dir):
        if '.fif' in file:
            return split_eo_eeg(os.path.join(fif_dir, file), n_outputs)

def split_n_back_eeg(dir_path):
    xdfs = {}
    fifs = {}
    
    for i in range(1, 6):
        substring = f'run-00{i}'
        for file in os.listdir(dir_path):
            if not substring in file:
                continue
            if '.xdf' in file:
                streams, _ = load_xdf(os.path.join(dir_path, file))
                marker_stream = next(s for s in streams if int(len(s['time_series'])) >= 100)
                xdfs[i] = marker_stream
            elif '.fif' in file:
                data = mne.io.read_raw_fif(os.path.join(dir_path, file)).get_data()
                fifs[i] = data

    cleaned_eegs = []
    for i in range(1, 6):
        start = 0
        start_time = xdfs[i]['time_series'][0][1]
        segment = None

        for j in range(len(xdfs[i]['time_series'])):
            if xdfs[i]['time_series'][j][0] == 0.0:
                start = int((xdfs[i]['time_series'][j][1] - start_time) * 512)
                
            if xdfs[i]['time_series'][j][0] == 400.0:
                end = int((xdfs[i]['time_series'][j][1] - start_time) * 512)
                
                if segment is None:
                    segment = data[:, start:end]
                else:
                    segment = np.concatenate((segment, data[:, start:end]), axis=1)
        
        cleaned_eegs.append(segment)

    return cleaned_eegs


def split_n_back_eeg_answers(dir_path, match_ans=100, mismatch_ans=200):
    xdfs = {}
    fifs = {}
    
    for i in range(1, 6):
        substring = f'run-00{i}'
        for file in os.listdir(dir_path):
            if not substring in file:
                continue
            if '.xdf' in file:
                streams, _ = load_xdf(os.path.join(dir_path, file))
                marker_stream = next(s for s in streams if int(len(s['time_series'])) >= 100)
                xdfs[i] = marker_stream
            elif '.fif' in file:
                data = mne.io.read_raw_fif(os.path.join(dir_path, file)).get_data()
                fifs[i] = data

    match_eegs = []
    mismatch_eegs = []
    for i in range(1, 6):
        start = 0
        start_time = xdfs[i]['time_series'][0][1]
        match = None
        match_segment = None
        mismatch_segment = None

        for j in range(len(xdfs[i]['time_series'])):
            if xdfs[i]['time_series'][j][0] == 0.0:
                start = int((xdfs[i]['time_series'][j][1] - start_time) * 512)
            
            if xdfs[i]['time_series'][j][0] == match_ans:
                match = True
            
            if xdfs[i]['time_series'][j][0] == mismatch_ans:
                match = False
                
            if xdfs[i]['time_series'][j][0] == 400.0:
                end = int((xdfs[i]['time_series'][j][1] - start_time) * 512)
                
                if match is None:
                    continue
                if match:
                    if match_segment is None:
                        match_segment = data[:, start:end]
                    else:
                        match_segment = np.concatenate((match_segment, data[:, start:end]), axis=1)
                else:
                    if mismatch_segment is None:
                        mismatch_segment = data[:, start:end]
                    else:
                        mismatch_segment = np.concatenate((mismatch_segment, data[:, start:end]), axis=1)

                match = None

        match_eegs.append(match_segment)
        mismatch_eegs.append(mismatch_segment)
        
    return [match_eegs, mismatch_eegs]
        

if __name__ == '__main__':
    meditation_channels = ['AF3', 'AF4', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'PO3', 'POZ', 'PO4', 'O1', 'O2']
    tacs_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']

    times = ['pre', 'post']
    subj_stim = {
        1: ['Meditation', 'tACS'],
        2: ['tACS', 'Meditation'],
        3: ['Meditation', 'Meditation'],
        4: ['tACS', 'Meditation']
    }

    subject_eegs = {}
    for i in range(1, 5):
        days = []

        for j in range(1, 3):
            data = {}

            eo_path = f'data/Group 12/sub-P122{i}/ses-S00{j}eo/eeg'
            if not os.path.exists(eo_path):
                eo_path = f'data/Group 12/sub-P122{i}/ses-S001eo/eeg'

            n2_pre_path = f'data/Group 12/sub-P122{i}/ses-S00{j}N2pre/eeg'
            n2_post_path = f'data/Group 12/sub-P122{i}/ses-S00{j}N2post/eeg'

            data['eo'] = split_eo_eeg_dir(eo_path)
            data['N2pre'] = split_n_back_eeg(n2_pre_path)
            data['N2post'] = split_n_back_eeg(n2_post_path)
            data['N2pre-ans'] = split_n_back_eeg_answers(n2_pre_path)
            data['N2post-ans'] = split_n_back_eeg_answers(n2_post_path)

            days.append(data)
        subject_eegs[f'P122{i}'] = days

    with open('data/EO_EEG_data.pkl', 'wb') as file:
        pickle.dump(subject_eegs, file)
