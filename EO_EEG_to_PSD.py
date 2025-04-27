from scipy.signal import welch
import pickle
import numpy as np

def eeg_to_psd(eegs):
    psds = []
    freqs = None

    for eeg in eegs:
        sample_psds = []
        for channel in eeg:
            freqs, psd = welch(channel, fs=512, nperseg=512)
            sample_psds.append(psd)
            psd = 10 * np.log10(np.clip(psd, 1e-7, None))
        
        psds.append(np.array(sample_psds))
    
    return freqs, np.array(psds)

if __name__ == '__main__':
    data = None
    with open('data/EO_EEG_data.pkl', 'rb') as file:
        data = pickle.load(file)

    # print(data['P1222'][0].keys())
    # print(np.shape(eeg_to_psd(data['P1222'][0]['eo'])[1]))

    psd_pickle = {}
    freqs = None
    for sub in data.keys():
        days = []
        for day in range(2):
            signals = {}
            for sequence in data[sub][day].keys():
                freqs, signals[sequence] = eeg_to_psd(data[sub][day][sequence])

            days.append(signals)
        psd_pickle[sub] = days
    
    psd_pickle['freqs'] = freqs

    with open('data/EO_PSD_data.pkl', 'wb') as file:
        pickle.dump(psd_pickle, file)
