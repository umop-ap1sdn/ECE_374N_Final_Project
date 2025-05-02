import numpy as np
from EO_EEG_Analysis import psd_window_samples
from n_back_acc import calc_acc_dope
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

def accuracy_power_scatter(x_data, y_data, frequencies, plot_title, colors, indeces=None, subj_stim=None):
    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(plot_title)
    row = 0
    col = 0

    for band in frequencies:
        for sub in range(1, 5):
            power_sum = np.zeros(np.shape(x_data[sub])[0])
            
            x_data_small = x_data[sub]

            for i in range(np.shape(x_data[sub])[0]):
                if i < 10 and indeces is not None:
                    if subj_stim[sub][0] == 'Meditation':
                        x_data_small = x_data[sub][:, indeces[0]]
                    else:
                        x_data_small = x_data[sub][:, indeces[1]]
                
                if i >= 10 and indeces is not None:
                    if subj_stim[sub][1] == 'Meditation':
                        x_data_small = x_data[sub][:, indeces[0]]
                    else:
                        x_data_small = x_data[sub][:, indeces[1]]

                # print(np.shape(x_data_small), np.shape(x_data[sub]), np.shape(indeces))

                for j in range(np.shape(x_data_small)[1]):
                    power_sum[i] += x_data_small[i][j][band]
            power_sum /= np.shape(x_data_small)[1]
            
            # print(np.shape(power_sum), np.shape(y_data[sub]), np.shape(x_data_small))

            linreg = LinearRegression()
            linreg.fit(np.reshape(power_sum, (-1, 1)), y_data[sub])
            y_pred = linreg.predict(np.reshape(power_sum, (-1, 1)))

            r2 = r2_score(y_data[sub], y_pred)

            m = linreg.coef_[0]
            b = linreg.intercept_

            x = np.linspace(0, np.max(power_sum), 100)
            line = m * x + b

            ax[row, col].scatter(power_sum, y_data[sub], label=f'P122{sub} (R^2={r2:.2f})', color=colors[sub-1])
            ax[row, col].plot(x, line, color=colors[sub-1])
        ax[row, col].set_xscale('log')
        ax[row, col].set_xlabel(f'{band} Frequency Power (dB)')
        ax[row, col].set_ylabel('N-Back Performance (DOPE)')
        ax[row, col].legend(loc='lower left')

        fig.savefig(f'figs/{plot_title}.png', dpi=300)

        col += 1
        if col == 3:
            col = 0
            row += 1
    plt.show()

if __name__ == '__main__':
    meditation_channels = np.array(['AF3', 'AF4', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'PO3', 'POZ', 'PO4', 'O1', 'O2'])
    tacs_channels = np.array(['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2'])

    top_left_med = ['AF4', 'F4', 'F2', 'FZ',
                    'FC4', 'FC2', 'FCZ',
                    'C4', 'C2', 'CZ']
    
    top_right_med = ['AF3', 'F1', 'F3', 'FZ',
                    'FC3', 'FC1', 'FCZ',
                    'C3', 'C1', 'CZ']
    
    bot_left_med = ['CP4', 'CP2', 'CPZ',
                    'P4', 'P2', 'PZ',
                    'PO4', 'POZ', 'O2']
    
    bot_right_med = ['CP3', 'CP1', 'CPZ',
                    'P3', 'P1', 'PZ',
                    'PO3', 'POZ', 'O1']
    


    top_left_tcs = ['FP2', 'FPZ',
                    'F8', 'F4', 'FZ',
                    'FC6', 'FC2',
                    'T8', 'C4', 'M1']
    
    top_right_tcs = ['FP1', 'FPZ',
                    'F7', 'F3', 'FZ',
                    'FC5', 'FC1',
                    'T7', 'C3', 'M2']
    
    bot_left_tcs = ['O2', 'OZ',
                    'P8', 'P4', 'POZ',
                    'CP6', 'CP2',
                    'T8', 'C4', 'PZ']
    
    bot_right_tcs = ['O1', 'OZ',
                    'P7', 'P3', 'POZ',
                    'CP5', 'CP1',
                    'T7', 'C3', 'PZ']
    
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
    
    '''
    for val in top_left_med:
        print(val, val in meditation_channels)

    indices = [np.where(tacs_channels == val)[0][0] for val in bot_right_tcs]

    print(indices)
    exit()
    '''

    data = None
    with open('data/EO_PSD_data.pkl', 'rb') as file:
        data = pickle.load(file)

    freqs = data['freqs']



    subj_acc = {}
    subj_dope = {}
    subj_psds = {}

    for subj in range(1, 5):
        # print(f'Subject 122{i}:')
        accuracies = None
        dopes = None
        subj_psds[subj] = None

        for day in range(1, 3):
            for time in times:
                # print(f'{time} N-Back')
                acc, dope, timeout = calc_acc_dope(subj, day, time)

                if accuracies is None:
                    accuracies = acc
                else:
                    accuracies = np.concatenate((accuracies, acc), axis=0)
                
                if dopes is None:
                    dopes = dope
                else:
                    dopes = np.concatenate((dopes, dope), axis=0)
        
        subj_acc[subj] = accuracies
        subj_dope[subj] = dopes
        
        subj_psds[subj] = np.array(psd_window_samples(data[f'P122{subj}'][0]['N2pre'], freqs, frequency_bands))
        subj_psds[subj] = np.concatenate((subj_psds[subj], psd_window_samples(data[f'P122{subj}'][0]['N2post'], freqs, frequency_bands)), axis=0)
        subj_psds[subj] = np.concatenate((subj_psds[subj], psd_window_samples(data[f'P122{subj}'][1]['N2pre'], freqs, frequency_bands)), axis=0)
        subj_psds[subj] = np.concatenate((subj_psds[subj], psd_window_samples(data[f'P122{subj}'][1]['N2post'], freqs, frequency_bands)), axis=0)

        print(f'Finished processing Subject P122{subj}')

    
    colors = ['#0d0887', '#7e03a8', '#cc4778', '#f89441']

    '''
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in top_left_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in top_left_tcs])

    accuracy_power_scatter(subj_psds, subj_acc, frequency_bands, 'Subject N-Back Performance vs Front Left Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in top_right_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in top_right_tcs])

    accuracy_power_scatter(subj_psds, subj_acc, frequency_bands, 'Subject N-Back Performance vs Front Right Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in bot_left_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in bot_left_tcs])

    accuracy_power_scatter(subj_psds, subj_acc, frequency_bands, 'Subject N-Back Performance vs Back Left Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in bot_right_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in bot_right_tcs])

    accuracy_power_scatter(subj_psds, subj_acc, frequency_bands, 'Subject N-Back Performance vs Back Right Brainwave Powers', colors,
                           indeces, subj_stim)
    '''



    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in top_left_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in top_left_tcs])

    accuracy_power_scatter(subj_psds, subj_dope, frequency_bands, 'Subject N-Back Performance vs Front Left Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in top_right_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in top_right_tcs])

    accuracy_power_scatter(subj_psds, subj_dope, frequency_bands, 'Subject N-Back Performance vs Front Right Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in bot_left_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in bot_left_tcs])

    accuracy_power_scatter(subj_psds, subj_dope, frequency_bands, 'Subject N-Back Performance vs Back Left Brainwave Powers', colors,
                           indeces, subj_stim)
    
    indeces = []
    indeces.append([np.where(meditation_channels == val)[0][0] for val in bot_right_med])
    indeces.append([np.where(tacs_channels == val)[0][0] for val in bot_right_tcs])

    accuracy_power_scatter(subj_psds, subj_dope, frequency_bands, 'Subject N-Back Performance vs Back Right Brainwave Powers', colors,
                           indeces, subj_stim)




