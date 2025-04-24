import os
import numpy as np
from pyxdf import load_xdf

def calc_acc_dope(sub, day, time, dataset_base='data/Group 12'):
    path = os.path.join(dataset_base, f'sub-P122{sub}/ses-S00{day}N2{time}/eeg')
    
    accuracies = []
    dopes = []
    timeout_rate = []

    for file in os.listdir(path):
        streams, headers = load_xdf(os.path.join(path, file))
        trials = 0
        correct = 0
        timeouts = 0
        dope = 1
        start_timestamp = 0

        stream_num = 0
        while len(streams[stream_num]['time_series']) < 100:
            stream_num += 1

        for marker in streams[stream_num]['time_series']:
            if marker[0] == 0.0:
                trials += 1
                start_timestamp = marker[1]
            
            if marker[0] == 1.0:
                correct += 1
                dope *= 1.0 / (((marker[1] - 0.25) - start_timestamp) * 1)

            if marker[0] == 2.0 or marker[0] == 300.0:
                dope *= 0.01 / (((marker[1] - 0.25) - start_timestamp) * 1)

            if marker[0] == 300.0:
                timeouts += 1

            

        
        # print(trials)
        accuracies.append(correct / trials)
        dopes.append(float(dope ** (1.0 / trials)))
        timeout_rate.append(float(timeouts / trials))
    
    return accuracies, dopes, timeout_rate

if __name__ == '__main__':
    times = ['pre', 'post']
    z = 1.960
    for i in range(1, 5):
        print(f'Subject 122{i}:')
        for j in range(1, 3):
            print(f'Day {j}:')
            for time in times:
                print(f'{time} N-Back')
                acc, dope, timeout = calc_acc_dope(i, j, time)
                acc_avg = np.mean(acc)
                acc_ci = z * np.std(acc) / np.sqrt(5.0)

                dope_avg = np.mean(dope)
                dope_ci = z * np.std(dope) / np.sqrt(5.0)

                timeout_avg = np.mean(timeout)
                timeout_ci = z * np.std(timeout) / np.sqrt(5.0)

                print(f'N-Back Accuracies: {acc}, Avg: {acc_avg:.3f} +- {acc_ci:.3f}')
                print(f'N-Back Dope Scores: {dope}, Avg: {dope_avg:.3f} +- {dope_ci:.3f}')
                print(f'N-Back Timeout Rates: {timeout}, Avg: {timeout_avg:.3f} +- {timeout_ci:.3f}')
                
            
        print()
        
            
