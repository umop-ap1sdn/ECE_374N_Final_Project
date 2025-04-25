from n_back_acc import calc_acc_dope
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

if __name__ == '__main__':
    times = ['pre', 'post']
    z = 1.960

    bar_width = 0.15
    bar_sep = 0.12

    for subj in range(1, 5):
        # print(f'Subject 122{i}:')
        accuracies = {}
        dopes = {}

        acc_stats = {}
        dope_stats = {}

        accuracy_CIs = {}
        dope_CIs = {}

        for day in range(1, 3):
            accuracies[day] = {}
            dopes[day] = {}
            acc_stats[day] = {}
            dope_stats[day] = {}
            accuracy_CIs[day] = {}
            dope_CIs[day] = {}
            # print(f'Day {j}:')
            for time in times:
                # print(f'{time} N-Back')
                acc, dope, timeout = calc_acc_dope(subj, day, time)
                accuracies[day][time] = np.mean(acc)
                accuracy_CIs[day][time] = z * np.std(acc) / np.sqrt(5.0)

                dopes[day][time] = np.mean(dope)
                dope_CIs[day][time] = z * np.std(dope) / np.sqrt(5.0)

                timeout_avg = np.mean(timeout)
                timeout_ci = z * np.std(timeout) / np.sqrt(5.0)

                acc_stats[day][time] = acc
                dope_stats[day][time] = dope

            acc_stats[day] = ttest_ind(acc_stats[day]['post'], acc_stats[day]['pre'])
            dope_stats[day] = ttest_ind(dope_stats[day]['post'], dope_stats[day]['pre'])


        day1_acc_m = accuracies[1]['post'] - accuracies[1]['pre']
        day1_dope_m = dopes[1]['post'] - dopes[1]['pre']
        day2_acc_m = accuracies[2]['post'] - accuracies[2]['pre']
        day2_dope_m = dopes[2]['post'] - dopes[2]['pre']
        
        x = np.linspace(-0.5, 1.5, 100)
        day1_acc_line = day1_acc_m * (x - 0) + accuracies[1]['pre']
        day1_dope_line = day1_dope_m * (x - 0) + dopes[1]['pre']
        day2_acc_line = day2_acc_m * (x - 0) + accuracies[2]['pre']
        day2_dope_line = day2_dope_m * (x - 0) + dopes[2]['pre']
        

        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f'Subject 122{subj} N-Back Accuracy and Dope Scores')
        ax[0].set_title('Day 1')
        ax[0].set_xticks(np.arange(2))
        ax[0].set_xticklabels(times)
        ax[0].set_xlabel('N-Back Session')
        ax[0].set_ylim([0, 1])
        ax[0].set_ylabel('Score')
        ax[0].bar(np.arange(2) - bar_sep, accuracies[1].values(), bar_width, yerr=accuracy_CIs[1].values(), capsize=5, label='Accuracy')
        ax[0].bar(np.arange(2) + bar_sep, dopes[1].values(), bar_width, yerr=dope_CIs[1].values(), capsize=5, label='Dope Score')
        ax[0].plot(x, day1_acc_line, label=f'Accuracy Trendline, t={acc_stats[1][0]:.3f} p={acc_stats[1][1]:.3f}')
        ax[0].plot(x, day1_dope_line, label=f'Dope Score Trendline, t={dope_stats[1][0]:.3f} p={dope_stats[1][1]:.3f}')
        ax[0].legend()

        ax[1].set_title('Day 2')
        ax[1].set_xticks(np.arange(2))
        ax[1].set_xticklabels(times)
        ax[1].set_xlabel('N-Back Session')
        ax[1].set_ylim([0, 1])
        ax[1].set_ylabel('Score')
        ax[1].bar(np.arange(2) - bar_sep, accuracies[2].values(), bar_width, yerr=accuracy_CIs[2].values(), capsize=5, label='Accuracy')
        ax[1].bar(np.arange(2) + bar_sep, dopes[2].values(), bar_width, yerr=dope_CIs[2].values(), capsize=5, label='Dope Score')
        ax[1].plot(x, day2_acc_line, label=f'Accuracy Trendline, t={acc_stats[2][0]:.3f} p={acc_stats[2][1]:.3f}')
        ax[1].plot(x, day2_dope_line, label=f'Dope Score Trendline, t={dope_stats[2][0]:.3f} p={dope_stats[2][1]:.3f}')
        ax[1].legend()
        if not os.path.exists('figs'):
            os.mkdir('figs')
        fig.savefig(f'figs/subject122{subj}_accuracy.png', dpi=300)

        plt.show()
        
        

