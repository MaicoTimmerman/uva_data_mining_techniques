from scipy import stats
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASELINE = 0.745

ESN_errors = []
with open('ESN_100_runs_test_RMSE.csv', newline='') as csvfile:
    daan_reader = csv.reader(csvfile, delimiter=';')
    for row in daan_reader:
        temp = [float(x) for x in row]
        ESN_errors.append(temp)

ESN_errors = np.array(ESN_errors)

echo_state_N = [1,2,5,10,20,50,100,200]


ESN_one_sample = stats.ttest_1samp(ESN_errors, BASELINE)
ESN_descriptives = stats.describe(ESN_errors)

fig, subplot_axes = plt.subplots(3, 3, sharey='all', sharex='all')

for i, N in enumerate(echo_state_N):

    shapriro_p = stats.shapiro(ESN_errors[:,i])[1]
    effect_size = ESN_descriptives.mean[i] - BASELINE

    print("One sample t-test for ESN (nodes = {:d}): Baseline={:.3f}: P-Value of {:.3f}, Effect size: {:.3f}".format(
                N,
                BASELINE,
                ESN_one_sample.pvalue[i],
                effect_size))
    print("Descriptive statistics: n={:d}, Mean={:.3f}, Variance={:.3f}, Shapiro-P={:.3f}".format(
                ESN_descriptives.nobs,
                ESN_descriptives.mean[i],
                ESN_descriptives.variance[i],
                shapriro_p))
    print("\n\n")

    row = i % len(subplot_axes)
    column = i // len(subplot_axes)

    sns.set(color_codes=True)
    sns.distplot(ESN_errors[:,i], label='ESN errors nodes = {}'.format(N), ax=subplot_axes[row,column])



MLP_errors = []

with open('mlp_results.csv', newline='') as csvfile:
    maico_reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(maico_reader):
        if (i == 0):
            continue
        if (int(row[1]) == 249):
            MLP_errors.append(float(row[-1]))


mlp_one_sample = stats.ttest_1samp(MLP_errors, BASELINE)
mlp_descriptives = stats.describe(MLP_errors)
shapriro_p = stats.shapiro(MLP_errors)[1]
effect_size = mlp_descriptives.mean - BASELINE

print("One sample t-test for ESN. Baseline={:.3f}: P-Value of {:.3f}, Effect size: {:.3f}".format(
            BASELINE,
            mlp_one_sample.pvalue,
            effect_size))

print("Descriptive statistics: n={:d}, Mean={:.3f}, Variance={:.3f}, Shapiro-P={:.3f}".format(
            mlp_descriptives.nobs,
            mlp_descriptives.mean,
            mlp_descriptives.variance,
            shapriro_p))

# stats.probplot(MLP_errors, plot=plt)
sns.distplot(MLP_errors, label='MLP errors',ax = subplot_axes[-1,-1])

# figure = plt.figure(1, clear=True)
# plt.subplot(331)


fig, subplot_axes
for row in subplot_axes:
    for axe in row:
        bot, top = axe.get_ylim()
        # axe.vlines(0.745, bot, top, color='red',linestyles='dashed', label='Baseline')
        axe.axvline(0.745, color='red', linestyle='--', label='Baseline')
        # axe.set_ylabel('Count')
        axe.set_xlabel('RMSE')
        axe.legend()
# axe = plt.gca()
# bot, top = axe.get_ylim()
# axe.vlines(0.745, bot, top, color='red',linestyles='dashed', label='Baseline')
# axe.set_ylabel('Count')
# axe.set_xlabel('RMSE')
plt.show()


# print(ESN_errors[:,3])
