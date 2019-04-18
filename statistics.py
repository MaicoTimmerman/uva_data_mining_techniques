from scipy import stats
import csv
import numpy as np

ESN_errors = []

with open('ESN_100_runs_test_RMSE.csv', newline='') as csvfile:
    daan_reader = csv.reader(csvfile, delimiter=';')
    for row in daan_reader:
        ESN_errors.append([float(x) for x in row])

# ESN_errors

# print(ESN_errors)

# rvs = stats.norm.rvs(loc=5, scale=10, size=(50))

# print(rvs)

echo_state_N = [1,2,5,10,20,50,100,200]


ESN_one_sample = stats.ttest_1samp(ESN_errors, 0.745)
ESN_descriptives = stats.describe(ESN_errors)

for i, N in enumerate(echo_state_N):
    shapriro_p = stats.shapiro(ESN_errors[:][i])[1]
    effect_size = ESN_descriptives.mean[i] - 0.745

    print("One sample t-test for ESN. nodes = {:d}, Baseline=0.745: P-Value of {:.3f}, Effect size: {:.3f}".format(
                N,
                ESN_one_sample.pvalue[i],
                effect_size))
    print("Descriptive statistics: n={:d}, Mean={:.3f}, Variance={:.3f}, Shapiro-P={:.3f}".format(
                ESN_descriptives.nobs,
                ESN_descriptives.mean[i],
                ESN_descriptives.variance[i],
                shapriro_p))
    print("\n\n")



MLP_errors = []

with open('mlp_results.csv', newline='') as csvfile:
    maico_reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(maico_reader):
        if (i == 0):
            continue
        if (int(row[1]) == 249):
            MLP_errors.append(float(row[-1]))
        # print(row[2])

# print(MLP_errors)

mlp_one_sample = stats.ttest_1samp(MLP_errors, 0.745)
mlp_descriptives = stats.describe(MLP_errors)
shapriro_p = stats.shapiro(MLP_errors)[1]
effect_size = mlp_descriptives.mean - 0.745

print("One sample t-test for ESN. Baseline=0.745: P-Value of {:.3f}, Effect size: {:.3f}".format(
            mlp_one_sample.pvalue,
            effect_size))

print("Descriptive statistics: n={:d}, Mean={:.3f}, Variance={:.3f}, Shapiro-P={:.3f}".format(
            mlp_descriptives.nobs,
            mlp_descriptives.mean,
            mlp_descriptives.variance,
            shapriro_p))




# mean = np.mean(MLP_errors)
# variance = np.var(MLP_errors)
# effect_size = np.mean(MLP_errors)-0.745

# print("Descriptive statistics: Mean {} , Variance {} , Effect size {}".format(mean, variance, effect_size))
