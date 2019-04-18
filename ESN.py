from torchesn.nn import ESN
from torchesn.utils import prepare_target
import torch
import numpy as np


def try_ESN (training_set, test_set, mood_mean, mood_min, mood_max):

    # prepare target matrix for offline training
    seq_lengths = [len(data) for thing, data in training_set]
    washouts = torch.tensor(np.ones(len(seq_lengths)) * 5, dtype=torch.long)
    # x = [time_series['mood_interpolated'].values for person_id, time_series in training_set]
    max_sequence_length = max([len(time_series) for person_id, time_series in training_set])
    training_targets = torch.zeros(len(training_set), max_sequence_length)
    for i, (person_id, time_series) in enumerate(training_set):
        sequence_length = len(time_series)
        training_targets[i, 0:sequence_length] = torch.tensor(np.array(time_series['mood_interpolated']))
    training_targets = training_targets.unsqueeze(2)
    flat_target = prepare_target(training_targets, seq_lengths, washouts, batch_first=True)

    input_size = training_set[0][1].shape[1] - 1
    n_hidden = 10
    n_output = 1
    model = ESN(input_size, n_hidden, n_output, readout_training='cholesky', batch_first=True)

    # accumulate matrices for ridge regression
    # training_array = torch.tensor(np.array([time_series.values for person_id, time_series in training_set]), dtype=torch.long)
    hidden = torch.tensor(np.random.rand(1, 1, n_hidden), dtype=torch.float)
    for person_id, time_series in training_set:
        input = torch.tensor(time_series.drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0)
        # washout = 5
        # target = torch.tensor(time_series['mood'].values)
        model(input, washouts, hidden, flat_target)

    # train
    model.fit()

    def get_RMSE (set, mood_mean, mood_min, mood_max):
        E = np.array([0])
        for i in range(len(set)):
            # Do inference
            washout_length = 5
            df = set[i][1]
            input = torch.tensor(df.drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0)
            output, _ = model(input, torch.tensor(np.array([washout_length])), hidden)

            # Calculate E
            predictions = output.detach().numpy().squeeze()
            predictions = predictions * (mood_max-mood_min) + mood_mean
            targets = df['mood_interpolated'].values[washout_length:]
            targets = targets * (mood_max-mood_min) + mood_mean
            true_targets = df['mood'].values[washout_length:]
            errors = predictions - targets
            # Select only errors where true target exists
            errors = errors[~np.isnan(true_targets)]
            E = np.append(E, errors, axis=0)

        draw_plot = False
        if draw_plot:
            import matplotlib.pyplot as plt
            plt.figure(i)
            plt.title("training set example")
            plt.plot(targets, label='target')
            plt.plot(predictions, label='prediction')
            plt.legend()
            plt.show(block=True)

        # Calculate RMSE
        SE = np.square(E)
        MSE = np.mean(SE)
        RMSE = np.sqrt(MSE)

        return RMSE

    # inference training set
    print(f"ESN RMSE training set: {get_RMSE(training_set, mood_mean, mood_min, mood_max)}")
    print(f"ESN RMSE test set: {get_RMSE(test_set, mood_mean, mood_min, mood_max)}")


