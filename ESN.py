from torchesn.nn import ESN
from torchesn.utils import prepare_target
import torch
import numpy as np


def try_ESN (training_set, test_set):

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
    n_hidden = 100
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

    # inference training
    input = torch.tensor(training_set[0][1].drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0)
    output, hidden = model(input, torch.tensor(np.array([5])), hidden)
    target = training_set[0][1]['mood_interpolated'].values
    print(f"target: {target}\n output: {output}")
    import matplotlib.pyplot as plt
    plt.figure(10)
    plt.title("training set example")
    plt.plot(target.squeeze()[5:])
    plt.plot(output.squeeze().detach().numpy())
    plt.show(block=False)

    # inference test
    input = torch.tensor(test_set[0][1].drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0)
    output, hidden = model(input, torch.tensor(np.array([5])), hidden)
    target = test_set[0][1]['mood_interpolated'].values
    print(f"target: {target}\n output: {output}")
    import matplotlib.pyplot as plt
    plt.figure(11)
    plt.title("test set example")
    plt.plot(target.squeeze()[5:])
    plt.plot(output.squeeze().detach().numpy())
    plt.show(block=True)


def try_ESN_single (training_set, test_set):
    training_set = training_set[0]
    washout = 5
    x = torch.tensor(training_set[1]['mood_interpolated'].values / 10.0)

    flat_target = prepare_target(x, 5, washout, batch_first)

    input_size = training_set[0][1].shape[1] - 1
    n_hidden = 1000
    n_output = 1
    model = ESN(input_size, n_hidden, n_output)

    # accumulate matrices for ridge regression
    # training_array = torch.tensor(np.array([time_series.values for person_id, time_series in training_set]), dtype=torch.long)
    hidden = torch.tensor(np.random.rand(1, 1, n_hidden), dtype=torch.float)
    for person_id, time_series in training_set:
        input = torch.tensor(time_series.drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0) / 10.0
        # washout = 5
        # target = torch.tensor(time_series['mood'].values)
        model(input, washouts, hidden, flat_target)

    # train
    model.fit()

    # inference
    input = torch.tensor(training_set[0][1].drop('mood', axis=1).values, dtype=torch.float).unsqueeze(0) / 10.0
    output, hidden = model(input, torch.tensor(np.array([5])), hidden)
    target = training_set[0][1]['mood'].values / 10.0
    print(f"target: {target}\n output: {output}")
    import matplotlib.pyplot as plt
    plt.figure(10)
    plt.plot(target.squeeze()[5:])
    plt.plot(output.squeeze().detach().numpy())
    plt.show(block=True)