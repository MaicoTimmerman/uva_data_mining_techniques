from torchesn.nn import ESN
from torchesn.utils import prepare_target
import torch
import numpy as np


def try_ESN (training_set, test_set):

    # prepare target matrix for offline training
    seq_lengths = [len(data) for thing, data in training_set]
    washouts = torch.tensor(np.ones(len(seq_lengths)) * 5, dtype=torch.long)
    training_targets = torch.tensor(np.array([time_series['mood'].values for person_id, time_series in training_set]))
    training_targets = training_targets.unsqueeze(2)
    flat_target = prepare_target(training_targets, seq_lengths, washouts, batch_first=True)

    input_size = training_set[0][1].shape[1]
    n_batches = len(training_set)
    n_hidden = 1000
    n_output = 1
    model = ESN(input_size, n_hidden, n_output, readout_training='cholesky', batch_first=True)

    # accumulate matrices for ridge regression
    # training_array = torch.tensor(np.array([time_series.values for person_id, time_series in training_set]), dtype=torch.long)
    hidden = torch.tensor(np.random.rand(1, 1, n_hidden), dtype=torch.float)
    for person_id, time_series in training_set:
        input = torch.tensor(time_series.values, dtype=torch.float).unsqueeze(0)
        # washout = 5
        # target = torch.tensor(time_series['mood'].values)
        model(input, washouts, hidden, flat_target)

    # train
    model.fit()

    # inference
    input = torch.tensor(test_set[0][1].values, dtype=torch.float).unsqueeze(0)
    output, hidden = model(input, torch.tensor(np.array([5])), hidden)
    target = test_set[0][1]['mood'].values
    print(f"target: {target}\n output: {output}")