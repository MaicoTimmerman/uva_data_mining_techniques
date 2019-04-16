import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import InstanceDataset


class MLP(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, 1))

    def forward(self, input):
        return self.model(input)


def save_plot_samples(train_curve):
    print(train_curve)
    plt.plot(train_curve)
    plt.show()


def run_epoch(model, dataloader, criterion, optimizer: optim.Optimizer = None):
    losses = []
    for batch_input, batch_target in dataloader:
        model.zero_grad()
        batch_output = model.forward(batch_input)
        loss = criterion(batch_output.squeeze(), batch_target)
        losses.append(loss.item())

        if optimizer:
            loss.backward()
            # curve
            optimizer.step()

    return losses


def train():
    dataset = InstanceDataset()
    train_dataset, test_dataset = dataset.get_train_test_split(0.8)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=32,
                                   num_workers=1, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=32,
                                  num_workers=1, drop_last=True)
    epochs = 250
    model = MLP(dataset[0][0].size, num_hidden=50)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    avg_losses = []

    for epoch in range(epochs):
        model.train()
        t1 = time.time()
        avg_losses.extend(
            run_epoch(model, train_data_loader, criterion, optimizer))
        t2 = time.time()

        speed = float(t2 - t1)
        print(f"[Epoch {epoch:02d}] "
              f"loss {math.sqrt(avg_losses[-1]):03f} "
              f"speed: {speed:04.2f}s")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                # save_plot_samples(avg_losses)
                test_losses = run_epoch(model, test_data_loader, criterion)
                test_loss = math.sqrt(sum(test_losses) / len(test_losses))
                print(f"[Epoch {epoch:02d}] "
                      f"test loss {test_loss:03f} ")
        if epoch % 40 == 0:
            torch.save(model.state_dict(),
                       "mlp_epoch{}.pt".format(epoch))

    torch.save(model.state_dict(), "mlp_epoch_last.pt")
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------


if __name__ == '__main__':
    train()
