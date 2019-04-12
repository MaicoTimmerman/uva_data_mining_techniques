import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def train():
    dataset = InstanceDataset()
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1,
                             drop_last=True)

    epochs = 25
    model = MLP(dataset[0][0].size, num_hidden=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e4)
    criterion = nn.MSELoss()

    train_curve, val_curve = [], []
    for epoch in range(epochs):
        model.train()
        t1 = time.time()
        for batch_input, batch_target in data_loader:
            batch_output = model.forward(batch_input)
            loss = criterion(batch_output.squeeze(), batch_target)
            loss.backward()
            # curve
            train_curve.append(loss.item())
            optimizer.step()
        t2 = time.time()

        speed = float(t2 - t1)
        print(f"[Epoch {epoch:02d}] loss {loss:03f} speed: {speed:04.2f}s")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                save_plot_samples(train_curve)
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
