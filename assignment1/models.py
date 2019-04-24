import csv
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import InstanceDataset

seed = int(time.time())
torch.manual_seed(seed)


def write_run(seed, epoch, value):
    with open("mlp_results2.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([seed, epoch, value])


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


def run_epoch(epoch, datalen, model, dataloader, criterion,
              optimizer: optim.Optimizer = None):
    losses = []
    loss_step = []
    for i, (batch_input, batch_target) in enumerate(dataloader):
        model.zero_grad()
        batch_output = model.forward(batch_input)
        loss = criterion(batch_output.squeeze(), batch_target)
        loss_step.append(i + datalen * epoch)
        losses.append(math.sqrt(loss.item()))

        if optimizer:
            loss.backward()
            # curve
            optimizer.step()

    return loss_step, losses


def save_plot_samples(avg_test_losses, avg_losses):
    plt.plot(*avg_test_losses, label="Test loss")
    plt.plot(*avg_losses, label="Training loss")
    plt.xlabel("Steps")
    plt.ylabel("RMSE loss")
    plt.legend()
    plt.xlim(0)
    plt.savefig("test.png")
    plt.show()


def train():
    dataset = InstanceDataset()
    train_dataset, test_dataset = dataset.get_train_test_split(0.8)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=32,
                                   num_workers=1, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=32,
                                  num_workers=1, drop_last=True)
    epochs = 25
    model = MLP(dataset[0][0].size, num_hidden=100)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    avg_losses = []
    avg_losses_step = []
    avg_test_losses = []
    avg_test_losses_step = []

    for epoch in range(epochs):
        model.train()
        t1 = time.time()
        epoch_loss_step, epoch_loss = run_epoch(epoch, len(train_data_loader),
                                                model, train_data_loader,
                                                criterion, optimizer)

        # epoch_loss = sum(epoch_loss) / len(epoch_loss)
        avg_losses.extend(epoch_loss)
        avg_losses_step.extend(epoch_loss_step)
        t2 = time.time()

        speed = float(t2 - t1)
        print(f"[Epoch {epoch:02d}] "
              f"loss {epoch_loss[-1]:03f} "
              f"speed: {speed:04.2f}s")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        # if epoch % 5 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            # save_plot_samples(avg_losses)
            test_losses_step, test_losses = run_epoch(epoch,
                                                      len(train_data_loader),
                                                      model, test_data_loader,
                                                      criterion)
            avg_test_losses.extend(test_losses)
            avg_test_losses_step.extend(test_losses_step)
            print(f"[Epoch {epoch:02d}] "
                  f"test loss {test_losses[-1]:03f} ")
            # write_run(seed=seed, epoch=epoch, value=test_loss)
        if epoch % 40 == 0:
            torch.save(model.state_dict(),
                       "mlp_epoch{}.pt".format(epoch))

    torch.save(model.state_dict(), "mlp_epoch_last.pt")
    save_plot_samples((avg_test_losses_step, avg_test_losses),
                      (avg_losses_step, avg_losses))
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------


if __name__ == '__main__':
    for i in range(30):
        seed = int(time.time())
        train()
