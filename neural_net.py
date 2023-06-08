import gzip
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MNISTNet(nn.Module):

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10))

    def forward(self, x):
        return self.layers(x)


class Trainer:

    def __init__(self, model, train_data, test_data, criterion, optimizer):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer

    def train_network(self, epochs, batch_size, learning_rate):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data)
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        for epoch in range(epochs):
            self.optimize_step(train_loader)
            self.evaluate(test_loader, epoch, epochs)

    def optimize_step(self, train_loader):
        self.model.train()
        for x, y_true in train_loader:
            optimizer.zero_grad()
            output = self.model.forward(x)
            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()

    def evaluate(self, test_loader, epoch, epochs):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        for x, y_true in test_loader:
            with torch.no_grad():
                output = self.model.forward(x)
                loss = self.criterion(output, y_true)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total_correct += (predicted == y_true).sum().item()
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Loss: {total_loss / len(test_loader):.4f}, "
              f"Accuracy: {total_correct / len(test_loader):.4f}")


if __name__ == "__main__":

    model = MNISTNet().to('cuda')

    with gzip.open('mnist_data.pkl.gz', 'rb') as f:
        train_dataset, test_dataset = pickle.load(f)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, train_dataset, test_dataset, criterion, optimizer)

    # Define your criterion and optimizer
    epochs = 6
    batch_size = 128
    learning_rate = 0.01
    trainer.train_network(epochs, batch_size, learning_rate)

