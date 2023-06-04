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

    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_network(self, epochs, criterion, optimizer):
        for epoch in range(epochs):
            self.optimize_step(criterion, optimizer)
            self.evaluate(epoch, epochs, criterion)

    def optimize_step(self, criterion, optimizer):
        self.model.train()
        for x, y_true in self.train_loader:
            optimizer.zero_grad()
            output = self.model.forward(x)
            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()

    def evaluate(self, epoch, epochs, criterion):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        for x, y_true in self.test_loader:
            with torch.no_grad():
                output = self.model.forward(x)
                loss = criterion(output, y_true)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total_correct += (predicted == y_true).sum().item()
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Loss: {total_loss / len(self.test_loader):.4f}, "
              f"Accuracy: {total_correct / len(self.test_loader):.4f}")


if __name__ == "__main__":

    model = MNISTNet().to('cuda')

    # Create DataLoaders
    batch_size = 128  # specify your desired batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = Trainer(model, train_loader, test_loader)

    # Define your criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Start training
    trainer.train_network(epochs=10, criterion=criterion, optimizer=optimizer)
