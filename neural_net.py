import gzip
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


model = nn.Sequential(
    nn.Linear(784, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Get data from pkl and load into Dataloaders
with gzip.open('mnist_data.pkl.gz', 'rb') as f:
    train_data, test_data = pickle.load(f)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data)

epochs = 10000
for e in range(epochs):

    # Train
    for x, y_true in train_loader:
        output = model(x)
        loss = criterion(output, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    total_loss = 0
    total_correct = 0
    for x, y_true in test_loader:
        x = x.view(x.size(0), -1)
        output = model(x)
        loss = criterion(output, y_true)
        total_loss += loss.item()
        _, predicted = output.max(1)
        total_correct += (predicted == y_true).sum().item()

    # Print report
    print(f"Epoch {e + 1}/{epochs}, "
          f"Loss: {total_loss / len(test_loader):.4f}, "
          f"Accuracy: {total_correct / len(test_loader):.4f}")
