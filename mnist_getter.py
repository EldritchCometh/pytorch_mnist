import gzip
import torch
import pickle
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
features = mnist.data
labels = mnist.target

# Convert to list
features = features.values.tolist()
labels = labels.values.tolist()

# Normalize features and convert labels to integers
features = [[float(f)/255 for f in x] for x in features]
labels = [int(y) for y in labels]

# Split into testing and training samples
X_train, y_train = features[:60000], labels[:60000]
X_test, y_test = features[60000:], labels[60000:]

# Convert the features and labels to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')

# Create TensorDatasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Dump it to the pickle
data = (train_data, test_data)
with gzip.open("mnist_data.pkl.gz", 'wb') as f:
    pickle.dump(data, f)