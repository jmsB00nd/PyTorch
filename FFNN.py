#import needed functions
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

#Neural Network Class
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
#setting the device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameter of the model
input_size = 784
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 3

#load the data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#init the network

model = NN(input_size, num_classes)
model.to(device=device)
#loss and optimizer
CE = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#train the network

for epoch in range(num_epochs):
    for batch_idx , (data, targets) in enumerate(train_loader):
        #to device
        data = data.to(device=device)
        targets = targets.to(device=device)
        #reshape
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = CE(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #adam step
        optimizer.step()


def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
