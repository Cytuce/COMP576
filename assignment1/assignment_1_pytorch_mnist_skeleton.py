import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

# Hyperparameters
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.9
try_cuda = True
seed = 1000
logging_interval = 100
logging_dir = "runs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logging_dir)

# Setting up the device
cuda = torch.cuda.is_available() and try_cuda
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Data Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Data Loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=test_batch_size, shuffle=False)

# Modified Network with LeakyReLU activation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # Conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # Conv2
        self.conv2_drop = nn.Dropout2d()  # Dropout
        self.fc1 = nn.Linear(1024, 1024)  # Fully connected
        self.fc2 = nn.Linear(1024, 10)    # Output layer

        # Initialize weights using Xavier
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # First layer: Conv + LeakyReLU + MaxPool
        x = self.conv1(x)
        net_input1 = x.clone()
        x = F.leaky_relu(F.max_pool2d(x, 2))
        activation1 = x.clone()

        # Second layer: Conv + LeakyReLU + MaxPool + Dropout
        x = self.conv2_drop(self.conv2(x))
        net_input2 = x.clone()
        x = F.leaky_relu(F.max_pool2d(x, 2))
        activation2 = x.clone()

        x = x.view(-1, 1024)  # Flatten
        # Fully connected layers
        net_input3 = x.clone()
        x = F.leaky_relu(self.fc1(x))
        activation3 = x.clone()

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1), (net_input1, net_input2, net_input3), (activation1, activation2, activation3)

model = Net().to(device)

# Optimizer: SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.NLLLoss()

# Training function with tensorboard logging
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, net_inputs, activations = model(data)
        loss = criterion(torch.log(output + 1e-13), target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

            # Log statistics for each layer
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/weights', param.clone().cpu().data.numpy(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'{name}/min_weight', param.min().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'{name}/max_weight', param.max().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'{name}/mean_weight', param.mean().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'{name}/std_weight', param.std().item(), epoch * len(train_loader) + batch_idx)

            # Log net inputs and activations
            for idx, (net_input, activation) in enumerate(zip(net_inputs, activations)):
                writer.add_histogram(f'NetInput/layer{idx+1}', net_input.cpu().data.numpy(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'NetInput/layer{idx+1}/min', net_input.min().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'NetInput/layer{idx+1}/max', net_input.max().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'NetInput/layer{idx+1}/mean', net_input.mean().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'NetInput/layer{idx+1}/std', net_input.std().item(), epoch * len(train_loader) + batch_idx)

                writer.add_histogram(f'Activations/layer{idx+1}', activation.cpu().data.numpy(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Activations/layer{idx+1}/min', activation.min().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Activations/layer{idx+1}/max', activation.max().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Activations/layer{idx+1}/mean', activation.mean().item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Activations/layer{idx+1}/std', activation.std().item(), epoch * len(train_loader) + batch_idx)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            test_loss += criterion(torch.log(output + 1e-13), target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({test_accuracy:.0f}%)\n')
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

# Training and testing over epochs
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

writer.close()