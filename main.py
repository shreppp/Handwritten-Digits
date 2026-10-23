import torch # pytorch
import torch.nn as nn # torch.nn is the module of pytorch used for creating neural networks
import torch.optim as optim # the optimizer function
import torch.nn.functional as F # used for the activation functions (f.relu)
from torch.utils.data import DataLoader # dataloader basically makes it easier to iterate over datasets and allows for things like batch sizes, shuffle, etc and feeds it into the nn
import torchvision.datasets as datasets # torchvision datasets gives access to built in datasets like mnist
import torchvision.transforms as transforms # transforms data into tensors to be fed into the nn

print("hello world")

class Net(nn.Module): # the cnn class that contains its architecture and forward pass.
    def __init__(self, in_channels=1, num_classes=10): # in channels = 1 because its grayscale (it would be 3 with rgb and 4 with opacity) num_classes = 10 because there's 10 output neurons and 10 classes that each image can be classified as
        super().__init__() # super.__init__() because it needs to inherit the properties of the parent class nn.module. it can also be written as super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # defines the first conv layer
        # Same convolutions -> output -> same as input dimension

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # max pooling keeps the largest activation in each kernel so it decreases the size of the feature map
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # for conv1 there were 8 out channels so theres 8 in channels on conv2 and 16 out channels meaning that theres 16 features total on the feature map

        self.fc1 = nn.Linear(16*7*7, num_classes) # ok so i do not understand ts: in fc1 -> 16 bcoz that is outchannel in conv2 and 7*7 because -> 2 poolings will make it (28/2)/2 => 7

    def forward(self, x): # defines forward pass (dont call directly, call it with Net())
        x = F.relu(self.conv1(x)) # first conv layer with ReLU as the activation function
        x = self.pool(x) # first pooling layer
        x = F.relu(self.conv2(x)) # second conv layer with ReLU as the activation function
        x = self.pool(x) # second pooling layer
        x = x.reshape(x.shape[0], -1) # flattens it from a stack of feature maps to one long vector per image
        x = self.fc1(x) # fully connected dense layer that is connected to the output layer and makes predictions

        return x

batch_size = 64 # number of training images the model has a forward and backward pass (forward prop and backprop) over before updating the weights (gradient descent) after averaging the gradients across the batch

# the next four lines just load the data set, the dataset is imported from torchvision and the loader is pytorch's dataloader.
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# device uses cuda if nvidia gpus are available but otherwise uses the cpu to train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1  # in channels determine if the image is grayscale, rgb, or rgba (1, 3, and 4 channels respectively) idk what 2 channels is i think its grayscale with transparency
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 500

# Initialize Network
model = Net(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)  # => [64 , 1, 28, 28] => 64 : num_images, 1 -> num_channels, (28,28): (height, width)
        # No need to flatten
        # data = data.reshape(data.shape[0], -1)  # Flatten

        # forward
        scores = model(data)

        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # set all gradients to zero for each batch
        loss.backward()

        # gradient descent
        optimizer.step()


def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on testing data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # dont compute gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)

            # print(predictions)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100: .2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
