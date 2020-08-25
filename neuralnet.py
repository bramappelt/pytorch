import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# define convolutional neural network
class Net(nn.Module):
    def __init__(self, n=6):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, train_loader, nloops=2, gpu=False, verbose=True, filename=None):
    # set hardware device
    if torch.cuda.is_available() and gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device)

    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network (on the CPU)
    for epoch in range(nloops):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the data in a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999 and verbose:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    if verbose:
        print('Finished Training')

    if filename is not None:
        PATH = f"./models/cifarnet10_{filename}{'g' if gpu else 'c'}.pth"
        torch.save(net.state_dict(), PATH)

    return net


def verify(net, test_loader, classes, verbose=True):
    # infer device from the model
    device = next(net.parameters()).device.type

    # test the neural net for the complete dataset and split results per class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    performance = {}
    overall = 0.0
    nclasses = len(classes)
    for i in range(nclasses):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        performance[classes[i]] = class_accuracy
        overall += class_accuracy * 1 / nclasses

    if verbose:
        for i in range(nclasses):
            print('Accuracy of %5s : %2d %%' % (
                   classes[i], performance[classes[i]]))
        print('-' * 25)
        print(f"Overall accuracy: {overall:2.0f} %")

    performance['overall'] = overall
    return performance
