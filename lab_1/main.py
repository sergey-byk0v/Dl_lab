import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from conv import Net


def accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            output = net(images)
            _, prediction = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    return correct / total


def save_history(test, train, label):
    plt.plot(test)
    plt.plot(train)
    plt.legend(['test', 'train'])
    plt.ylabel(label, fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.grid()
    plt.savefig(label + '.png');


if __name__ == "__main__":
	transform = transforms.Compose([transforms.ToTensor(),
		                       transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
	test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=False)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.0001)

	train_losses = []
	test_losses = []
	accuracy_test = []
	accuracy_train = []
	for epoch in range(50):

	    running_loss = 0.0
	    test_loss = 0.
	    for data in trainloader:
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

	    for data in testloader:
		inputs, labels = data
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		test_loss += loss.item()

	    train_losses.append(running_loss / len(trainloader))
	    test_losses.append(test_loss / len(testloader))
	    test_accuracy = accuracy(testloader, net)
	    train_accuracy = accuracy(trainloader, net)
	    accuracy_test.append(test_accuracy)
	    accuracy_train.append(train_accuracy)

	    print(f"epoch - {epoch}, train_loss - {running_loss / len(trainloader)}, test_loss - {test_loss / len(testloader)}, train - {train_accuracy}, test - {test_accuracy}")
	
	save_history(accuracy_test, accuracy_train, 'Accuracy')
	save_history(test_losses, train_losses, 'Loss')


