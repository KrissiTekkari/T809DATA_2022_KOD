import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('C:\gagnanam_datasets', download=False, train=True, transform=transform)
valset = datasets.MNIST('C:\gagnanam_datasets', download=False, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, 128),
                      nn.SELU(),
                      nn.Linear(128, 128),
                      nn.Tanh(),
                      nn.Linear(128, 64),
                      nn.Tanh(),
                      nn.Linear(64, output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


### found this plotting code on the internet, but did not write down where from ###

def out(model, images):
    images = images.view(images.shape[0], -1)
    output = model(images)
    top_p, top_class = output.topk(1, dim=1)
    return top_p, top_class

def plot_misclassifications(model, test):
    misclassifications = np.zeros(10)
    for images, labels in test:
        top_p, top_class = out(model, images)
        equals = top_class == labels.view(*top_class.shape)
        for i in range(len(labels)):
            if not equals[i]:
                misclassifications[labels[i]] += 1
    misclassifications = misclassifications / len(test.dataset)
    #plt.plot(misclassifications)
    plt.bar(np.arange(10), misclassifications, color='darkred')
    
    plt.xticks(np.arange(10))
    plt.xlabel('digit')
    plt.ylabel('misclassification rate')
    plt.show()

plot_misclassifications(model, valloader)

def confusion_matrix_print(model, test):
    confusion_matrix = np.zeros((10,10))
    for images, labels in test:
        top_p, top_class = out(model, images)
        for i in range(len(labels)):
            confusion_matrix[labels[i], top_class[i]] += 1
    confusion_matrix = confusion_matrix.astype(int)
    print('    0   1   2   3   4   5   6   7   8   9')
    for i in range(10):
        print(i, confusion_matrix[i])
        
        
confusion_matrix_print(model, valloader)

def error(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            top_p, top_class = out(model, images)
            equals = top_class == labels.view(*top_class.shape)
            correct += torch.sum(equals.type(torch.FloatTensor))
            total += labels.size(0)
    print('Error: ', (1-float(correct/total))*100, '%')

error(model, valloader)