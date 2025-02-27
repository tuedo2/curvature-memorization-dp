import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet import Bottleneck, ResNet50

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

train = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train) # change flag to True to download
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

# test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)


if __name__ == '__main__':

    net = ResNet50(100).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Loss [{epoch+1}, {i+1}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0
        
        if (epoch+1) % 10 == 0:
            path = f'./models/epoch_{epoch+1}.pth'
            torch.save(net.state_dict(), path)