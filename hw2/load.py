from sklearn.utils import shuffle
import torch
import torchvision
import sklearn.metrics as metrics
import numpy as np
import sys

from torch.utils.data import Dataset, random_split
import student
from config import device


def test_network(net,testloader,print_confusion=True):
    net.eval()
    total_images = 0
    total_correct = 0
    conf_matrix = np.zeros((8,8))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                labels.cpu(),predicted.cpu(),labels=[0,1,2,3,4,5,6,7])
    model_accuracy = total_correct / total_images * 100
    print('{0} test {1:.2f}%'.format(total_images,model_accuracy))
    if print_confusion:
        np.set_printoptions(precision=2, suppress=True)
        print(conf_matrix)



if __name__ == '__main__':
    model = student.net.to(device)
    model.load_state_dict(torch.load("checkModel_old.pth"))
    data = torchvision.datasets.ImageFolder(root='./data', transform=student.transform('test'))
    testloader = torch.utils.data.DataLoader(data, batch_size=student.batch_size, shuffle=True)
    test_network(model, testloader)

