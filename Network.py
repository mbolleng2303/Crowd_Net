import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        #self.batchnorm1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(1,hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.batchnorm3 = nn.BatchNorm1d(1,num_classes)

    def forward(self, x):
        #x = self.batchnorm1(x)
        out = self.fc1(x.float())
        out = self.relu(out)
        out = self.batchnorm2(out.t())
        out = self.fc2(out.t())
        out = self.batchnorm3(out.t())
        out = torch.softmax(out.t(), dim = 1)
        return out