import numpy as np
import torch
import torch.nn as nn
from Crowd_data import Crowd_Dataset
from Network import NeuralNet
from accurancy import binary_acc
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 4
hidden_size = 5
num_classes = 3
num_epochs = 100
batch_size = 1  # don't touch !
learning_rate = 0.001

# Dataset from csv
dataset = Crowd_Dataset(num_classes, input_size)

# Split into train and test set
ratio = 0.25
N = len(dataset)
# TODO : add validation
test_dataset, train_dataset = torch.utils.data.random_split(dataset, [int(ratio*N),int((1-ratio)*N)])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader): # TODO : reduceONplateau, weight decay
        # Move tensors to the configured device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % len(train_dataset) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the training results
y_pred_list = []
label_list = []

y = []
y_true = []
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in train_loader: # only diff with code above
        features = features.to(device)
        labels = labels.to(device)
        y_true.append(labels.int().detach().numpy())  # TODO : adapte for batch size
        #label_list.append(np.where(labels.numpy()==1)[1])
        outputs = model(features)
        outputs = torch.round(outputs)
        y.append(outputs.int().detach().numpy())  # TODO : adapte for batch size
        #y_pred_list.append(np.where(outputs.numpy()==1)[1].item())
        #acc = binary_acc(outputs, labels)

    #print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    #y_pred_list = np.array(y_pred_list).squeeze().tolist()
    #label_list = np.array(label_list).squeeze().tolist()

    conf = multilabel_confusion_matrix(np.array(y_true).squeeze(axis=1), np.array(y).squeeze(axis=1))
    #print(classification_report(label_list, y_pred_list))
    print('trainset result')
    print(conf)
    for i in range(num_classes):
        score = (conf[i, 0, 0]+conf[i, 1, 1])
        elem = conf[i, 0, 0]+conf[i, 1, 1]+conf[i, 1, 0]+conf[i, 0, 1]
        acc = score/elem
        print("accurancy classes ", i, "=", acc)

    # TODO : make more visual output
# Test the model
y_pred_list = []
label_list = []

y = []
y_true = []
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        y_true.append(labels.int().detach().numpy())  # TODO : adapte for batch size
        #label_list.append(np.where(labels.numpy()==1)[1])
        outputs = model(features)
        outputs = torch.round(outputs)
        y.append(outputs.int().detach().numpy())  # TODO : adapte for batch size
        #y_pred_list.append(np.where(outputs.numpy()==1)[1].item())
        #acc = binary_acc(outputs, labels)

    #print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    #y_pred_list = np.array(y_pred_list).squeeze().tolist()
    #label_list = np.array(label_list).squeeze().tolist()

    conf = multilabel_confusion_matrix(np.array(y_true).squeeze(axis=1), np.array(y).squeeze(axis=1))
    # print(classification_report(label_list, y_pred_list))
    print('testset result')
    print(conf)
    for i in range(num_classes):
        score = (conf[i, 0, 0] + conf[i, 1, 1])
        elem = conf[i, 0, 0] + conf[i, 1, 1] + conf[i, 1, 0] + conf[i, 0, 1]
        acc = score / elem
        print("accurancy classes ", i, "=", acc)
    # TODO : make more visual output



# Save the model checkpoint
torch.save(model.state_dict(), 'Checkpoint/model.ckpt') # TODO : save best checkpoint during training (not last one)