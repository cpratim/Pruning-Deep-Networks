import torch
import torch.nn as nn
import torch.optim as optim


class FullyConnectedOneLayer(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedOneLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, 28 * 28)
        else:
            x = x.view(-1)
            x = x.unsqueeze(0)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def forward_return_hidden(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        h1 = self.relu(out)
        out = self.fc2(h1)
        out = self.softmax(out)
        return h1, out
    


class FullyConnectedTwoLayer(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedTwoLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)   # Output layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
    def forward_return_hidden(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        h1 = self.relu(out)
        out = self.fc2(h1)
        h2 = self.relu(out)
        out = self.fc3(h2)
        return h1, h2, out
        

def load_fully_connected_two_layer(input_size, hidden_size, num_classes, learning_rate):
    model = FullyConnectedTwoLayer(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def load_fully_connected_one_layer(input_size, hidden_size, num_classes, learning_rate):
    model = FullyConnectedOneLayer(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer
