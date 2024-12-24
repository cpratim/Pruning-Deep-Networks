import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def test(model, test_loader):

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

