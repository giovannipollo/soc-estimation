import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer with 1 input feature and 10 hidden units
        self.fc2 = nn.Linear(10, 1)  # Output layer with 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the hidden layer
        x = self.fc2(x)  # Output layer without activation
        return x

# Create the neural network
net = Net()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Generate training data
x_train = torch.unsqueeze(torch.linspace(-3000, 5000, 80), dim=1)  # Input data
y_train = -x_train**2  # Target data

# Training loop
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(x_train)
    loss = criterion(outputs, y_train)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the results
x_test = torch.unsqueeze(torch.linspace(-3000, 5000, 20), dim=1)
y_pred = net(x_test)
plt.scatter(x_train, y_train, label='Training Data')
plt.plot(x_test, y_pred.detach().numpy(), 'r', label='Model Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Approximation of -x^2')
plt.savefig('xsquared.png')
